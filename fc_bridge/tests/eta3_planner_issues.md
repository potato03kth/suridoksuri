# eta3 플래너 문제 분석: curvature 허위 값 & heading 진동

대상 파일: `vtol_sim_checkpoint1_1/vtol_sim/path_planning/eta3clothoid_v3_1_planner.py`  
발견 경위: `fc_bridge/run_phase1.py --save-plot` 결과 이미지(`eta3_05041422.png`) 분석

---

## 1. 플래너 파이프라인 구조

```
waypoints_ned
  │
  ▼
[Stage 0] _insert_wps_if_infeasible()   ← κ 초과 구간에 중간 WP 삽입 (최대 4회)
  │
  ▼
[Stage 1] _solve_g2_nr()                ← 전역 Newton-Raphson, G2 연속 해 탐색
  │         θ (inner), κ (inner, tanh 변환), L_seg (exp 변환) 최적화
  │         위치 잔차(m) + 헤딩 잔차(rad) 동시 최소화
  ▼
_clothoid_sample()   per-segment         ← 해석적 clothoid 생성
  │   κ(s) = κ_i + (κ_j−κ_i)/L · s  (선형 보간)
  │   θ(s) = θ_i + κ_i·s + ½·(Δκ/L)·s²
  ▼
affine 보정 (per-segment)                ← 끝점 잔차를 선형 분배
  │   err = target_end − clothoid_end
  │   correction(s) = (s/L)·err
  ▼
pts_arr concat → s_arr → PathPoint 생성
```

---

## 2. 문제 1: curvature 필드가 실제 경로 기하와 무관했음 (수정 완료)

### 원인

`all_kappa`에 누적된 `seg_kp`는 `_clothoid_sample()`이 반환하는 **설계 κ** 배열이다.
`seg_kp[i] = κ_i + (κ_j−κ_i)/L_seg · s[i]` — affine 보정 **이전** 값.

affine 보정은 **위치**만 이동시키며, `kappa_arr`에는 반영되지 않는다.

### 영향

NR 수렴 실패 → affine 보정량 폭발 → 실제 경로 기하학적 κ >> 저장된 설계 κ.

하위 소비자가 받는 curvature:

| 소비자 | 사용 방식 | 영향 |
|---|---|---|
| `fc_bridge/run_phase2.py` | 속도 프로필 계산 | 실제보다 낮은 κ → 과속 허용 |
| `mpc_controller._get_preview_curvatures()` | MPC 피드포워드 | 횡방향 제어력 과소 추정 |
| `vtol_sim/metrics.py` | 횡방향 가속도 a_n 계산 | 메트릭 허위 |

### 수정 내용

`pts_arr`와 `s_arr` 확정 후, 실제 보정된 위치로 곡률을 유한차분 재계산:

```python
_dN = np.gradient(pts_arr[:, 0], s_arr)
_dE = np.gradient(pts_arr[:, 1], s_arr)
_d2N = np.gradient(_dN, s_arr)
_d2E = np.gradient(_dE, s_arr)
_spd = np.sqrt(np.maximum(_dN**2 + _dE**2, 1e-24))
kappa_arr = (_dN * _d2E - _dE * _d2N) / _spd**3
```

`iterpin_planner._compute_curvature()`와 동일 공식. `all_kappa` 리스트 전체 제거.

---

## 3. 문제 2: NR 발산 — WP3→WP4 구간

### 기하학적 설정

```
WP2=(N=200, E=200)  →  WP3=(N=0, E=200)  →  WP4=(N=0, E=0)
방향 WP2→WP3: (−N) = θ≈π  (south)
방향 WP3→WP4: (−E) = θ≈−π/2  (west)
```

### 헤딩 충돌

NR 잔차 함수에서 헤딩 연속성 조건:
```
F[3k+2] = w_head · wrap(θ_end_of_seg_k − θ_{k+1})
```

- `w_head = min(mean_chord, 50) = 50` (강한 가중치)
- WP2→WP3 세그먼트가 수렴하면 θ_WP3 ≈ π (south)로 고정됨
- WP3→WP4 세그먼트는 θ_WP3 = −π/2 (west)가 필요함
- 두 조건의 각도 충돌: |wrap(π − (−π/2))| = π/2 rad = 90°

### κ_max 한계 초과

κ_max = a_max_g × g × accel_tol / v_cruise² = 0.3 × 9.81 × 0.9 / 225 ≈ 0.01177 1/m

WP3→WP4를 θ=π에서 시작해 θ=−π/2로 끝내려면 (n=1 감기 기준):
```
0.5 · κ_WP3 · L_seg ≈ wrap(−π/2 − π) = π/2
κ_WP3 ≈ π / L_seg ≈ π / 200 ≈ 0.0157 1/m  >  κ_max
```

tanh 변환이 κ를 ±0.98·κ_max로 클리핑하므로 NR이 요구 값에 도달 불가.

### 결과

NR은 헤딩 잔차를 희생하고 위치 잔차만 최소화하는 방향으로 수렴을 시도하나 실패.
`pos_max_final`이 수십~수백 m → affine 보정량도 동등하게 폭발 → 경로 형상 붕괴.

### 다른 구간이 괜찮은 이유

| WP | θ 초기값 근거 | 결과 |
|---|---|---|
| WP1 (WP0→WP1→WP2 코너) | _initial_thetas() bisector ≈ π/4 | NR 자유도 있음, 수렴 |
| WP2 (WP1→WP2→WP3 코너) | bisector ≈ 3π/4 | 동일 |
| WP3 | WP2→WP3 끝 헤딩 ≈ π에 고착, WP3→WP4 요구와 충돌 | **발산** |
| WP0 시작 | θ_0 = arctan2(WP1−WP0) = 0, kappa0=0 | chord 방향과 일치 → 잔차 0 |

---

## 4. 문제 3: heading 진동 (미해결)

### 진동의 출처

헤딩 배열 `chi_arr`는 보정된 위치의 유한차분에서 계산된다:
```python
chi[i] = arctan2(pts[i+1,1]−pts[i,1], pts[i+1,0]−pts[i,0])
```

진동은 `_clothoid_sample(θ_WP3_bad, κ_WP3_bad, ...)` 자체에서 발생한다.
bad NR 파라미터 → θ(s)가 물리적으로 불가능한 궤적을 그림 → pts가 지그재그.

### affine 보정이 진동을 유발하지 않는 이유

linear correction의 접선 편향:
```
d(correction)/ds = err/L  ← 상수 벡터
```
각 스텝에 동일한 상수 바이어스가 더해질 뿐이다. 진동은 만들지 않는다.
보정 자체는 깨끗하며 진동은 보정 이전 clothoid에서 이미 존재한다.

### smoothstep 보정으로도 해결 불가

smoothstep `f(t) = 3t²−2t³`의 접선 편향:
```
d(correction)/ds = 6t(1−t)/L · err
  t=0, t=1: 0  (양끝 접선 보존)
  t=0.5:    1.5·err/L  (linear보다 1.5배 큼)
```

- **장점**: 세그먼트 접합점 헤딩 점프 제거 (정상 호에서 미세 개선)
- **단점**: 발산 구간 중간부 보정 기울기 1.5배 증폭 → 오히려 악화

어떤 affine 방식이든 clothoid sample 자체의 진동은 그대로 살아남는다.

### 진짜 해결 방법

| 방법 | 설명 | 난이도 |
|---|---|---|
| 중간 WP 자동 삽입 | `_insert_wps_if_infeasible()`이 이미 있으나 헤딩 충돌 기준이 아닌 κ 기준으로 동작 | 중 |
| terminal κ=0 조건 완화 | `kappa_N = 0` 대신 이전 세그먼트 κ 연속 허용 | 하 (부작용 있음) |
| 잔차 임계 초과 시 fallback | pos_res > threshold → diterpin 또는 dubins으로 해당 구간 대체 | 중 |
| diterpin 플래너 사용 | 우회 WP 자동 삽입, 헤딩 충돌 회피 | 즉시 가능 |

---

## 5. 현재 상태 요약

| 항목 | 상태 |
|---|---|
| curvature 필드 허위 값 | **수정됨** — 보정 위치 유한차분으로 재계산 |
| WP3→WP4 경로 붕괴 | **미해결** — NR 발산 근본 원인 그대로 |
| heading 진동 | **미해결** — affine 레이어에서 해결 불가 |
| 정상 구간 (WP0→3) | 정상 동작, curvature 정확도 향상됨 |
