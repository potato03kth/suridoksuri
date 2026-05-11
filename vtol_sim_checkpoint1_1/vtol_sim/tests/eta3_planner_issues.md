# eta3 플래너 문제 분석: curvature 허위 값 & heading 진동

대상 파일: `path_planning/eta3clothoid_v3_1_planner.py`  
발견 경위: fc_bridge 시각화 결과(`eta3_05041422.png`) 분석 — WP3→WP4 구간 경로 붕괴 및 curvature 패널 이상

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

```python
kappa_arr = kappa_i + dk * s   # affine 보정 이전 값
```

affine 보정은 **위치**만 이동시키고, `kappa_arr`에는 반영되지 않는다.
결과적으로 `PathPoint.curvature`는 보정된 실제 경로가 아닌 NR이 "의도한" κ를 저장했다.

### NR 수렴 성공 vs 실패에 따른 괴리

| 상황 | affine 보정량 | 설계 κ vs 실제 κ |
|---|---|---|
| NR 수렴 (정상 구간) | 수 cm ~ 수 m | 거의 동일 |
| NR 발산 (WP3→WP4) | 수십~수백 m | 설계 κ ≪ 실제 κ |

### 하위 소비자 영향

| 소비자 | 사용 방식 | 영향 |
|---|---|---|
| `fc_bridge/run_phase2.py` | 속도 프로필 계산 | 실제보다 낮은 κ → 과속 허용 |
| `mpc_controller._get_preview_curvatures()` | MPC 피드포워드 | 횡방향 제어력 과소 추정 |
| `vtol_sim/metrics.py` `a_n` | 횡방향 가속도 계산 | 메트릭 허위 |

### 수정 내용

`pts_arr`, `s_arr` 확정 후 실제 보정된 위치로 유한차분 재계산 (`_compute_curvature()`와 동일 공식):

```python
_dN = np.gradient(pts_arr[:, 0], s_arr)
_dE = np.gradient(pts_arr[:, 1], s_arr)
_d2N = np.gradient(_dN, s_arr)
_d2E = np.gradient(_dE, s_arr)
_spd = np.sqrt(np.maximum(_dN**2 + _dE**2, 1e-24))
kappa_arr = (_dN * _d2E - _dE * _d2N) / _spd**3
```

`all_kappa` 리스트 전체 제거 (선언, 4개 append, concat).

---

## 3. 문제 2: NR 발산 — WP3→WP4 구간

### 기하학적 설정 (기본 테스트 웨이포인트 기준)

```
WP2=(N=200, E=200)  →  WP3=(N=0, E=200)  →  WP4=(N=0, E=0)
WP2→WP3 진행 방향: −N  ∴ θ_end ≈ π   (south)
WP3→WP4 진행 방향: −E  ∴ θ 필요값 ≈ −π/2  (west)
```

### NR 헤딩 잔차 구조

```python
F[3k+2] = w_head · wrap(theta_end_of_seg_k − theta[k+1])
# w_head = min(mean_chord, 50) = 50  (강한 가중치)
```

WP2→WP3 세그먼트가 수렴하면 `theta[WP3] ≈ π`로 강하게 고착된다.
WP3→WP4 세그먼트는 `theta[WP3] = −π/2`가 필요하다.
**두 요구의 충돌: `|wrap(π − (−π/2))| = π/2 rad`** — NR이 동시에 만족 불가.

### κ_max 범위 내 해 부재

κ_max = a_max_g × g × accel_tol / v_cruise²  
= 0.3 × 9.81 × 0.9 / 225 ≈ **0.01177 1/m**  (R_min ≈ 85 m)

θ = π에서 출발해 −π/2로 도달하려면 (n=1 감기 기준):
```
Δθ = wrap(−π/2 − π) = π/2  →  κ_avg · L_seg ≈ π/2
κ_WP3 ≈ π / L_seg ≈ π / 200 ≈ 0.0157 1/m  >  κ_max
```

tanh 변환이 κ를 `±0.98·κ_max`로 클리핑하므로 NR이 요구 값에 도달 불가.
NR은 3회 clip_attempt(v_lo/v_hi 조정)를 시도하지만 모두 실패.

### 다른 구간이 괜찮은 이유

WP1, WP2는 **내부 WP** — θ와 κ 모두 NR 자유 변수.  
`_initial_thetas()`가 bisector 방향으로 초기화 → 각도 충돌 없음 → 수렴.

| WP | θ 초기값 | NR 자유도 | 결과 |
|---|---|---|---|
| WP0 (시작) | arctan2(WP1−WP0) = 0 | 고정 (BC) | chord 방향과 일치 → 잔차 ≈ 0 |
| WP1 (코너) | bisector ≈ π/4 | θ, κ 모두 자유 | 수렴 |
| WP2 (코너) | bisector ≈ 3π/4 | θ, κ 모두 자유 | 수렴 |
| WP3 (penultimate) | WP2→WP3 끝 헤딩 π에 의해 고착 | **충돌** | **발산** |
| WP4 (끝) | arctan2(WP4−WP3) = −π/2 | 고정 (BC), kappa_N=0 | — |

---

## 4. 문제 3: heading 진동 (미해결)

### 진동의 출처

헤딩 배열은 보정된 위치의 유한차분으로 계산된다:
```python
chi[i] = arctan2(pts[i+1, 1]−pts[i, 1],  pts[i+1, 0]−pts[i, 0])
```

진동은 `_clothoid_sample(θ_WP3_bad, κ_WP3_bad, ...)` **자체**에서 발생한다.  
bad NR 파라미터 → `θ(s) = θ_i + κ_i·s + ½·(Δκ/L)·s²` 가 물리적으로 불가능한 궤적을 생성 → pts 지그재그 → chi 진동.

### affine 보정이 진동을 만들지 않는다

linear correction의 접선 편향:
```
d(correction)/ds = err/L  ← 상수 벡터 (모든 s에서 동일)
```
각 스텝에 동일한 상수 바이어스가 더해질 뿐이다. 새로운 진동을 만들지 않는다.
진동은 보정 이전 clothoid에 이미 존재한다.

### smoothstep 보정으로도 해결 불가

`f(t) = 3t²−2t³`의 접선 편향:
```
df/dt = 6t(1−t)
  t=0, t=1: 0         → 양끝 접선 보존 (세그먼트 접합점 점프 제거)
  t=0.5:    1.5        → 중간부 기울기 linear 대비 1.5배 증폭
```

발산 구간에서는 중간부 보정 기울기가 커져 오히려 악화된다.  
어떤 affine 방식이든 clothoid sample 자체의 진동은 통과된다.

### 실질적 해결 방법

| 방법 | 설명 | 비고 |
|---|---|---|
| `_insert_wps_if_infeasible()` 기준 확장 | 현재 κ 초과만 검사 → **헤딩 충돌 감지** 추가 | 중간 WP 삽입 시 각도 분산 |
| terminal `kappa_N = 0` 완화 | 이전 세그먼트 κ 연속 허용 | 착륙 정밀도 하락 가능 |
| 잔차 임계 초과 시 구간 fallback | `pos_res > 0.5m` → dubins/diterpin으로 해당 세그먼트 대체 | 중 |
| `diterpin` 플래너 사용 | 우회 WP 자동 삽입으로 헤딩 충돌 회피 | 즉시 가능 |

---

## 5. 현재 상태 요약

| 항목 | 상태 | 위치 |
|---|---|---|
| `PathPoint.curvature` 허위 값 | **수정됨** — 보정 위치 유한차분 재계산 | `eta3clothoid_v3_1_planner.py` L431–440 |
| WP3→WP4 경로 붕괴 | **미해결** — NR 발산 근본 원인 존재 | NR 또는 플래너 선택 레이어 |
| heading 진동 | **미해결** — affine 레이어에서 해결 불가 | 동일 |
| 정상 구간 (WP0→WP3) | 정상 동작, curvature 정확도 향상 | — |
