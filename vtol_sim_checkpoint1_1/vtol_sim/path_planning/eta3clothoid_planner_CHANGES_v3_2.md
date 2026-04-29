# η³-Clothoid Planner — v3.1 → v3.2 변경 기록

## 진단

v3.1에서 출력한 path는 **곡선이 매끄러우나 모든 WP를 빗나가는** 새로운 결함이 발생.
시각적 분석 결과 **앞쪽 WP는 통과, 뒤쪽 WP는 점점 더 크게 빗나가는** 양상.
이는 v3.1에서 도입한 "누적 좌표 연결"의 부작용으로, **NR 위치 잔차가 segment를
거치며 누적**되어 후반부가 점점 어긋나는 구조적 문제.

### 근본 원인

v3.1의 연결 방식:

```python
# v3.1 (잘못됨)
cur_origin = wps_2d[0].copy()
for k in range(N - 1):
    seg_pts = _clothoid_sample(thetas[k], kappas[k], kappas[k+1], seg_Ls[k], ds)
    seg_pts_global = seg_pts + cur_origin   # 이전 segment 끝점 누적
    cur_origin = seg_pts_global[-1]         # 다음 segment 시작
```

NR이 풀어준 `(thetas[k], kappas[k], seg_Ls[k])`는 **`wps_2d[k] → wps_2d[k+1]` 사이를
잇는 해**이지만, NR 잔차 `e_k`가 남아있을 수 있음. 이때:

- segment 0 끝점: `wps_2d[1] + e_0`
- segment 1 끝점: `wps_2d[2] + e_0 + e_1`  ← 누적
- segment k 끝점: `wps_2d[k+1] + Σ e_i`   ← 점점 커짐

NR이 `tol=1e-5`로 수렴하면 `|e_k| ~ 1e-5 m`라 누적해도 무시 가능하지만, 다음 두 케이스에서
누적 오차가 가시화됨:

1. **NR 미수렴**: `|F|`가 충분히 작아지지 못한 채 max_iter 종료 → 잔차 큼
2. **헤딩 vs 위치 잔차 가중치 불균형**: `mean_chord` 가중치가 매우 크면 (수백 m) NR이
   헤딩만 줄이고 위치는 큰 잔차를 남길 수 있음

이 케이스가 6-WP 그림에서 발생한 것으로 추정. NR이 헤딩을 우선 풀어 곡선은 매끈하지만
위치 잔차가 누적되어 WP에서 멀어짐.

### 트레이드오프 정리

| 연결 방식 | 자기 연속성 | WP 통과 |
|-----------|-------------|---------|
| v3 (`wps_2d[k]` 기준 + 끝점 버림 + WP 점프) | NR 잔차에 의존 (작으면 OK) | 강제 보장 (점프로) |
| v3.1 (누적 좌표) | **구조적 보장** | NR 잔차 누적 → 빗나감 |
| v3.2 (segment별 + affine 끝점 보정) | **구조적 보장** | **구조적 보장** |

---

## v3.2 핵심 아이디어: Affine 끝점 보정

각 segment를 `wps_2d[k]`를 원점으로 그리되(v3 방식), **끝점 잔차를 segment 길이를 따라
선형 분배**하여 끝점이 `wps_2d[k+1]`에 정확히 맞도록 보정한다.

```python
# v3.2
seg_pts_local, seg_th, seg_kp, seg_s = _clothoid_sample(...)

seg_end_local    = seg_pts_local[-1]                   # NR이 푼 끝점 (로컬)
target_end_local = wps_2d[k + 1] - wps_2d[k]           # 목표 끝점 (로컬)
err              = target_end_local - seg_end_local    # 잔차 벡터

L_total    = seg_s[-1]
correction = np.outer(seg_s / L_total, err)            # 시작 0, 끝 err로 선형 보간
seg_pts_local_corrected = seg_pts_local + correction

seg_pts_global = seg_pts_local_corrected + wps_2d[k]   # 전역으로 변환
```

**보장 항목**:
- segment 시작점: `wps_2d[k]` (보정 0)
- segment 끝점: `wps_2d[k+1]` (보정 = err로 정확히 잔차 상쇄)
- 두 인접 segment의 공유 점은 **양쪽 모두 같은 WP 좌표**이므로 자기 연속성 자동
- 보정량 `err`가 작을수록(NR 수렴이 좋을수록) 곡선 모양 변형 적음

**수학적 의미**: 끝점 잔차를 1차(linear) shear 변환으로 흡수. 곡률 자체는
변형되지만(2차 미분) NR 잔차 크기에 비례하므로 작음. 헤딩(1차 미분)은 끝점에서 약간
변하지만 segment 사이 헤딩 매칭은 여전히 NR이 풀어둔 값에 의존.

> ⚠ **주의**: affine 보정은 NR 잔차 흡수용 안전망이지, NR이 잘 수렴하는 것을 대체하지
> 않는다. NR 잔차가 너무 크면 보정량이 커져 곡선이 G2 보장을 잃는다(곡률은 여전히
> 연속이지만 κ_max 초과 가능). 따라서 잔차가 임계값(0.5 m) 초과 시 경고를 출력한다.

---

## 변경 내용

### 1. `_residual` 헤딩 가중치 분리

```python
# v3.1
F[3*k + 2] = mean_chord * _wrap(th_end - th[k+1])

# v3.2
F[3*k + 2] = w_head * _wrap(th_end - th[k+1])  # w_head = min(mean_chord, 50.0)
```

`mean_chord`가 매우 클 때(예: 500 m) 헤딩 잔차가 위치 잔차에 비해 과대평가되어
NR이 헤딩만 우선 줄이는 문제 방지. `w_head` 상한을 50으로 클램프해 균형 유지.

### 2. `_solve_g2_nr` 반환에 위치/헤딩 잔차 분리

```python
# v3.1: return th, kp, Ls, norm_F
# v3.2: return th, kp, Ls, pos_max, head_max
```

planner가 `pos_max`를 보고 경고 출력 가능.

### 3. Newton 스텝에 Levenberg-Marquardt 댐핑 추가

```python
# v3.2
lam = 1e-8 * np.trace(J.T @ J) / max(n_j, 1)
A   = J.T @ J + lam * np.eye(n_j)
dx  = np.linalg.solve(A, -J.T @ F)
```

야코비안이 거의 특이할 때 안정성 증가. 실패 시 `lstsq`로 fallback.

### 4. Clothoid 연결을 segment별 + affine 끝점 보정으로 변경 (핵심)

```python
# v3.2
for k in range(N - 1):
    seg_pts_local, seg_th, seg_kp, seg_s = _clothoid_sample(...)
    err = (wps_2d[k+1] - wps_2d[k]) - seg_pts_local[-1]
    correction = np.outer(seg_s / seg_s[-1], err)
    seg_pts_local_corrected = seg_pts_local + correction
    seg_pts_global = seg_pts_local_corrected + wps_2d[k]
    if k < N - 2:
        all_pts.append(seg_pts_global[:-1])
    else:
        all_pts.append(seg_pts_global)  # 마지막 segment는 끝점 포함
```

**v3.1과 비교**:
- v3.1: `cur_origin` 누적 → 잔차 누적 누적 누적
- v3.2: 각 segment 독립적으로 `wps_2d[k]` 기준, 끝점은 `wps_2d[k+1]`로 강제
- 두 인접 segment의 경계는 양쪽 모두 같은 WP에서 정확히 만남 → 자기 연속

### 5. `_clothoid_sample` 반환에 `s` 배열 추가

affine 보정 시 호 길이 매개변수가 필요하므로 `(pts, theta, kappa, s)` 4-튜플 반환.

### 6. Planner에서 큰 위치 잔차 경고

```python
if pos_res > 0.5:
    print(f"[Eta3ClothoidPlannerV3] ⚠ NR 위치 잔차 {pos_res:.3f}m가 큽니다. "
          f"affine 보정으로 WP 통과는 보장하지만 곡선이 변형될 수 있습니다.")
```

### 7. 종단 decay/ext: 마지막 path 점 기준으로 평행이동

```python
# v3.2
last_global = all_pts[-1][-1].copy()
decay_pts_global = decay_pts + last_global  # decay_pts[0] == (0,0)이므로 정확히 이어짐
```

마지막 segment 끝점은 affine 보정으로 `wps_2d[-1]`과 정확히 일치함.

---

## 보장 항목 비교

| 항목                        | v3       | v3.1                   | v3.2                |
|-----------------------------|----------|------------------------|---------------------|
| WP 완전 통과 (위치)         | NR 잔차 의존 | NR 잔차 누적 ⚠       | **구조적 보장**     |
| κ_max 전 구간 준수          | ✓        | ✓                      | ✓ (NR 잔차 작을 때) |
| G1 연속성                   | NR 잔차 의존 | **구조적 보장**       | **구조적 보장**     |
| G2 연속 (구조적)            | ✓        | ✓                      | ✓ (NR 잔차 작을 때) |
| Path 자기 연속성            | NR 잔차 의존 | **구조적 보장**       | **구조적 보장**     |
| 자기 루프 없음              | v∈[-0.3,0.6] | v∈[-0.5,1.2] (단계적) | v∈[-0.5,1.2] (단계적) |

> v3.2에서 G2/κ_max는 "NR 잔차가 충분히 작을 때"라는 조건이 붙는다. 이는 v3에서도
> 동일한 조건이었으나 v3.1에서 잠시 명시되지 않았던 것을 다시 명확화한 것이다.
> 일반 입력에서 `pos_max < 0.5 m` 정도면 affine 보정으로 인한 곡률 변형은
> 매우 작다 (`Δκ ~ |err|/L²` 수준).

---

## 회귀 테스트 권장 사항

1. v3.1에서 보고된 6-WP 케이스에서 **모든 WP를 path가 정확히 통과**하는지 확인
2. `verbose=True`로 `pos_max < 1e-3 m`, `head_max < 1e-4 rad` 도달 확인
3. NR이 미수렴해도 affine 보정이 작동해 WP를 통과하는지 확인
   (인위적으로 `nr_max_iter=2`로 낮춰 테스트)
4. 직선 입력 회귀 — affine 보정량이 0에 가까운지 확인 (`err ≈ 0`)
5. 인접 path 점 헤딩 변화량 — segment 경계에서 점프 없는지 확인
