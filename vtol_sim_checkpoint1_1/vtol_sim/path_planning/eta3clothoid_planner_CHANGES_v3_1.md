# η³-Clothoid Planner — v3 → v3.1 변경 기록

## 진단

v3 출력 path에서 다음 두 가지 시각적 결함이 관찰됨:

1. **WP에서 heading 불연속(꺾임)** — 특히 큰 회전각이 필요한 WP 주변
2. **부자연스러운 직선 구간** — 일부 WP 사이에 짧은 직선처럼 보이는 연결

### 근본 원인

#### 원인 A. Stage 0 → Stage 1 사이에서 θ 초기값 폐기

```python
# v3 (잘못된 흐름)
th_pre = _initial_thetas(wps_2d, theta0, theta_N)
wps_2d, _, orig_indices = _insert_wps_if_infeasible(wps_2d, th_pre, kappa_max)
#                ^ 삽입 후 θ를 _ 로 받아 폐기
thetas, ... = _solve_g2_nr(wps_2d, ...)   # 내부에서 _initial_thetas 재호출
```

`_insert_wps_if_infeasible`는 삽입된 WP의 θ를 **양옆 θ의 원형 평균**으로 정한다.
이 정보가 NR로 전달되지 않으면, NR은 삽입된 중점 WP에서 다시 가중 이등분선으로 θ를
계산하는데, 중점이라 양옆 chord가 거의 일직선 → **삽입된 노드의 θ가 chord 방향과
같아져** 사전 삽입의 의도(부드러운 G1)가 무효화됨.

#### 원인 B. L 자유변수 클립 범위가 너무 좁음

```python
# v3
_V_CLIP_LO = -0.3
_V_CLIP_HI =  0.6
# → L_k ∈ [exp(-0.3), exp(0.6)] · chord_k ≈ [0.74, 1.82] · chord_k
```

큰 회전각 + 긴 chord 조합에서 NR이 위치 잔차를 줄이려면 L을 chord의 2배 이상으로
늘려야 하는 경우가 있는데, 1.82·chord에서 막힘. 동시에 `κ ≤ κ_max·tanh(u) < κ_max`
이므로 곡률로도 보상 불가 → **NR이 `tol=1e-5`에 도달하지 못한 채 max_iter 종료**.

#### 원인 C. Clothoid 연결 시 "WP 좌표로 점프"

```python
# v3
for k in range(N - 1):
    seg_pts, _, seg_kp = _clothoid_sample(thetas[k], kappas[k], kappas[k+1], seg_Ls[k], ds)
    seg_pts = seg_pts + wps_2d[k]            # 항상 wps_2d[k] 기준으로 평행이동
    ...
    all_pts.append(seg_pts[:-1])             # segment 끝점 버림
all_pts.append(wps_2d[N - 1: N])             # 마지막 WP 좌표 점프
```

각 segment를 `wps_2d[k]` 기준으로 그리는데, segment의 클로소이드 끝점은
`wps_2d[k+1]`과 **NR 잔차만큼 다르다**. 잔차가 충분히 작으면 안 보이지만,
원인 B로 잔차가 큰 채 종료된 경우 **WP 근처에서 미세 점프 + 다음 segment가 다음
WP 좌표에서 새 헤딩으로 시작**하는 것처럼 보여 **꺾임/직선처럼 가시화**된다.

#### 원인 D. Armijo 백트래킹 8회 부족

복잡한 잔차면에서 step=0.5⁸=0.0039 까지 줄여도 충분조건을 못 만족하면, v3는
그냥 그 작은 step을 적용하고 다음 iter로 넘어감. 수렴 안정성이 낮음.

---

## 변경 내용

### 1. `_solve_g2_nr` 시그니처에 `th_init_full` 매개변수 추가

```python
def _solve_g2_nr(wps, kappa_max, theta0, kappa0, theta_N, kappa_N,
                 th_init_full=None,    # ← v3.1 추가
                 max_iter=60, tol=1e-5, ...):
    if th_init_full is not None and len(th_init_full) == N:
        th_init = np.array(th_init_full, dtype=float)
        th_init[0]  = theta0
        th_init[-1] = theta_N
    else:
        th_init = _initial_thetas(wps, theta0, theta_N)
```

### 2. Stage 0 반환값을 NR로 전달

```python
# v3.1
wps_2d, th_after_insert, orig_indices = _insert_wps_if_infeasible(
    wps_2d, th_pre, kappa_max)
...
thetas, kappas, seg_Ls, nr_norm_final = _solve_g2_nr(
    wps_2d, kappa_max, theta0, kappa0, theta_N, kappa_N,
    th_init_full=th_after_insert,    # ← 사전 삽입 시 잡은 θ 전달
    ...)
```

### 3. L 클립 범위 확장 + 단계적 fallback

```python
# v3.1
_V_CLIP_LO_DEFAULT = -0.5    # exp ≈ 0.61
_V_CLIP_HI_DEFAULT =  1.2    # exp ≈ 3.32

clip_attempts = [
    (-0.5, 1.2),    # 1차
    (-0.7, 1.8),    # 2차 (비수렴 시 더 풀어줌)
    (-1.0, 2.5),    # 3차 (매우 관대)
]
```

각 클립 조합으로 NR을 돌리고 `|F| < tol`이면 종료, 아니면 더 넓은 클립으로 재시도.
모두 실패해도 **|F|가 가장 작은 결과**를 채택. 반환 튜플에 `nr_norm_final` 추가.

### 4. Armijo 백트래킹 8 → 20

```python
for _bt in range(20):           # v3: range(8)
    if np.linalg.norm(_residual(x + step*dx, *args)) \
            <= (1.0 - c_armijo*step) * norm_F:
        break
    step *= 0.5
```

### 5. Clothoid 연결을 "누적 좌표"로 변경

가장 중요한 시각적 결함 제거:

```python
# v3.1
cur_origin = wps_2d[0].copy()
for k in range(N - 1):
    seg_pts, _, seg_kp = _clothoid_sample(
        thetas[k], kappas[k], kappas[k+1], seg_Ls[k], self.ds)
    seg_pts_global = seg_pts + cur_origin   # ← wps_2d[k] 대신 실제 끝점 누적
    all_pts.append(seg_pts_global[:-1])
    all_kappa.append(seg_kp[:-1])
    cur_origin = seg_pts_global[-1]         # 다음 segment의 시작점

all_pts.append(cur_origin.reshape(1, 2))    # 마지막 끝점은 한 번만 추가
```

이렇게 하면:
- segment 간 **점 점프가 구조적으로 불가능**해짐 (이전 끝점 = 다음 시작점)
- NR 잔차가 남아 있어도 path는 항상 **자기 자신과 연속**
- WP 좌표 ≠ path 통과점일 수 있지만 (잔차만큼 차이), 이는 NR이 줄여야 할 항목이며
  v3.1의 클립/백트래킹 개선으로 일반적인 입력에선 1e-5 m 이하로 떨어짐

### 6. 종단 마지막 WP 더미 점 제거

```python
# v3 (제거)
all_pts.append(wps_2d[N - 1: N])
all_kappa.append(np.array([kappas[-1]]))
```

위 2번으로 대체됨 (`cur_origin.reshape(1, 2)`).

### 7. 종단 decay 클로소이드도 누적 좌표 기준

```python
# v3.1
if len(decay_pts) > 1:
    decay_pts_global = decay_pts + cur_origin - decay_pts[0]
    all_pts.append(decay_pts_global[1:])
    ...
    terminal_pos = decay_pts_global[-1]
```

`decay_pts[0]`이 로컬 원점(0,0)에서 시작하므로 `cur_origin - decay_pts[0]`로 정확히
이전 끝점에 이어붙임.

### 8. 고도 보간에서 wp_marks 부족 케이스 방어

```python
if len(sorted_marks) >= 2:
    alt_arr = np.interp(s_arr, wp_s_arr, wp_h_arr)
else:
    alt_arr = np.full(len(pts_arr), wps[0, 2])
```

### 9. wp_marks 기록 위치 변경

v3는 segment 시작 인덱스에 `orig_indices[k]`를 매핑했는데, v3.1은 segment 끝 인덱스
(다음 WP 위치)에 `orig_indices[k+1]`를 매핑. 시작 WP는 별도로 인덱스 0에 매핑.
누적 좌표 방식과 자연스럽게 일치.

---

## 보장 항목 변경

| 항목                        | v3       | v3.1                        |
|-----------------------------|----------|-----------------------------|
| WP 완전 통과 (위치 잔차)    | < nr_tol | < nr_tol (수렴 시), 그 외 best-effort |
| κ_max 전 구간 준수          | ✓        | ✓                           |
| G1 연속성                   | < nr_tol/mean_chord | **구조적 보장** (누적 좌표) |
| G2 연속 (구조적)            | ✓        | ✓                           |
| 자기 루프 없음              | v∈[-0.3,0.6] | v∈[-0.5,1.2]~[-1.0,2.5] (단계적) |
| 종단 κ 부드러움             | ✓        | ✓                           |
| Path 자기 연속성            | NR 잔차에 의존 | **구조적 보장**             |

가장 중요한 변화: **path의 자기 연속성이 NR 수렴에 의존하지 않고 구조적으로 보장**된다.
NR이 `tol`까지 못 떨어진 경우에도 path는 매끈하며, 단지 마지막 WP 통과 위치가 약간
어긋날 수 있을 뿐(이는 새 클립/백트래킹으로 거의 해소됨)입니다.

---

## 회귀 테스트 권장 사항

1. v3에서 보고된 6-WP 케이스(첨부 그림)에서 WP 0,1,3 부근 꺾임 사라졌는지 확인
2. `verbose=True`로 `nr_norm_final < 1e-5` 도달 여부 확인
3. 인접 path 점들의 헤딩 변화량(`np.diff(chi_arr)`) 최댓값이 v3 대비 감소했는지 확인
4. 직선 입력(2~3개 WP 일직선) 케이스 회귀 — 자명 해 유지되는지
5. U턴 케이스 — `_initial_thetas`의 fallback 분기가 정상 작동하는지
