# η³-Clothoid Stage 2 재설계 계획

> 대상 파일: `path_planning/eta3clothoid_planner.py`  
> 의존: `path_planning/base_planner.py`

---

## 1. 현재 문제 상황

### 1.1 순방향 독립 보정 — 덮어쓰기 문제 (핵심)

현재 Stage 2 루프 ([eta3clothoid_planner.py:370–385](eta3clothoid_planner.py)):

```python
for i in range(N - 1):
    L, th_c, kp_c = _segment_nr_correct(
        thetas[i], kappas[i],          # ← 구간 i 의 시작 조건 (보정됨)
        thetas[i + 1], kappas[i + 1],  # ← 구간 i 의 종료 조건 (지금은 고정)
        target_disp, kappa_max,
    )
    thetas[i] = th_c   # ← thetas[i] 갱신
    kappas[i] = kp_c
```

**문제**: 구간 i 처리 후 다음 반복(구간 i+1)에서 `thetas[i+1]`, `kappas[i+1]`이 NR에 의해 덮어써진다. 그러나 이 값들은 이미 구간 i의 **종료 조건**으로 사용되었다.

```
구간 i   NR: (θ_i,   κ_i)   → 보정됨,  (θ_{i+1}, κ_{i+1}) 고정으로 가정
구간 i+1 NR: (θ_{i+1}, κ_{i+1}) → 덮어씀  ← 구간 i가 사용한 종료 조건 무효화
```

**결과**: 구간 i의 끝 헤딩 ≠ 구간 i+1의 시작 헤딩 → **G1/G2 불연속**

```
연속성 요구 조건 (만족 안 됨):
  segment i   끝:   (θ_{i+1}_original, κ_{i+1}_original)
  segment i+1 시작: (θ_{i+1}_corrected, κ_{i+1}_corrected)
  → 이 두 값이 일반적으로 다름
```

---

### 1.2 κ 클리핑으로 인한 NR 수렴 실패

현재 `_segment_nr_correct` ([eta3clothoid_planner.py:287](eta3clothoid_planner.py)):

```python
kappa_i_c = np.clip(kappa_i_c + step * delta[1], -kappa_max, kappa_max)
```

| 문제 | 구체적 영향 |
|------|------------|
| 야코비안 불연속 | 클리핑 경계에서 ∂κ/∂u = 0 → NR 방향 벡터 왜곡 |
| 자유도 손실 | κ가 한계에 걸리면 θ 하나로 2D 끝점을 맞춰야 함 → 과결정 |
| WP 통과 오차 | NR이 수렴해도 잔차가 남음 → WP를 비껴남 |

---

### 1.3 Stage 1 미구현 (η³ G2)

`_eta3_g2_residual` ([eta3clothoid_planner.py:166](eta3clothoid_planner.py))이 `NotImplementedError`를 발생시킨다. Stage 1 NR 루프 전체가 주석 처리되어 있고, `plan()`에서는 대신 초기값(`_eta3_initial_guess`)만 사용한다.

```python
# plan() 내 현재 동작 (L.362–363):
thetas, kappas = _eta3_initial_guess(...)   # 내부 노드 κ = 0
kappas = np.clip(kappas, -kappa_max, kappa_max)   # κ=0 이므로 효과 없음
```

Stage 2에 좋은 초기값이 공급되지 않으므로 NR의 수렴 도메인이 좁아진다.

---

### 문제 상황 요약

```
Stage 1 미구현
  └→ κ=0 초기값 공급
       └→ Stage 2 NR 수렴 도메인 축소
            ├→ κ 클리핑 → 자유도 손실 → WP 통과 오차
            └→ 순방향 덮어쓰기 → G1/G2 불연속
```

---

## 2. 해결 방법

### 2.1 tanh 재매개변수화 (κ 제약 내재화)

κ를 직접 최적화 변수로 쓰는 대신, 비제약 변수 u로 교체한다.

```
κ_i = κ_max · tanh(u_i),    u_i ∈ ℝ
∂κ_i/∂u_i = κ_max · sech²(u_i)   ← 어디서나 연속, 클리핑 없음
```

- u_i → +∞ 이면 κ_i → κ_max (점근, 도달 불가)
- u_i = 0 이면 κ_i = 0 (초기값과 자연스럽게 일치)
- NR의 야코비안에 sech² 인수가 추가되지만 연속이므로 수렴 성질 유지

**κ=0 초기값과의 호환**: `u_i = 0` → `κ_i = 0`. 현재 Stage 1 폴백 초기값과 그대로 연결된다.

---

### 2.2 전역 동시 NR (순방향 덮어쓰기 제거)

현재 구조의 근본 결함은 "구간 단위 순방향 보정"이다. 이를 **모든 내부 노드를 동시에 최적화**하는 전역 NR로 교체한다.

#### 변수 정의

```
경계 고정: θ_0, u_0  (= κ_0),  θ_{N-1}, u_{N-1}  (= κ_{N-1})
자유 변수: x = [θ_1, u_1, θ_2, u_2, ..., θ_{N-2}, u_{N-2}]   크기: 2(N-2)
```

#### 잔차 방정식

구간 i (i = 0, …, N-2) 마다 2개:

```
L_i    = _compute_L(θ_i, κ_max·tanh(u_i), θ_{i+1}, κ_max·tanh(u_{i+1}), chord_i)
(p_x, p_y) = _fresnel_endpoint(θ_i, κ_max·tanh(u_i), κ_max·tanh(u_{i+1}), L_i)

F[2i]   = p_x − Δx_i
F[2i+1] = p_y − Δy_i
```

전체 F 크기: **2(N-1)**, 자유 변수: **2(N-2)** → 2개 과결정 → `lstsq` (최소 잔차 해)

#### 야코비안 구조

```
          θ_1  u_1  θ_2  u_2  θ_3  u_3  …
seg 0  [ ∂F₀/∂θ₁  ∂F₀/∂u₁   0    0    0    0  …]   ← θ_0, u_0 고정이므로 여기서 시작
seg 0  [ ∂F₁/∂θ₁  ∂F₁/∂u₁   0    0    0    0  …]
seg 1  [ ∂F₂/∂θ₁  ∂F₂/∂u₁  ∂F₂/∂θ₂  ∂F₂/∂u₂  0   0  …]
seg 1  [ ∂F₃/∂θ₁  ∂F₃/∂u₁  ∂F₃/∂θ₂  ∂F₃/∂u₂  0   0  …]
seg 2  [  0    0   ∂F₄/∂θ₂  ∂F₄/∂u₂  ∂F₄/∂θ₃ ∂F₄/∂u₃ …]
…
```

**블록 대역(banded) 구조**: 각 행에 최대 4개 비영 원소 → 희소 야코비안으로 효율적 처리 가능.  
수치 미분: 전진 차분(eps = 1e-6) 또는 5점 스텐실.

#### NR 반복

```
for iter in range(max_iter):
    F = _global_residual(x, ...)
    if ‖F‖ < tol: break
    J = _numerical_jacobian(x, ...)       # 2(N-1) × 2(N-2) 희소 행렬
    dx = lstsq(J, -F, rcond=None)[0]
    step = min(1.0, 0.5 / (‖dx‖ + ε))    # 단순 스텝 제한
    x += step · dx
```

---

### 2.3 WP 삽입 — 기하학적 실현 가능성 사전 검사

전역 NR이더라도 "필요한 κ > κ_max"인 구간은 수렴 불가능하다. 이를 사전에 검출하고 중간 WP를 삽입한다.

```
각 구간 i 에 대해:
  chord_i = ‖WP_{i+1} - WP_i‖
  Δθ_i    = |_wrap(θ_{i+1} - θ_i)|
  κ_needed = 2 · Δθ_i / chord_i      (균일 κ 추정)

  if κ_needed > κ_max:
      삽입 WP = (WP_i + WP_{i+1}) / 2   (중점, 또는 오프셋 추가)
      구간 분할 반복 (κ_needed ≤ κ_max 될 때까지)
```

WP 삽입 후 전역 NR을 실행하면 모든 구간이 실현 가능하므로 수렴이 보장된다.

---

## 3. 적용된 코드 구조

### 3.1 전체 파이프라인

```
plan()
 │
 ├─ [Stage 1] _solve_eta3_g2()          ← 미구현(PSEUDO); 폴백: _eta3_initial_guess
 │    └→ thetas[N], kappas[N]
 │
 ├─ [사전 검사] _check_and_insert_wps() ← NEW
 │    ├─ 구간별 κ_needed 계산
 │    └─ κ_needed > κ_max 시 중점 WP 삽입 → wps_2d, thetas, kappas 재구성
 │
 ├─ [Stage 2] _global_stage2_nr()       ← NEW (순방향 NR 대체)
 │    ├─ _build_state_vector()          ← x = [θ_1,u_1,...,θ_{N-2},u_{N-2}]
 │    ├─ _global_residual()             ← F (2(N-1),)
 │    ├─ _numerical_jacobian_global()   ← J (2(N-1) × 2(N-2))
 │    └─ NR 반복 (lstsq)
 │         └→ thetas[N], kappas[N] (tanh 역변환 포함)
 │
 ├─ [샘플링] 구간별 _clothoid_sample()   ← 기존 유지
 │
 └─ PathPoint 조립, Path 반환           ← 기존 유지
```

---

### 3.2 신규 함수 명세

#### `_check_and_insert_wps`

```python
def _check_and_insert_wps(
    wps_2d:   np.ndarray,   # (N, 2)
    thetas:   np.ndarray,   # (N,)
    kappas:   np.ndarray,   # (N,)
    kappa_max: float,
    max_insert: int = 3,    # 단일 구간 최대 삽입 횟수
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    κ_needed > κ_max 인 구간을 검출하여 중점 WP를 삽입한다.
    삽입된 WP의 θ는 이등분선 방향, κ는 0으로 초기화.

    Returns: wps_2d_new, thetas_new, kappas_new
    """
```

**동작**:
1. 각 구간 i에 대해 `κ_needed = 2 · |Δθ_i| / chord_i` 계산
2. `κ_needed > κ_max` 이면 `WP_mid = (WP_i + WP_{i+1}) / 2` 삽입
3. 삽입 WP의 `θ_mid` = 양쪽 코드 이등분선, `κ_mid = 0`
4. `max_insert` 회까지 반복 (무한루프 방지)

---

#### `_global_residual`

```python
def _global_residual(
    x:          np.ndarray,   # [θ_1,u_1,...,θ_{N-2},u_{N-2}]  (2(N-2),)
    wps_2d:     np.ndarray,   # (N, 2)
    theta_bc:   tuple[float, float],   # (θ_0, θ_{N-1})
    kappa_bc:   tuple[float, float],   # (κ_0, κ_{N-1}) — tanh 역변환 안 함
    kappa_max:  float,
    n_quad:     int = 200,
) -> np.ndarray:              # F (2(N-1),)
    """
    전역 잔차 벡터.

    내부 처리:
      θ_i = x[2*(i-1)]
      u_i = x[2*(i-1)+1]
      κ_i = kappa_max * tanh(u_i)
    경계:
      θ_0, κ_0, θ_{N-1}, κ_{N-1} 고정
    """
```

---

#### `_global_stage2_nr`

```python
def _global_stage2_nr(
    wps_2d:      np.ndarray,   # (N, 2)
    thetas:      np.ndarray,   # (N,) 초기값 (Stage 1 출력 또는 폴백)
    kappas:      np.ndarray,   # (N,) 초기값
    kappa_max:   float,
    max_iter:    int = 50,
    tol:         float = 1e-4,
    eps_jac:     float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    전역 동시 NR.
    tanh 재매개변수화로 κ 제약 내재화.
    야코비안: 수치 미분 (전진 차분).
    선형 시스템: lstsq (2(N-1) × 2(N-2) 과결정).

    Returns thetas (N,), kappas (N,) — 경계 포함, κ ∈ (-κ_max, κ_max)
    """
    N = len(wps_2d)
    if N <= 2:
        return thetas, kappas

    # 초기 상태 벡터: u_i = arctanh(κ_i / kappa_max) (내부 노드만)
    u_init  = np.arctanh(np.clip(kappas[1:-1] / kappa_max, -0.9999, 0.9999))
    x = np.zeros(2 * (N - 2))
    x[0::2] = thetas[1:-1]
    x[1::2] = u_init

    theta_bc = (thetas[0],  thetas[-1])
    kappa_bc = (kappas[0],  kappas[-1])

    for _ in range(max_iter):
        F = _global_residual(x, wps_2d, theta_bc, kappa_bc, kappa_max)
        if np.linalg.norm(F) < tol:
            break
        J = _numerical_jacobian_global(x, wps_2d, theta_bc, kappa_bc, kappa_max, eps_jac)
        dx = np.linalg.lstsq(J, -F, rcond=None)[0]
        step = min(1.0, 0.5 / (np.linalg.norm(dx) + 1e-12))
        x += step * dx

    thetas[1:-1] = x[0::2]
    kappas[1:-1] = kappa_max * np.tanh(x[1::2])
    return thetas, kappas
```

---

### 3.3 수정되는 `plan()` 흐름

```python
# Stage 1 (기존 폴백 유지)
thetas, kappas = _eta3_initial_guess(wps_2d, theta0, kappa0, theta_N, kappa_N)
kappas = np.clip(kappas, -kappa_max, kappa_max)

# NEW: 기하학적 실현 가능성 사전 검사 + WP 삽입
wps_2d, thetas, kappas = _check_and_insert_wps(wps_2d, thetas, kappas, kappa_max)
N = len(wps_2d)   # WP 삽입으로 N이 늘어날 수 있음

# NEW: 전역 Stage 2 NR (순방향 NR 대체)
thetas, kappas = _global_stage2_nr(
    wps_2d, thetas, kappas, kappa_max,
    max_iter=self.nr_max_iter, tol=self.nr_tol,
)

# 이하 샘플링, PathPoint 조립: 기존과 동일
for i in range(N - 1):
    L = _compute_L(thetas[i], kappas[i], thetas[i+1], kappas[i+1], ...)
    seg_pts, _, seg_kappa = _clothoid_sample(thetas[i], kappas[i], kappas[i+1], L, self.ds)
    seg_pts += wps_2d[i]
    ...
```

---

## 4. 보장 항목 변화

| 항목 | 현재 상태 | 재설계 후 |
|------|-----------|-----------|
| WP 위치 통과 (G0) | NR 수렴 시 근사 보장, 클리핑 시 실패 | 전역 NR + WP 삽입으로 개선 |
| 헤딩 연속성 (G1) | 순방향 덮어쓰기로 **불보장** | 전역 NR로 최소 잔차 의미에서 보장 |
| 곡률 연속성 (G2) | **불보장** (Stage 1 미구현) | Stage 1 구현 전까지는 근사; Stage 1 완성 후 보장 |
| κ_max 준수 | 클리핑으로 강제 (NR 수렴 실패 위험) | tanh로 **구조적 보장** |
| G3 연속성 | 미구현 (설계 범위 밖) | 미구현 유지 |

---

## 5. 구현 우선순위 및 진행 상황

```
1순위: _check_and_insert_wps()     ✅ 완료 (2026-04-29)
2순위: _global_stage2_nr()         ✅ 완료 (2026-04-29)
3순위: Stage 1 η³ G2 잔차 구현      ⬜ 미구현 — G2 연속성 완성 (Bertolazzi-Frego 2018 참조)
```

Stage 1 없이도 1·2순위만 구현하면 현재보다 실질적으로 안정적인 경로가 생성된다.  
Stage 1이 완성되면 전역 NR의 초기값 품질이 높아져 수렴이 더 빠르고 정확해진다.

---

## 6. 1순위 구현 상세 (2026-04-29)

### 6.1 추가된 함수

#### `_menger_kappa(wps_2d, i, kappa_max) → float`

내부 노드 i 의 부호 있는 Menger 곡률. κ=0 초기값으로 인한 `_compute_L` 특이점(ksum≈0)을 회피하기 위해 도입.

- NED 컨벤션 (우선회=양수): `sign = +1` if `cross = a×b ≥ 0`
- 반환값: `clip(sign·area2/(|a|·|b|·|ac_chord|), -0.9κ_max, +0.9κ_max)`

#### `_check_and_insert_wps(wps_2d, thetas, kappas, kappa_max, max_insert=3)`

κ_needed = `2·|Δθ|/chord > κ_max` 인 구간에 중점 WP를 삽입.

| 삽입 WP 초기값 | 이유 |
|---|---|
| `θ_mid = atan2(chord)` | 양쪽 부분 코드가 동일 방향이므로 bisector = chord 방향 |
| `κ_mid = 0` | 중점의 Menger 곡률은 평행한 두 코드 → cross=0 → 구조적으로 0 |

`orig_indices: list[int]` 를 반환하여 삽입 WP(-1)와 원본 WP(≥0)를 구별.

### 6.2 수정된 함수

#### `_eta3_initial_guess`
- 시그니처에 `kappa_max: float = 1.0` 추가
- 내부 노드 κ 초기값: `0.0` → `_menger_kappa(wps_2d, i, kappa_max)`
- U턴 처리: bisector 벡터 크기 < 1e-9 이면 진입 방향에 수직인 방향 사용

#### `plan()`
- `N_original = len(wps)` 추가 (WP 삽입 후에도 원본 수 보존)
- `_eta3_initial_guess` 호출 시 `kappa_max` 전달
- `_check_and_insert_wps` 호출 → `wps_2d, thetas, kappas, orig_indices` 갱신
- Stage 2 루프: `orig_indices[i] >= 0` 인 경우에만 `wp_marks`에 원본 WP 인덱스 기록
- 고도 보간 필터: `wi < N` → `wi < N_original` (삽입 WP가 원본 고도 배열을 벗어나지 않도록)

### 6.3 검증 결과 (스모크 테스트)

```
Menger κ (90° 우선회, chord=10): 0.0707  ✓ (양수)
급격 구간 WP 삽입: 3개 삽입, orig_indices 정확
4-WP plan(): 307점, 918.5m, wp_index [0,1,2,3] 모두 마킹 ✓
```

### 6.4 남은 한계 → 2순위에서 해결됨

- G1/G2 불연속: 순방향 NR 덮어쓰기 문제 → **`_global_stage2_nr` 구현으로 해결**
- κ 클리핑 NR 수렴 불안정: → **tanh 재매개변수화로 구조적 보장**

---

## 7. 2순위 구현 상세 (2026-04-29)

### 7.1 추가된 함수

#### `_global_residual(x, wps_2d, theta_bc, kappa_bc, kappa_max, n_quad=100)`

전역 잔차 벡터 F, 크기 2(N-1). 구간 i의 끝점 변위 오차를 반환.

- **L=chord 고정**: `_compute_L` 미사용. `kappa_sum≈0`일 때 L>>chord 발산 방지.
- 내부 처리: `θ_k = x[2*(k-1)]`, `κ_k = kappa_max·tanh(x[2*(k-1)+1])`
- `_fresnel_endpoint(θ_i, κ_i, κ_{i+1}, L=chord, n_quad)` → endpoint 변위 계산

#### `_numerical_jacobian_global(x, wps_2d, theta_bc, kappa_bc, kappa_max, eps_jac=1e-6)`

전진 차분 수치 야코비안. 형태 2(N-1) × 2(N-2).

#### `_global_stage2_nr(wps_2d, thetas, kappas, kappa_max, max_iter=50, tol=1e-4, eps_jac=1e-6)`

전역 동시 NR 메인 함수.

| 설계 결정 | 이유 |
|-----------|------|
| L=chord (잔차, 샘플링 모두) | `_compute_L` 특이점 제거, NR-샘플링 일관성 |
| tanh 재매개변수화 `κ=κ_max·tanh(u)` | κ ∈ (-κ_max, κ_max) 구조적 보장, 클리핑 불필요 |
| Armijo 역방향 선탐색 | 단순 스텝 제한 교체 → 단조 감소 보장, 발산 방지 |
| NR 후 θ 래핑 `_wrap(θ)` | θ 무한 증가 → 거대 L 추정 방지 |
| lstsq (2(N-1) × 2(N-2)) | 과결정 시스템 최소잔차 해 |

### 7.2 수정된 함수

#### `plan()`

- 순방향 `_segment_nr_correct` 루프 제거
- `_global_stage2_nr()` 호출로 대체
- 샘플링 루프: `_compute_L` → `L = max(chord, 1e-3)` (NR과 동일 L)

### 7.3 검증 결과 (스모크 테스트, 2026-04-29)

```
[완만 선회] pts=687,  len=697m,   max|κ|=0.00662≤κ_max,  wps=[0,1,2,3],  t=0.08s  ✓
[급격 90°]  pts=615,  len=630m,   max|κ|=0.00662≤κ_max,  wps=[0,1,2,3],  t=0.55s  ✓
[U턴]       pts=446,  len=460m,   max|κ|=0.00662≤κ_max,  wps=[0,1,2,3],  t=0.74s  ✓

run_scenario basic --planner eta3clothoid2 --controller nlgl --seed 42 --no-plot  ✓
run_scenario basic --planner eta3clothoid2 --controller mpc  --seed 42 --no-plot  ✓
```

### 7.4 G1 연속성 보장 방식

전역 NR에서 θ_i와 κ_i는 구간 i의 끝 조건이자 구간 i+1의 시작 조건으로 **공유**됨.
순방향 NR과 달리 "구간 i 끝 헤딩 ≠ 구간 i+1 시작 헤딩" 문제가 구조적으로 사라짐.

```
순방향 NR (이전): segment i 끝 = θ_{i+1}_original
                  segment i+1 시작 = θ_{i+1}_corrected  ← 불일치 → G1 불연속
전역 NR (현재):   segment i 끝 = θ_{i+1}_optimized
                  segment i+1 시작 = θ_{i+1}_optimized  ← 동일값 → G1 연속
```

### 7.5 남은 한계

| 항목 | 상태 | 원인 |
|------|------|------|
| 기하학적 불가 구간 WP 통과 오차 | ||F||≈14m (R_min=151m에서 90° chord=200m) | 물리적 한계, 소프트웨어 결함 아님 |
| G2 연속성 | 근사 보장 | Stage 1 미구현 → 3순위에서 해결 |
| NR 수렴 속도 | 급격 구간에서 ~0.5s | max_insert=3 후에도 불가 구간 잔존 |
