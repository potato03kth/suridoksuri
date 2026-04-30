# 제약 최적화 (로그 장벽법) — κ_max 적용 방법

> 대상 파일: `path_planning/eta3clothoid_planner.py`  
> 목적: L.287 `np.clip` 제거 → 로그 장벽 내재화로 NR 수렴 중 κ ∈ (−κ_max, κ_max) 구조적 보장

---

## 1. 왜 클리핑 대신 장벽법인가

### 현재 코드의 문제 (`_segment_nr_correct`, L.287)

```python
kappa_i_c = np.clip(kappa_i_c + step * delta[1], -kappa_max, kappa_max)
```

| 결함 | 원인 |
|------|------|
| 야코비안 불연속 | 경계에서 ∂κ/∂step = 0 → NR 방향 왜곡 |
| 자유도 손실 | κ가 경계에 붙으면 θ 하나로 2D 끝점을 맞춰야 함 |
| WP 통과 오차 | 잔차가 남아도 κ 경계에 걸리면 NR이 멈춤 |

### 로그 장벽법의 원리

κ를 최적화 변수로 유지하면서, 목적함수에 장벽 항을 추가한다.

```
κ ∈ (−κ_max, κ_max)에서만 유한한 함수:
B(κ) = −log(κ_max − κ) − log(κ_max + κ)
```

| κ 위치 | B(κ) 값 |
|--------|---------|
| κ = 0 | 2·log(κ_max) (유한) |
| κ → ±κ_max | +∞ |
| κ > κ_max | 정의 없음 (NR이 자동으로 회피) |

---

## 2. 수학적 기반

### 장벽 함수와 미분

```
B(κ)  = −log(κ_max − κ) − log(κ_max + κ)

dB/dκ = 1/(κ_max − κ) − 1/(κ_max + κ)
       = 2κ / (κ_max² − κ²)          ← g_B (스칼라)

d²B/dκ² = 1/(κ_max − κ)² + 1/(κ_max + κ)²   ← h_B (스칼라)
```

### 증가된 목적함수

NR의 각 반복을 다음 미최소화 문제의 Newton 스텝으로 해석한다.

```
f(θ_i, κ_i) = ½ · ‖res‖² + μ · B(κ_i)

∇f  = [J^T res + μ·[0, g_B]]         (2,)  벡터
∇²f ≈ J^T J + μ·diag([0, h_B])       (2,2) 행렬  (Gauss-Newton 근사)
```

Newton 스텝:

```
(J^T J + μ·diag([0, h_B])) · δx = −(J^T res + μ·[0, g_B])
```

이를 풀면 κ_i가 ±κ_max에 접근할수록 h_B → ∞ → δκ_i → 0이 되어 경계를 자연스럽게 회피한다.

### μ 감소 스케줄

```
μ_0    = 1.0 × κ_max²     ← κ 스케일에 맞춰 초기화
μ_{k+1} = max(μ_k × 0.5, μ_min)   ← 매 outer 반복마다 감소
μ_min  = 1e-4 × κ_max²   ← 하한 (너무 작으면 수치 불안정)
```

μ가 클수록 장벽이 강하게 작용 → 초기에 경계 침범 방지.  
μ가 줄어들수록 본래 WP 통과 조건(잔차 최소화)에 집중.

---

## 3. 적용 위치

```
eta3clothoid_planner.py
│
├─ _segment_nr_correct()   ← Stage 2 per-segment NR (1차 적용, 즉시 효과)
│     L.283–291: J 계산 + solve + clip
│                → J^T J + barrier 방식으로 교체
│
└─ _solve_eta3_g2()        ← Stage 1 NR (Stage 1 구현 후 적용)
      L.206–215: NR 루프 (현재 주석 처리)
                → _eta3_g2_residual 구현 시 같은 방식 적용
```

---

## 4. Stage 2 수정 — `_segment_nr_correct`

### 현재 코드 (L.279–291)

```python
try:
    delta = np.linalg.solve(J, -res)
except np.linalg.LinAlgError:
    delta = np.linalg.lstsq(J, -res, rcond=None)[0]

step = min(1.0, 0.3 / (np.linalg.norm(delta) + 1e-12))
theta_i_c += step * delta[0]
kappa_i_c  = np.clip(kappa_i_c + step * delta[1], -kappa_max, kappa_max)  # ← 제거 대상
```

### 변경 후 코드

```python
# 장벽 스칼라 계산
kappa_gap = kappa_max**2 - kappa_i_c**2 + 1e-12   # 0 나눔 방지
g_B = 2.0 * kappa_i_c / kappa_gap
h_B = (1.0 / (kappa_max - kappa_i_c + 1e-12)**2
     + 1.0 / (kappa_max + kappa_i_c + 1e-12)**2)

# 증가된 정규 방정식  (2×2)
H_aug = J.T @ J + mu * np.diag([0.0, h_B])
g_aug = J.T @ res + mu * np.array([0.0, g_B])

try:
    delta = np.linalg.solve(H_aug, -g_aug)
except np.linalg.LinAlgError:
    delta = np.linalg.lstsq(H_aug, -g_aug, rcond=None)[0]

step = min(1.0, 0.3 / (np.linalg.norm(delta) + 1e-12))
theta_i_c += step * delta[0]
kappa_i_c += step * delta[1]           # clip 없음 — 장벽이 내재적으로 막음
kappa_i_c  = np.clip(kappa_i_c,        # 수치 오류 완충용 소프트 가드만 유지
                     -kappa_max * 0.999,
                      kappa_max * 0.999)
```

### 함수 시그니처 변경

```python
def _segment_nr_correct(theta_i, kappa_i, theta_j, kappa_j,
                        target_disp, kappa_max,
                        max_iter=30, tol=1e-4,
                        mu_init=None,          # ← 추가
                        mu_min=None            # ← 추가
                        ) -> tuple[float, float, float]:
    mu     = mu_init if mu_init is not None else kappa_max ** 2
    mu_min = mu_min  if mu_min  is not None else 1e-4 * kappa_max ** 2
    ...
    for it in range(max_iter):
        ...
        # 장벽 스텝 (위 코드)
        ...
        # outer μ 감소: 일정 반복 후
        if it % 5 == 4:
            mu = max(mu * 0.5, mu_min)
```

---

## 5. Stage 2 전역 NR에 적용 (`_global_stage2_nr`)

ETA3CLOTHOID_STAGE2_REDESIGN.md의 전역 NR 구조에 장벽을 추가한다.  
tanh 재매개변수화와 **선택적** — 둘 중 하나만 사용.

### 변수 정의 (장벽 버전, tanh 없음)

```
x = [θ_1, κ_1, θ_2, κ_2, ..., θ_{N-2}, κ_{N-2}]   크기: 2(N-2)
κ_i = x[2*(i-1)+1]   (직접 사용, tanh 변환 없음)
```

### NR 반복 코드

```python
def _global_stage2_barrier_nr(
    wps_2d, thetas, kappas, kappa_max,
    max_iter=50, tol=1e-4, eps_jac=1e-6,
    mu_init=None, mu_min=None,
):
    N = len(wps_2d)
    if N <= 2:
        return thetas, kappas

    mu     = mu_init if mu_init is not None else kappa_max ** 2
    mu_min = mu_min  if mu_min  is not None else 1e-4 * kappa_max ** 2

    x = np.zeros(2 * (N - 2))
    x[0::2] = thetas[1:-1]
    x[1::2] = np.clip(kappas[1:-1], -kappa_max * 0.99, kappa_max * 0.99)

    theta_bc = (thetas[0],  thetas[-1])
    kappa_bc = (kappas[0],  kappas[-1])

    for it in range(max_iter):
        F = _global_residual_direct(x, wps_2d, theta_bc, kappa_bc)
        if np.linalg.norm(F) < tol:
            break

        J = _numerical_jacobian_global(
            x, wps_2d, theta_bc, kappa_bc, eps_jac
        )   # shape: (2(N-1), 2(N-2))

        # 장벽 기울기 및 헤시안 대각 (κ 위치만 비영)
        g_bar = np.zeros(2 * (N - 2))
        h_bar = np.zeros(2 * (N - 2))
        for k in range(N - 2):
            kp = x[2 * k + 1]
            gap = kappa_max ** 2 - kp ** 2 + 1e-12
            g_bar[2 * k + 1] = 2.0 * kp / gap
            h_bar[2 * k + 1] = (1.0 / (kappa_max - kp + 1e-12) ** 2
                               + 1.0 / (kappa_max + kp + 1e-12) ** 2)

        # 증가된 정규 방정식
        H_aug = J.T @ J + mu * np.diag(h_bar)
        g_aug = J.T @ F + mu * g_bar

        dx = np.linalg.lstsq(H_aug, -g_aug, rcond=None)[0]
        step = min(1.0, 0.5 / (np.linalg.norm(dx) + 1e-12))
        x += step * dx

        # 소프트 가드 (수치 안전)
        x[1::2] = np.clip(x[1::2], -kappa_max * 0.999, kappa_max * 0.999)

        # μ 감소
        if it % 5 == 4:
            mu = max(mu * 0.5, mu_min)

    thetas[1:-1] = x[0::2]
    kappas[1:-1] = x[1::2]
    return thetas, kappas
```

### `_global_residual_direct` (tanh 없는 버전)

```python
def _global_residual_direct(
    x, wps_2d, theta_bc, kappa_bc, n_quad=200
):
    """κ를 직접 사용 (tanh 없음)."""
    N = len(wps_2d)
    thetas_full = np.empty(N)
    kappas_full = np.empty(N)
    thetas_full[0],  kappas_full[0]  = theta_bc[0], kappa_bc[0]
    thetas_full[-1], kappas_full[-1] = theta_bc[1], kappa_bc[1]
    thetas_full[1:-1] = x[0::2]
    kappas_full[1:-1] = x[1::2]

    F = np.zeros(2 * (N - 1))
    for i in range(N - 1):
        chord     = wps_2d[i + 1] - wps_2d[i]
        chord_len = np.linalg.norm(chord)
        L = _compute_L(thetas_full[i], kappas_full[i],
                       thetas_full[i + 1], kappas_full[i + 1], chord_len)
        dp = _fresnel_endpoint(thetas_full[i], kappas_full[i],
                               kappas_full[i + 1], L, n_quad)
        F[2 * i]     = dp[0] - chord[0]
        F[2 * i + 1] = dp[1] - chord[1]
    return F
```

---

## 6. Stage 1 적용 방법 (참고, 미구현 구간)

`_solve_eta3_g2` 내 NR 루프가 활성화된 후 다음과 같이 적용.

```python
# _solve_eta3_g2 NR 루프 내
for it in range(max_iter):
    thetas[1:-1] = x[0::2]
    kappas[1:-1] = x[1::2]        # κ 직접 사용

    F = _eta3_g2_residual(thetas, kappas, wps_2d)   # shape: (2(N-2),)
    if np.linalg.norm(F) < tol:
        break

    J = _numerical_jacobian(...)   # shape: (2(N-2), 2(N-2))

    # 장벽: κ 위치 (x[1::2])에만 적용
    kappas_int = x[1::2]
    g_bar = np.zeros_like(x)
    h_bar = np.zeros_like(x)
    for k, kp in enumerate(kappas_int):
        gap = kappa_max ** 2 - kp ** 2 + 1e-12
        g_bar[2*k+1] = 2.0 * kp / gap
        h_bar[2*k+1] = (1/(kappa_max - kp + 1e-12)**2
                      + 1/(kappa_max + kp + 1e-12)**2)

    # Stage 1은 정방 시스템 (N-2 내부 노드, 2(N-2) 방정식 = 2(N-2) 변수)
    H_aug = J.T @ J + mu * np.diag(h_bar)
    g_aug = J.T @ F + mu * g_bar
    dx = np.linalg.lstsq(H_aug, -g_aug, rcond=None)[0]
    x += min(1.0, 0.5 / (np.linalg.norm(dx) + 1e-12)) * dx
    x[1::2] = np.clip(x[1::2], -kappa_max * 0.999, kappa_max * 0.999)

    if it % 5 == 4:
        mu = max(mu * 0.5, mu_min)
```

---

## 7. 수정되는 `plan()` 흐름

Stage 2 per-segment NR 교체 버전 (전역 NR 미적용 시 최소 변경):

```python
# _segment_nr_correct 호출 시 mu 파라미터 추가
for i in range(N - 1):
    target_disp = wps_2d[i + 1] - wps_2d[i]
    L, th_c, kp_c = _segment_nr_correct(
        thetas[i], kappas[i],
        thetas[i + 1], kappas[i + 1],
        target_disp, kappa_max,
        max_iter=self.nr_max_iter, tol=self.nr_tol,
        mu_init=kappa_max ** 2,          # ← 추가
        mu_min=1e-4 * kappa_max ** 2,    # ← 추가
    )
    thetas[i] = th_c
    kappas[i] = kp_c
```

전역 NR 적용 버전 (ETA3CLOTHOID_STAGE2_REDESIGN.md의 파이프라인):

```python
# Stage 1 폴백 유지
thetas, kappas = _eta3_initial_guess(wps_2d, theta0, kappa0, theta_N, kappa_N)
kappas = np.clip(kappas, -kappa_max * 0.99, kappa_max * 0.99)

# WP 삽입 (WP_INSERTION_KAPPA.md 참조)
wps_2d, thetas, kappas = _check_and_insert_wps(wps_2d, thetas, kappas, kappa_max)
N = len(wps_2d)

# 전역 NR + 장벽 (per-segment NR 전체 대체)
thetas, kappas = _global_stage2_barrier_nr(
    wps_2d, thetas, kappas, kappa_max,
    max_iter=self.nr_max_iter, tol=self.nr_tol,
)
```

---

## 8. 보장 항목 및 한계

| 항목 | 보장 여부 | 조건 |
|------|-----------|------|
| κ ∈ (−κ_max, κ_max) | **구조적 보장** | NR이 수렴하는 한 장벽이 경계 침범 차단 |
| κ = ±κ_max 도달 | **불가** | 점근적 — μ → 0이어도 정확히 도달 안 됨 |
| WP 위치 통과 (G0) | NR 수렴 시 보장 | 초기값 품질과 μ 스케줄 영향 받음 |
| G1/G2 연속성 | per-segment NR 시 불보장 | 전역 NR 적용 후에도 과결정 잔차 |
| NR 수렴 | 초기값 의존 | μ 너무 크면 장벽이 지배 → 수렴 느림 |

### μ 튜닝 지침

```
μ_0 너무 크면 → NR이 장벽만 최소화, WP 통과 조건 무시
μ_0 너무 작으면 → 초기 반복에서 κ 경계 침범 → g_B, h_B 발산
권장 범위: μ_0 = 0.1 × κ_max² ~ 10 × κ_max²
```

### tanh 방법과의 비교

| 항목 | 장벽법 | tanh 재매개변수화 |
|------|--------|-----------------|
| κ = ±κ_max 도달 | 불가 (점근) | 불가 (점근) |
| 구현 복잡도 | 높음 (μ 스케줄, H_aug 구성) | 낮음 (변수 변환만) |
| μ 튜닝 필요 | 필요 | 불필요 |
| 야코비안 연속성 | 연속 (h_B가 연속) | 연속 (sech² 연속) |
| 경계 근처 수치 안정성 | h_B → ∞ 위험 (소프트 가드 필요) | sech² → 0 (수렴 느림) |

---

## 9. 구현 순서 권장

```
1. _segment_nr_correct 내 L.287 클리핑 → 장벽 스텝 교체
   (per-segment NR 구조 유지, 최소 변경)

2. 동작 확인 후 _global_stage2_barrier_nr 구현
   (순방향 덮어쓰기 문제 동시 해결)

3. Stage 1 (_eta3_g2_residual) 구현 완료 후 동일 장벽 패턴 적용
```
