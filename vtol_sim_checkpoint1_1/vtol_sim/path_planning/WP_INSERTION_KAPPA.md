# WP 삽입 — κ_max 기하학적 실현 가능성 보장

> 대상 파일: `path_planning/eta3clothoid_planner.py`  
> 목적: 기하학적으로 불가능한 회전을 사전 탐지하여 중간 WP를 삽입, NR이 수렴 불가능한 구간을 원천 제거

---

## 1. 왜 WP 삽입인가

### 전역 NR도 수렴 불가능한 경우

어떤 NR(tanh 또는 장벽법 포함)도 다음 조건에서는 수렴하지 못한다.

```
구간 i에 필요한 최소 곡률 > κ_max
```

이 경우 물리적으로 해당 WP를 통과할 수 있는 클로소이드 구간이 존재하지 않는다.  
NR은 잔차를 줄이려 하지만, 해 자체가 없으므로 무한히 진동하거나 발산한다.

### 클로소이드의 선형 κ 보장

클로소이드의 핵심 성질:

```
κ(s) = κ_i + (κ_j − κ_i)/L · s      ← s에 대해 선형
```

| 조건 | 결론 |
|------|------|
| `|κ_i| ≤ κ_max` **AND** `|κ_j| ≤ κ_max` | 전 구간: `max|κ(s)| = max(|κ_i|, |κ_j|) ≤ κ_max` |

즉 **끝점 곡률만 제약 내에 있으면 전 구간이 자동으로 보장된다.**  
따라서 "κ_max 이내에서 해결 가능한가"는 끝점 조건이 실현 가능한지로 귀결된다.

### WP 삽입의 아이디어

회전이 너무 큰 구간 → 두 개의 작은 회전 구간으로 분할 → 각 구간이 독립적으로 κ_max 이내에서 실현 가능.

```
WP_i ──────(회전 너무 큼)────── WP_{i+1}
        ↓ 중점 WP 삽입
WP_i ──(절반 회전)── WP_mid ──(절반 회전)── WP_{i+1}
```

---

## 2. 기하학적 판정 기준

### κ_needed 유도

단순화: 구간 i를 균일한 곡률(원호)로 근사.

```
chord_i   = ‖WP_{i+1} − WP_i‖
Δθ_i      = wrap(θ_{i+1} − θ_i)        ← 전체 선회각 (rad)
arc_length ≈ chord_i                    ← 선회각 작을 때 근사

원호 공식:  Δθ = κ · L  →  κ = Δθ / L

단, chord와 arc의 관계: chord = 2R · sin(Δθ/2)
R = chord / (2 · sin(Δθ/2))  →  κ = 1/R = 2 · sin(Δθ/2) / chord

중소 각도 근사 (sin(x) ≈ x):
κ_needed ≈ 2 · |Δθ| / chord_i
```

**판정 기준:**

```
κ_needed = 2 · |Δθ_i| / chord_i

if κ_needed > κ_max:
    → 이 구간은 κ_max 이내에서 WP 통과 불가 → WP 삽입 필요
```

### 정밀 판정 (선택)

```python
# sin 버전 (더 정확, 큰 각도에서 차이)
sin_half = np.sin(abs(delta_theta) / 2.0)
if chord > 1e-3:
    kappa_needed = 2.0 * sin_half / chord
else:
    kappa_needed = np.inf
```

---

## 3. WP 삽입 전략

### 3.1 삽입 위치: 중점

```python
WP_mid = (WP_i + WP_{i+1}) / 2.0
```

**선택 근거**: 두 구간에 회전을 균등 분배하여 각 구간의 κ_needed를 절반으로 줄임.

더 정밀하게: 삽입 WP의 위치를 κ_needed가 균등해지도록 조정할 수 있으나 중점이 충분히 효과적.

### 3.2 삽입 WP의 초기 θ와 κ

```python
# θ_mid: 두 코드 방향의 이등분선
d_in  = _unit(WP_mid - WP_i)
d_out = _unit(WP_{i+1} - WP_mid)
bis   = _unit(d_in + d_out)
theta_mid = np.arctan2(bis[1], bis[0])

# κ_mid: 0 (내부 노드 초기값 기본값)
kappa_mid = 0.0
```

### 3.3 반복 삽입

단일 삽입으로 κ_needed ≤ κ_max가 되지 않으면 재분할.

```
초기 구간: κ_needed = K > κ_max
1회 삽입 후: 각 구간 κ_needed ≈ K/2  (완전 균등 분할 가정)
n회 삽입 후: 각 구간 κ_needed ≈ K/2^n

필요 삽입 횟수: n = ceil(log2(K / κ_max))
```

무한루프 방지를 위해 `max_insert` 횟수 제한.

---

## 4. `_check_and_insert_wps` 구현

```python
def _check_and_insert_wps(
    wps_2d:    np.ndarray,    # (N, 2) — 수평 WP
    thetas:    np.ndarray,    # (N,)   — 초기 헤딩
    kappas:    np.ndarray,    # (N,)   — 초기 곡률
    kappa_max: float,
    max_insert: int = 3,      # 단일 구간 최대 삽입 횟수
    use_sin:   bool = True,   # True: sin 판정, False: 선형 근사
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    κ_needed > κ_max 구간을 검출하여 중점 WP를 삽입한다.

    삽입된 WP:
      - θ_mid: 이등분선 방향
      - κ_mid: 0

    Returns
    -------
    wps_2d_new  : (M, 2)  M ≥ N
    thetas_new  : (M,)
    kappas_new  : (M,)
    """
    wps    = list(wps_2d)
    thetas = list(thetas)
    kappas = list(kappas)

    i = 0
    insert_counts = {}    # 구간별 삽입 횟수 추적

    while i < len(wps) - 1:
        chord_vec  = np.array(wps[i + 1]) - np.array(wps[i])
        chord      = np.linalg.norm(chord_vec)

        if chord < 1e-3:
            i += 1
            continue

        delta_theta = _wrap(thetas[i + 1] - thetas[i])

        if use_sin:
            sin_half      = np.sin(abs(delta_theta) / 2.0)
            kappa_needed  = 2.0 * sin_half / chord
        else:
            kappa_needed  = 2.0 * abs(delta_theta) / chord

        seg_key = i    # 삽입 전 원본 구간 인덱스 (근사 추적용)
        if kappa_needed > kappa_max:
            n_done = insert_counts.get(seg_key, 0)
            if n_done >= max_insert:
                # 최대 삽입 횟수 초과 — 경고 출력 후 통과
                import warnings
                warnings.warn(
                    f"WP 구간 {seg_key}: κ_needed={kappa_needed:.4f} > κ_max={kappa_max:.4f}, "
                    f"최대 삽입({max_insert}회) 초과. 이 구간은 실현 불가."
                )
                i += 1
                continue

            # 중점 삽입
            wp_mid    = (np.array(wps[i]) + np.array(wps[i + 1])) / 2.0
            d_in      = _unit(wp_mid - np.array(wps[i]))
            d_out     = _unit(np.array(wps[i + 1]) - wp_mid)
            bis       = _unit(d_in + d_out)
            theta_mid = np.arctan2(bis[1], bis[0])
            kappa_mid = 0.0

            wps.insert(i + 1, wp_mid)
            thetas.insert(i + 1, theta_mid)
            kappas.insert(i + 1, kappa_mid)

            insert_counts[seg_key] = n_done + 1
            # i를 증가시키지 않고 동일 구간 재검사
        else:
            i += 1

    return np.array(wps), np.array(thetas), np.array(kappas)
```

---

## 5. `plan()`과의 연결

### 호출 위치

```python
def plan(self, waypoints_ned, aircraft_params, initial_state=None):
    ...
    wps_2d = wps[:, :2]
    N      = len(wps)

    # Stage 1 폴백 (기존)
    thetas, kappas = _eta3_initial_guess(
        wps_2d, theta0, kappa0, theta_N, kappa_N
    )
    kappas = np.clip(kappas, -kappa_max * 0.99, kappa_max * 0.99)

    # ── NEW: 기하학적 실현 가능성 사전 검사 ──────────────────────────
    wps_2d_ext, thetas, kappas = _check_and_insert_wps(
        wps_2d, thetas, kappas, kappa_max,
        max_insert=3,
    )
    N_ext = len(wps_2d_ext)   # 삽입 후 WP 수 (N_ext ≥ N)

    # ── Stage 2: 전역 NR (또는 기존 per-segment NR) ──────────────────
    thetas, kappas = _global_stage2_nr(
        wps_2d_ext, thetas, kappas, kappa_max, ...
    )
    ...
```

### N 변화 처리

삽입 후 N이 늘어나므로 이후 루프를 `N_ext`로 변경:

```python
# Stage 2 샘플링 루프
for i in range(N_ext - 1):
    target_disp = wps_2d_ext[i + 1] - wps_2d_ext[i]
    ...
```

---

## 6. wp_index 마킹 처리

삽입된 WP는 원본 WP가 아니므로 `wp_index` 마킹에서 제외해야 한다.

### 방법 A: 삽입 WP 추적 플래그

`_check_and_insert_wps` 반환에 `is_inserted` 배열 추가:

```python
def _check_and_insert_wps(...) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ...
    is_inserted = [False] * len(original_wps)   # 원본 WP는 False
    ...
    is_inserted.insert(i + 1, True)             # 삽입된 WP는 True
    ...
    return np.array(wps), np.array(thetas), np.array(kappas), np.array(is_inserted)
```

`plan()`에서 `wp_index` 마킹 시:

```python
# 원본 WP 인덱스 복원 (is_inserted=False 인 항목만 카운트)
orig_idx = 0
for k, inserted in enumerate(is_inserted):
    if not inserted:
        wp_marks[seg_start_idx[k]] = orig_idx
        orig_idx += 1
```

### 방법 B: 원본 WP 위치 직접 비교 (더 단순)

```python
# plan() 조립 단계에서
for k, wp_mark_idx in enumerate(orig_wp_positions_in_path):
    points[wp_mark_idx].wp_index = k
# 삽입 WP 위치 점들은 wp_index = None (기본값 유지)
```

---

## 7. 엣지 케이스

### 7.1 chord → 0 (거의 동일 위치 WP)

```python
if chord < 1e-3:
    i += 1
    continue    # 판정 건너뜀, κ_needed 계산 불가
```

### 7.2 180° 이상 회전 (Δθ ≈ π)

sin 버전: `2·sin(π/2)/chord = 2/chord` → chord가 짧으면 매우 큰 κ_needed.  
최대 삽입 횟수에 걸리기 전에 chord가 충분히 짧아지면 자연스럽게 해결.

```python
# 반대 방향 WP (역방향 선회): Δθ ≈ π 탐지
if abs(abs(delta_theta) - np.pi) < 0.1:
    # 항상 삽입 (정확한 방향 계산이 불안정)
    # 중점에 오프셋 추가하여 한쪽으로 치우침
    perp = np.array([-chord_vec[1], chord_vec[0]])
    perp_unit = _unit(perp)
    wp_mid = (np.array(wps[i]) + np.array(wps[i+1])) / 2.0 + 0.1 * perp_unit
```

### 7.3 연속 삽입으로 WP 수 폭발

실제 비행 시나리오에서 일반적이지 않지만 극단적 회전이 많으면 WP 수가 지수 증가.  
`max_insert = 3` → 최대 구간당 7개 WP (2³-1 삽입). 총 WP 증가 상한 추가 권장:

```python
MAX_TOTAL_WPS = 500
if len(wps) > MAX_TOTAL_WPS:
    warnings.warn("WP 삽입 한계 도달")
    break
```

---

## 8. 수학적 보장 정리

### 삽입 후 κ_max 준수 보장

**전제**: `_check_and_insert_wps` 완료 후 모든 구간 i에 대해:
```
κ_needed_i ≤ κ_max
```

**클로소이드 선형성으로부터**:
- 전역 NR이 수렴하여 `|κ_i| ≤ κ_max`, `|κ_{i+1}| ≤ κ_max` 를 달성하면
- 구간 전체: `κ(s) = κ_i + (κ_{i+1} − κ_i)/L · s` → `max|κ(s)| = max(|κ_i|, |κ_{i+1}|) ≤ κ_max`

즉 **WP 삽입 + 전역 NR 수렴 = 전 구간 κ_max 준수**.

### WP 통과 보장

| WP 종류 | 통과 보장 여부 |
|---------|----------------|
| 원본 WP | 전역 NR 수렴 시 잔차 < tol 범위 내 통과 |
| 삽입 WP | 동일하게 전역 NR 대상에 포함되어 통과 보장 |

삽입 WP는 실제 임무 요구 WP가 아니므로, `wp_index = None`으로 마킹되어 비행 체계에 임무 WP로 보고되지 않는다.

---

## 9. 장벽법과의 비교 및 조합

| 항목 | WP 삽입 | 장벽법 (CONSTRAINED_OPTIMIZATION_KAPPA.md) |
|------|---------|-------------------------------------------|
| 적용 레이어 | NR 이전 (사전 처리) | NR 내부 (최적화 수정) |
| κ_max 초과 구간 | 분할로 제거 | NR 내에서 회피 시도 |
| NR 수렴 불가 구간 처리 | **사전 제거** (핵심) | 처리 불가 (NR이 실패) |
| 연속성 보장 | 직접 보장 안 함 (전역 NR 필요) | 직접 보장 안 함 |
| 구현 복잡도 | 낮음 | 높음 |
| WP 수 증가 | 가능 | 없음 |

### 권장 조합

```
_check_and_insert_wps()     ← 1단계: 실현 불가 구간 제거
  +
_global_stage2_barrier_nr() ← 2단계: 수렴 + κ_max 내재화
또는
_global_stage2_nr()         ← 2단계: tanh 버전
```

WP 삽입 단독으로는 순방향 NR 덮어쓰기 문제를 해결하지 못한다.  
**반드시 전역 NR과 함께 사용해야 G1 연속성까지 개선된다.**

---

## 10. 구현 순서 권장

```
1. _check_and_insert_wps() 구현 + plan()에 연결
   → 즉시 효과: 수렴 불가 구간 제거, NR 안정성 향상

2. _global_stage2_nr() (또는 _global_stage2_barrier_nr()) 구현
   → WP 삽입과 조합하여 G0 + κ_max 구조적 보장

3. is_inserted 플래그로 wp_index 마킹 정확화

4. Stage 1 (_eta3_g2_residual) 구현
   → 더 좋은 초기값으로 삽입 WP 수 감소
```
