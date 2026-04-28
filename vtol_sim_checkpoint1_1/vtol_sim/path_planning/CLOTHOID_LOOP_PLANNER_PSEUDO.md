# ClothoidLoopPlanner — Pseudo Code

> 새 플래너 설계 문서  
> 파일명 예정: `clothoid_loop_planner.py`  
> 클래스명 예정: `ClothoidLoopPlanner`

---

## 설계 원칙 요약

| 항목 | 방침 |
|------|------|
| 곡률 상한 | `κ_max = 1/R_min` 하드 제한, 수식 구조적 보장 |
| R_min 계산 | `a_max = a_max_g × 9.81 × accel_tol`, `R_min = v² / a_max` |
| 코너 분류 | T_L ≤ T_max → 일반 코너 / T_L > T_max → 루프 코너 |
| WP 통과 | **보장 불가** (아래 분석 참조) — 검증·재생성 자리 확보 |
| 중복점 | `_straight_segment` 반환 시 P0 포함, P1 제외로 고정 |

---

## WP 정확 통과 불가 분석

```
WP_prev ────────── P_entry ~~[나선-원호-나선]~~ P_exit ─────────── WP_next
                      ↑                              ↑
              WP - d_in × T_L              WP + d_out × T_L

          WP 꼭짓점은 P_entry와 P_exit 사이의 직선 교점 —
          경로는 그 안쪽을 부드럽게 잘라서 통과함.
          즉, WP 꼭짓점은 경로 위에 없다.
```

**구조적 이유:**
- 클로소이드 코너 블렌딩은 WP를 "통과점"이 아닌 "방향 전환 기준점"으로 사용
- T_L이 클수록 WP와 경로 사이의 거리가 커짐
- T_L = 0이면 경로가 WP를 통과하지만, 그러면 곡률이 0→κ_max 전환이 순간적(Dubins와 동일)

**결론:**  
WP를 정확히 통과하는 클로소이드 경로를 만들려면, 나선 파라미터를 WP 통과 조건으로 역산하는
별도의 비선형 풀이가 필요하다. 이 플래너에서는 자리만 확보한다.

---

## 클래스 구조 개요

```
ClothoidLoopPlanner(BasePlanner)
│
├── __init__(ds, accel_tol, spiral_fraction, min_turn_deg, end_extension, loop_margin)
│
├── [PUBLIC]  plan(waypoints_ned, aircraft_params, initial_state) → Path
│
├── [PRIVATE] _build_2d_path(wps_2d, R_min, κ_max) → (pts, s, κ, wp_marks)
│   ├── PHASE 1: 각 내부 WP 코너 분류
│   └── PHASE 2: 세그먼트 조립
│
├── [PRIVATE] _corner_params(α, R_min) → (T_L, L_s)
├── [PRIVATE] _build_normal_corner(P_entry, P_exit, WP, L_s, R_min, α) → (pts, κ)
├── [PRIVATE] _build_loop_corner(WP, d_in, d_out, R_min, α) → (pts, κ, loop_exit)
├── [PRIVATE] _straight_segment(P0, P1) → (pts, κ)   ← P0 포함 / P1 제외
├── [PRIVATE] _tangent_length(L_s, R, α) → T_L
├── [PRIVATE] _headings_from_pts(pts) → chi_arr
│
├── [PSEUDO]  _check_wp_passage(corner_pts, WP, tol) → bool
└── [PSEUDO]  _regenerate_corner_through_wp(P_entry, WP, P_exit, R_min, α) → (pts, κ)
```

---

## Pseudo Code

---

### `__init__`

```
PARAMETERS:
    ds              : float = 1.0     # 경로 샘플 간격 (m)
    accel_tol       : float = 0.9     # κ 하드제한 여유율 (0 < tol ≤ 1.0)
    spiral_fraction : float = 0.4     # 코너 각도 중 나선이 차지하는 비율
    min_turn_deg    : float = 2.0     # 이 이하 선회각은 직선으로 처리
    end_extension   : float = 15.0    # 마지막 WP 이후 경로 연장 (m)
    loop_margin     : float = 0.45    # T_L 상한 비율 (코너 겹침 방지)

INIT:
    self.ds              = ds
    self.accel_tol       = accel_tol
    self.spiral_fraction = spiral_fraction
    self.min_turn_rad    = deg2rad(min_turn_deg)
    self.end_extension   = end_extension
    self.loop_margin     = loop_margin
```

---

### `plan`

```
FUNCTION plan(waypoints_ned, aircraft_params, initial_state=None) → Path:

    t_start = now()

    # ── 1. 제한값 계산 ──────────────────────────────────────────────────
    v       = aircraft_params["v_cruise"]
    g       = aircraft_params.get("gravity", 9.81)
    a_max_g = aircraft_params["a_max_g"]

    a_max   = a_max_g * 9.81 * self.accel_tol   # 하드 가속도 상한 (m/s²)
    R_min   = v² / a_max                         # 최소 회전 반경 (m)
    κ_max   = 1.0 / R_min                        # 최대 곡률 (1/m) — 하드 제한

    # ── 2. 입력 검증 ────────────────────────────────────────────────────
    wps = asarray(waypoints_ned, float)
    IF wps.ndim != 2 OR wps.shape[1] != 3:
        RAISE ValueError("waypoints_ned must be (N, 3)")
    IF len(wps) < 2:
        RAISE ValueError("waypoints는 최소 2개 필요")

    # ── 3. 수평 경로 생성 ───────────────────────────────────────────────
    pts_2d, s_arr, κ_arr, wp_marks = _build_2d_path(wps[:, :2], R_min, κ_max)

    # ── 4. 고도 보간 (호 길이 기준 linear interpolation) ────────────────
    wp_s    = cumulative arc-lengths at each WP (2D projected)
    alt_arr = interp(s_arr, wp_s, wps[:, 2])

    # ── 5. 상승각 계산 ──────────────────────────────────────────────────
    gamma_arr[k] = arctan2(alt_arr[k+1] - alt_arr[k],
                           s_arr[k+1]   - s_arr[k])   for k in 0..N-2
    gamma_arr[-1] = gamma_arr[-2]

    # ── 6. 방위각 계산 ──────────────────────────────────────────────────
    chi_arr = _headings_from_pts(pts_2d)

    # ── 7. PathPoint 조립 ───────────────────────────────────────────────
    FOR k in 0..len(pts_2d)-1:
        points.append(PathPoint(
            pos      = [pts_2d[k,0], pts_2d[k,1], alt_arr[k]],
            v_ref    = v,
            chi_ref  = chi_arr[k],
            gamma_ref= gamma_arr[k],
            curvature= κ_arr[k],
            s        = s_arr[k],
            wp_index = wp_marks.get(k, None),
        ))

    RETURN Path(
        points        = points,
        waypoints_ned = wps,
        total_length  = s_arr[-1],
        planning_time = now() - t_start,
    )
```

---

### `_build_2d_path`

```
FUNCTION _build_2d_path(wps_2d, R_min, κ_max) → (pts, s, κ, wp_marks):

    N_wp = len(wps_2d)

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 1: 코너 분류
    # ═══════════════════════════════════════════════════════════════════

    corners = [None] * N_wp     # None=직선처리, dict=코너 정보

    FOR i in 1 .. N_wp-2:      # 내부 WP만

        d_in  = unit(wps_2d[i]   - wps_2d[i-1])
        d_out = unit(wps_2d[i+1] - wps_2d[i])
        α     = signed_turn(d_in, d_out)   # 좌선회 음 / 우선회 양

        IF |α| < self.min_turn_rad:
            corners[i] = None              # 직선 처리
            CONTINUE

        seg_in  = ||wps_2d[i]   - wps_2d[i-1]||
        seg_out = ||wps_2d[i+1] - wps_2d[i]||
        T_max   = self.loop_margin * min(seg_in, seg_out)

        T_L, L_s = _corner_params(α, R_min)

        IF T_L ≤ T_max:
            # ── 일반 코너 ──────────────────────────────────────────────
            P_entry = wps_2d[i] - d_in  * T_L
            P_exit  = wps_2d[i] + d_out * T_L
            corners[i] = {
                type    : NORMAL,
                P_entry : P_entry,
                P_exit  : P_exit,
                L_s     : L_s,
                α       : α,
                d_in    : d_in,
                d_out   : d_out,
                wp      : wps_2d[i],
            }
        ELSE:
            # ── 루프 코너 (직접 전환 불가) ─────────────────────────────
            corners[i] = {
                type    : LOOP,
                α       : α,
                d_in    : d_in,
                d_out   : d_out,
                wp      : wps_2d[i],
            }

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 2: 세그먼트 조립 (P0 포함 / P1 제외 원칙으로 중복점 제거)
    # ═══════════════════════════════════════════════════════════════════

    all_pts  = []
    all_κ    = []
    wp_marks = {}
    seg_start = wps_2d[0].copy()

    # 첫 점 (WP0) 삽입 — 루프 최초 시작점
    all_pts.append(wps_2d[0])
    all_κ.append(0.0)
    wp_marks[0] = 0

    FOR i in 0 .. N_wp-1:

        IF i == N_wp-1:
            # ── 마지막 WP: 연장 직선 ───────────────────────────────────
            d_last  = unit(wps_2d[-1] - wps_2d[-2])
            seg_end = wps_2d[-1] + d_last * self.end_extension
            s_pts, s_κ = _straight_segment(seg_start, seg_end)   # P1 제외
            all_pts += s_pts
            all_κ   += s_κ
            wp_marks[len(all_pts) - 1] = N_wp - 1    # 마지막 WP 마킹
            BREAK

        c = corners[i+1]   # 다음 WP의 코너 정보

        IF c is None:
            # ── 직선 구간 ──────────────────────────────────────────────
            next_is_last = (i+1 == N_wp-1)
            IF next_is_last:
                seg_end = wps_2d[i+1]
            ELSE IF corners[i+2] is NORMAL:
                seg_end = corners[i+2]["P_entry"]
            ELSE IF corners[i+2] is LOOP:
                seg_end = corners[i+2]["wp"]       # 루프는 WP에서 시작
            ELSE:
                seg_end = wps_2d[i+1]

            s_pts, s_κ = _straight_segment(seg_start, seg_end)
            # 중간 WP 마킹 (직선 통과 WP)
            IF 0 < i+1 < N_wp-1 AND corners[i+1] is None:
                idx_local = argmin(||s_pts - wps_2d[i+1]||)
                wp_marks[len(all_pts) + idx_local] = i+1
            all_pts += s_pts
            all_κ   += s_κ
            seg_start = seg_end

        ELSE IF c["type"] == NORMAL:
            # ── 일반 코너 ──────────────────────────────────────────────
            # 직선: seg_start → P_entry (P1 제외)
            s_pts, s_κ = _straight_segment(seg_start, c["P_entry"])
            all_pts += s_pts
            all_κ   += s_κ

            # 코너 (나선-원호-나선)
            c_pts, c_κ = _build_normal_corner(
                c["P_entry"], c["P_exit"], c["wp"],
                c["L_s"], R_min, c["α"]
            )

            # [PSEUDO] WP 통과 검증 및 재생성 자리
            IF NOT _check_wp_passage(c_pts, c["wp"], tol=R_min * 0.1):
                c_pts, c_κ = _regenerate_corner_through_wp(
                    c["P_entry"], c["wp"], c["P_exit"], R_min, c["α"]
                )

            # WP 마킹: 코너 내 WP에 가장 가까운 점
            idx_local = argmin(||c_pts - c["wp"]||)
            wp_marks[len(all_pts) + idx_local] = i+1

            all_pts += c_pts       # c_pts[0]은 P_entry → 이전 직선 P1과 동일점이므로
            all_κ   += c_κ        # c_pts[1:]로 교체하거나 조립 단계에서 처리
            seg_start = c["P_exit"]

        ELSE IF c["type"] == LOOP:
            # ── 루프 코너 ──────────────────────────────────────────────
            # 직선: seg_start → WP (루프 진입점, P1 제외)
            s_pts, s_κ = _straight_segment(seg_start, c["wp"])
            all_pts += s_pts
            all_κ   += s_κ

            # 루프 (나선-큰원호-나선)
            l_pts, l_κ, loop_exit = _build_loop_corner(
                c["wp"], c["d_in"], c["d_out"], R_min, c["α"]
            )

            # WP 마킹: 루프 시작점 = WP에 가장 가까운 점
            idx_local = argmin(||l_pts - c["wp"]||)
            wp_marks[len(all_pts) + idx_local] = i+1

            all_pts += l_pts
            all_κ   += l_κ
            seg_start = loop_exit

    pts = array(all_pts)
    κ   = array(all_κ)
    ds  = ||diff(pts)||
    s   = concat([0.0], cumsum(ds))

    RETURN pts, s, κ, wp_marks
```

---

### `_corner_params`

```
FUNCTION _corner_params(α, R_min) → (T_L, L_s):

    θ_s = |α| * self.spiral_fraction
    L_s = 2.0 * R_min * θ_s
    T_L = _tangent_length(L_s, R_min, α)

    RETURN T_L, L_s
```

---

### `_build_normal_corner`

```
FUNCTION _build_normal_corner(P_entry, P_exit, WP, L_s, R_min, α) → (pts, κ):

    # 기존 ClothoidPlanner._build_corner()와 동일 구조
    sign    = sign(α)
    abs_α   = |α|
    θ_s     = abs_α * self.spiral_fraction
    arc_ang = abs_α - 2*θ_s               # 원호 각도 (음이면 원호 없음)
    χ_in    = arctan2(d_in[1], d_in[0])   # d_in = unit(WP - P_entry)

    # 진입 나선: κ(s) = sign * s / (R_min * L_s),  s ∈ [0, L_s]
    #   헤딩: h(s) = χ_in + sign * s² / (2 * R_min * L_s)
    entry_pts, entry_κ = integrate_spiral(P_entry, χ_in, L_s, R_min, sign)

    # 원호 (arc_ang > 0 인 경우만)
    IF arc_ang > 1e-4:
        arc_pts, arc_κ = integrate_arc(last(entry_pts), last(entry_h),
                                       arc_ang, R_min, sign)

    # 이탈 나선: κ(s) = sign * (L_s - s) / (R_min * L_s),  s ∈ [0, L_s]
    #   헤딩: h(s) = χ_out - sign * (L_s - s)² / (2 * R_min * L_s)
    χ_out = χ_in + sign * abs_α
    exit_pts, exit_κ = integrate_spiral_exit(last_pos, χ_out, L_s, R_min, sign)

    pts = concat(entry_pts, arc_pts[1:], exit_pts[1:])
    κ   = concat(entry_κ,   arc_κ[1:],  exit_κ[1:])

    # ── 하드 곡률 제한 검증 ─────────────────────────────────────────────
    ASSERT max(|κ|) ≤ 1/R_min + ε    # 구조적으로 보장되어야 하나 안전망

    RETURN pts, κ
```

---

### `_build_loop_corner`

```
FUNCTION _build_loop_corner(WP, d_in, d_out, R_min, α) → (pts, κ, loop_exit):

    # ── 루프 방향 및 각도 결정 ─────────────────────────────────────────
    # 직접 선회 방향의 반대로 루프
    loop_sign = -sign(α)
    α_loop    = 2π - |α|      # 루프 선회 총 각도 (항상 > π)

    # ── 루프 나선 파라미터 ─────────────────────────────────────────────
    θ_s_loop  = α_loop * self.spiral_fraction
    L_s_loop  = 2.0 * R_min * θ_s_loop
    arc_ang   = α_loop - 2 * θ_s_loop         # 루프 원호 각도 (> π)

    # ── 루프 시작 헤딩 ─────────────────────────────────────────────────
    χ_in  = arctan2(d_in[1], d_in[0])
    χ_out = χ_in + loop_sign * α_loop         # 이탈 헤딩 (= d_out 방향이어야 함)

    # ── 진입 나선: WP에서 시작 ─────────────────────────────────────────
    entry_pts, entry_κ = integrate_spiral(WP, χ_in, L_s_loop, R_min, loop_sign)

    # ── 큰 원호 (> π) ─────────────────────────────────────────────────
    arc_pts, arc_κ = integrate_arc(last(entry_pts), last(entry_h),
                                   arc_ang, R_min, loop_sign)

    # ── 이탈 나선: d_out 방향으로 정렬되며 종료 ──────────────────────
    exit_pts, exit_κ = integrate_spiral_exit(last(arc_pts), χ_out,
                                             L_s_loop, R_min, loop_sign)

    loop_exit = exit_pts[-1]   # 루프 이탈점 → 이후 seg_start로 사용

    pts = concat(entry_pts, arc_pts[1:], exit_pts[1:])
    κ   = concat(entry_κ,   arc_κ[1:],  exit_κ[1:])

    # ── 하드 곡률 제한 검증 ─────────────────────────────────────────────
    ASSERT max(|κ|) ≤ 1/R_min + ε

    RETURN pts, κ, loop_exit

    # ※ 루프 출구 방향이 d_out과 일치하는지 검증 필요
    # ※ loop_exit이 WP_next 방향 직선 위에 있는지 검증 필요
    # ※ L_s_loop가 매우 클 경우(α_loop ≈ 2π) 별도 처리 필요
```

---

### `_straight_segment` (중복점 제거 버전)

```
FUNCTION _straight_segment(P0, P1) → (pts, κ):
    # P0 포함, P1 제외 — 조립 단계에서 중복점 방지
    # 예외: 첫 점 삽입은 호출 전 별도로 처리

    dist = ||P1 - P0||
    IF dist < 1e-6:
        RETURN [], []     # 빈 배열 (P0는 이미 삽입됨)

    n    = max(1, ceil(dist / self.ds))
    t    = linspace(0, 1, n+1)[:-1]      # 0 ~ (n-1)/n, P1 제외
    pts  = P0 + t[:, None] * (P1 - P0)
    κ    = [0.0] * n

    RETURN pts, κ
```

---

### `_tangent_length`

```
FUNCTION _tangent_length(L_s, R, α) → T_L:
    # 로컬 프레임에서 코너 끝점 (ex, ey) 산출 후 접선 교점까지 거리 계산
    corner_pts = _corner_pts_local(L_s, R, |α|, sign=1)
    ex, ey = corner_pts[-1]
    sa = sin(|α|)
    IF |sa| < 1e-9: RETURN L_s
    T_L = ex - (ey / sa) * cos(|α|)
    RETURN max(T_L, L_s * 0.1)
```

---

### `_headings_from_pts`

```
FUNCTION _headings_from_pts(pts) → chi_arr:
    dx = diff(pts[:, 0])
    dy = diff(pts[:, 1])
    h  = arctan2(dy, dx)          # (N-1,)
    RETURN concat(h, [h[-1]])     # 마지막 점은 직전 값 복사
```

---

### [PSEUDO] `_check_wp_passage`

```
# ═══════════════════════════════════════════════════════════════════════
# [PSEUDO FUNCTION] — 미구현 자리 표시
# ═══════════════════════════════════════════════════════════════════════
#
# 클로소이드 코너 블렌딩은 WP 꼭짓점을 경로가 정확히 통과하지 않음.
# 이 함수는 "가장 가까운 경로 점"이 허용 오차 tol 이내인지 확인한다.
#
FUNCTION _check_wp_passage(corner_pts, WP, tol) → bool:
    min_dist = min(||corner_pts[k] - WP|| for k in all)
    RETURN min_dist ≤ tol

    # 참고: tol 기준 예시
    #   엄격 통과: tol = 0.5 m
    #   근사 통과: tol = R_min * 0.1
    #   미사용(항상 True): tol = inf
```

---

### [PSEUDO] `_regenerate_corner_through_wp`

```
# ═══════════════════════════════════════════════════════════════════════
# [PSEUDO FUNCTION] — 미구현 자리 표시
# ═══════════════════════════════════════════════════════════════════════
#
# WP를 정확히 통과하는 클로소이드 경로 재생성.
# _check_wp_passage가 False를 반환할 때 호출된다.
#
# 구현 후보 전략 (미결정):
#
#   Option A: 나선 파라미터 역산
#             WP 통과 조건 f(L_s, R_min) = 0 을 비선형 방정식으로 풀어
#             WP를 지나는 (L_s, R) 조합을 찾는다.
#             단, κ_max = 1/R_min 제약을 라그랑주 조건으로 포함해야 함.
#
#   Option B: WP 강제 보간점 삽입
#             P_entry → WP → P_exit 를 각각 독립적인 나선으로 연결.
#             연결부에서 C1 연속(방향 연속)은 보장할 수 있으나
#             C2(곡률 연속)는 보장 어려움.
#
#   Option C: 루프 경로로 강제 전환
#             WP를 통과하는 루프(_build_loop_corner)로 대체.
#             경로 길이가 증가하지만 WP 통과가 보장됨.
#
FUNCTION _regenerate_corner_through_wp(P_entry, WP, P_exit, R_min, α) → (pts, κ):
    RAISE NotImplementedError(
        "WP 정확 통과 재생성 미구현 — Option A/B/C 중 선택 후 구현"
    )
```

---

## 코너 분류 흐름도

```
내부 WP i 처리
│
├─ |α| < min_turn_rad?
│   YES → STRAIGHT (직선 처리, 코너 없음)
│
├─ T_L ≤ T_max?
│   YES → NORMAL CORNER
│           직선 → 진입나선(0→κ_max) → 원호(κ_max) → 이탈나선(κ_max→0) → 직선
│           κ_max 구조적 보장
│           [PSEUDO] WP 통과 검증 → 실패 시 [PSEUDO] 재생성
│
└─ T_L > T_max
    → LOOP CORNER
        직선 → 진입나선(0→κ_max) → 큰원호(≥180°, κ_max) → 이탈나선(κ_max→0) → 직선
        loop_sign = -sign(α),  α_loop = 2π - |α|
        κ_max 구조적 보장
```

---

## 구현 체크리스트

- [ ] `BasePlanner` 상속, `plan()` 시그니처 일치
- [ ] `a_max = a_max_g × 9.81 × accel_tol` 적용
- [ ] `κ_max = 1/R_min` — 모든 구간에서 `max(|κ|) ≤ κ_max` 검증
- [ ] `_straight_segment`: P0 포함, P1 제외 (중복점 제거)
- [ ] `_build_loop_corner`: α_loop = 2π - |α|, loop_sign = -sign(α)
- [ ] 루프 이탈 방향 = d_out 검증
- [ ] `PathPoint.s` 단조 증가
- [ ] `PathPoint.wp_index` — 원본 WP마다 정확히 하나 (`argmin` 패턴)
- [ ] `Path.total_length` = s_arr[-1]
- [ ] `Path.planning_time` 기록
- [ ] `_check_wp_passage` 자리 확보 (PSEUDO)
- [ ] `_regenerate_corner_through_wp` 자리 확보 (PSEUDO, NotImplementedError)
- [ ] `run_scenario.py` — `build_planner()` 분기 + `argparse choices` 등록
- [ ] `python run_scenario.py basic --planner clothoid_loop --no-plot` 오류 없음
