# ClothoidPlanner — 구조 및 컴포넌트 요약

> 파일: `path_planning/clothoid_planner.py`  
> 의존: `path_planning/base_planner.py` (`BasePlanner`, `Path`, `PathPoint`)

---

## 수학적 배경

클로소이드(오일러 나선)는 호 길이에 비례하여 곡률이 선형 증가하는 곡선이다.

```
κ(s) = s / A²,   A² = R_min × L_s
κ_max = 1 / R_min = a_max / v²
```

각 웨이포인트(WP) 코너를 다음 5-구간으로 구성해 최대 곡률을 구조적으로 보장한다.

```
직선 → 진입 나선 → 원호 → 이탈 나선 → 직선
```

---

## 데이터 구조 (`base_planner.py`)

### `PathPoint`

경로 위 단일 점을 표현하는 dataclass.

| 필드 | 타입 | 설명 |
|------|------|------|
| `pos` | `ndarray (3,)` | NED 좌표 `[x_N, x_E, h]` |
| `v_ref` | `float` | 참조 속도 (m/s) |
| `chi_ref` | `float` | 참조 방위각 (rad) |
| `gamma_ref` | `float` | 참조 상승각 (rad) |
| `curvature` | `float` | 곡률 (1/m); 좌선회 음, 우선회 양 |
| `s` | `float` | 경로 시작점으로부터의 호 길이 (m) |
| `wp_index` | `int \| None` | 이 점이 대응하는 WP 번호 (중간 점은 None) |

### `Path`

완성된 경로 전체를 담는 dataclass.

| 필드 / 메서드 | 설명 |
|---------------|------|
| `points` | `PathPoint` 리스트 |
| `waypoints_ned` | 원본 WP 배열 `(N, 3)` |
| `total_length` | 전체 호 길이 (m) |
| `planning_time` | 경로 생성 소요 시간 (s) |
| `positions_array()` | 모든 점의 위치를 `(N, 3)` 배열로 반환 |
| `waypoint_indices_in_path()` | `wp_index != None`인 점의 경로 내 인덱스 목록 |

---

## `ClothoidPlanner` 클래스

`BasePlanner`를 상속. 전체 흐름: `plan` → `_build_2d_path` → 직선/코너 조립 → 고도·상승각 보간 → `Path` 반환.

---

### `__init__`

**입력 파라미터**

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `ds` | `1.0 m` | 경로 점 간격 (해상도) |
| `accel_tol` | `0.9` | 최대 가속도 여유율 (0 < tol ≤ 1) |
| `spiral_fraction` | `0.4` | 전체 코너 각도 중 나선이 차지하는 비율 |
| `min_turn_deg` | `2.0 deg` | 이 이하의 선회각은 직선으로 처리 |
| `end_extension` | `15.0 m` | 마지막 WP 이후 경로 연장 거리 |

---

### `plan` — 공개 인터페이스

**입력**

| 인자 | 타입 | 설명 |
|------|------|------|
| `waypoints_ned` | `(N, 3) ndarray` | WP 시퀀스 `[x_N, x_E, h]` |
| `aircraft_params` | `dict` | `v_cruise`, `a_max_g`, `gravity`(선택) |
| `initial_state` | `dict \| None` | (현재 미사용, 확장 예비) |

**출력**: `Path` 객체

**핵심 동작**
1. `R_min = v² / (a_max_g × g × accel_tol)` 계산
2. `_build_2d_path`로 수평 경로(x, y) 및 곡률 배열 생성
3. WP 간 호 길이로 고도 선형 보간 → `alt_arr`
4. 연속 점 차분으로 상승각 `gamma_arr` 산출 (trapezoidal)
5. 각 점을 `PathPoint`로 조립 → `Path` 반환

---

### `_build_2d_path` — 수평 경로 조립

**입력**: `wps_2d (N,2)`, `R_min`  
**출력**: `(pts, s_arr, kappa_arr, wp_marks)`

| 출력 | 설명 |
|------|------|
| `pts` | 경로 점 배열 `(M, 2)` |
| `s_arr` | 각 점의 누적 호 길이 `(M,)` |
| `kappa_arr` | 각 점의 곡률 `(M,)` |
| `wp_marks` | `{점 인덱스: WP 인덱스}` 매핑 딕셔너리 |

**핵심 동작**
1. 내부 WP마다 코너 파라미터 사전 계산 (`P_entry`, `P_exit`, `L_s`, `alpha`)
2. 인접 구간 길이의 45%로 접선 길이 `T_L` 상한 설정 (코너 겹침 방지)
3. 세그먼트 루프: 직선 구간 → `_straight_segment`, 코너 → `_build_corner`
4. 마지막 WP 이후 `end_extension` 직선 추가
5. 전체 점 연결 후 `np.cumsum`으로 호 길이 계산

---

### `_straight_segment` — 직선 구간 생성

**입력**: `P0 (2,)`, `P1 (2,)`  
**출력**: `(pts (n,2), kappa list)`

- `ds` 간격으로 등분한 점 생성 (`linspace`)
- 곡률 = 0 일정

---

### `_tangent_length` — 코너 접선 길이 계산

**입력**: `L_s`, `R`, `alpha`  
**출력**: `T_L (float)`

- 국소 프레임에서 코너 끝점 좌표 `(ex, ey)` 산출
- 진입 방향(+x)과 이탈 방향(`alpha`)의 교점까지 거리 해석적으로 계산
- `T_L = ex - (ey / sin(alpha)) * cos(alpha)`

---

### `_corner_pts_local` — 로컬 프레임 코너 점 생성

**입력**: `L_s`, `R`, `abs_alpha`, `sign`  
**출력**: `pts (M, 2)` — 진입점 원점, 진입 방향 +x 기준

**단계별 헤딩 프로파일**

| 구간 | 헤딩 공식 |
|------|-----------|
| 진입 나선 | `h(s) = sign × s² / (2·R·L_s)` |
| 원호 | `h(s) = h_end_entry + sign × s / R` |
| 이탈 나선 | `h(s) = chi_out − sign × (L_s−s)² / (2·R·L_s)` |

각 구간 `_integrate_heading` 호출로 점 좌표 산출.

---

### `_build_corner` — 월드 프레임 코너 생성

**입력**: `P_entry (2,)`, `P_exit (2,)`, `WP (2,)`, `L_s`, `R`, `alpha`  
**출력**: `(pts (M,2), kappa list, wp_local_idx int)`

| 출력 | 설명 |
|------|------|
| `pts` | NED 프레임 코너 경로 점 |
| `kappa list` | 각 점 곡률 — 나선: `sign·s/(R·L_s)`, 원호: `sign/R`, 이탈 나선: `sign·(L_s−s)/(R·L_s)` |
| `wp_local_idx` | WP 꼭짓점에 가장 가까운 점의 로컬 인덱스 |

`_corner_pts_local`과 유사하나 세계 좌표계에서 직접 계산하고 곡률 배열을 함께 생성.

---

### `_headings_from_pts` — 연속 점에서 방위각 산출

**입력**: `pts (N, 2)`  
**출력**: `h (N,)` — 각 점의 진행 방위각 (rad)

- `np.arctan2(dy, dx)` 차분으로 계산
- 마지막 점은 직전 값 복사

---

## 모듈 레벨 헬퍼 함수

### `_unit(v)`
- **입력**: 임의 벡터 `v`
- **출력**: 단위 벡터 (영벡터 안전 처리)

### `_signed_turn(d_in, d_out)`
- **입력**: 진입/이탈 단위 방향 벡터
- **출력**: 부호 있는 선회각 (rad)
- 외적 부호로 방향 판정: NE 평면에서 왼쪽 선회 → 음수

### `_integrate_heading(x0, y0, h_arr, s_arr)`
- **입력**: 시작 좌표 `(x0, y0)`, 헤딩 배열 `h_arr`, 호 길이 배열 `s_arr`
- **출력**: `(x_arr, y_arr)` — 사다리꼴 적분으로 산출한 경로 좌표
- 공식: `x[i+1] = x[i] + 0.5·(cos h[i] + cos h[i+1])·ds`

---

## 컴포넌트 의존 흐름

```
plan()
 ├─ _build_2d_path()
 │   ├─ _tangent_length()          # 코너마다 T_L 계산
 │   │   └─ _corner_pts_local()    # 로컬 프레임 기하
 │   │       └─ _integrate_heading()
 │   ├─ _straight_segment()        # 직선 구간
 │   └─ _build_corner()            # 코너 구간 (월드 프레임)
 │       └─ _integrate_heading()
 ├─ _headings_from_pts()           # 방위각 산출
 └─ np.interp()                    # 고도 보간
```
