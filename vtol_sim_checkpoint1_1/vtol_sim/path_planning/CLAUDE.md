# path_planning — Claude 작업 가이드

이 디렉터리는 VTOL 시뮬레이터의 경로 생성 모듈이다. 새 planner를 만들거나 기존 planner를 수정할 때 이 문서를 참고한다.

---

## 디렉터리 구조

```
path_planning/
├── base_planner.py          ← PathPoint, Path, BasePlanner 정의 (수정 금지)
├── waypoint_generator.py    ← WP 생성 유틸리티
├── dubins_planner.py        ← 직선+원호, 에너지 고도 보상
├── spline_planner.py        ← Cubic Spline C2
├── bspline_planner.py       ← B-Spline (degree-5)
├── bspline_2_planner.py     ← B-Spline (degree-3) + 반복 정제
├── hermite_bspline.py       ← Hermite + B-Spline 혼합
├── Qhermite_bspline.py      ← Quintic Hermite + B-Spline 혼합
├── iterpin_planner.py       ← Quintic Hermite + 반복 핀, C2 연속
├── D_iterpin_planner.py     ← 반복 핀 + 2D 파라미터 탐색 + 우회 WP
├── clothoid_planner.py      ← Euler Spiral (κ ∝ s), 곡률 구조적 보장
└── CREATE_PLANNER.md        ← planner 추가 절차 상세 명령서
```

---

## 핵심 인터페이스 (`base_planner.py`)

### PathPoint 필드

| 필드        | 타입           | 단위 | 규칙                                      |
| ----------- | -------------- | ---- | ----------------------------------------- |
| `pos`       | `ndarray (3,)` | m    | `[x_N, x_E, h]` NED (h는 고도, 부호 양수) |
| `v_ref`     | `float`        | m/s  | 보통 `aircraft_params["v_cruise"]` 고정   |
| `chi_ref`   | `float`        | rad  | `arctan2(dE, dN)`, 범위 −π~π              |
| `gamma_ref` | `float`        | rad  | 상승각, 수평=0                            |
| `curvature` | `float`        | 1/m  | **우선회 +, 좌선회 −** (NED 컨벤션)       |
| `s`         | `float`        | m    | 단조 증가 필수                            |
| `wp_index`  | `int \| None`  | —    | 각 원본 WP에 정확히 하나                  |

### wp_index 마킹 표준

```python
for wi, wp in enumerate(wps):
    d = np.linalg.norm(sampled_2d - wp[:2], axis=1)
    points[int(np.argmin(d))].wp_index = wi
```

첫 점 `wp_index=0`, 마지막 WP `wp_index=N-1`, 중간 WP 각 1개씩 필수.

### R_min 계산 표준

```python
g        = float(aircraft_params.get("gravity", 9.81))
v_cruise = float(aircraft_params["v_cruise"])
a_max_g  = float(aircraft_params["a_max_g"])
R_min    = v_cruise**2 / (a_max_g * g)          # 기본
R_min    = v_cruise**2 / (a_max_g * g * accel_tol)  # 여유율 포함
```

---

## 새 Planner 추가 절차

> 상세 절차는 [CREATE_PLANNER.md](CREATE_PLANNER.md) 참고. 여기선 핵심만 요약.

### 1. 파일 생성

```
path_planning/<name>_planner.py
```

### 2. 클래스 골격

```python
from __future__ import annotations
import time
import numpy as np
from .base_planner import BasePlanner, Path, PathPoint

class MyPlanner(BasePlanner):

    def __init__(self, ds: float = 1.0, **kwargs):
        self.ds = ds

    def plan(self, waypoints_ned: np.ndarray,
             aircraft_params: dict,
             initial_state: dict | None = None) -> Path:
        t0 = time.perf_counter()
        wps = np.asarray(waypoints_ned, dtype=float)
        if len(wps) < 2:
            raise ValueError("waypoints는 최소 2개 필요")

        # ... 경로 계산 ...

        return Path(
            points=points,
            waypoints_ned=wps,
            total_length=float(s_arr[-1]),
            planning_time=time.perf_counter() - t0,
        )
```

### 3. `run_scenario.py` 등록

파일: [../run_scenario.py](../run_scenario.py)

두 곳 수정:

```python
# import 추가
from path_planning.my_planner import MyPlanner

# build_planner() 분기 추가
if name == "myplanner":
    return MyPlanner(ds=1.0)

# argparse choices 추가
choices=[..., "myplanner"]
```

### 4. 실행 테스트

```bash
cd vtol_sim_checkpoint1_1/vtol_sim
python run_scenario.py basic --planner myplanner --controller nlgl --seed 42 --no-plot
python run_scenario.py basic --planner myplanner --controller mpc  --seed 42 --no-plot
```

---

## 구현 체크리스트

- [ ] `BasePlanner` 상속, `plan()` 시그니처 일치
- [ ] `waypoints_ned` shape 검사 (`N >= 2`)
- [ ] `PathPoint.s` 단조 증가
- [ ] `PathPoint.wp_index` — 각 원본 WP에 정확히 하나
- [ ] `Path.total_length` = 마지막 `s` 값
- [ ] `Path.planning_time` 기록
- [ ] `curvature` 부호: 우선회 +, 좌선회 −
- [ ] `chi_ref` 범위: −π ~ π
- [ ] `build_planner()` + `argparse choices` 등록
- [ ] `--no-plot` 오류 없이 실행

---

## 기존 Planner 참조 가이드

| 파일                   | `--planner` 이름 | 특징                                  |
| ---------------------- | ---------------- | ------------------------------------- |
| `dubins_planner.py`    | `dubins`         | 단순, 직선+원호, 기준선               |
| `spline_planner.py`    | `spline`         | Cubic Spline C2 연속                  |
| `clothoid_planner.py`  | `clothoid`       | 곡률 선형 증가, 최대 곡률 구조적 보장 |
| `D_iterpin_planner.py` | `diterpin`       | 가장 복잡, 우회 WP 자동 삽입          |

<!-- 새 planner의 복잡도가 낮다면 `dubins_planner.py`를, 곡률 제어가 목표라면 `clothoid_planner.py`를 참고 출발점으로 삼아라. -->

---

## 주의사항

- `base_planner.py`는 수정하지 않는다.
- `curvature` 부호 컨벤션은 NED 프레임 기준: 오른쪽 선회(동쪽 방향 회전) = **양수**.
- 고도(`h`)는 NED의 −D 값이므로 양수가 위쪽이다.
- 경로 점 간격 `ds`는 생성자 파라미터로 노출하고, 기본값 `1.0 m` 권장.
