# 새 경로 생성기(Planner) 추가 명령서

새 planner를 구현하고 시뮬레이션에 등록하기 위한 절차와 계약을 정의한다.

---

## 1. 파일 위치

```
vtol_sim_checkpoint1_1/vtol_sim/path_planning/<name>_planner.py
```

명명 규칙: `<알고리즘 이름>_planner.py` (소문자, 언더스코어)

---

## 2. 필수 구현 계약

### 2-1. 클래스 골격

```python
from __future__ import annotations
import time
import numpy as np
from .base_planner import BasePlanner, Path, PathPoint

class MyPlanner(BasePlanner):

    def __init__(self, ds: float = 1.0, **kwargs):
        # ds: 경로 샘플링 간격 (m) — 모든 planner에 필수
        self.ds = ds
        # ...

    def plan(self,
             waypoints_ned: np.ndarray,
             aircraft_params: dict,
             initial_state: dict | None = None) -> Path:
        t_start = time.perf_counter()

        wps = np.asarray(waypoints_ned, dtype=float)   # shape (N, 3)
        if len(wps) < 2:
            raise ValueError("waypoints는 최소 2개 필요")

        # --- 경로 계산 ---
        points: list[PathPoint] = []
        s_total = 0.0

        # ... 구현 ...

        return Path(
            points=points,
            waypoints_ned=wps,
            total_length=s_total,
            planning_time=time.perf_counter() - t_start,
        )
```

### 2-2. PathPoint 필드 규칙

| 필드        | 타입              | 단위 | 설명                                                 |
| ----------- | ----------------- | ---- | ---------------------------------------------------- |
| `pos`       | `np.ndarray (3,)` | m    | `[x_N, x_E, h(표준 NED와 부호 반대)]` NED 좌표       |
| `v_ref`     | `float`           | m/s  | 계획 속도 — 보통 `aircraft_params["v_cruise"]`       |
| `chi_ref`   | `float`           | rad  | 방위각, `atan2(dE, dN)`                              |
| `gamma_ref` | `float`           | rad  | 상승각, 수평 비행이면 0.0                            |
| `curvature` | `float`           | 1/m  | 부호 있는 곡률 — **우선회 +, 좌선회 −** (NED 컨벤션) |
| `s`         | `float`           | m    | 경로 시작에서 누적 호 길이 (단조 증가)               |
| `wp_index`  | `int \| None`     | —    | 원본 WP에 가장 가까운 점에만 WP 번호 부여            |

**wp_index 마킹 최소 요구사항:**

- 첫 번째 PathPoint: `wp_index=0`
- 마지막 WP: `wp_index=N-1`
- 중간 WP: 각 WP에 가장 가까운 단일 점 하나에만 부여

```python
# wp_index 마킹 표준 패턴
for wi, wp in enumerate(wps):
    d = np.linalg.norm(sampled_2d - wp[:2], axis=1)
    points[int(np.argmin(d))].wp_index = wi
```

### 2-3. aircraft_params 딕셔너리 키

| 키            | 타입  | 설명                                |
| ------------- | ----- | ----------------------------------- |
| `v_cruise`    | float | 순항 속도 (m/s)                     |
| `gravity`     | float | 중력 가속도 (m/s²), 보통 9.81       |
| `a_max_g`     | float | 최대 측면 가속도 (g 단위), 보통 0.3 |
| `phi_max_deg` | float | 최대 뱅크각 (°)                     |
| `tau_phi`     | float | 뱅크각 시정수 (s)                   |

최소 회전 반경 계산 표준:

```python
g        = float(aircraft_params["gravity"])
v_cruise = float(aircraft_params["v_cruise"])
a_max_g  = float(aircraft_params["a_max_g"])
phi      = np.arctan(a_max_g)                   # 운용 뱅크각
R_min    = (v_cruise ** 2) / (g * np.tan(phi))  # 최소 회전 반경 (m)
kappa_max = 1.0 / R_min                         # 최대 곡률 (1/m)
```

---

## 3. 시뮬레이션 등록

### 3-1. `run_scenario.py` — `build_planner()` 함수에 추가

```python
# 파일: vtol_sim_checkpoint1_1/vtol_sim/run_scenario.py

from path_planning.my_planner import MyPlanner   # ← import 추가

def build_planner(name: str):
    # ... 기존 분기 ...
    if name == "myplanner":                        # ← 분기 추가
        return MyPlanner(ds=1.0, param_a=0.5)
    raise ValueError(f"Unknown planner: {name}")
```

### 3-2. `argparse choices` 목록에 추가

```python
parser.add_argument(
    "--planner", default="dubins",
    choices=[..., "myplanner"],   # ← 추가
)
```

---

## 4. 실행 테스트

```bash
cd vtol_sim_checkpoint1_1/vtol_sim
python run_scenario.py basic --planner myplanner --controller nlgl --seed 42
python run_scenario.py basic --planner myplanner --controller mpc  --seed 42
```

---

## 5. 체크리스트

구현 완료 전 다음 항목을 모두 확인한다.

- [ ] `BasePlanner` 상속, `plan()` 시그니처 일치
- [ ] `waypoints_ned.shape` 검사 (`N >= 2`, `shape[1] >= 2`)
- [ ] `PathPoint.s` 단조 증가
- [ ] `PathPoint.wp_index` — 각 원본 WP에 정확히 하나
- [ ] `Path.total_length` = 마지막 PathPoint의 `s` 값
- [ ] `Path.planning_time` 기록 (`time.perf_counter()` 차이)
- [ ] `curvature` 부호: 우선회 +, 좌선회 −
- [ ] `chi_ref` 범위: `wrap_angle()` 또는 `atan2` 결과 (−π ~ π)
- [ ] `build_planner()` 분기 + `argparse choices` 등록
- [ ] `python run_scenario.py basic --planner <name> --no-plot` 오류 없이 실행

---

## 6. 기존 플래너 참고 목록

| 파일                   | 클래스                 | 알고리즘 요약                                  |
| ---------------------- | ---------------------- | ---------------------------------------------- |
| `dubins_planner.py`    | `DubinsPlanner`        | 직선 + 원호 모서리, 에너지 고도 보상           |
| `spline_planner.py`    | `SplinePlanner`        | Cubic Spline C2, 가이드 포인트 직선 통과       |
| `iterpin_planner.py`   | `IterativePinPlanner`  | Quintic Hermite + 반복 핀, C2 연속             |
| `D_iterpin_planner.py` | `DIterativePinPlanner` | 반복 핀 + 2D 파라미터 탐색 + 우회 WP 자동 삽입 |
| `bspline_planner.py`   | `BSplinePlanner`       | B-Spline (degree-5)                            |
| `bspline_2_planner.py` | `BSplinePlanner`       | B-Spline (degree-3) + 반복 정제                |
| `hermite_bspline.py`   | `BSplinePlanner`       | Hermite + B-Spline 혼합                        |
| `Qhermite_bspline.py`  | `BSplinePlanner`       | Quintic Hermite + B-Spline 혼합                |
