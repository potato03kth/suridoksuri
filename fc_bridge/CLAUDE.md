# fc_bridge — Claude 작업 가이드

ROS2 의존 없는 **순수 Python 비행 제어 라이브러리**.  
`fc_ros/` 노드들이 이 패키지를 import해 경로 계획·유도 알고리즘을 사용한다.

---

## 모듈 구조

| 모듈 | 역할 |
|---|---|
| `config.py` | 전역 설정 (연결 주소, 기체 파라미터) |
| `planning/planner_runner.py` | 경로 생성 진입점 (`run_planner()`) |
| `planning/speed_profile.py` | 곡률 기반 속도 프로파일 생성 |
| `guidance/l1_guidance.py` | L1 Guidance 알고리즘 |
| `comm/vehicle_state.py` | `VehicleState` 데이터클래스 |
| `comm/telemetry.py` | pymavlink 기반 텔레메트리 (SITL 직접 실행용) |
| `execution/offboard_follower.py` | SITL 직접 실행용 경로 추종 (pymavlink 기반) |
| `execution/mission_uploader.py` | PX4 미션 업로드 |
| `utils/rotation.py` | `quat_to_euler_xyz()` (NumPy 2.x 호환) |
| `run_phase1.py` | 경로 생성 + 미션 업로드 + 시각화 스크립트 |
| `run_phase2.py` | Offboard 경로 추종 스크립트 (pymavlink 직접 사용) |

> `execution/`, `comm/` 의 pymavlink 기반 코드는 SITL 직접 실행용 레거시다.  
> ROS2 환경에서는 `fc_ros/` 노드가 대신하므로 건드리지 않는다.

---

## 핵심 인터페이스

### `run_planner()` — 경로 생성

```python
from fc_bridge.planning.planner_runner import run_planner

path = run_planner(
    planner_name="eta3",           # "eta3" | "diterpin"
    waypoints_ned=np.array([       # shape (N, 3), NED 좌표 [N, E, h_up]
        [0.0,   0.0, 50.0],
        [200.0, 0.0, 50.0],
    ]),
    vehicle_params={
        "v_cruise": 15.0,    # 순항 속도 (m/s)
        "a_max_g":  0.3,     # 횡방향 가속도 상한 (g)
        "gravity":  9.81,
    },
    # planner_kwargs={}   # 플래너 생성자 추가 파라미터 (선택)
    # initial_state={"initial_heading": 0.0}  # 초기 헤딩 (선택)
)
```

### `Path` 반환값

| 속성 | 타입 | 설명 |
|---|---|---|
| `path.points` | `list[PathPoint]` | 경로점 목록 (약 1m 간격) |
| `path.total_length` | `float` | 총 호길이 (m) |
| `path.planning_time` | `float` | 계획 소요 시간 (s) |

### `PathPoint` 속성

| 속성 | 타입 | 설명 |
|---|---|---|
| `pt.pos` | `ndarray (3,)` | NED 위치 `[N, E, h_up]` |
| `pt.v_ref` | `float` | 해당 경로점 목표 속도 (m/s) |
| `pt.gamma_ref` | `float` | 상승각 (rad) |
| `pt.curvature` | `float` | 곡률 κ (1/m, 오른쪽=양수) |
| `pt.chi_ref` | `float` | 기준 헤딩 (rad) |
| `pt.s` | `float` | 호길이 누적값 (m) |
| `pt.wp_index` | `int \| None` | WP 통과점이면 WP 인덱스, 아니면 `None` |

### `OffboardNode.main()`에서 사용하는 패턴

```python
path = run_planner(planner_name, waypoints, vehicle_params)

path_pts      = np.array([pt.pos[:2]    for pt in path.points])  # (N, 2) NE
v_profile     = np.array([pt.v_ref      for pt in path.points])  # (N,)
gamma_profile = np.array([pt.gamma_ref  for pt in path.points])  # (N,)
```

---

## 세션 D: dry-run 경로 검증

Windows(또는 WSL)에서 SITL 없이 실행 가능.

```bash
cd fc_bridge
python run_phase1.py --dry-run --plot --planner eta3
```

| 옵션 | 설명 |
|---|---|
| `--dry-run` | MAVLink 연결 없이 경로 생성만 수행 |
| `--plot` | 6패널 시각화 창 표시 (TkAgg) |
| `--save-plot results/path.png` | 파일 저장 |
| `--planner eta3 \| diterpin` | 플래너 선택 |

**시각화 6패널 구성:**
- Top-down 2D (|κ| 컬러맵, WP 통과점 lime 마킹)
- 고도 vs arc length
- 곡률 (부호) vs arc length
- 위치잔차 CTE
- 속도 프로파일
- 헤딩 χ_ref

`run_phase1.py`의 기본 웨이포인트 (`DEFAULT_WAYPOINTS`):
```python
[[0, 0, 50], [200, 0, 50], [200, 200, 50], [0, 200, 50], [0, 0, 50]]  # 200×200m 사각형 귀환
```

세션 D의 직선/L자/사각형 테스트 경로는 이 값을 교체하거나 별도 스크립트로 주입.

---

## config.py 기본값

```python
CONNECTION_STR = "udp:127.0.0.1:14550"   # SITL 연결 주소
VEHICLE_PARAMS = {"v_cruise": 15.0, "a_max_g": 0.3, "gravity": 9.81}
CONTROL_HZ = 10       # 세트포인트 주파수
L1_DISTANCE = 20.0    # L1 lookahead 거리 (m)
```

`run_phase1.py`는 `config.py`를 직접 읽는다.  
ROS2 경로(`fc_ros_params.yaml`)와 값을 맞출 때 두 곳 모두 확인.

---

## 테스트

```bash
# 리포지토리 루트에서 (ROS2 없이 실행 가능)
cd fc_bridge
pytest tests/
```
