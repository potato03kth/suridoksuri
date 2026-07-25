---
doc_type: onboarding
project: suridoksuri-1
scope: "이 노트북에서 rclpy(fc_ros 노드) 테스트가 돌아가게 만들기 — 전용 세션용 자기완결 브리프"
last_updated: 2026-07-25
---

# 온보딩 — 이 노트북에 rclpy 환경 만들기

**이 문서만 읽고 시작하면 된다.** FC 트랙 보드(`docs/session_status.md`)나
`flight_plan.md`를 정독하지 말 것.

---

## 1. 왜 필요한가 (해결하려는 실제 통증)

`fc_ros/fc_ros/nodes/offboard_node.py`(1000줄, 실기체 제어 상태기계)를
**이 노트북에서 단 한 줄도 실행해볼 수 없다.** `rclpy`가 없어서 import조차 안 된다.

그래서 지금까지 이렇게 우회해왔다:

- 판정 로직을 전부 `fc_bridge/execution/state_logic.py`(rclpy 무관 순수함수)로 빼고
  `fc_ros/test/test_offboard_node.py`는 그 순수함수만 테스트한다.
- 노드 본체(상태 전이, 콜백, 세트포인트 발행, 파라미터 처리)는 **테스트가 0개다.**

2026-07-25 사고 수정 때 실제로 막혔던 지점:
`_apply_path_origin()`이 경로 배열을 제대로 평행이동하는지, `_State.PILOT_TAKEOVER`
전이가 실제로 세트포인트를 끊는지를 **직접 검증하지 못하고** 순수함수로 쪼개서
간접 검증했다. 상태기계 자체의 회귀는 여전히 못 잡는다.

**목표: `pytest fc_ros/test/`가 노드를 실제로 인스턴스화해서 돌아가게 만든다.**

---

## 2. 지금 환경 (측정된 사실, 추측 아님)

| 항목 | 값 |
|---|---|
| 기본 WSL 배포판 | `Ubuntu` — **Ubuntu 24.04.1 LTS (Noble)**, Python **3.12.3** |
| 저장소 위치 | `/home/suri/suridoksuri` (이 배포판 안) |
| `/opt/ros` | **없음** — ROS2 미설치 |
| 두 번째 배포판 | **`Ubuntu-22.04` (Stopped)** — E드라이브 `E:\wsl\Ubuntu-22.04` |
| 그 안에 있는 것 | **ROS2 Humble + MAVROS + PX4-Autopilot(gz_x500)** — 2026-07-24 구축 |

```
$ wsl.exe -l -v
  NAME            STATE           VERSION
* Ubuntu          Running         2
  Ubuntu-22.04    Stopped         2
```

**즉 rclpy는 이미 이 노트북에 있다 — 다른 배포판에.**
맨땅에서 설치할 필요 없다. 구축 명령 전체는 `docs/wsl_dev_env_setup.md` **섹션 F**.

### 왜 기본 배포판에 그냥 못 까나

ROS2 **Humble은 Ubuntu 22.04(jammy) + Python 3.10** 대상이다. 기본 배포판은 24.04라
`apt install ros-humble-*`가 안 된다. 24.04용은 **Jazzy**인데, 실기체·개발컴·SITL이
전부 Humble이라 버전을 갈라놓으면 "여기선 되는데 기체에선 안 된다"가 생긴다.

---

## 3. 선택지 (이 셋 중 하나를 사용자와 정하고 시작할 것)

| 안 | 방법 | 장점 | 단점 |
|---|---|---|---|
| **A** | 기존 `Ubuntu-22.04` 배포판에서 저장소를 열고 거기서 pytest | 새로 깔 게 없음, Humble 일치 | 저장소가 배포판 간 분리 — 어느 쪽에서 편집하는지 규율 필요 |
| **B** | 기본 24.04에 **Docker**로 Humble 컨테이너 (실기체와 같은 방식) | 실기체 `fc` 컨테이너와 환경 동일, 저장소는 볼륨 마운트 | Docker 셋업, 이미지 용량 |
| **C** | 기본 24.04에 `pip install rclpy` | 제일 간단해 보임 | **거의 확실히 실패한다** — rclpy는 순수 파이썬이 아니라 C 확장 + ROS 미들웨어(rmw/DDS)에 의존. PyPI 배포본은 사실상 없다 |

**추천은 A.** 배포판이 이미 있고 Humble이 실기체와 일치한다.
저장소 접근은 배포판 간 `\\wsl.localhost\Ubuntu\home\suri\suridoksuri` 경로로 되지만
**9p 파일시스템이라 느리다** — 별도 clone 후 브랜치 동기화로 가는 편이 나을 수 있다.
이건 실측해보고 정할 것.

---

## 4. ⚠️ 먼저 읽을 함정 (같은 걸 세 번째 겪지 말 것)

### 4-1. `fc_bridge`를 `pip install -e .`로 깔면 `ros2` CLI가 깨진다

`fc_bridge/setup.py`가 `fc_bridge/` **디렉터리 안에서** `find_packages()`를 돌리기 때문에
하위 패키지가 최상위로 평평하게 깔린다(`execution/`, `comm/`, `guidance/`…).
`fc_bridge.` 네임스페이스가 아예 안 생긴다.

그렇다고 저장소 루트를 라이브 `PYTHONPATH`로 주입하면
`ros2` CLI가 `importlib.metadata.PackageNotFoundError: ros2cli`로 죽는다
(단독 `python3 -c`로는 재현 안 되고 `ros2` CLI 경로에서만 터진다).

**해결(2026-07-24 검증됨):** 저장소 루트 절대경로를 담은 `.pth` 파일을
`site.getusersitepackages()`에 만든다.

```bash
python3 -c "import site,os;p=site.getusersitepackages();os.makedirs(p,exist_ok=True);
open(os.path.join(p,'suridoksuri.pth'),'w').write('/절대/경로/suridoksuri\n')"
```

> 실기체(RPi)는 이 문제를 **launch를 저장소 루트에서 실행**하는 것으로 우회하고 있다
> (cwd가 sys.path에 들어감). 상세는 `docs/rpi_deploy.md` §4.

### 4-2. 노드를 그냥 실행하면 안 된다

`offboard_node`는 **인스턴스화되는 순간 ARM 명령을 보낸다.**
테스트에서는 MAVROS 서비스가 없으니 실제 기체가 움직일 일은 없지만,
SITL이 붙어 있으면 진짜로 뜬다. 테스트는 반드시
**MAVROS 없는 상태 / 목(mock)** 으로 짤 것.

### 4-3. SITL 프리플라이트 우회 파라미터를 실기체에 넣지 말 것

`CBRK_SUPPLY_CHK=894281`, `NAV_DLL_ACT=0`은 **SITL 전용**이다.

---

## 5. 완료 기준 (이게 되면 끝)

```bash
# 1) rclpy import
python3 -c "import rclpy; print(rclpy.__file__)"

# 2) 노드 모듈 import (인스턴스화 없음 — 안전)
python3 -c "import fc_ros.nodes.offboard_node as m; print(m._State.PILOT_TAKEOVER)"

# 3) 기존 테스트가 여전히 통과 (회귀 없음)
python3 -m pytest fc_bridge/tests/ fc_ros/test/ -q     # 현재 286개 통과 중

# 4) (본 목적) 노드를 인스턴스화하는 테스트를 1개라도 추가해 통과시킨다
```

**4번의 첫 테스트로 뭘 짤지 — 바로 이걸 권한다** (2026-07-25 사고 회귀 방지):

- `waypoint_frame="takeoff"`로 노드를 띄우고 `_apply_path_origin([8.53,-6.84,-10.55])`
  호출 후 `_mc_wps[0]`이 이륙지점과 일치하고 `_cruise_alt == -7.55`인지
- `_current_mode="POSCTL"` + `_offboard_engaged_once=True` 상태에서 `_control_callback()`을
  돌리면 `_sm == _State.PILOT_TAKEOVER`가 되고 **세트포인트 퍼블리시가 0회**인지
  (퍼블리셔를 목으로 바꿔 호출 횟수를 센다)

두 케이스 다 실제 사고에서 나온 것이고, 지금은 순수함수로 간접 검증만 돼 있다.
근거 수치는 `logs/2026-07-25_flight01/notes.md`.

---

## 6. 참고 문서 (필요할 때만 열 것)

| 문서 | 언제 |
|---|---|
| `docs/wsl_dev_env_setup.md` **섹션 F** | Ubuntu-22.04 배포판에 뭐가 어떻게 깔렸는지 |
| `docs/rpi_deploy.md` | 실기체 배포·`fc_bridge` import 구조 |
| `fc_bridge/CLAUDE.md` | fc_bridge 모듈 구조·인터페이스 |
| `logs/2026-07-25_flight01/notes.md` | 왜 노드 테스트가 필요한지의 구체 사례 |

**작업 후:** `fc_ros/test/` 실행 방법이 바뀌면 이 문서 §5와 `fc_bridge/CLAUDE.md` §테스트를
갱신하고, 새로 발견한 함정은 §4에 추가한다.
