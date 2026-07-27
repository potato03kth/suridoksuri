---
doc_type: interface_spec
project: suridoksuri-1
scope: vision → fc_ros 정밀착륙 인터페이스 — 메시지 타입·좌표계·페일세이프·런타임 환경 정찰 결과 (vision_plan.md §9 7번 착수 전 사실확정)
status: 🔍 정찰 완료 / 구현 미착수 — 사용자 결정 대기 항목 4건 (§8)
created: 2026-07-28
last_updated: 2026-07-28
---

# vision ↔ fc_ros 정밀착륙 인터페이스 — 정찰·설계 문서

> **이 문서는 구현이 아니라 사실확정이다.** `docs/vision_plan.md` §9 빌드순서 7번
> (offboard 정밀착륙 서브상태 연결)에 들어가기 전에, 잘못된 전제로 구현했다가 통째로
> 버리는 것을 막기 위해 실물을 확인한 결과다. 이 세션은 `.py` 파일을 하나도 만들거나
> 고치지 않았다. `fc_ros/`·`fc_bridge/`는 **읽기만** 했다(도메인 격리, 루트 `CLAUDE.md`).
>
> **모든 주장에 근거(파일:줄 또는 실행한 명령)를 붙였다.** 근거가 없는 항목은 §7
> "미확인 목록"에 명시적으로 남겼다 — 추측으로 메우지 않았다.

---

## 0. 세 줄 요약

1. 🔴 **최대 리스크(런타임 환경)가 현실이었다.** RPi 호스트에는 **ROS2가 아예 없고**
   (`ls -d /opt/ros/*` → 없음), `picam-venv`에도 `rclpy`가 없으며(Python **3.12**),
   `fc` 컨테이너는 **ROS Humble / Python 3.10**이다. 지금 상태로는 vision 쪽 ROS2 노드를
   **띄울 곳 자체가 없다.** 이건 "패키지 하나 설치"로 끝나는 문제가 아니라 배포판 선택 문제다(§5).
2. 🔴 **`mavros_msgs/msg/LandingTarget`의 `frame` 상수가 실제 MAVLink `MAV_FRAME` 값과
   1씩 어긋나 있다**(같은 기체에서 두 파일을 직접 대조해 확인, §3.1). 계획서가 명시한
   "네이티브 precision-land 피벗" 경로를 문서만 보고 짰다면 **조용히 오작동**했을 자리다.
   추가로 `~/raw` 구독은 `listen_lt: true`일 때만 생성되는데 실기체 설정은 **`false`**다.
3. **권고: ROS 메시지 타입 선택을 지금 확정하지 말고, vision 코어를 transport-agnostic으로
   두고 "얇은 소켓 + 컨테이너 안 shim 노드"로 잇는다**(§8 R4). 그러면 A/B/C 선택이 전부
   컨테이너 내부 문제로 축소돼, 런타임 배포판 문제와 메시지 타입 문제가 서로를 막지 않는다.

---

## 1. 현재 양쪽 끝단의 실제 상태

| | vision 쪽 | fc_ros 쪽 |
|---|---|---|
| ROS2 노드 | **없음** (`rclpy` import 0건 — 오케스트레이터 확인) | `OffboardNode`(`fc_ros/fc_ros/nodes/offboard_node.py:138`) |
| 산출물 | `TargetEstimate` (`vision/core/target.py:52-71`), JSONL `chosen.target_estimate`에만 실림 | — |
| 상태 힌트 | `Decision.command` (`vision/core/state_machine.py:86-94`) — **소비자 없음** | `_State` enum (`offboard_node.py:111-124`) |
| 정밀착륙 자리 | — | `HOLD`(`:1230`) → **여기** → `LANDING`(`:1300`, AUTO.LAND) |
| 포트(§7.2) | `TargetSink`는 **이름만 있고 미구현** (`grep -rn "TargetSink" vision/ --include=*.py` → 0건) | — |

`vision_plan.md` §7.2 "변화 흡수 지도"가 이 이음매를 이미 이렇게 규정해 뒀다:

> `offboard 통합` | 코어=transport-agnostic 라이브러리, ROS 노드=얇은 래퍼 | in-process 호출로 전환

아래 권고(§8)는 이 줄을 그대로 따른 것이다 — 새 아키텍처가 아니다.

---

## 2. 조사 D — 🔴 런타임 환경 (가장 중요, 먼저 읽을 것)

**결론: 두 프로세스는 지금 같은 ROS2 도메인에서 통신할 수 없다. 통신할 ROS2가 한쪽에 없다.**

### 2.1 확인한 사실 (전부 실행 출력 근거 있음)

실행: `ssh suri@100.67.27.83 ...` (읽기 전용 확인만. 설치·설정변경·서비스 조작 일절 없음)

| 항목 | 값 | 확인 명령 |
|---|---|---|
| 호스트 OS | **Ubuntu 24.04.4 LTS (noble)**, aarch64, 커널 6.8.0-1060-raspi | `grep -E "^(NAME\|VERSION)=" /etc/os-release`; `uname -a` |
| 호스트 ROS2 | **없음** | `ls -d /opt/ros/*` → `NO /opt/ros on host` |
| `picam-venv` 경로 | `/home/suri/local-libcamera-src/picam-venv` | `local-libcamera/env.sh`의 `PICAM_PYTHON` |
| `picam-venv` python | **3.12.3**, `--system-site-packages`, 심볼릭 링크 `/usr/bin/python3` | `cat picam-venv/pyvenv.cfg` |
| `picam-venv` 내용 | picamera2 0.3.36, cv2(opencv-contrib 4.13.0.92), numpy **2.5.1**, av 18.0.0, simplejpeg, pidng … | `ls .../site-packages/` |
| `picam-venv`의 `rclpy` | **없음** | `$PICAM_PYTHON -c "import rclpy"` → `ModuleNotFoundError` |
| 컨테이너 `fc` | 이미지 `ros:humble`, `Up 26 hours` | `docker ps -a` |
| 컨테이너 네트워크 | **`NetworkMode=host`** | `docker inspect fc --format ...` |
| 컨테이너 IPC | **`IpcMode=private`** (독립 `/dev/shm`, **64MB**) | `docker inspect`; 컨테이너 `df -h /dev/shm` → 64M / 호스트 → 3.9G |
| 컨테이너 python | **3.10.12**, `rclpy` 있음(`/opt/ros/humble/local/lib/python3.10/dist-packages/rclpy`) | 컨테이너 안 `python3 -c "import rclpy,sys"` |
| 컨테이너 cv2 / picamera2 | **둘 다 없음** (numpy는 1.21.5) | 컨테이너 안 `import cv2` / `import picamera2` → `ModuleNotFoundError` |
| 컨테이너 디바이스 | **`/dev/ttyACM0` 하나뿐** (카메라 디바이스 패스스루 없음), `Privileged=false` | `docker inspect fc --format "Devices={{.HostConfig.Devices}}"` |
| 컨테이너 마운트 | `/home/suri/drone_ws:/drone_ws` 하나 | `docker inspect ... Mounts` |
| `ROS_DOMAIN_ID` | **어디에도 설정 없음** → 기본값 0 (양쪽 동일) | 컨테이너 `printenv \| grep -Ei "ROS\|RMW\|DDS"` → `ROS_DISTRO=humble` 뿐; 호스트 `.bashrc`/`.profile` grep → 0건 |
| RMW | `librmw_fastrtps_cpp.so`만 설치 (기본 Fast DDS) | 컨테이너 `ls /opt/ros/humble/lib/librmw_*.so` |
| 현재 실행 중 | mavros·offboard_node·vision **전부 미실행**. `fc` 컨테이너 안엔 idle `bash` 하나 | `ps -eo pid,args --no-headers > /tmp/_ps_snap.txt` 후 grep; `docker top fc` |

### 2.2 이게 왜 "설치 하나로 안 끝나는" 문제인가

- ROS 2 **Humble**의 타깃 플랫폼은 Ubuntu **22.04**다. 호스트는 **24.04(noble)** 이고, noble의
  네이티브 배포판은 **Jazzy**다. 즉 호스트에 apt로 깔 수 있는 건 Humble이 아니라 Jazzy다.
- **Humble ↔ Jazzy 교차 통신은 ROS 2가 지원하지 않는다**(같은 DDS를 써도 rosidl 타입 해시/
  IDL 세대가 달라 토픽이 매칭되지 않는다). 즉 "호스트에 ROS 깔면 되지"는 성립하지 않는다.
- 컨테이너 쪽을 `ros:jazzy`로 올리는 것은 **FC 도메인 전체(SITL 검증 완료본)를 다시 검증**해야
  하는 변경이고, 애초에 vision 세션의 권한 밖이다.
- 반대로 vision을 컨테이너 안으로 옮기는 것도 싸지 않다: 컨테이너에 **cv2도 picamera2도 없고**,
  **카메라 디바이스가 하나도 패스스루되어 있지 않으며**(`/dev/ttyACM0`뿐), 호스트의 libcamera는
  Python 3.12용으로 **로컬 소스빌드**한 것이라(`local-libcamera/env.sh`) Python 3.10 컨테이너에서
  재사용할 수 없다. 이 스택을 세우는 데 실제로 세션 하나가 통째로 들었다
  (`docs/vision_camera_bringup.md`).

### 2.3 ⚠️ 설령 양쪽에 같은 배포판을 맞춰도 남는 함정 (미리 기록)

`NetworkMode=host` + `IpcMode=private` 조합은 ROS2+Docker의 알려진 함정 자리다. Fast DDS는
같은 호스트로 판단되면 **공유메모리(SHM) 전송을 먼저 쓰는데**, 컨테이너가 독립 `/dev/shm`
(64MB)을 갖고 있어 호스트 프로세스와 SHM 세그먼트를 공유하지 못한다. **디스커버리는 성공하는데
데이터만 안 흐르는** 증상으로 나타날 수 있다. 회피책은 `--ipc=host` 또는 UDP 강제 XML 프로파일.
**이건 이번 세션에서 재현 검증하지 않았다**(§7 미확인) — 다만 §8 권고(R4)를 택하면 이 함정을
아예 만나지 않는다.

---

## 3. 조사 A — 메시지 타입 선택

### 3.1 🔴 (a) `mavros_msgs/msg/LandingTarget` — 실물 확인 결과

**mavros 2.14.0**(`/opt/ros/humble/share/mavros/package.xml`, `mavros_msgs` 동일 버전).
아래는 전부 실기체에서 직접 읽은 것이다.

**필드 구성** (`/opt/ros/humble/share/mavros_msgs/msg/LandingTarget.msg`, 컨테이너에서 `cat`):

```
std_msgs/Header header
uint8   target_num
uint8   frame
float32[2] angle      # X/Y 축 각도 오프셋 (rad)
float32 distance      # 거리 (m)
float32[2] size       # 타겟 각크기 (rad)
geometry_msgs/Pose pose
uint8   type          # LANDING_TARGET_TYPE
```

**`type` 상수** (msg 안에 정의됨):
`LIGHT_BEACON=0`, `RADIO_BEACON=1`, **`VISION_FIDUCIAL=2`**, **`VISION_OTHER=3`**.

**MAVROS 플러그인** — `landing_target`은 `mavros_extras`에 존재하고
(`/opt/ros/humble/share/mavros_extras/mavros_plugins.xml:109`) **denylist에 없다**
(`px4_pluginlists.yaml` denylist = `image_pub, vibration, distance_sensor, rangefinder,
wheel_odometry`) → `ros2 launch mavros px4.launch`(실기체 절차,
`docs/mc_flight_procedure.md:38`) 시 **로드된다**. 네임스페이스는 `mavros`
(`px4.launch`의 `<arg name="namespace" default="mavros"/>`) → 토픽 풀네임 `/mavros/landing_target/*`.

**토픽 4개** (`strings libmavros_extras_plugins.so` + 소스 원문 대조, 아래 참조):

| 토픽 | 방향 | 타입 | 비고 |
|---|---|---|---|
| `/mavros/landing_target/raw` | **구독**(ROS→FCU) | `mavros_msgs/LandingTarget` | 🔴 **`listen_lt: true`일 때만 생성** |
| `/mavros/landing_target/pose` | **구독**(ROS→FCU) | `geometry_msgs/PoseStamped` | 항상 존재. angle/distance/size를 **파라미터로 재계산** |
| `/mavros/landing_target/pose_in` | 발행(FCU→ROS) | `geometry_msgs/PoseStamped` | 우리가 쓸 게 아님 |
| `/mavros/landing_target/lt_marker` | 발행(FCU→ROS) | `geometry_msgs/Vector3Stamped` | 〃 |

소스 원문(mavros 2.14.0 `mavros_extras/src/plugins/landing_target.cpp`, WebFetch로 확인):

```cpp
node_declare_and_watch_parameter(
  "listen_lt", false, [&](const rclcpp::Parameter & p) {
    auto listen_lt = p.as_bool();
    land_target_sub.reset();
    if (listen_lt) {
      land_target_sub = node->create_subscription<mavros_msgs::msg::LandingTarget>(
        "~/raw", 10, std::bind(&LandingTargetPlugin::landtarget_cb, this, _1));
    }
  });
```

🔴 **실기체 `px4_config.yaml`은 `listen_lt: false`다**
(`/opt/ros/humble/share/mavros/launch/px4_config.yaml:214`, 노드 키는 `:212`, 직접 읽음).
**즉 지금 `/mavros/landing_target/raw`에 아무리 발행해도 구독자가 없다.** 쓰려면 파라미터를
`true`로 바꿔야 하고, 그건 시스템 yaml 수정 또는 launch override — **FC 도메인 작업**이다.

**필드 통과 여부** — `landtarget_cb`는 `angle/distance/size/target_num/frame/type`을 그대로
MAVLink로 넘긴다(`landing_target()`가 `lt.frame = frame; lt.distance = distance; ... uas->send_message(lt)`).
반면 `~/pose` 경로(`pose_cb`)는 `angle/distance/size`를 **카메라 FOV·`target_size` 파라미터로
재계산**한다 — 즉 `~/pose`를 쓰면 `image.width/height`(기본 640×480), `camera.fov_x/y`(기본 115°),
`target_size`(기본 0.3×0.3m)를 **우리 카메라 실측값으로 다 고쳐야** 한다. 우리 실측 HFOV는 75°이고
타겟은 0.5m(ArUco)/3.0m(초록매트)라 **기본값과 전부 다르다.**

#### 🔴 `frame` 상수가 실제 MAV_FRAME과 1씩 어긋나 있다 (같은 기체에서 두 파일 대조)

`mavros_msgs/msg/LandingTarget.msg`의 상수 블록 vs
`/opt/ros/humble/include/mavlink/v2.0/common/common.hpp`의 `enum class MAV_FRAME`:

| 이름 | `LandingTarget.msg` | 실제 `MAV_FRAME` (common.hpp) |
|---|---|---|
| `GLOBAL` | 0 | 0 ✅ |
| `LOCAL_NED` | **2** | **1** ❌ |
| `MISSION` | 3 | 2 ❌ |
| `LOCAL_ENU` | 5 | 4 ❌ |
| `BODY_NED` | 9 | 8 ❌ |
| `GLOBAL_TERRAIN_ALT_INT` | 12 | 11 ❌ |
| `BODY_FRD` | **정의 없음** | **12** |

`landtarget_cb`는 `static_cast<MAV_FRAME>(req->frame)`으로 **실제 enum 기준**으로 분기한다:

```cpp
const auto data_frame = static_cast<MAV_FRAME>(req->frame);
switch (data_frame) {
  case MAV_FRAME::LOCAL_NED:  // = 1
      position = ftf::transform_frame_enu_ned(...); position_valid = true; break;
  case MAV_FRAME::BODY_FRD:   // = 12
      position = ftf::transform_frame_baselink_aircraft(...); position_valid = true; break;
  default:
      if (data_frame != MAV_FRAME::GLOBAL)
        RCLCPP_WARN_STREAM(get_logger(), "LT: Landing target frame '" << req->frame << "' is not supported");
}
```

**따라서 `msg.frame = LandingTarget.LOCAL_NED`(=2)로 채우면 → `MAV_FRAME::MISSION`으로 해석 →
`default` 분기 → "not supported" 경고 + position 0 + `position_valid=false`. 그런데도
`lt.frame = 2`가 그대로 FCU로 나간다.** 조용히 틀린 게 아니라 경고는 나오지만, 상수 이름을
믿고 짜면 100% 이 함정에 빠진다.

**정확한 사용법:** `msg.frame = 1`(LOCAL_NED) 또는 `msg.frame = 12`(BODY_FRD)를 **정수 리터럴로**
쓰고 주석을 단다. **msg 상수를 쓰지 마라.**

#### 좌표 규약 (플러그인이 뭘 기대하는가)

MAVROS 표준대로 **ROS 쪽은 항상 ENU/baselink(FLU)** 이고, 플러그인이 NED/aircraft(FRD)로
변환한다:

- `frame = 1 (LOCAL_NED)` → 우리가 넣을 것은 **local ENU** 벡터
  (`/mavros/local_position/pose`와 같은 프레임). `transform_frame_enu_ned`가 NED로 바꾼다.
- `frame = 12 (BODY_FRD)` → 우리가 넣을 것은 **body FLU** 벡터(x=전방, y=좌, z=상).
  `transform_frame_baselink_aircraft`가 FRD로 바꾼다.

👉 **우리에게 자연스러운 건 `12 (BODY_FRD)`다.** `TargetEstimate.position`은 이미 "기체(카메라)
기준 상대 벡터"라, 카메라→body 고정 회전만 곱하면 끝이고 **기체 자세도 시간동기도 필요 없다**
(§4.4 참조 — 이게 body 프레임의 결정적 이점이다).

#### `confidence` / `calib_accuracy` / `not_for_closed_loop_30cm` / `target_type` / `command`를 어디 싣나

| 필드 | LandingTarget에 실을 자리 | 결론 |
|---|---|---|
| `confidence` | **없음** | ❌ 못 실음 |
| `calib_accuracy` (str) | **없음** | ❌ 못 실음 |
| `not_for_closed_loop_30cm` (bool) | **없음** | ❌ 못 실음 |
| `target_type` (`"aruco_23"` 등) | `type` (0~3 enum) + `target_num`(uint8) | △ **손실 압축만 가능** — ArUco→`VISION_FIDUCIAL(2)`, 초록매트/흰박스→`VISION_OTHER(3)`. `"aruco_23"` vs `"distress_green"` 구분은 `target_num`에 우리 사설 코드북을 얹어야 하고, 그건 MAVLink가 "타겟 ID"로 쓰는 필드를 전용하는 것 |
| `Decision.command` (str) | **없음** | ❌ 못 실음 |

**→ (a)만으로는 안전 계약을 실을 수 없다. 별도 채널이 반드시 필요하다.** 이건 (a)의 결격이
아니라 (a)의 **용도가 다르다**는 뜻이다 — (a)는 "PX4에게 타겟 위치를 알려주는 네이티브 피벗
경로"이지 "우리 폐루프의 안전 계약 채널"이 아니다.

### 3.2 (b) 표준 메시지 조합

핵심 후보는 `geometry_msgs/PoseWithCovarianceStamped`
(= `header` + `pose.pose`(Pose) + `pose.covariance` float64[36], 6×6 순서 `[x,y,z,rot_x,rot_y,rot_z]`).

- **장점 — `uncertainty` 자리가 딱 맞는다.** `TargetEstimate.uncertainty`는 지금 항상 `None`이고
  (`vision/core/target.py:32-33,65`) 실측 캘리브레이션 후 채울 자리로 예약돼 있다. 그때 산출될
  것은 solvePnP 재투영 잔차 기반 공분산이므로 **6×6 공분산에 그대로 들어간다.** 커스텀 필드를
  새로 정의할 필요가 없고, rviz·`robot_localization` 등 기성 도구가 즉시 읽는다.
  → **이 장점은 실질적이다.** 다만 지금은 채울 값이 없어 "미리 잡아두는 자리"라는 점은 정직히 봐야 한다.
- **한계** — 나머지 5개 필드는 여전히 갈 곳이 없다. 해결책은 **동반 상태 토픽**:
  - `diagnostic_msgs/DiagnosticStatus` — `level`(OK/WARN/ERROR), `name`, `message`,
    `values`(KeyValue 배열). `calib_accuracy`/`not_for_closed_loop_30cm`/`target_type`/
    `confidence`/`state`/`command`를 **전부 KeyValue로** 실을 수 있고, `level`이 그대로
    페일세이프 신호가 된다. **새 패키지 0개.**
  - 또는 `std_msgs/String`에 JSON — 더 단순하지만 스키마가 없다.
- **주의:** 두 토픽으로 나누면 **동기 문제가 생긴다.** 반드시 두 메시지의 `header.stamp`를
  같은 값으로 채우고, 소비자는 stamp가 일치하는 쌍만 유효로 본다(불일치 = stale). 이건
  §6 페일세이프 계약에 넣었다.

### 3.3 (c) 커스텀 `.msg` 패키지 신설 — 구체적 비용

| 비용 항목 | 내용 |
|---|---|
| 새 패키지 | `vision_msgs_local`(ament_cmake) 신설. `package.xml`+`CMakeLists.txt`+`msg/*.msg` |
| 빌드 의존 | **양쪽 다** 이 패키지를 빌드해야 한다. `fc_ros/package.xml`에 `<depend>` 추가 → `fc_ros`가 vision 패키지에 빌드 의존을 갖게 된다(루트 `CLAUDE.md` "도메인 간 의존 기록" 대상) |
| RPi 컨테이너 절차 | `docs/rpi_deploy.md` 절차가 `colcon build --packages-select fc_ros` 한 줄인데 → `--packages-select vision_msgs_local fc_ros`(순서 의존)로 바뀐다. `--symlink-install` 금지 규칙(§4)은 그대로 |
| ament 없는 쪽 | 🔴 **결정타** — vision 프로세스는 지금 호스트 `picam-venv`(ROS 없음)에서 돈다. 커스텀 msg의 Python 바인딩은 `rosidl`이 생성하므로 **ROS2 환경 안에서만** import된다. 즉 **§2의 런타임 문제를 먼저 풀지 않으면 (c)는 착수 자체가 불가능하다** |
| 유지비 | msg 수정 시 양쪽 재빌드 + 배포 검증(`docs/rpi_deploy.md` §5 md5 대조) |

**결론: (c)는 표현력은 완벽하지만 지금 비용/차단요인이 가장 크다. 그리고 §2를 풀기 전엔
(b)도 (c)도 (a)도 전부 착수 불가라는 점이 (c)의 문제가 아니라 이 조사의 진짜 발견이다.**

### 3.4 세 후보 비교표

| | (a) `mavros_msgs/LandingTarget` | (b) `PoseWithCovarianceStamped` + `DiagnosticStatus` | (c) 커스텀 msg |
|---|---|---|---|
| 새 빌드 의존 | 없음 (mavros 기설치) | 없음 (전부 기본 패키지) | **양쪽 빌드 의존 신설** |
| 불확실성(공분산) | ❌ 자리 없음 | ✅ 6×6 그대로 | ✅ 자유 |
| `confidence` | ❌ | ✅ (KeyValue 또는 공분산 팽창) | ✅ |
| `calib_accuracy` | ❌ | ✅ (KeyValue) | ✅ |
| `not_for_closed_loop_30cm` | ❌ | ✅ (KeyValue + `level`) | ✅ |
| `target_type` 원문 | △ enum 2종으로 손실 압축 | ✅ | ✅ |
| `command`/`state` | ❌ | ✅ (KeyValue) | ✅ |
| PX4 네이티브 피벗 | ✅ **유일한 경로** | ❌ (변환 노드 필요) | ❌ |
| 함정 | 🔴 frame 상수 오프바이원 / `listen_lt:false` | 두 토픽 stamp 동기 필요 | rosidl 없는 환경에선 import 불가 |

**→ (a)와 (b)는 배타적이지 않고 역할이 다르다. 둘 다 쓰는 게 맞다**(§8 권고).

---

## 4. 조사 B — 자세 소스와 좌표계 계약

### 4.1 fc_ros가 기체 자세를 받는 방식 (읽기만 함)

- 구독: `/mavros/local_position/pose` (`geometry_msgs/PoseStamped`) —
  `offboard_node.py:432-435`
- **QoS: `BEST_EFFORT` / `KEEP_LAST` / `depth=10`** — `offboard_node.py:68-72`의 `_MAVROS_QOS`.
  같은 프로파일을 `velocity_local`·`state`·`extended_state`·`altitude`에 공용으로 쓴다
  (`:436-451`). 발행 쪽 `/mavros/setpoint_position/local`도 같은 프로파일(`:462-463`).
  → **vision이 만드는 토픽도 BEST_EFFORT/KEEP_LAST/depth=1~10으로 맞춰야 한다.**
  RELIABLE 발행자 ↔ BEST_EFFORT 구독자는 붙지만, 그 반대(BEST_EFFORT 발행 ↔ RELIABLE 구독)는
  **QoS 비호환으로 아예 연결되지 않는다.** 정밀착륙 스트림은 최신값이 전부이므로 BEST_EFFORT +
  `depth=1`이 옳다.
- ⚠️ `/mavros/local_position/pose`는 **volatile(래치 없음)** 이다 — `offboard_node.py:478-485`가
  이 사실 때문에 겪은 사고(2026-07-25 flight01)를 주석으로 남겨 뒀다. 노드 기동 직후엔
  **자세가 아예 없다.** 정밀착륙 서브상태도 "한 번이라도 받았는가"를 확인해야 한다
  (`_pose_received` 패턴, `:485`).

### 4.2 ENU/NED 변환 관례 (`fc_bridge/utils/rotation.py`, `vehicle_state_bridge.py`)

```
ENU → 내부 저장 (vehicle_state_bridge.py:16-41)
  pos_ned = [ y_enu, x_enu, z_enu ]      # = [N, E, h_up]   ⚠️ 3번째가 '위'다
  vel_ned = [ vy_enu, vx_enu, -vz_enu ]  # = [vN, vE, vD]   ⚠️ 3번째가 '아래'다
  roll  =  roll_enu
  pitch = -pitch_enu
  yaw   = wrap(pi/2 - yaw_enu)
역변환 (rotation.py:23-35)
  yaw_enu = pi/2 - yaw_ned  →  quat (w,0,0,z)
```

🔴 **이 저장소의 "ned"는 표준 NED가 아니다.** `pos_ned`의 3번째 성분은 **h_up**(위가 양수)이고
`vel_ned`의 3번째는 **vD**(아래가 양수)다. 같은 접미사 `_ned`가 위치와 속도에서 반대 부호
규약을 쓴다(`vehicle_state_bridge.py:30, 51`의 주석이 둘 다 명시적으로 그렇게 말한다).
**vision이 이 배열을 그대로 받아 쓰면 고도 부호를 반드시 틀린다.** vision 쪽 계약은 이
내부 표현을 **참조하지 않는다**로 못박는 게 안전하다 — vision은 ROS 표준 ENU/FLU만 쓰고,
NED 변환은 fc_ros 안에서만 일어나게 한다.

**쿼터니언 순서 함정:** `quat_to_euler_xyz(w, x, y, z)`는 **w가 먼저**다(`rotation.py:11`).
반면 `TargetEstimate.orientation`은 **(x, y, z, w)** 다(`vision/core/target.py:58`) — ROS
`geometry_msgs/Quaternion`도 x,y,z,w다. **두 규약이 파일 하나 사이로 다르다.** 어댑터에서
반드시 명시적으로 재배열할 것.

### 4.3 좌표계 체인 전체 명세

**단위: 전부 미터(SI), 각도는 라디안. 회전은 능동회전(active), 벡터를 프레임 간 이동시킬 때
`p_B = R_{B←A} · p_A` 표기를 쓴다.**

```
① 카메라 광학 프레임 (cam) — OpenCV 표준
     X = 이미지 오른쪽, Y = 이미지 아래, Z = 렌즈 바깥(전방)
     TargetEstimate.position 이 바로 이 프레임 (vision/core/target.py:26-28)
     orientation = quat (x,y,z,w), cv2.Rodrigues(rvec) 유래

         │  R_{flu←cam}(ψ_m) : 고정 마운트 회전 (나디르 하드마운트)
         ▼
② body FLU (baselink, ROS 표준) — X=전방(기수), Y=좌, Z=상
     MAVROS/ROS가 기체 body로 쓰는 프레임. mavros landing_target 플러그인이
     BODY_FRD 분기에서 기대하는 입력 프레임(= baselink)

         │  (x, -y, -z)  : FLU → FRD  ← mavros ftf::transform_frame_baselink_aircraft
         ▼
③ body FRD (aircraft, PX4) — X=전방, Y=우, Z=하
     MAVLink LANDING_TARGET frame=12(BODY_FRD)가 기대하는 프레임

  ── 또는 ENU 경로 ──
② body FLU  ──  R(q_enu) : /mavros/local_position/pose 의 orientation ──▶
④ local ENU (map) — X=동, Y=북, Z=상
         │  (y, x, -z) : ENU → NED  ← mavros ftf::transform_frame_enu_ned
         ▼
⑤ local NED — X=북, Y=동, Z=하   (MAVLink frame=1(LOCAL_NED))
```

**① → ② 의 명시적 형태 (나디르 하드마운트, 마운트 요각 ψ_m = 0인 공칭 케이스):**

"ψ_m = 0" = 이미지의 위쪽(-Y_cam 방향)이 기수 전방을 가리키도록 장착된 경우.

```
x_flu = -y_cam
y_flu = -x_cam
z_flu = -z_cam

R_{flu←cam}(ψ_m=0) = [[ 0, -1,  0],
                      [-1,  0,  0],
                      [ 0,  0, -1]]      det = +1 (진짜 회전)
```

같은 것을 FRD로 바로 쓰면:

```
R_{frd←cam}(ψ_m=0) = [[ 0, -1, 0],
                      [ 1,  0, 0],
                      [ 0,  0, 1]]      (= Z축 +90° 회전), det = +1
```

일반 ψ_m에 대해서는 `R_{frd←cam}(ψ_m) = Rz(ψ_m) · R_{frd←cam}(0)`.

🔴 **ψ_m 은 실측해야 하는 값이지 가정할 값이 아니다.** 지금 저장소 어디에도 이 값이 없다
(§7 미확인). 카메라를 기수 기준 몇 도 돌려 달았는지가 그대로 **착륙 오프셋의 방향**이 되므로,
잘못 잡으면 "정확히 90° 틀린 방향으로 정확하게" 이동한다. 90°/180° 오차는 실비행에서 아주
알아보기 쉬운 증상(수정할수록 멀어짐)이니 첫 폐루프 시험의 1번 체크항목으로 둔다.

**왕복검증(round-trip) 명세 — `fc_bridge/tests/test_rotation.py` 관례를 그대로 따른다.**
그 파일의 `test_round_trip_through_decode`(`:30-36`)는 "인코드 → 디코드 → 원값 복원"을
`abs=1e-9`로 확인한다. vision 쪽도 같은 형태로 쓸 수 있게 명세한다:

| 검증 | 내용 | 통과 기준 |
|---|---|---|
| RT-1 | `R_{flu←cam}(ψ)` 가 모든 ψ에서 `Rᵀ·R = I`, `det = +1` | `1e-12` |
| RT-2 | `p_cam → p_flu → p_frd → (역변환) → p_cam` 왕복 | `1e-9` |
| RT-3 | ψ_m=0에서 알려진 방향쌍이 맞는가 — 카메라 정중앙 바로 아래 타겟 `p_cam=(0,0,h)` → `p_frd=(0,0,h)`(정하방), `p_flu=(0,0,-h)` | `1e-12` |
| RT-4 | 이미지 **오른쪽**에 보이는 타겟 `p_cam=(+d,0,h)` → `p_frd=(0,+d,h)`(기체 **우측**), `p_flu=(0,-d,-h)` | `1e-12` |
| RT-5 | 이미지 **위쪽**에 보이는 타겟 `p_cam=(0,-d,h)` → `p_frd=(+d,0,h)`(기체 **전방**) | `1e-12` |
| RT-6 | quat 순서 왕복: `(x,y,z,w)` → `quat_to_euler_xyz(w,x,y,z)` → 재구성 quat | `1e-9` |
| RT-7 | ENU 경로: 자세 `q_enu`가 수평(roll=pitch=0)일 때 `p_enu`의 수평성분이 `p_flu`를 yaw만큼 돌린 것과 같은가 | `1e-9` |

RT-3~RT-5는 **부호 실수를 잡는 실질적 그물**이다(축 이름만 맞고 부호가 뒤집힌 회전행렬도
RT-1·RT-2는 통과한다).

### 4.4 ⭐ 왜 body 프레임 경로가 결정적으로 유리한가

`TargetEstimate.position`은 **카메라에서 타겟까지의 3D 상대 벡터**다. 이걸 `R_{frd←cam}` 하나만
곱하면 **body FRD 상대 벡터**가 된다.

- **기체 자세가 필요 없다.** → `/mavros/local_position/pose`를 읽을 필요가 없다.
- **시간 동기가 필요 없다.** → 프레임 타임스탬프 ↔ 자세 타임스탬프 정렬 문제가 **사라진다.**
- 기울기(tilt) 보정은 **PX4가 자기 자세로 알아서** 한다(PX4는 자기 IMU 자세를 지연 없이 안다).

반대로 ENU/NED로 변환해 넘기려면 vision이 기체 자세를 읽어 곱해야 하고, 그 순간
§4.2(자세 오차)와 §7.8(시간 동기) 문제가 **vision 쪽으로 이관된다.**

**→ 좌표계 계약의 정식 권고: vision은 `body FLU` 상대 벡터를 낸다. NED 변환은 하지 않는다.**

### 4.5 🔴 남는 물리 리스크 — "고무 마운트의 역설"

`vision_plan.md` §4.2 4번:

> **고무 마운트의 역설:** 진동댐핑이 카메라를 IMU와 분리시켜 **IMU 자세 ≠ 실제 카메라 자세** 잔차.

§4.4의 body-프레임 경로는 "시간 동기"와 "자세 읽기"를 없애주지만 **이 잔차는 없애지 못한다.**
`R_{frd←cam}`을 상수로 두는 것 자체가 "카메라가 기체에 강체로 붙어 있다"는 가정이고, 고무
마운트는 정확히 그 가정을 깬다. 정량 영향은 §4.2의 식 그대로다:

```
지상오차 = 고도 × tan(θ_잔차)
  40m · 5° → 3.50 m      (40·tan5° = 40×0.08749 = 3.50)
  40m · 2° → 1.40 m
  40m · 1° → 0.70 m
   3m · 1° → 0.052 m
```

**고도가 낮아질수록 이 오차는 선형으로 줄어든다** — 즉 이 리스크는 "40m에서 정확히 찍는 것"을
막지, "3m에서 폐루프로 수렴하는 것"은 막지 않는다. §4.3(30cm는 폐루프로만)의 논지와 정확히
일치한다. **대응은 회피가 아니라 게이팅이다: 고고도 추정치로 최종 커밋하지 않는다**(§6의
`not_for_closed_loop_30cm` 계약과 상태머신 커밋 게이트가 이미 이 역할).

미확인: 실제 잔차 각도가 몇 도인지 아무도 측정한 적 없다(§7).

### 4.6 시간 동기 (§7.8) — 현재 양쪽 다 "받은 시각"을 쓰고 있다

| 쪽 | 지금 쓰는 시각 | 근거 | 문제 |
|---|---|---|---|
| vision | `time.time()` — **프레임을 넘겨주는 순간의 wall clock** | `vision/utils/frame_source.py:178` | 센서 노출 시각이 아니다. picamera2가 주는 `SensorTimestamp`(CLOCK_BOOTTIME ns)를 **안 쓴다** — `capture_metadata()` 호출은 `tools/calib_capture.py`에만 있고 `frame_source.py`엔 없다 |
| fc_ros | `time.monotonic()` — **콜백이 도는 순간** | `fc_ros/.../vehicle_state_bridge.py:41,52` | `msg.header.stamp`(MAVROS가 FCU 시각으로 채운 값)를 **버린다** |

**→ 두 쪽이 서로 다른 클록(wall vs monotonic)의, 서로 다른 지점(핸드오프 vs 콜백)의 시각을
쓰고 있어서 지금은 정렬 자체가 불가능하다.** §4.4 경로를 택하면 이 문제가 **폐루프 유도에서는
사라지지만**, 로그 상관·재생·사후분석에는 여전히 필요하다.

**계약:**
1. vision이 내보내는 모든 메시지의 시각은 **`SensorTimestamp` 기반 단조 클록**으로 통일한다
   (프레임 핸드오프 시각이 아니라 **노출 시각**). picamera2 `capture_metadata()["SensorTimestamp"]`는
   `CLOCK_BOOTTIME` ns이고, ROS 쪽 `rclcpp::Clock(RCL_STEADY_TIME)`도 BOOTTIME/MONOTONIC 계열이라
   같은 기준으로 비교 가능하다. **오프셋을 한 번 측정해 기록**하고, 그 오프셋 자체를 메시지에
   실어 보낸다(`clock_offset_ns`) — 소비자가 그걸 보고 자기 클록으로 환산한다.
2. 시간 어긋남의 허용치는 각도로 환산해 정한다: `허용 시차 = 허용 각오차 / 기체 각속도`.
   기체 각속도는 **미측정**(§7)이므로 값은 실측 후 확정한다. 형태만 못박는다 —
   `max_clock_skew_s`를 파라미터로 두고, 초과 시 추정치를 **무효 처리**한다.
3. 40m·5°→3.5m라는 §4.2 계산이 곧 시간 예산이다: **시차는 곧 자세차다.**

---

## 5. 조사 C — 페일세이프 계약 (메시지 수준)

`vision_plan.md` §8:
> 비전 노드 크래시/검출 상실/브라운아웃 → offboard가 홀드·재상승(기존 OVERRIDE 계열 재사용).
> "추측 후 커밋" 금지.

### 5.1 ⚠️ 먼저 정정: 기존 `OVERRIDE`는 "홀드·재상승"이 아니다

`_request_override()`(`offboard_node.py:1081-1099`)는 **MC→POSCTL / FW→MANUAL을 요청하고,
거부되면 10틱 뒤 AUTO.LOITER로 폴백**한 뒤 **setpoint 발행을 멈춘다**. 즉 **제어권을 완전히
내려놓는 조종사 인계 경로**이지 홀드가 아니다. 그리고 그 바로 아래 `_safety_fallback()`
docstring(`:1104-1113`)은 이렇게 못박는다:

> **새 폴백 경로를 만들지 않는다.** S7 장애주입에서 실증된 안전경로는 OVERRIDE(FW)/
> OVERRIDE(MC)/PILOT_TAKEOVER 3종뿐이고, 새 경로를 만들면 그 경로부터 다시 실증해야 한다.

**따라서 §8의 "OVERRIDE 계열 재사용"을 문자 그대로 구현하면 검출 한 번 놓쳤다고 조종사 인계가
난다 — 과잉이다.** 하지만 새 폴백 경로를 만드는 것도 금지다. 이 모순의 해법은 **계층을 나누는
것**이다: 홀드/재상승은 **OFFBOARD 안에 머무는 정상 동작**(새 폴백 경로가 아님)이고,
OVERRIDE는 그 위의 최종 탈출구로 남긴다.

```
1단 (정상)   PRECISION_SERVO  — 추정치 유효 → 폐루프 하강
2단 (홀드)   HOLD_ON_TARGET   — 추정 무효/stale → 현재 위치 유지 setpoint 계속 발행
                               ※ _step_hold(:1230-1265)의 slew_setpoint 패턴 그대로 재사용
                                 = 새 경로가 아니라 이미 검증된 패턴의 재사용
3단 (재상승) REASCEND         — 2단이 t_hold 초과 → 목표 고도만 올린 같은 setpoint
4단 (포기)   기존 LANDING(AUTO.LAND) 또는 _request_override()
                               ※ 여기서만 검증된 3종 경로를 쓴다
```

**핵심 불변식: 1→2 전이는 "메시지가 왔다"가 아니라 "메시지가 안 왔다"로 트리거된다.**
소비자는 매 제어틱(10Hz, `control_hz` 기본 10.0 — `offboard_node.py:175`)마다 **자기가 가진
마지막 추정치의 나이를 재고**, 나이가 임계를 넘으면 아무도 아무 말 안 해도 2단으로 간다.

### 5.2 stale 판정

```
age = t_now(소비자 클록) - t_estimate(§4.6 계약대로 환산된 노출 시각)
유효 조건: age <= stale_timeout_s   AND   |clock_skew| <= max_clock_skew_s
```

- `stale_timeout_s` **권고 초기값 0.5 s**. 근거: 제어루프가 10Hz(`control_hz=10.0`)이므로
  0.5초 = 5틱 = 발행이 10Hz라면 5프레임 연속 결손. **이 값은 vision 발행 주파수를 측정한 뒤
  확정해야 한다** — `vision_plan.md` §10이 "폐루프 servo 목표 주파수(Hz)"를 **미정**으로
  남겨뒀고(§7 미확인), 실측 없이 못박으면 안 된다. 파라미터로 노출하고 기본값에 근거 주석을 단다.
- **두 토픽 방식(§3.2)이면 stamp 일치도 유효 조건에 포함한다**: pose와 status의
  `header.stamp`가 다르면 둘 중 하나가 유실된 것 → 무효.

### 5.3 `confidence` 하한 — 🔴 지금은 사실상 무의미하다

**측정한 사실:**
- `Detection.confidence` 기본값 = **1.0** (`vision/core/state.py:10`)
- `vision/modules/aruco.py`와 `vision/modules/distress_box.py`는 **`confidence`를 한 번도 설정하지
  않는다** (`grep -rn "confidence=" vision/modules/*.py` → `detector.py:46`, `vertiport_field.py:41` 둘뿐)
- `main.py`의 ArUco 경로는 `confidence=det.confidence`를 그대로 넘긴다(`vision/main.py:146`)

**→ ArUco `TargetEstimate.confidence`는 항상 정확히 1.0이다. `min_confidence` 게이트는 현재
ArUco 경로에서 no-op다.**
그리고 실제로 값이 들어가는 `detector.py:43`의 공식은
`confidence = solidity × min(1, area/min_area)` — **확률이 아니라 형상 휴리스틱**이고,
`vertiport_field.py:41`은 `min(1, circularity)`로 **의미가 아예 다르다.**

**계약:**
1. `confidence`는 **검출기 간 비교 불가**로 명시한다. 단일 전역 임계값을 쓰지 않는다.
2. 폐루프 게이팅은 `confidence`가 아니라 **상태머신의 커밋 게이트**(`lock_confirm_frames`
   연속 `fine_locked`, `max_candidates_for_lock` 모호 거절 — `core/state_machine.py:55-64`)에
   맡긴다. 이건 이미 구조적으로 강제돼 있고 테스트도 있다(`vision/CLAUDE.md` 테스트 규칙표).
3. `confidence`는 **관측/로깅용으로만** 싣는다. 하한은 파라미터로 노출하되 **기본값 0.0(비활성)**
   으로 두고, 검출기별 캘리브레이션이 생긴 뒤에 켠다. 지금 0.9 같은 값을 넣으면 "동작하는 것처럼
   보이지만 아무것도 거르지 않는" 가짜 안전장치가 된다.

### 5.4 노드가 죽은 걸 소비자가 어떻게 아는가 — **침묵이 정답, 명시적 무효는 보조**

| | 토픽 침묵(deadline miss) | 명시적 무효 메시지 |
|---|---|---|
| 프로세스 SIGKILL/OOM/커널 패닉 | ✅ 감지됨 | ❌ **감지 불가** — 죽은 프로세스는 메시지를 못 보낸다 |
| 브라운아웃(전원 순단) | ✅ | ❌ |
| 무한루프/데드락(프로세스는 살아있음) | ✅ | ❌ (발행 스레드가 멈춰도 못 보냄) |
| 검출 상실(노드는 건강) | ✅ (느림 — timeout 대기) | ✅ (즉시) |
| 오탐 자각(자기 거절) | ✅ (느림) | ✅ (즉시 + 사유 전달) |

**결론: 침묵을 유일한 권위로 삼고, 명시적 무효는 지연 단축용 최적화로만 얹는다.**
"명시적 무효 메시지를 받아야 안전동작에 들어간다"는 설계는 **생산자가 죽는 바로 그 순간에
실패한다.** 반대로 침묵 기반은 어떤 죽음이든 잡는다.

구체 계약:
1. **소비자는 `stale_timeout_s` 초과 시 자동으로 2단(HOLD)으로 간다.** vision의 협조가 전혀
   필요 없다. (ROS QoS `Deadline`으로 강제할 수도 있지만, 어차피 소비자가 매 틱 age를 재므로
   불필요한 복잡도다.)
2. **vision은 검출을 잃어도 발행을 멈추지 않는다.** `valid=false` + `reason`을 담은 메시지를
   같은 주기로 계속 보낸다 → 소비자는 "죽음"과 "안 보임"을 구분할 수 있고, 사유가 로그에 남는다.
   (이걸 안 보내면 두 경우가 똑같이 침묵으로 보인다.)
3. 🔴 **소켓 전송(§8 R4)을 쓰면 이게 더 강해진다** — 프로세스가 죽으면 커널이 소켓을 닫아
   소비자가 **EOF를 즉시** 받는다. timeout을 기다릴 필요조차 없다. DDS 토픽 침묵보다 엄밀하게 우월하다.

### 5.5 `not_for_closed_loop_30cm=True` 의 전파와 소비자 의무

🔴 **먼저 직시할 사실: 이 플래그는 지금 100% 확률로 `True`다.** 실측 캘리브레이션이 보류됐고
(`docs/vision_plan.md` §9, 메모리 `project_vision_calibration_deferred`), `nominal.yaml`이
`not_for_closed_loop_30cm: true`를 담고 있으며, `TargetEstimate` 기본값도 `True`
(`vision/core/target.py:69`)다.

**→ "True면 폐루프 금지"로 계약하면 정밀착륙 통합 전체가 착수 즉시 죽는다.** 그렇다고 무시하면
provenance echo(§7.3)를 만든 이유가 사라진다.

**계약 — 플래그는 "서보 금지"가 아니라 "최종 커밋 금지"다:**

| `not_for_closed_loop_30cm` | 소비자가 해도 되는 것 | 소비자가 하면 안 되는 것 |
|---|---|---|
| `True` (현재) | ACQUIRE/CENTER/SERVO로 **`unverified_calib_floor_agl_m`(권고 초기값 = 상태머신 `terminal_agl_m` 기본 3.0m)까지** 하강·정렬 | 그 고도 아래로 **비전만 믿고** 계속 하강. 바닥 고도에 닿으면 정렬 상태를 유지한 채 **AUTO.LAND/GPS+라이다에 인계** |
| `False` (실측 캘리브 후) | TERMINAL까지 비전 폐루프로 커밋 | — |

이 설계의 장점: **지금 당장 가치가 있고**(3m까지 비전으로 정렬하면 GPS만 쓰는 것보다 훨씬 나음),
**실측 캘리브레이션이 끝나면 플래그 하나로 자동 해금**된다. 코드 변경이 필요 없다.

전파 경로: `nominal.yaml` → `load_camera_calibration()` → `solve_target_pose(calib_accuracy=,
not_for_closed_loop_30cm=, calib_id=)` → `TargetEstimate` → **인터페이스 메시지** → 소비자.
`calib_id`도 같이 실어 로그에서 "어느 캘리브로 난 비행인가"를 사후에 특정할 수 있게 한다
(§7.3 provenance echo의 원래 목적).

### 5.6 계약 파라미터 요약 (전부 파라미터화, 매직넘버 금지 — §7.3)

| 파라미터 | 권고 초기값 | 근거 / 상태 |
|---|---|---|
| `stale_timeout_s` | 0.5 | 제어루프 10Hz의 5틱. **vision 발행률 실측 후 확정 필요** |
| `max_clock_skew_s` | — | **실측 후 확정** (§4.6) |
| `min_confidence` | 0.0 (비활성) | §5.3 — 지금 켜면 가짜 안전장치 |
| `unverified_calib_floor_agl_m` | 3.0 | 상태머신 `terminal_agl_m` 기본값과 정렬 (`state_machine.py:66`) |
| `hold_before_reascend_s` | — | **미정** — FC 도메인이 정할 값 |
| `mount_yaw_psi_m_rad` | — | **실측 필요** (§4.3) |

---

## 6. 조사 E — 상태머신 `command`를 인터페이스에 실을 것인가

### 6.1 코드가 이미 답의 절반을 적어 뒀다

`vision/core/state_machine.py:20-21` (파일 최상단 docstring):

> 이 파일이 뱉는 `Decision.command`는 **문자열 힌트일 뿐이다** — 이걸 소비해 실제 기체를 움직이는
> 쪽(`fc_ros`/`fc_bridge`)은 이번 범위 밖(§9 7번, 다른 세션).

`Decision` dataclass 주석(`:88`)도 같은 말을 반복한다:
> `command`는 소비자(FC 세션)를 위한 **문자열 힌트일 뿐**, 여기선 뱉기만 한다.

즉 원작자는 이미 "이건 명령이 아니다"라고 두 번 못박았다. 하지만 그건 "안 보낸다"가 아니라
"보내도 명령으로 취급하지 말라"는 뜻이다.

### 6.2 찬반

**찬 (보내자):**
- vision은 이미 판단을 했다 — `lock_confirm_frames` 연속 확정, `max_candidates_for_lock` 모호
  거절, `loss_tolerance_frames` 흔들림 허용, 블라인드 드리프트 추정. 이건 **프레임 시퀀스를
  본 쪽만 할 수 있는 판단**이다. FC가 단일 추정치만 받아서는 재현할 수 없다.
- FC에서 재구현하면 **같은 로직이 두 벌**이 되고, 둘이 어긋나는 순간 디버깅 지옥이다.
- `state`(`LandingState`)는 JSONL에 이미 실려 있어(`vision/CLAUDE.md` 상태머신 배선 절)
  비행 후 로그 상관에 반드시 필요하다.

**반 (보내지 말자):**
- **제어 권한이 두 곳으로 쪼개진다.** "vision이 descend라고 했는데 FC가 안전상 못 내려간다"에서
  누가 이기는가 — 이게 답이 없으면 실비행에서 갈린다.
- FC 도메인엔 이미 강한 규칙이 있다: `_safety_fallback()` docstring(`offboard_node.py:1104-1113`)
  "**새 폴백 경로를 만들지 않는다** ... 새 경로를 만들면 그 경로부터 다시 실증해야 한다."
  외부 문자열이 상태전이를 직접 몰면 정확히 그 "새 경로"가 된다.
- `command` 문자열은 스키마가 없다. 오타/새 값 추가가 **소비자 쪽에서 조용히 무시**된다.

### 6.3 권고 — **거부권(veto)으로만 싣는다**

두 입장은 사실 배타적이지 않다. 갈리는 지점은 "보내느냐"가 아니라 **"방향이 있느냐"** 다.

```
✅ vision의 상태는 FC 하강의 [필요조건]이다.  (withhold 가능)
❌ vision의 상태는 FC 하강의 [충분조건]이 아니다. (compel 불가)
```

구체적으로:
- **싣는 것:** `state`(LandingState enum의 문자열) + `reason`(사유 문자열, 로깅 전용).
  `Decision.reason`은 지금 JSONL에도 안 실린다(`vision/CLAUDE.md`: "로그 스키마 변경 금지"로
  in-process에만 남음) — 인터페이스에는 싣는 게 맞다. 사후 분석에서 "왜 거절했나"가 가장 중요하다.
- **싣지 않는 것(또는 싣되 소비자가 무시):** `command` 동사 문자열(`"descend"` 등).
- **소비자 규칙:**
  ```
  FC가 하강해도 되는가 = (FC 자체 안전조건 전부 통과)
                        AND (vision.state ∈ {PRECISION_SERVO, TERMINAL})
                        AND (추정치 fresh)
                        AND (not_for_closed_loop_30cm 게이트 통과, §5.5)
  ```
  vision이 `HOLD`/`ABORT_ASCEND`면 FC는 **반드시 멈추거나 올라간다**(거부권 행사).
  vision이 `PRECISION_SERVO`여도 FC 자체 조건이 안 되면 **FC가 이긴다**.

**"누가 이기는가"의 답: 항상 더 보수적인 쪽이 이긴다.** 이건 단조(monotone) 성질이라 조합 폭발이
없고, 새 폴백 경로도 안 만든다(vision은 기존 FC 안전조건에 AND 항을 하나 더할 뿐이다).

🔀 **이건 사용자 결정이 필요한 갈림길이다** — §8 D3.

---

## 7. 미확인으로 남은 것 (추측으로 메우지 않았다)

| # | 미확인 항목 | 왜 확인 못 했나 | 확인 방법 |
|---|---|---|---|
| U1 | `NetworkMode=host` + `IpcMode=private`에서 Fast DDS SHM 전송이 실제로 실패하는지 | 호스트에 ROS2가 없어 실험 자체가 불가능 | 양쪽 배포판을 맞춘 뒤 `ros2 topic pub/echo` 실측 |
| U2 | Humble↔Jazzy 교차 통신이 이 조합에서 정확히 어떻게 실패하는지 | 위와 동일 | (권고 R4를 택하면 확인 불필요) |
| U3 | 카메라 마운트 요각 **ψ_m** (기수 기준 카메라 회전) | 물리 측정 필요, 저장소에 기록 없음 | 기수 방향에 표식 두고 1프레임 촬영 |
| U4 | 고무 마운트 자세 잔차(§4.5)의 실제 크기 | 카메라 IMU 없음, 실비행 데이터 없음 | 실비행 중 알려진 지상 타겟 vs IMU 자세 비교 |
| U5 | vision 파이프라인의 실제 발행 주파수(폐루프 servo Hz) | `vision_plan.md` §10에도 "미정"으로 남아 있음 | RPi에서 `main.py live` 실행해 JSONL latency 통계 |
| U6 | 기체 최대 각속도 → `max_clock_skew_s` 산출 근거 | 실비행 로그 미분석 | 기존 비행로그의 자세 미분 |
| U7 | picamera2 `SensorTimestamp`와 ROS steady clock의 실제 오프셋 | 카메라 배타적 + 이번 세션 카메라 미점유 원칙 | 동시 캡처 스크립트 1회 실행 |
| U8 | PX4 쪽 precision landing 요구사항(`PLD_*` 파라미터, `landing_target_estimator` 활성 여부, 이 기체 PX4 버전) | FC 도메인 + 기체 파라미터 미조회 | FC 세션이 `ros2 param`/QGC로 확인 |
| U9 | `listen_lt: true`로 바꿨을 때 부작용 | 설정 변경 금지(읽기 전용 세션) | FC 세션이 launch override로 시험 |
| U10 | `mavros_msgs` frame 상수 오프바이원이 upstream에서 고쳐졌는지 / 다른 버전에서도 같은지 | 이 기체의 2.14.0만 확인함 | (실무상 불필요 — 우리는 2.14.0 고정) |
| U11 | vision→fc 링크의 성능 예산(지연·CPU) | 구현 전 | 구현 후 실측 |

**확인한 것과 안 한 것의 경계를 다시 강조한다:** §2·§3의 표에 든 값은 **전부 실기체에서 명령을
돌려 얻은 출력**이거나 **저장소 파일의 직접 인용**이다. §3.1의 mavros 소스 인용은 upstream
`mavlink/mavros` 태그 2.14.0 원문이고, 그 결론(frame 오프바이원)은 **기체 안의 두 파일을 직접
대조해 독립적으로 재확인**했다.

---

## 8. 권고안 (1개) + 사용자 결정 갈림길

### 🎯 권고: **R4 — transport-agnostic 코어 + localhost 소켓 + 컨테이너 안 얇은 shim 노드**

```
[호스트 / picam-venv / Py3.12 / ROS 없음]        [fc 컨테이너 / Humble / Py3.10]
 vision main.py                                   vision_bridge_node (fc_ros, 신규·얇음)
   └ LandingStateMachine ─┐                          │  ← 소켓 서버 (127.0.0.1:PORT)
   └ TargetEstimate ──────┤                          ├─▶ /vision/target_pose
                          │                          │     (PoseWithCovarianceStamped, BEST_EFFORT/depth=1)
                    SocketTargetSink ──── TCP ──────▶├─▶ /vision/target_status
                    (§7.2의 TargetSink 포트 구현)     │     (DiagnosticStatus, 같은 header.stamp)
                                                     └─▶ /mavros/landing_target/raw  [피벗 시에만]
                                                           (LandingTarget, frame=12 정수리터럴)
                                                                     │
                                                          OffboardNode 정밀착륙 서브상태
```

**왜 이것인가:**

1. **§2의 차단요인을 우회한다** — 호스트에 ROS2를 깔 필요도, 컨테이너를 Jazzy로 올릴 필요도,
   카메라를 컨테이너에 패스스루할 필요도 없다. **U1·U2가 아예 무관해진다.**
2. **`vision_plan.md` §7.2가 이미 지정한 구조다** — "코어=transport-agnostic 라이브러리,
   ROS 노드=얇은 래퍼". 새 아키텍처가 아니라 명세 이행이다. `TargetSink`(§7.2)와
   `CommandSource`(CC 연동용) 포트가 계획서에 이미 이름으로 존재하고 미구현 상태다.
3. **메시지 타입 A/B/C 선택이 컨테이너 내부 문제로 축소된다** — shim 노드 한 파일만 고치면
   (a)든 (b)든 (c)든 바꿀 수 있고, vision 코어는 절대 안 바뀐다. **지금 A/B/C를 확정하지 않아도
   진도가 나간다**는 게 핵심 가치다.
4. **페일세이프가 더 강해진다** — 프로세스 사망 시 소켓 EOF로 **즉시** 감지(§5.4). DDS 토픽
   침묵보다 엄밀하다. shim을 **서버**로, vision을 **클라이언트**로 두는 이유가 이것이다
   (FC 스택이 장수명, vision이 재시작 대상).
5. **디버깅이 쉽다** — 와이어 포맷을 **line-delimited JSON**으로 두면 `nc 127.0.0.1 PORT`로
   눈으로 본다. 이미 `chosen.target_estimate` dict 직렬화 코드가 있다
   (`vision/main.py:115` `_target_estimate_to_dict`) → **새 직렬화 코드가 거의 필요 없다.**
   `schema_version` 필드를 넣는 것도 §7.2가 이미 요구한 것이다.
6. **양쪽 도메인 격리를 지킨다** — vision은 `SocketTargetSink` 하나만 추가(vision 도메인),
   FC는 `vision_bridge_node` 하나만 추가(FC 도메인). 서로의 파일을 안 건드린다.
   빌드 의존도 안 생긴다.

**메시지 타입에 대한 부수 권고(shim 노드 내부):** **(a)와 (b)를 둘 다 쓴다.**
- 폐루프 유도 = **(b)** `PoseWithCovarianceStamped` + `DiagnosticStatus`. 커스텀 패키지 0개,
  `uncertainty`가 갈 공분산 자리가 있고, 안전 계약 5개 필드가 전부 KeyValue로 들어간다.
- 네이티브 precision-land 피벗 = **(a)** `/mavros/landing_target/raw`,
  `frame = 12` **정수 리터럴**(msg 상수 사용 금지), `listen_lt: true` 필요.
- **(c) 커스텀 msg는 지금 만들지 않는다.** (b)로 표현이 되고, 빌드 의존 비용이 실익보다 크다.
  나중에 필요해지면 그때 (b)→(c)는 shim 노드 안에서만 바뀐다.

**좌표계 부수 권고:** vision은 **body FLU 상대 벡터**를 낸다(§4.4). NED 변환도, 기체 자세 구독도,
시간 정렬도 하지 않는다.

### 🔀 사용자에게 물어야 할 갈림길

**D1. 런타임 배포판 — 🔴 가장 중요, 다른 모든 것이 여기 걸려 있다**

| 선택지 | 트레이드오프 |
|---|---|
| **R4 (권고)** 소켓 + 컨테이너 shim | ✅ 지금 당장 착수 가능, 도메인 격리 유지, 페일세이프 강함, 되돌리기 쉬움 · ❌ ROS 생태계 도구(rosbag/rviz)가 vision 프로세스를 직접 못 봄(shim이 재발행하므로 컨테이너 안에서는 다 보임), 소켓 재접속 로직을 우리가 짜야 함 |
| R1 호스트에 ROS2 Jazzy 설치 | ✅ vision이 진짜 ROS 노드가 됨 · ❌ **Humble↔Jazzy 교차 통신 미지원 → 사실상 불가**. 게다가 `picam-venv`가 Py3.12 system-site-packages라 Jazzy와 섞이는 리스크 |
| R2 `fc` 컨테이너를 Jazzy로 | ✅ 양쪽 한 배포판 · ❌ **SITL 검증 완료된 FC 스택 전체 재검증**. vision 도메인 권한 밖. 대회 일정상 자살행위 |
| R3 vision을 컨테이너 안으로 | ✅ 양쪽 한 배포판, 정공법 · ❌ libcamera+picamera2를 Py3.10으로 **다시 소스빌드**(호스트에서 세션 하나 걸림), 카메라 디바이스 패스스루 추가, cv2 설치, `--privileged` 검토. **비용이 가장 큼** |

**D2. `listen_lt: true`로 바꿔 네이티브 피벗 경로를 열어둘 것인가**
- 열어둔다: 정밀도 미달 시 PX4 네이티브 precision-land로 즉시 피벗 가능(계획서 §8의 명시된
  보험). 비용은 mavros 파라미터 override 한 줄 + PX4 쪽 `PLD_*` 확인(U8).
- 안 연다: 지금은 폐루프 한 경로만. 나중에 급하게 열어야 할 때 검증할 시간이 없을 수 있음.
- ⚠️ FC 도메인 결정 사항 — vision 세션이 바꿀 수 없다.

**D3. vision `state`를 FC에 거부권으로 넘길 것인가 (§6)**
- 넘긴다(권고): 커밋 게이트 로직 중복 없음, 항상 보수적인 쪽이 이김 · FC가 외부 신호에 종속됨
- 안 넘긴다: FC가 완전 자율 · vision의 `lock_confirm_frames`/모호 거절 판단을 FC가 다시 구현해야 함
  (단일 추정치만으로는 원리적으로 불가능한 부분이 있음)
- `command` 동사(`"descend"` 등)를 명령으로 쓰는 안은 **양쪽 다 권고하지 않는다.**

**D4. `not_for_closed_loop_30cm=True` 상태에서 어디까지 허용할 것인가 (§5.5)**
- 권고안(3m까지 정렬 후 AUTO.LAND 인계): 지금 당장 가치 있고, 캘리브 완료 시 자동 해금
- 더 보수적(비전 폐루프 아예 금지): 안전하지만 **실측 캘리브 전까지 통합 전체가 무의미해짐**
- 더 공격적(플래그 무시): §7.3 provenance echo를 만든 이유가 사라짐. 권고하지 않음

---

## 9. 다음 세션(실제 구현) 작업 목록

**전제: D1이 R4로 결정된 경우.** 다른 선택지면 1~2번이 통째로 바뀐다.

### vision 도메인 (이 도메인 세션이 할 일)

| # | 작업 | 산출물 | 테스트 |
|---|---|---|---|
| V1 | `vision/utils/target_sink.py` 신설 — `TargetSink` 포트(§7.2) + `SocketTargetSink`(TCP 클라이언트, 비차단 bounded queue + drop-oldest). **`utils/blackbox.py`의 `_DropOldestQueueHandler` 패턴 재사용** (`vision/CLAUDE.md`에 전례 명시) | `TargetSink` 추상 + 소켓 구현 + `NullSink` | 실제 소켓 서버 띄워 실제 바이트 수신·JSON 파싱·재접속·비차단(실측 시간)·서버 부재 시 무크래시 |
| V2 | 와이어 스키마 확정 — `schema_version`, `stamp_ns`(SensorTimestamp 기반), `clock_offset_ns`, `valid`, `reason`, `position_flu`, `orientation_xyzw`, `covariance`(6×6 또는 null), `confidence`, `target_type`, `calib_accuracy`, `not_for_closed_loop_30cm`, `calib_id`, `state`, `command`, `frame_id` | JSON 스키마 문서(이 파일에 부록으로) + 직렬화 함수 | 왕복(dict→JSON→dict) + 필수키 회귀 |
| V3 | `vision/core/frames.py`(신규, core 규칙 준수 — numpy만) — `R_frd_cam(psi_m)`/`R_flu_cam(psi_m)` + `cam_to_flu()`/`cam_to_frd()` | 순수 회전 유틸 | **§4.3의 RT-1~RT-7 전부** (`fc_bridge/tests/test_rotation.py` 스타일) |
| V4 | `LiveFrameSource`가 `SensorTimestamp`를 쓰도록 — `frame_source.py:178`의 `time.time()` 대체 + `FrameRecord`에 `sensor_ts_ns` 추가 | 노출 시각 기반 타임스탬프 | 가짜 picamera2 주입(`sys.modules` 패턴, 기존 전례)으로 metadata 경로 검증 |
| V5 | `main.py --target-sink socket://127.0.0.1:PORT` 배선 (기존 `--display stream` opt-in 패턴과 동일하게 **기본 off**) | CLI 옵션 | `main.py` end-to-end로 실제 소켓에 실제 JSON이 흐르는지 |
| V6 | U3(마운트 요각 ψ_m) 실측 — 기수 표식 1프레임 촬영 후 config에 기록 | `vision/config/` 또는 calibration yaml에 값 | — (물리 개입 필요) |
| V7 | U5(발행 주파수) 실측 — RPi에서 `main.py live` JSONL latency 통계 | `stale_timeout_s` 근거 수치 | — (RPi 실행, 카메라 배타성 주의) |

### FC 도메인 (FC 세션에 넘길 일 — vision 세션은 손대지 않는다)

| # | 작업 |
|---|---|
| F1 | `fc_ros/fc_ros/nodes/vision_bridge_node.py` 신설 — 소켓 **서버**, JSON→ROS 변환, `/vision/target_pose`(PoseWithCovarianceStamped) + `/vision/target_status`(DiagnosticStatus) 발행. **QoS는 `_MAVROS_QOS`와 같은 BEST_EFFORT 계열** |
| F2 | `OffboardNode`에 `PRECISION_LAND` 서브상태 — `HOLD`(`:1230`)와 `LANDING`(`:1300`) 사이. §5.1의 4단 사다리. **`_step_hold`의 `slew_setpoint` 패턴 재사용, 새 폴백 경로 신설 금지** |
| F3 | §5.6 파라미터 6종을 `fc_ros_params.yaml` + `phase2.launch.py`에 추가 (`docs/rpi_deploy.md` §6 절차) |
| F4 | (D2가 yes면) `listen_lt: true` override + `/mavros/landing_target/raw` 발행 경로. **`frame = 12` 정수 리터럴, msg 상수 사용 금지 — §3.1** |
| F5 | U8 — PX4 `PLD_*` 파라미터·`landing_target_estimator` 활성 여부·기체 PX4 버전 확인 |
| F6 | SITL 장애주입 — vision 프로세스 SIGKILL, 소켓 끊김, stale, `valid=false` 각각에 대해 2단/3단/4단 전이 실증 |

### 루트 문서

| # | 작업 |
|---|---|
| R-1 | 루트 `CLAUDE.md` "도메인 간 의존 관계"에 실제 이음매(소켓 + shim 노드, 빌드 의존 없음)를 확정 형태로 갱신 |
| R-2 | `docs/vision_plan.md` §7.1의 "미정" 3건(단위·부호 규약 / 프레임 매핑 / 불확실성 필드) 해소 표시 → 이 문서 참조 |

---

## 10. 부록 — 재현용 명령 모음

이 문서의 사실을 **직접 재현**하려면 (전부 읽기 전용):

```bash
# 런타임 환경 (§2)
ssh suri@100.67.27.83 'grep -E "^(NAME|VERSION)=" /etc/os-release; ls -d /opt/ros/* 2>/dev/null || echo "NO /opt/ros on host"'
ssh suri@100.67.27.83 'cat /home/suri/local-libcamera-src/picam-venv/pyvenv.cfg;
  /home/suri/local-libcamera-src/picam-venv/bin/python3 -c "import rclpy" 2>&1 | tail -1'
ssh suri@100.67.27.83 'docker inspect fc --format "NetworkMode={{.HostConfig.NetworkMode}} IpcMode={{.HostConfig.IpcMode}} Privileged={{.HostConfig.Privileged}}";
  docker inspect fc --format "Devices={{.HostConfig.Devices}}"'
ssh suri@100.67.27.83 'docker exec fc bash -lc "python3 -c \"import sys;print(sys.version)\"; python3 -c \"import cv2\" 2>&1|tail -1"'

# LandingTarget 실물 (§3.1)
ssh suri@100.67.27.83 'docker exec fc cat /opt/ros/humble/share/mavros_msgs/msg/LandingTarget.msg'
ssh suri@100.67.27.83 'docker exec fc bash -lc "sed -n \"/enum class MAV_FRAME/,/^};/p\" /opt/ros/humble/include/mavlink/v2.0/common/common.hpp"'
ssh suri@100.67.27.83 'docker exec fc bash -lc "sed -n \"211,228p\" /opt/ros/humble/share/mavros/launch/px4_config.yaml"'   # listen_lt: false
ssh suri@100.67.27.83 'docker exec fc bash -lc "strings /opt/ros/humble/lib/libmavros_extras_plugins.so | grep -E \"^~/(raw|pose_in|pose|lt_marker)$\" | sort -u"'
```

```bash
# 저장소 쪽 (§1, §4, §5)
sed -n '52,71p'    vision/core/target.py            # TargetEstimate
sed -n '20,21p'    vision/core/state_machine.py     # "문자열 힌트일 뿐"
sed -n '68,72p'    fc_ros/fc_ros/nodes/offboard_node.py    # _MAVROS_QOS = BEST_EFFORT
sed -n '432,435p'  fc_ros/fc_ros/nodes/offboard_node.py    # /mavros/local_position/pose 구독
sed -n '1104,1113p' fc_ros/fc_ros/nodes/offboard_node.py   # "새 폴백 경로를 만들지 않는다"
sed -n '16,52p'    fc_ros/fc_ros/adapters/vehicle_state_bridge.py  # ENU→[N,E,h_up] / [vN,vE,vD]
sed -n '178p'      vision/utils/frame_source.py     # ts=time.time()
grep -n 'confidence' vision/core/state.py           # 기본값 1.0
grep -rn 'confidence=' vision/modules/*.py          # aruco/distress_box에 없음
grep -rn 'TargetSink' vision/ --include=*.py        # 0건 (미구현)
```

---

## 11. 참조

- `docs/vision_plan.md` §4.2(tilt) · §4.3(30cm 폐루프) · §7.1(ports & adapters) · §7.2(변화 흡수 지도) · §7.8(시간 동기) · §8(통합) · §9(빌드순서 7번) · §10(열린 항목)
- `docs/vision_next_session_brief.md` §3 (이 환경의 함정 9가지 — 5번 `pgrep` 자기매칭, 8번 RPi `git pull` 막힘, 9번 카메라 배타성)
- `docs/rpi_deploy.md` (배포 절차 · `--symlink-install` 금지 · `PYTHONPATH` 함정)
- `docs/mc_flight_procedure.md:38` (`ros2 launch mavros px4.launch fcu_url:=/dev/ttyACM0:57600`)
- `vision/CLAUDE.md` (파일 역할표 · 상태머신/ArUco 배선 절 · import 규칙 · 테스트 규칙표)
- 루트 `CLAUDE.md` "도메인 간 의존 관계"
