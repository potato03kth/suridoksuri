---
doc_type: orchestrator_handoff
project: suridoksuri-1
track: 👁 vision-정밀착륙
scope: 2026-07-28 vision 오케스트레이션 세션 인수인계 — 이 문서 하나로 다음 세션이 완전히 이어받는다
status: ▶ 진행 중 (3개 세션 백그라운드 실행 중, §4 참조)
created: 2026-07-28
last_updated: 2026-07-28
---

# vision 오케스트레이션 인수인계

> **다음 세션 진입 방법:** 사용자가 "너는 오케스트레이터이다"로 시작하면 **이 문서 하나만** 읽고 §4(진행 중)
> → §5(다음 할 일) 순으로 이어받는다. `docs/vision_status.md`·`docs/vision_plan.md`·`docs/vision_fc_interface.md`는
> **필요한 절만** 열어라(각각 크다).
>
> **이 문서는 이전 브리프 `docs/vision_next_session_brief.md`(2026-07-25 작성)를 대체한다.** 그 문서의
> "다음 행동 후보" 2a·2c·2d는 이번 세션에서 대부분 소화됐다. 여전히 유효한 건 §3의 환경 함정
> 목록뿐이고, 그것도 이 문서 §7에 갱신본이 있다.

---

## 0-a. ▶ 다음 세션이 **가장 먼저** 할 일

**2차 웨이브 백그라운드 트랙 3개(A/B/C)가 돌고 있다**(§4-0). 1차 웨이브는 전부 종료·검증 완료다.

1. **`git fetch origin && git log --oneline -20`** 로 브랜치 tip을 확인하고 `pytest vision/tests/`를
   돌린다. **기준선은 886 passed**(2차 웨이브 착수 시점). 그보다 늘었으면 A/B/C가 올린 것이다.
2. **§4-0의 A/B/C 완료분을 §1-1대로 전건 직접 재현 검증한다.** 세션 자기보고 수용 0건이 이 세션의
   규율이고 실제로 그 과정에서 오보를 여러 건 잡았다. 특히 확인할 것:
   - **A**: 안전창 부등식이 실제로 코드에서 계산되는가 / `R` 기본값이 지어낸 값이 아닌가 /
     창이 빌 때(`R≈1.0`) 정말 `infeasible`로 떨어지는가
   - **B**: **부호**다. 알려진 방향쌍으로 왕복 검증했는가(축 이름만 맞고 부호가 뒤집혀도
     노름 검사는 통과한다) / `pos_ned`의 3번째가 `h_up`인 것을 반영했는가 /
     **절대 좌표를 기억하지 않는가**(기체 위치를 바꾸면 목표점이 따라 움직여야 한다)
   - **C**: attitude 항을 무시하도록 파괴해도 통과하면 pseudo다 / 스코어가 낮은 후보를
     **버리지 않는가**(§5 8번의 핵심 조건)
3. **§4-0의 트랙 D를 착수한다** (A 완료가 선행). 상태머신 밀어붙이기 + `vision_plan.md` §5.1/§8
   반대 기조 서술 갱신.
4. **§3-4의 FC 변동 확인 명령**을 돌린다. 바뀌었으면 §3-4의 "볼 것" 목록대로 계약 영향을 판단한다.
5. **§5-1의 U6(기체 반경 정확값)·U7(허용 드리프트)을 사용자에게 확인한다** — 둘 다 물리 측정이
   필요하고 A/D 트랙의 임계값을 직접 결정한다. **지어내지 마라.**
6. 그 다음 §8 백로그. **폐루프를 닫는 마지막 칸은 FC 도메인**(`OffboardNode` 정밀착륙 서브상태 +
   이제는 `/vision/landing_setpoint` 구독까지)이라 vision 세션이 할 수 없다 — 사용자에게 FC 트랙
   착수를 제안하는 것까지가 범위다.

---

## 0. 세 줄 요약

1. **§9 7번(offboard 정밀착륙 연결)의 vision 쪽 절반이 끝났다.** 초록구역을 검출해 상대 pose를 만들고
   JSONL 소켓으로 내보내는 것까지 **종단간 실증 완료**. 남은 건 컨테이너 shim(진행 중)과 FC 쪽 소비(FC 도메인).
2. **실행 경로 위의 결함 3건을 찾아 고쳤다** — `KalmanTracker` numpy 2.x 즉사, `TemporalFusion` 승격 불가,
   초록구역 pose 부재. **셋 다 유닛테스트로는 안 잡히고 실제로 돌려봐야 보이는 것들이었다.**
3. `pytest vision/tests/` **462 → 770 passed**, 회귀 0. FC 인터페이스 계약은 세션 내내 **무변경**.

---

## 1. 이 세션의 역할과 프로토콜 (그대로 이어갈 것)

사용자가 **"너는 오케스트레이터이다"**로 역할을 지정했다. 메모리 `feedback_orchestrator_protocol`이 정본이고,
이번 세션에서 **실제로 값을 한** 운용 규칙은 아래다.

### 1-1. 🔴 세션 자기보고를 절대 그대로 믿지 않는다 — 전건 직접 재현

이번 세션은 **8개 서브에이전트 보고를 전부 오케스트레이터가 직접 재현 검증**했다. 그 과정에서 실제로 잡아낸 것:

- 에이전트 두 개가 **같은 사고를 서로 자기가 냈다고 보고**했다 — reflog상 reset은 **1건뿐**이라 한쪽은 착각이었다.
- 한 세션이 "플래키는 9/10 실패"라고 통계만 보고했는데, 오케스트레이터가 **결정론적 재현에 성공**해
  원인을 확정했다(§6-4).
- 한 세션이 "40m 오차 +1.44%"라고 했는데, 오케스트레이터가 재현하니 **거리 라벨과 calib 해상도 조합**에
  따라 값이 달라지는 것을 발견했다(§6-5).

**구체적 방법 (그대로 쓸 것):**
```bash
# 메인 워킹트리는 FC 세션이 쓸 수 있으므로 검증은 항상 격리 worktree 에서
git worktree add --detach /tmp/.../verify origin/dev--vision-computing-module
cd /tmp/.../verify && /home/suri/suridoksuri/.venv/bin/python -m pytest vision/tests/ -q
# 파괴검증 재현: 코드를 직접 깨고 -> red 확인 -> 원복 -> green 확인
# 반드시 종료코드를 변수로 받는다.  `| tail` 파이프는 종료코드를 삼킨다.
git worktree remove --force /tmp/.../verify   # 끝나면 반드시 정리
```

### 1-2. 서브에이전트 프롬프트에 반드시 넣을 것 (이번 세션에서 확립)

1. **git 절차 전문** — §7-1. 특히 `checkout -B` 금지. 이걸 안 넣어서 사고가 났다.
2. **파괴검증 의무 + 테스트↔파괴 대응표** — "테스트를 만들었다"는 주장은 그 자체로 무가치하다.
   대응표를 요구하면 세션이 **스스로 pseudo 테스트를 찾아낸다**(실제로 3개 세션이 자기 pseudo를 잡았다).
3. **`__pycache__` 함정** — 파괴/원복 편집이 길이가 같고 같은 초에 일어나면 `.pyc` 유효성 판정(mtime 초+크기)이
   통과해 **깨진 바이트코드가 재사용**된다. 캐시 삭제 + `PYTHONDONTWRITEBYTECODE=1`.
4. **종단간 실증을 합격 기준으로** — 유닛테스트만으로는 불합격. 이번 세션의 큰 발견 3건이 전부 종단간에서만 나왔다.
5. **"못 한 것"을 별도 항목으로 요구** — 정직 보고를 구조적으로 유도한다. 잘 작동했다.
6. **알려진 플래키를 미리 알려줄 것** — 안 알려주면 세션이 자기 회귀로 오판한다.

### 1-3. 실행 모드

- **전부 백그라운드**(`run_in_background: true`). 포그라운드는 결과 받고 같은 턴에 바로 이어칠 때만.
- **전부 `isolation: "worktree"`** — 메인 워킹트리는 FC 도메인 세션과 공유된다. 이번 세션에서 실제로
  대화 중에 FC가 커밋을 쌓았다.
- 모델·effort는 **지정하지 않는 게 기본**(사용자 Max 구독).
- 병렬 3개까지는 무리 없었다. 단 **파일이 겹치면 안 된다** — `vision/CLAUDE.md` 테스트 규칙표처럼 여러
  세션이 만지는 파일은 "네가 건드린 행만" 식으로 범위를 쪼개 지시하라.

---

## 2. 이 세션이 만든 것 (전부 검증 완료, 재조사 금지)

커밋 순서대로. **✅ = 오케스트레이터가 직접 재현 확인.**

| 커밋 | 내용 | 검증 |
|---|---|---|
| `1678539` | 색 캘리브레이터 마진 `hue=6`/`sat=0`/`val=0` 확정 | ✅ 부호 반전 red 2건 / **기본값 0 되돌림 red 4건** / 원복 green |
| `657721c` | vision↔fc 인터페이스 **정찰 문서**(`docs/vision_fc_interface.md`, 852줄) | ✅ 실기체에서 런타임 사실·off-by-one 전건 재현 |
| `72f6721` | `fusion`/`tracker` 회귀망 + **`KalmanTracker` numpy 2.x 크래시 수정** | ✅ 수정 되돌리니 첫 검출 프레임에서 즉사 재현 |
| `8ce333e` | **`TemporalFusion` 구조적 결함 수정** | ✅ min_frames=5에 정확히 5프레임째 확정 / 깜빡임 20프레임 미확정 |
| `ce43654` | `registry`/`image_loader`/`video_reader` 커버리지 74건 | ✅ 스위트 통과 |
| `ac66a12` | 트랙 보드 기록 | — |
| `e17c573` | **정밀착륙 인터페이스 Phase 1** — wire/frames/target_sink | ✅ 순수 stdlib 소비자로 종단간 수신 |
| `b45fdc4` | `main.py --target-sink` 배선 + blackbox 플래키 결정론화 | ✅ 종단간 + 부하 아래 5/5 |
| `e1f8471` | **초록구역 상대 pose 산출**(`modules/distress_mat.py`) | ✅ 10m→9.9755m / 20m→19.885m / D2 파괴 red 3건 |
| `6bf751f` | 트랙 보드 기록 | — |

### 2-1. 핵심 산출물 지도

```
vision/core/wire.py          JSONL 와이어 포맷 + 페일세이프 계약 (SCHEMA_VERSION=1)
vision/core/frames.py        카메라 광학 → body FLU → FRD (마운트 요각 ψ_m, MEASURED=False)
vision/utils/target_sink.py  localhost TCP 서버 127.0.0.1:8091 (비차단·drop-oldest·재접속·SIGTERM)
vision/modules/distress_mat.py  초록구역 4코너 + 실측 3.0m solvePnP → TargetEstimate
vision/main.py --target-sink    세 실행경로 전부에 배선 (기본 꺼짐)
docs/vision_fc_interface.md     정찰 사실확정 문서 (852줄, 필요한 절만 읽을 것)
```

---

## 3. 🔒 확정 사실 — **재조사 금지**

전부 오케스트레이터가 실기체/실행으로 직접 확인했다.

### 3-1. 런타임 환경 (이게 아키텍처를 결정했다)

| 항목 | 값 |
|---|---|
| RPi 호스트 | Ubuntu **24.04 noble** / Py **3.12.3** / **ROS2 없음**(`/opt/ros/*` 부재) |
| `picam-venv` | `/home/suri/local-libcamera-src/picam-venv` — **rclpy 없음** |
| `fc` 컨테이너 | `ros:humble` / Py **3.10.12** / rclpy 있음(`source /opt/ros/humble/setup.bash` 필요) / **cv2·picamera2 없음** / `NetworkMode=host` / `IpcMode=private` |
| 함의 | noble에 깔리는 건 **Jazzy**이고 **Humble↔Jazzy 교차통신 미지원** → 두 프로세스가 같은 ROS2 그래프에 못 올라간다 |

**🔵 사용자 결정: localhost 소켓 + 컨테이너 shim.** 호스트 ROS2·컨테이너 교체·카메라 패스스루 전부 불필요.
`docs/vision_plan.md` §7.2("코어=transport-agnostic, ROS 노드=얇은 래퍼")의 이행이다.

⚠️ `docker exec fc bash -lc "python3 -c 'import rclpy'"`는 **ROS setup을 source하지 않아 실패한다.**
`bash -c "source /opt/ros/humble/setup.bash && ..."`를 써야 한다. (오케스트레이터가 한 번 물렸다.)

### 3-2. 🔴 `mavros_msgs/LandingTarget` frame 상수 off-by-one

| 출처 | LOCAL_NED | MISSION | BODY_FRD |
|---|---|---|---|
| `LandingTarget.msg` 상수 | **2** | 3 | — |
| 실제 `MAV_FRAME` enum (`common.hpp:154-167`) | **1** | 2 | **12** |

**상수 이름을 믿고 짜면 MAVLink가 `MISSION`으로 읽어 조용히 `position_valid=false`로 떨어진다.**
정수 리터럴 `1`/`12`를 써라. 또 `px4_config.yaml:214`가 **`listen_lt: false`**라 `~/raw` 구독이 아예 생성되지 않는다.

### 3-3. 인터페이스 계약

- 레코드 2종: `type="target"`(필수키 = `wire.REQUIRED_TARGET_KEYS`) / `type="state_hint"`. `schema_version != 1`이면 거절.
- **유도 입력은 `position_flu`**(body FLU).
  > 🔴 **정정 (2026-07-28, Phase 2에서 실기체 소스 대조로 확인).** 이 문서 초판은 *"`position_frd`는
  > `LandingTarget` 피벗용"* 이라고 썼는데 **틀렸다.** mavros `landtarget_cb`가
  > `case MAV_FRAME::BODY_FRD: position = ftf::transform_frame_baselink_aircraft(...)` 로
  > **플러그인이 FLU→FRD 변환을 직접 한다.** `position_frd`를 넣으면 변환이 **두 번** 걸려 y·z 부호가 뒤집힌다.
  > → **pose 토픽이든 `LandingTarget`이든 전부 `position_flu`를 넣는다.** `position_frd`는 진단·기록용이다.
  >
  > 독립 교차검증: 이 저장소가 이미 같은 규약을 기록하고 있다 — `sitl_vtol_remediation_plan.md:101`
  > *"`coordinate_frame = FRAME_LOCAL_NED`(=1), **값은 ENU로 채운다**(MAVROS가 NED 변환)"*,
  > `offboard_node.py:461`도 ENU PoseStamped를 발행한다. **MAVROS는 ROS쪽 FLU/ENU를 받아
  > 스스로 FRD/NED로 바꾼다** — 이게 mavros 전역 관례다.
- `orientation`은 **카메라 광학 프레임 그대로**(`orientation_frame:"cam_optical"`) — body 자세로 오인 금지.
  `fc_bridge`의 쿼터니언은 `(w,x,y,z)`, `TargetEstimate`는 `(x,y,z,w)`로 **순서가 다르다.**
- stale 판정은 `stamp_monotonic_ns`. **`valid=false` = "안 보임", 침묵/EOF = "죽음"** — 이 구분이 페일세이프의 핵심.
- `command_hint`는 **advisory, 소비 금지**. 거부권은 `state`만.
- `confidence`는 **ArUco 경로에서 항상 정확히 1.0**(`Detection` 기본값을 `aruco.py`가 안 덮음) → `min_confidence` 게이트는 **현재 no-op**.
- `not_for_closed_loop_30cm`은 **현재 100% True**. "True면 폐루프 금지"로 계약하면 **즉사한다** →
  "**최종 커밋** 금지"(`closed_loop_floor_agl_m`=3.0m까지 정렬 후 AUTO.LAND 인계)로 해석돼 있다.
- 초록구역 레코드엔 `plane_reference:"mat_top_surface"` / `platform_height_m:0.105` —
  **라이다 AGL(지면 기준)과 정확히 0.105m 어긋난다.**
- 저장소 내부 `pos_ned`는 `[N,E,h_up]`(위 양수)인데 `vel_ned`는 `[vN,vE,vD]`(아래 양수) — **같은 접미사가 반대 부호 규약.**

### 3-4. FC 도메인 경계 — 세션 내내 **무변경**

기준선 `893a5eb`(2026-07-28 01:13) 이후:
- `fc_ros/`·`fc_bridge/` 커밋 **0건**
- 인터페이스 파일 3종 md5 전부 일치:
  ```
  922388ff55cda07de9fe9922d9476f89  fc_ros/fc_ros/nodes/offboard_node.py
  a2c658c8396762cac4b84aafdce215ae  fc_bridge/execution/state_logic.py
  3ec8282aef7a77f7b58ae01e133dda0b  fc_bridge/utils/rotation.py
  ```
- 같은 기간 FC 커밋 6건은 전부 PX4 툴체인·패치빌드·파라미터 백업

**FC 변동 재확인 명령 (다음 세션이 주기적으로 돌릴 것):**
```bash
git fetch origin -q
git log --oneline 893a5eb..origin/dev--vision-computing-module -- fc_ros/ fc_bridge/
md5sum fc_ros/fc_ros/nodes/offboard_node.py fc_bridge/execution/state_logic.py fc_bridge/utils/rotation.py
```
**바뀌었다면 볼 것:** `offboard_node.py`의 `HOLD`/`LANDING`(정밀착륙이 붙는 자리) · `_step_hold()`의
`self._cruise_alt`(R5가 이 줄을 건드린다) · `_RANGE_GUARDED_STATES` · `_publish_pos_setpoint(pos_ned, yaw)`
시그니처 · **A안(`setpoint_raw/local`) 기각이 뒤집혔는지**(뒤집히면 MC 구간 setpoint 채널이 통째로 바뀐다).

---

### 3-5. 🔴 발행 주파수는 10Hz가 아니라 **4.4Hz** (2026-07-28 실기체 실측)

저장소가 여러 곳에서 "10Hz"를 가정해 왔는데 **실측으로 반증됐다.** 실카메라 `main.py live`(4608×2592)
44.2초: **간격 median 0.2207s(4.53Hz) / p95 0.310s**, 컨테이너 `ros2 topic hz` **4.35Hz**.

- `stale_warn_s` 를 0.5 → **0.75** 로 올렸다(p95의 2.4배). 커밋 `ec59efc`.
- `target_sink` 큐 8은 "0.8초"가 아니라 실제 **1.8초**치다(안전 방향이라 미변경, 기록만).
- 병목은 **4608×2592 전해상도**다. 낮추면 빨라지지만 `nominal.yaml` 캘리브와 어긋나 **거리가 통째로 틀린다**(§6-5).
  → 열린 결정.
- **타임아웃·stale 판정을 새로 만들 때 10Hz를 가정하지 마라.**

### 3-6. CLOCK_MONOTONIC — 호스트↔컨테이너 **완전 동일** (미확인 해소)

Phase 1이 최대 리스크로 남긴 항목. 오케스트레이터가 직접 확인:
```
host      /proc/self/ns/time -> time:[4026531834]
container /proc/self/ns/time -> time:[4026531834]      # 동일 inode
timens_offsets: 양쪽 다 monotonic 0 0 / boottime 0 0
container monotonic 147864.14  ==  host /proc/uptime 147864.14
```
**`clock_offset_ns` 환산 불필요.** 종단간 실측 `age_s = 0.00067`(0.67ms).

---

## 4. ▶ 지금 돌고 있는 것

### 🔄 4-0. 2차 웨이브 — 2026-07-28 설계 논의 후속 (백그라운드 3개, **진행 중**)

§5 6~10번 결정을 구현하는 트랙이다. **다음 세션은 이 3건의 완료 보고를 §1-1대로 직접 재현 검증하는
것부터 시작한다 — 세션 자기보고를 그대로 믿지 마라.**

| # | 트랙 | 소유 파일 (겹침 0으로 쪼갰다) | 근거 |
|---|---|---|---|
| **A** | **착륙점 기하 재설계** — `interior_margin_ratio` 비율값을 폐기하고 기체반경 `R` 기반 안전창에서 도출. 창이 비면 지어내지 말고 `meta`에 사유 남기고 상위가 `valid=false`로 발행 | `modules/distress_box.py` · `presets/distress_*.yaml` · `tests/test_distress_box.py` | §5-2. **C 트랙의 선행**(밀어붙이기 임계가 여기서 나온다) |
| **B** | **shim setpoint 변환** — `/mavros/local_position/pose` 구독 → body FLU + 자세 + 현재위치 → `/vision/landing_setpoint`. **절대 좌표 기억 금지**(매 레코드 재계산) | `ros/shim_core.py` · `ros/shim_node.py` · `tests/test_shim_core.py` | §5 7번 |
| **C** | **사전정보 스코어링** — 호모그래피 예상 형상으로 후보 랭킹. **거절 금지**, 입력 없으면 자동 비활성 | `modules/<신규>.py` · `registry.py` · `main.py` · `replay.py` · 해당 테스트 | §5 8번 |

**아직 안 띄운 트랙 D (A 완료 후 착수):** 상태머신 밀어붙이기 —
`core/state_machine.py`의 TERMINAL `ABORT_ASCEND` 2경로를 데드레코닝 지속으로 전환 +
밀어붙이기 임계를 `max_drift_estimate_m`과 **분리된 별도 파라미터**로 신설 + `vision_plan.md`
§5.1/§8의 반대 기조 서술 갱신. A가 안전창을 확정해야 임계를 도출할 수 있어 순차로 둔다.

⚠️ **세 세션이 `vision/CLAUDE.md`를 동시에 만진다.** 각자 "네가 건드린 절만" 지시를 받았고
새 절은 파일 맨 끝 근처에 추가하게 했다. 완료 보고를 받으면 `git show --stat`으로 각 커밋이
자기 범위만 담았는지 확인하라.

---

### 4-1~4-3. 1차 웨이브 (2026-07-28 오전 착수) — **전부 종료·검증 완료**

| # | 트랙 | 내용 | 특기사항 |
|---|---|---|---|
| A | **Phase 2 — 컨테이너 shim** | ✅ **완료·검증됨** (`036b276`→`ec59efc`→`b14f42a`) | 아래 4-1 |
| B | **bind 하드페일 + 화면 경고 + replay 배선** | ✅ **완료·검증됨** (`3cda638`) | 아래 4-2 |
| C | **잔여 갭 5건** | ✅ **완료·검증됨** (`9c82337` `1d8d891` `30eb0eb` `f6dd2ae` `ada8ed6`) | 아래 4-3 |

**세 트랙 전부 종료. `pytest vision/tests/` 최종 886 passed, 0 failed. 다음 세션은 §5(다음 할 일)부터 시작한다.**

### 4-3. 잔여 갭 결과 (오케스트레이터 재현 확인)

| 항목 | 결과 |
|---|---|
| **착륙점 등거리 축퇴** (`9c82337`) | 1px 흔들림에서 **2.970m 점프 → 0.000m**. `tie_tolerance_ratio=0.02`(동률 허용오차) + `corner_hysteresis`(직전 선택 유지) = 폭 2·tol **슈미트 트리거**. 편심 박스 동작·`interior_margin_ratio=0.3`·"박스 옆" 규약 **무변경**. ✅ D1 재현(`tie_tolerance_ratio`→0.0 시 5건 red, 원복 886) |
| **`drift_estimate` tan(HFOV/2)** (`1d8d891`) | 계수를 화각(도)이 아니라 **인트린식에서 유도**(`(width/2)/fx`) — `hfov_assumption`이 수평인지 대각인지 미해결인 문제를 우회한다. 🔴 **`max_drift_estimate_m` 1.0→0.75 동반 조정** — 안 고쳤으면 게이트가 **1.303배 헐거워진다**(보수 배율만큼 정확해진 대가) |
| **골든 리프 재생성** (`30eb0eb`) | 생성기에 블록 추가 → **골든 20파일 md5 전량 무변화**, `git status` 변경 0건. 조용한 드리프트 방지 전수 무결성 테스트 추가 |
| **`LiveFrameSource` AF** (`f6dd2ae`) | `--af-mode` CLI. AF 실패는 캡처를 죽이지 않고 `af_error`에만 남긴다. AF 순수로직을 `utils/frame_source.py`로 단일 출처화(`h264_stream.py`가 import — import 규칙상 방향을 뒤집었다). 🔴 **실기체 미검증** |
| **`geo_project.py` 폐기** (`ada8ed6`) | AST 전수 감사 **참조 0건** 확인 후 삭제 + 부활 방지 묘비 테스트. ✅ 파일 부재·참조 0 확인 |

**🔴 오케스트레이터가 찾은 잔여 테스트 갭:** `corner_hysteresis` **기본값**을 `True→False`로 뒤집어도
**886 passed로 아무 테스트도 안 잡는다** — 테스트가 파라미터를 전부 명시로 넘겨 기본값을 한 번도 안 밟는다.
메커니즘은 D1~D5가 지키지만 **기본값 자체는 무방비**다. (Phase 2에서 세션이 스스로 찾은 D6와 **완전히 같은 계열**
— 이 저장소에서 두 번째다. **파괴검증 설계 시 "기본값을 밟는 테스트가 있는가"를 항상 따로 물어라.**)
비행 리스크는 낮다(기본값이 안전한 쪽) — 다음 세션이 여유 있을 때 닫으면 된다.

**🔴 FC 도메인에 남은 stale 참조 (vision 세션이 못 닫는다):** `geo_project` 삭제 후에도
`docs/pixhawk6c_rpi4_integration_guide.md`(`pixel_to_gps_with_attitude()` 확장을 "필수"로 적고 P2/P3 작업목록·블록도에 포함)
와 `docs/flight_plan.md:241`이 남아 있다. 코드 재발은 묘비 테스트가 막지만 **문서는 FC 세션이 닫아야 한다.**

### 4-1. Phase 2 결과 (오케스트레이터 재현 확인)

| 토픽 | 타입 |
|---|---|
| `/vision/target_pose` | `geometry_msgs/PoseWithCovarianceStamped` |
| `/vision/target_status` | **`diagnostic_msgs/DiagnosticArray`** — `DiagnosticStatus`엔 header가 없어 stamp 동기가 불가능하다. 배열에 `vision/target`+`vision/state`를 같은 stamp로 싣는다 |
| `/mavros/landing_target/raw` | `mavros_msgs/LandingTarget` — **기본 꺼짐**(`listen_lt: false`라 구독자 없음) |

배치는 `vision/ros/`(`shim_core`=stdlib 순수로직 / `shim_node`=rclpy 어댑터). **`fc_ros/` 무수정.**

- **페일세이프 3분법 실증:** 안 보임=WARN / 생산자 사망=**pose 침묵 + status ERROR 계속**(실측 1.0Hz) / shim 사망=둘 다 침묵.
- **실기체에서 잡은 버그 1건:** `rclpy.init()`이 기본 `SignalHandlerOptions.ALL`로 **SIGTERM 핸들러를 자기 것으로 덮어써서** 종료 시 `RCLError`. 핸들러 등록을 `init()` **뒤로** 옮기고 AST 순서 회귀로 고정. ⚠️ **`main.py`의 D-A5와 완전히 같은 계열의 함정**이다 — 이 저장소에서 두 번째다.
- **첫 파괴검증에서 2건이 green으로 통과했다** — 진짜 테스트 갭이었고 세션이 스스로 찾아 닫았다.

**⚠️ 미검증:** `LandingTarget` 경로는 `listen_lt: false`라 구독자가 없어 **실제 MAVLink 송신을 확인 못 했다**(단위테스트 + mavros 원문 대조까지만). `size`는 물리 크기가 와이어에 없어 0(지어내지 않음).

### 4-2. bind 하드페일 결과 (오케스트레이터 재현 확인)

- 점유 포트 + `--target-sink` → **exit 3** + 포트 번호·진단 명령(`ss -ltnp | grep <port>`) 포함 stderr. `replay.py`도 동일.
- 기본 실행(`--target-sink` 미지정) → **exit 0, 소켓 미바인드** 무변경.
- **오버레이 육안 확인:** 소비자 0명 → 빨간 대형 `CONSUMERS 0 - GUIDANCE GOES NOWHERE`, 접속 시 초록 `CONSUMERS 1` + `seq/dropped`, **ArUco 검출 박스·신뢰도 라벨 그대로 살아있음.** 증거 파일 `vision/results/sink_overlay_demo/`.
- **`replay.py` 종단간 `TERMINAL` 도달** — `CENTER_DESCEND → LOCK → PRECISION_SERVO → TERMINAL`, 160건 드롭 0. `main.py`는 AGL을 못 받아 여기까지 못 가므로 **`state_hint`를 회귀로 잡는 유일한 경로**다.
- ⚠️ **cv2 Hershey 폰트는 한글을 못 그린다** — 오버레이 문자열은 **ASCII 전용**(소스 레벨 회귀테스트로 고정).

**⚠️ 세 세션이 같은 브랜치에 동시에 push한다.** 완료 보고를 받으면 `git log --oneline` 으로 순서를 확인하고,
각 커밋이 `vision/` 파일만 담았는지 `git show --stat`으로 확인하라.

---

## 5. 🔵 사용자 결정 (2026-07-28 확정 — 재질의 금지)

1. **런타임 연결 = localhost 소켓 + 컨테이너 shim.** (§3-1)
2. **소켓 bind 실패 = 하드 페일(죽인다).** 사용자 원문: *"'유도 좌표가 아무 데도 안 나가는 상황'에 화면은
   뜰 수 있는가? 만약 그렇다면, 지금은 디버깅이 활발한 상태이니까, 걍 안되면 죽여버릴 수 있도록 하여라.
   화면에도 보이도록."*
   → 오케스트레이터 확인: `--display`와 `--target-sink`는 **완전 독립**이라 유도 좌표가 허공으로 가는 채
   화면만 멀쩡히 뜨는 상황이 실제로 가능하다. **bind 실패는 죽이고, 소비자 0명은 화면 오버레이로 경고**
   (시작 직후엔 소비자가 아직 안 붙었을 수 있어 죽이면 안 된다).
3. **고아 FC 커밋 `7929e4c` 처리 = FC 세션에 맡긴다** → **해소 완료**(§6-1).
4. **Phase 2 착수 승인** — FC의 PX4 재플래시 준비가 끝났다.
5. **서버/클라이언트 방향 = vision이 서버**(오케스트레이터 판단, 사용자 미이의). 정찰문서 §8 권고 4번은
   반대(shim=서버)였으나 구현 전 작성분이고, EOF로 죽음을 감지하는 성질은 방향 무관하게 유지됨을
   테스트로 확인했다. 되돌리기 쉬운 내부 세부다.

---

### 🔵 2026-07-28 후속 설계 논의에서 확정된 것 (사용자 승인 완료 — 재질의 금지)

6. **발행 주파수 = 4.4Hz로 만족한다. 해상도 조정은 차후.**
   사용자 원문: *"해상도 문제는 차후 해결할 것이다. 다만, 현재 2차 직전인 지금은 진행하지도 않은
   캘리를 틀어버리는 것에 부담이 있기 때문에, 일단 4.4Hz로 만족한다."*
   → §3-5의 열린 결정이 **닫혔다.** 4608×2592 전해상도 유지. 여전히 **타임아웃·stale 판정에
   10Hz를 가정하지 마라.**

7. 🔴 **FC 커뮤니케이션 = "변환은 shim에서, 절대 좌표는 기억하지 않는다".**
   사용자 원안은 *"attitude를 vision이 받아 setpoint까지 연산"*이었고, 오케스트레이터가
   **연산 지점만 소켓 뒤(컨테이너 ROS 그래프 안)로 옮기는 절충안**을 제시해 사용자가 수용했다.

   ```
   vision/main.py ──position_flu(무변경)──▶ shim_node.py + /mavros/local_position/pose 구독
                                                  ▼ 여기서 변환
                                          /vision/landing_setpoint (목표 pos_ned)
                                                  ▼
                                          offboard_node._publish_pos_setpoint()  [FC 소관, 미구현]
   ```

   **원안을 기각한 이유 3가지(재논의 금지):**
   - attitude가 소켓 왕복 + 4.4Hz만큼 지연 → 10°/s 기동에 250ms 지연이면 2.5° 오차 →
     10m AGL에서 **44cm**. 30cm 요구를 지연 하나로 날린다. 같은 그래프면 지연 0.
   - 와이어가 절대좌표가 되면 `LandingTarget` 네이티브 precision-land **피벗 경로가 막힌다**
     (§8이 "정밀 미달 시 네이티브 피벗"을 명시적 안전장치로 설계).
   - attitude 역방향 채널은 vision→FC 의존을 만든다. 지금은 FC가 죽어도 vision이 계속 뱉는다.

   🔴 **"절대 setpoint를 만들되 절대 좌표를 기억하지 않는다"** — 목표점을 한 번 잡아 고정하면
   EKF 드리프트가 그대로 오차로 남는다. 매 레코드마다 `목표 = 그 순간 최신 pose + 그 순간 상대오차`
   로 재계산해야 드리프트가 상쇄된다. **이게 이 설계의 핵심이고 회귀테스트 대상이다.**

8. **사전정보 기반 후보 스코어링 도입 — 단 "거절이 아니라 스코어링".**
   사용자 제안: *"이미 객체가 지상에 놓여있는 각도와 모양과 크기를 안다"*는 점 + 기체 위치·attitude로
   화면상 예상 형상을 예측해 검출 정확도를 높인다. (사용자는 "쌍곡함수"라 했으나 정확히는
   **호모그래피 = 평면 사영변환** `H = K·[r₁ r₂ t]`.)

   **이미 축소판이 있다** — `distress_coarse.yaml`의 `min_area`/`max_area`가 정확히 이 논리의
   나디르·광대역 버전이다. 이번 작업은 그걸 attitude로 일반화 + 실시간 AGL로 대역 축소.

   🔴 **하드 게이트 금지**(오케스트레이터 조건, 사용자 수용): AGL/attitude 오차가 게이트를 통째로
   오프셋시켜 **진짜 타겟을 거절**할 수 있고, 그건 9번 기조와 정면 충돌한다. ψ_m 미측정 + 골든셋
   전량 합성이라 튜닝 근거도 없다. → **후보 랭킹 점수로만**, 입력 없으면 **자동 비활성**.
   부수 이득: 상태머신의 `n_candidates>1 → HOLD` 거절 빈도가 줄어 9번과 시너지.

9. 🔴 **비행 기조 전환 — "실패해도 재시도" → "마지막 값으로 밀어붙이기".**
   사용자 원문: *"좋은 비행 시퀀스는 실패하지 않는 시퀀스라고 생각한다. 우리의 현재 시퀀스는
   '실패해도 다시 시도한다'는 기조가 묻어있다(ex. 정밀착륙 중 목표 놓칠 시 재상승). 이건 좋지
   않다고 본다. ('최종적으로 실패하더라도 그냥 마지막 값을 기반으로 밀어붙이기') 기조는 어떠한가?
   나는 이것이 더 좋다고 본다. 그래야 표면적으로 성공한 것처럼 보일 것이다."*

   대회 판정이 **"매끄럽게 보이면"(정성)**이고 `ABORT_ASCEND`는 명백한 감점 신호이므로 타당하다.
   **오케스트레이터가 붙인 조건 2가지(사용자 수용):**
   - **TERMINAL(AGL≤3m)에서만 밀어붙인다.** 그 위 상태의 폴백은 `HOLD`(제자리)이지 재상승이
     아니라 이미 기조와 충돌하지 않고, 눈에도 "신중하게 보는 것"으로 보인다.
     (참고: 현재 코드에서 `ABORT_ASCEND`가 발동하는 곳은 `state_machine.py:258-261`
     **TERMINAL 두 조건뿐**이다 — 나머지는 전부 `HOLD`다.)
   - **밀어붙이기 임계는 `max_drift_estimate_m`(안전 폴백 게이트)과 분리된 별도 값**이어야 하고,
     "매트 안에 확실히 들어가는가"에서 도출해야 한다. **무제한 밀어붙이기는 안 된다** —
     초록구역이 0.105m 라이즈드 구조물이라 가장자리에 걸치면 기울어져 넘어진다(진짜 사고).

   ⚠️ **문서도 함께 고쳐야 한다.** `vision_plan.md` §5.1이 *"안 보이는데 계속 내려간다를 금지하는
   게 핵심"*, §8이 *"추측 후 커밋 금지"*라고 **명시적으로 반대 기조를 박아 놨다.** 코드만 바꾸면
   다음 세션이 문서를 보고 되돌린다.

10. **기체 최외곽(다리/프롭 끝) 반경 = 0.5m 초과** (사용자 확인, **정확한 값은 미상**).

### 🔴 5-2. 10번이 드러낸 구조적 결함 — 착륙점이 정상 착륙조차 매트를 넘는다

10번을 현재 착륙점 규칙에 대입하니 **밀어붙이기 이전의 문제**가 나왔다:

```
매트 반변                        1.50 m
현재 착륙점 (중심에서)             1.05 m   ← interior_margin_ratio=0.3
착륙점 → 매트 가장자리 여유          0.45 m
기체 최외곽 반경                 > 0.50 m
──────────────────────────────────────────
유도 오차가 0이어도 기체가 매트 밖으로 0.05m 이상 삐져나온다.
```

`interior_margin_ratio=0.3`은 "대회측 미회신 잠정값"으로 들어간 숫자이고 **기체 크기를 한 번도
고려한 적이 없다.** 안전창(착륙점 중심거리 `d`, 기체반경 `R`, 흰 박스 반변 `b`≈0.10m,
허용 드리프트 `δ`):

```
하한(박스 회피):   √2·d ≥ b·√2 + R        상한(매트 이탈):  d ≤ 1.50 − R − δ
R=0.5 → 창 [0.45, 1.00]   현재 d=1.05 는 창 밖
R=0.7 → 창 [0.60, 0.80]   드리프트 여유 0.20m 뿐
R=1.0 → 창이 빈다 — 박스를 피하면서 매트 안에 서는 것이 물리적으로 불가능
```

착륙점은 bbox 모서리 방향(대각선)이라 좌표가 `(d, d)` 꼴이다 — **박스까지는 `√2·d`, 매트
가장자리까지는 축 방향으로** 재야 한다(대각선으로 재면 틀린다).

→ **R을 파라미터로 두고 안전창이 비면 명시적으로 알리도록** 재설계 중(아래 §4-A 트랙).
`R`은 지어내지 말 것 — `core/frames.py`의 `MOUNT_YAW_PSI_M_MEASURED=False` 패턴을 따라
`측정됨=False` 플래그를 meta까지 전파한다.

### 5-1. 아직 열린 결정

| # | 항목 | 상태 |
|---|---|---|
| D2 | `listen_lt: true`로 네이티브 precision-land 피벗 경로를 열 것인가 | **FC 도메인 결정.** 지금은 `false` |
| D3 | vision `state`를 FC에 **거부권**으로 넘길 것인가 | 포맷은 양쪽 다 지원. `command_hint`를 명령으로 쓰는 건 양쪽 다 비권고 |
| — | 캘리브레이션 해상도 불일치 자동 보정 | 현재 **보정 안 함**(의도적). 다운스케일 운용 도입 시 결정 필요 (§6-5) |
| — | 엣지 기반 프리셋(`video.yaml`/`single_frame.yaml`/`default.yaml`) 존치·수정 | **대회 경로는 영향 없음** → 급하지 않음 (§6-3) |
| — | ~~해상도 vs 발행 주파수~~ | ✅ **닫힘** — 4.4Hz 유지 확정(§5 6번) |
| — | ~~`interior_margin_ratio=0.3` 잠정값 유지~~ | 🔴 **뒤집힘** — 기체 반경 0.5m+ 에서 구조적으로 불안전함이 드러났다(§5-2). "대회측 미회신"은 더 이상 유지 사유가 못 된다 — **물리적으로 틀린 값**이다 |
| 🔴 **U6** | **기체 최외곽 반경 `R`의 정확한 값** | 사용자 확인은 **"0.5m 초과"**뿐. `R`이 안전창의 폭을 직접 결정하고 `R≈1.0m`이면 **창이 아예 빈다**(박스를 피하면서 매트 안에 서는 것이 불가능). 물리 측정 필요 — **ψ_m과 같은 성격의 미측정 물리량** |
| 🔴 **U7** | **밀어붙이기 허용 드리프트 `δ`** | §5-2 안전창의 상한을 결정. `R` 확정 후 도출 가능. 지금은 `max_drift_estimate_m=0.75`를 그대로 쓰면 **안 된다**(여유가 0.45m뿐) |

---

## 6. 이번 세션에서 밝혀낸 것 (다음 세션이 알아야 할 사실)

### 6-1. 🔧 프로세스 사고와 해소 (완전 종료)

오케스트레이터가 worktree 세션에게 `git checkout -B dev--vision-computing-module origin/...`을 시켰다.
이 ref는 **메인 워킹트리와 공유**되고 **FC 세션이 실시간 커밋 중**이라, `-B`가 ref를 되돌리며 FC의 미push
커밋 `7929e4c`를 고아로 만들었다.

**해소 상태 (전부 확인됨):**
- 내용 손실 **0** — 인덱스·디스크·rescue 브랜치 3중 보존이었고, 이후 FC 세션이 재커밋해
  **브랜치 tip이 `7929e4c`의 상위집합**이 됐다(SITL 로그 4런 동일, 문서는 +313줄 확장).
- `rescue/fc-7929e4c-orphaned-by-vision-worktree` **삭제 완료**(로컬+origin).
- FC 세션이 남긴 stash(= vision 작업 852줄+ 되돌리는 내용)를 **태그 `archive/fc-session-stash-2026-07-28`으로
  영구 보존한 뒤 드롭**했다. **절대 apply하지 마라** — 적용하면 이번 세션 작업이 날아간다.
- 이번 세션 에이전트 worktree·브랜치 전부 정리 완료. 워킹트리 clean, origin 동기, 미push 0건.

**재발 방지는 메모리 `feedback_worktree_base_branch`에 기록됨.** §7-1이 그 요약이다.

### 6-2. 실행 경로 위 결함 3건 (전부 수정됨)

| 결함 | 성격 | 어떻게 발견됐나 |
|---|---|---|
| `KalmanTracker` numpy 2.x 즉사 | `int(pred[0])`이 4x1 배열 원소를 스칼라 변환 → `TypeError`. **첫 검출 프레임에서 죽어 모듈 100% 불능** | 테스트를 쓰려다 대상이 안 돌아가서 |
| `TemporalFusion` 승격 불가 | `_decay()`가 방금 매칭된 후보까지 깎아 **프레임당 검출 1개면 영원히 미확정.** `main.py:184` 폴백 때문에 크래시는 안 나고 **흔들림 억제만 조용히 무효** | 같은 세션 |
| 초록구역 pose 부재 | pose 산출이 **ArUco 전용**이라 초록구역은 좌표가 항상 `null` | **오케스트레이터의 종단간 실행에서만** 드러남 |

⚠️ **`KalmanTracker` 건은 이 저장소 numpy 2.x 비호환의 3번째 사례**다(선례: `np.trapz` 제거로 플래너 크래시,
커밋 `0785777`). **numpy 2.x 비호환을 계속 의심하라.**

### 6-3. 엣지 기반 프리셋이 단색 타겟을 못 잡는다 (미수정, 대회 경로 무관)

`morphology`의 `open`이 Canny 1px 엣지를 통째로 지운다. 오케스트레이터 실측
(단색 사각형 → Canny 320px → close 320px → **open 0px → 컨투어 0**):

| 프리셋 | ops / kernel | 단색 타겟 |
|---|---|---|
| `video.yaml` | `[close, open]` k=7 | ❌ 소멸 |
| `single_frame.yaml` | `[close, open]` k=5 | ❌ 소멸 |
| `default.yaml` | `[close, open]` k=5 | ❌ 소멸 |
| `low_light.yaml` | `[dilate, close, open]` k=7 | ✅ 생존(dilate가 살림) |

**🟢 대회 타겟 프리셋 4종은 영향 없다** — `distress_coarse`/`distress_fine`/`vertiport_coarse`/`vertiport_fine`은
**`edge_detector`를 안 쓴다**(색 마스크 → morphology → rect). 채워진 blob에 `open(5)`는 무해하다.
또 `video.yaml`의 MOG2가 정지 타깃 마스크를 0으로 만드는 것도 관찰됨(호버링 정밀착륙 국면과 상충).

### 6-4. `test_blackbox` 플래키 — 원인 결정론적 확정

`vision/utils/blackbox.py:59-60`이 `__init__`에서 `QueueListener.start()`를 한다. 오케스트레이터 실측:

| 조건 | 파일 건수 | 마지막 frame_id | `len(records) <= 5` |
|---|---|---|---|
| 생산 루프 양보 없음 | 5 | 49 | 통과 |
| 틱당 0.2ms 양보 | **50** | 49 | **실패** |

**drop-oldest 자체는 정상**이고 단언이 틀렸다 — "**큐에** 최대 N건"을 "**싱크에** 최대 N건"으로 혼동.
CPU 부하 의존이라 **서브에이전트 3개가 돌 때 3/5 실패, 유휴 시 8/8 통과**한다. `b45fdc4`에서 결정론 3건으로
대체됐고 부하 아래 5/5 통과 확인.

### 6-5. 캘리브레이션 해상도 불일치의 서명

오케스트레이터가 일부러 460px용 calib을 320px 프레임에 먹여봤다: **z는 맞고 x/y만 1.4m 틀어진다**
(주점 불일치). 골든 생성기가 canvas 크기와 무관하게 같은 `fx`를 쓰기 때문이다.
**자동 재스케일은 의도적으로 안 한다** — 프레임이 다운스케일인지 크롭인지 코드가 알 수 없고, 잘못 보정하면
진짜 불일치가 숨는다. 대신 `frame_size_px`/`calib_image_size_px`를 meta에 실어 레코드만 봐도 진단된다.
골든 전용 가상 카메라 yaml: `vision/tests/golden/distress/synthetic_calib/canvas{460,320,200}.yaml`.

### 6-6. 🔴 미검증으로 남은 것 (억지로 메우지 말 것)

- **실촬영 검증 전무.** 골든셋이 전부 합성(원근·왜곡 없는 축평행 사각형)이다.
  코너 감김 오류가 **합성에서는 position에 안 나타나고 orientation에만** 나타난다 —
  **"합성에선 안 보이는데 실기체에서 터지는" 종류**다. `vision/CLAUDE.md`에 명시됨.
- **카메라 마운트 요각 ψ_m 미측정**(`MEASURED=False`, 기본 0). 물리 측정 필요.
- **체커보드 실측 캘리브레이션 보류 중** — 사용자가 2026-07-24에 결정했고 유지 중.
  **사용자가 먼저 꺼내기 전까지 재제안 금지**(메모리 `project_vision_calibration_deferred`).
- `not_for_closed_loop_30cm=True`가 계속 붙어 나간다. **30cm 정밀도 주장은 전혀 하지 않는다.**
- **`distress_box` 등거리 축퇴** — 박스가 매트 정중앙이라 네 모서리가 등거리이고 1px 흔들림에 착륙점이
  매트 반대편으로 점프한다. §4의 트랙 C가 처리 중.

---

## 7. 🔴 이 환경의 함정 (다시 조사하지 말 것)

### 7-1. git — 서브에이전트 프롬프트에 반드시 복사할 것

```
❌ git checkout -B dev--vision-computing-module origin/...   ← 절대 금지 (FC 커밋을 고아로 만든다)
❌ git reset --hard / 전역 git checkout . / 전역 git restore  ← 공유 인덱스에 남의 staged 작업이 있다
✅ 격리 worktree 전용 브랜치에 머문다 (브랜치 바꾸지 않는다)
✅ git fetch origin && git rebase origin/dev--vision-computing-module
✅ git push origin HEAD:dev--vision-computing-module     (거절 시 force 금지 → fetch·rebase·재시도)
✅ 커밋 직전: git fetch origin  +  git show --stat 으로 FC 파일 미포함 확인
```
**동시 push로 인덱스가 스테일해진다** — 한 세션이 그대로 커밋해 남의 작업 363줄을 되돌릴 뻔했다.

### 7-2. 네트워크·원격

1. **랩탑(WSL2)은 tailscale 피어가 아니다.** 경로는 항상 **"RPi가 listen, 클라이언트가 connect"** 방향으로.
2. **비대화형 SSH 백그라운드 자식은 SIGINT가 SIG_IGN으로 막힌다.** 종료·종료검증은 **반드시 SIGTERM**.
3. **`pgrep -f <패턴>`이 자기 자신(SSH 명령줄)을 매칭한다.** 대괄호 트릭(`ps -eo pid,args --no-headers | grep "[v]ision"`)을
   쓰고 죽이기 전에 `args`를 눈으로 확인. **엉뚱한 PID에 SIGTERM을 보내 원격 셸을 죽인 사고**가 있었다.
4. SSH로 장기 프로세스는 **런처 스크립트 파일**을 만들어 `setsid nohup /path/launcher.sh > log 2>&1 < /dev/null &`.
   heredoc은 SSH 인용 중첩에서 조용히 실패하니 `printf '%s\n' ... > file`.

### 7-3. 실기체(RPi)

5. **카메라는 배타적이다** — `main.py live`와 `tools/h264_stream.py`를 동시에 못 띄운다.
6. **RPi 저장소는 `git pull`이 막혀 있다.** vision 갱신은
   `cd /home/suri/drone_ws/src/suridoksuri && git fetch origin && git checkout origin/dev--vision-computing-module -- vision/`.
7. **RPi `picam-venv`에 `pytest`가 없다** — 실기체 검증은 "실행해서 로그/산출물 확인" 방식. 억지로 설치 금지.
8. **컨테이너에서 rclpy를 쓰려면 `source /opt/ros/humble/setup.bash`가 필요**하다(§3-1).
9. **PX4는 절대 업그레이드·플래시하지 않는다**(vision 세션 기준). 컨테이너 이미지 교체·호스트 ROS2 설치도 금지.

### 7-4. 테스트

10. **파괴검증을 `set -e` + `| tail`로 돌리면 실패를 놓친다** — 파이프라인은 마지막 명령의 종료코드만 본다.
    **종료코드를 변수로 받아 명시 확인.**
11. **`__pycache__` 함정** — §1-2 3번.
12. **알려진 플래키는 이제 없다**(`b45fdc4`에서 해소). 새로 플래키가 보이면 **CPU 부하 의존을 먼저 의심**하라
    — 서브에이전트를 여러 개 돌리는 중이면 그것 때문일 수 있다.

### 7-5. 절대 하지 말 것

- **`vision/utils/stream.py`(MjpegStreamer) 폐기 금지** — H.264와 **병행 운용**이 확정(MJPEG=검출 오버레이 관찰 /
  H.264=카메라 원본 저지연). 카메라 배타성 때문에 역할 분리가 실측으로 확정됐다.
- **`vision/core/state_machine.py`에 타겟별 분기 추가 금지** — "타겟 종류 무관 공통 골격"이 §9 6번의 핵심 요구.
  타겟 특수성은 `main.py`/`replay.py`의 `_build_observation()`이 흡수한다.
- **`fc_ros`/`fc_bridge` 수정 금지**(읽기는 허용). 정밀착륙 서브상태 구현은 **FC 트랙 소관**이다.
- **체커보드 캘리브레이션 재제안 금지**(§6-6).

---

## 8. 남은 백로그 (§4의 3건 이후)

### 8-1. 통합 완주까지 남은 것

1. **FC 쪽 소비 — `offboard_node` 정밀착륙 서브상태** ← **FC 도메인 세션이 해야 한다.**
   붙는 자리는 `HOLD`(`offboard_node.py:1230`) → `LANDING`(`:1298`) 사이. 진입점은
   `_publish_pos_setpoint(pos_ned, yaw)`(`:1007`). 새 상태를 `_RANGE_GUARDED_STATES`(`:132`)에 넣을지 결정 필요.
   **기본 off 파라미터로 넣어 배포할 것** — 실기체 FW+OFFBOARD 실적이 **0건**이라
   `sitl_vtol_remediation_plan.md` §4-1 4번이 "첫 실비행에 미검증 변수를 둘로 만들지 말라"고 명문화하고 있다.
2. **§9 8번 폐루프 30cm 검증** — 실측 캘리브레이션 선행 필요. **예선 통과 후**로 보류 확정.

### 8-2. 물리 개입 필요 (사용자 대기)

`docs/vision_status.md` 맨 위 "🔴 미실시 항목" 표 참조. **그 표를 지우지 말 것**(사용자 지시).

| # | 항목 | 비고 |
|---|---|---|
| 1 | **골든셋 실촬영 교체** | 지금 전부 합성. 실제 자갈/조명/그림자 오탐률을 아무도 모른다. **§6-6의 "합성에선 안 보이는 결함"이 여기 걸린다** |
| 2 | 체커보드 실측 캘리브레이션 | 보류 중, 재제안 금지 |
| 3 | **H.264 스트림 육안 확인** | 정량(30fps·25~41ms)은 검증됐고 화질·색·체감지연만 미확인. `ffplay tcp://100.67.27.83:8082` |
| 4 | AF 윈도우 분리 증명 | 서로 다른 거리에 두 물체 물리 배치 필요 |
| 5 | 40cm급 근접 초점 피크 | 체커보드를 40cm/210cm에 들고 있어야 함 |
| 6 | 실기체 데이터 기반 검출기 재검증 | 1번에 종속 |
| — | **카메라 마운트 요각 ψ_m 측정** | 신규. `frames.py`가 파라미터로 대기 중 |

### 8-3. 연기 확정

- **빠께스(소형 단일물체) 창의적 탐색설계** — vision 파트 완료 후. 2026-07-22 정정으로 순서 복귀.
  **다음 세션은 빠께스 트랙에 들어가지 마라.**

---

## 9. 참조

- `docs/vision_status.md` — 트랙 보드(라이브). 2026-07-28 절에 이번 세션 전건 기록. 맨 위 "🔴 미실시 항목" 표 먼저.
- `docs/vision_fc_interface.md` — 정찰 사실확정(852줄). **필요한 절만.**
- `docs/vision_plan.md` — §5.1(상태머신)·§5.3(②조난자)·§7.2(ports&adapters)·§7.5(기록·재생)·§8(FC통합)·§9(빌드순서)·§12(폐기방침)
- `vision/CLAUDE.md` — 파일역할표·테스트 규칙표·import 규칙·각 결정의 근거
- `docs/session_status.md` 🛩 sitl-vtol 트랙 + `docs/sitl_vtol_remediation_plan.md` — **FC 쪽 진행 상황**(읽기만)
- 메모리: `feedback_orchestrator_protocol` · `feedback_worktree_base_branch`(🔴 2026-07-28 갱신) ·
  `project_vision_dev_env` · `project_vision_calibration_deferred` · `project_rpi5_ubuntu_camera_stack` ·
  `project_vision_2nd_qualifier_bucket_target`
- ~~`docs/vision_next_session_brief.md`~~ — **이 문서로 대체됨.** §3 환경 함정만 §7에 흡수됐다.
