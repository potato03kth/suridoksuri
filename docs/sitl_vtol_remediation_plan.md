---
doc_type: handoff_plan
project: suridoksuri-1
track: 🛩 sitl-vtol → 🔧 main-code / ✈ vtol-실기체
scope: SITL-7 후속 — AUTO_PATH 검증 + A안 도입 + 결함 전체 해소 계획 (R1~R7)
created: 2026-07-27
status: 계획 확정, 실행 대기
---

# SITL-7 후속 개선 계획 — 인수인계

> **이 문서 하나로 다음 오케스트레이션 세션을 시작할 수 있게 쓴 것이다.**
> §1(확정 사실)은 **재조사 금지** — 전부 실측·소스 검증을 마친 것이다.
> §3의 R1~R7을 순서대로 굴리면 된다. §4는 굴리기 전에 사용자에게 받아둘 결정이다.

---

## 0. 목표

SITL-7 캠페인(24런)이 도출한 **결함 16건**을 해소하고, 그 중심에 있는
**A안(`setpoint_raw/local` 위치+속도 동시 발행 → `FW_POSCTRL_MODE_AUTO_PATH`)** 을 도입한다.

**A안을 맨 앞에 두는 이유:** FW 위치제어 모드 자체가 바뀌므로 cte·고도·코너·setpoint 계단 등
**거의 모든 정량 지표의 기준선이 새로 잡힌다.** 다른 결함을 먼저 고치면 그 검증을 두 번 해야 한다.

**A안이 해결하는 것:** PX4 버전에 안전이 의존하는 상태(F-4)의 제거.
**A안이 해결하지 못하는 것:** 나머지 15건 — 타임아웃 부재, setpoint 계단, 고도 프로파일 폐기 등은
전부 우리 코드 문제이고 별도 작업이 필요하다.

---

## 1. 인계 시점의 확정 사실 (재조사 금지)

### 1-1. 환경

| 항목 | 값 |
|---|---|
| SITL 호스트 | WSL 배포판 `Ubuntu-22.04` (E드라이브). `wsl.exe -d Ubuntu-22.04 -- bash -lc '...'` |
| **실기체 PX4 빌드** | `/root/PX4-vehicle` (worktree, `c890d9db0a` = 실기체 탑재본, course 가드 **있음**) |
| 기존 SITL 빌드 | `/root/PX4-Autopilot` (`9bb0d365c4`, 태그 `sitl7-orig-head`, 가드 **없음**) |
| 런 실행 | `wsl.exe -d Ubuntu-22.04 -- bash /root/s4/run_vehicle.sh <ID> --run-id <ID>_<tag>` |
| 워크스페이스 | `/root/drone_ws`, 저장소 클론 `/root/drone_ws/src/suridoksuri` |
| MAVROS | `udp://:14540@localhost:14580` |
| 하니스 | `tools/sitl/` — `run_scenario.py`(주입훅·RangeGuard) / `analyze_run.py` / `path_geometry.py` / `scenarios.yaml` |

### 1-2. PX4 상류 회귀 (근본원인, 전문은 `sitl_vtol_fw_offboard_rootcause.md`)

`FixedWingModeManager.cpp`의 오프보드 분기가 `_pos_sp_triplet = {}`로 제로초기화 후
`cruising_speed/vx/vy/vz/lat/lon/alt/gliding_enabled`에만 NaN을 넣는다. **`course`가 빠졌다.**
`msg/PositionSetpoint.msg:36` 규약이 `NaN = unused`이므로 **0.0f = "코스 0 rad(정북) 유지" 유효명령**이 되고,
그 분기의 경로 기준점이 기체 현재 위치라 **횡오차가 구조적으로 항상 0 → 경보·페일세이프가 안 걸린다.**

- 가드 `2e59c98b7c`(05-27) → revert `1499238f1c`(07-03)
- **실기체 `c890d9db0a` = 가드 O, revert X → 안전.** SITL 기존 `9bb0d365c4` = 취약.
- **실기체 PX4 업그레이드 금지** (A안 선행 없이는).
- **실기체 FW+OFFBOARD 비행 실적 0건** (ulog 78건 전수 스캔). 이 캠페인 전체가 SITL 근거다.

### 1-3. A안 실측 근거 (S3 프로브, `logs/2026-07-27_s3_fw_offboard_probe/`)

목표 정동 800m, 각 페이즈 ~55초:

| 페이즈 | 결과 |
|---|---|
| **위치+속도** | dE +794.6m, 지상코스 90.0°, **횡오차 0.2m** — `FW_POSCTRL_MODE_AUTO_PATH` 진입 |
| 위치만 (현행) | dN +814.7m, 코스 359.9° — 버그 발현 |
| 속도만 | `FW_POSCTRL_MODE_OTHER`로 빠져 **setpoint 발행 자체가 52.96초 정지** — 기각 |

**A안 구현 규약 (소스 근거 확정분):**
1. `/mavros/setpoint_raw/local` (`mavros_msgs/msg/PositionTarget`)
2. `coordinate_frame = FRAME_LOCAL_NED`(=1), **값은 ENU로 채운다**(MAVROS가 NED 변환 — 현행 PoseStamped와 동일 규약)
3. `type_mask = 3520` = `IGNORE_AFX|AFY|AFZ|YAW|YAW_RATE` (64+128+256+1024+2048). **위치·속도 활성**
4. `velocity`는 `velocity_2d.normalized()`만 쓰인다 → **크기는 무시, 방향만 유효**
5. **가속도는 반드시 마스크로 무시** — finite면 곡률(`loiter_radius`)을 계산해 선회를 걸어버린다
6. `cruising_speed`는 PX4가 무조건 NaN으로 덮는다 → **오프보드에서 순항속도 지정 불가.**
   대기속도는 `FW_AIRSPD_TRIM`으로만 정해진다. **`v_cruise`는 플래너 경로 형상에만 영향**
7. **MC 구간은 현행 위치-only 유지**(`mc_pos_control`은 `course`와 무관). MC/FW가 채널을 공유하면 `type_mask`를 구간별로 달리 줄 것

### 1-4. 캠페인 실측 기준선 (A안 도입 후 재측정 대상)

실기체 PX4 빌드 기준. **A안 적용 후 이 표와 나란히 비교해야 개선/악화를 판정할 수 있다.**

| 지표 | 현행 실측 |
|---|---|
| FW cte 최대 → 마지막 | A3 7.2→0.3 / B3 7.0→0.1 / B4 13.9→0.9 / B5 8.6→1.0 / C2 19.6→0.1 m |
| 코너 오버슈트 / WP 최근접 | 90°: 0.085~0.915m / 19~22m · 135°: 4.79m / 37.6m |
| **선회 중 고도 침하** | 직선 2.16 / 28° 1.95 / 90° 4.73 / 폐곡선 6.82 / **135° 7.45** / **직선인데 C2 6.76** m |
| setpoint 계단 | TRANSITION_FW→STREAMING **214~300m** · TRANSITION_MC→HOLD **60~117m** |
| 정천이 / 역천이 | 2.42~2.60s / 4.92~6.09s |
| 역천이 종점 통과량 | ≈ `57 − d_end_thresh` (선형). 현재 기본 10 → **+46.8m 통과** |
| 헤딩 정렬 잔류오차 | 회전량 비례: 4° 회전→0.4° / 95° 회전→2.4~2.5° (허용치 2.86°) |
| 바람 8m/s 영향 | cte 1.2→4.0m, 고도편차 2.24→3.99m, **정렬 허용대 재이탈 0→12.5°** |
| 제어틱 p95 | 104ms (N=301·501 동일). `_find_segment` O(N) 우려는 **기각됨** |
| 플래너 `__init__` 블로킹 | 2WP 즉시 / L자 50~69s / U턴 45~73s / **폐곡선 263.5s**. `v_cruise`에 160배 민감 |

### 1-5. 검증된 안전 경로 (건드리면 회귀 확인 필수)

- **OVERRIDE(FW)** MANUAL 거부 → 10틱 후 AUTO.LOITER 폴백 → DONE. 스트림 정지
- **OVERRIDE(MC)** POSCTL 분기 (S7 최초 검증)
- **PILOT_TAKEOVER** POSCTL 실제 진입 → 인계 감지 1회 → 스트림 영구 정지 → `vehicle_command` 0건
- **OFFBOARD 강제 이탈 복구** 상실 0.97s 후 복구, 완주. mode flapping 없음
- **C8 home_amsl/geoid** `alt=53.0m AMSL (지면 3.0+50.0)` 정확 — 작업 H-2 VTOL 이관 합격

---

## 2. 결함 → 작업 패키지

| WP | 대상 결함 | 내용 |
|---|---|---|
| **WP-A** | F-4 | **A안 도입** — FW 구간 `setpoint_raw/local` 위치+속도 |
| **WP-B** | F-1, F-2, F-3 | **타임아웃 + 거리 상한** — `TRANSITION_FW`/`TRANSITION_MC`/`ENTRY`/`CLIMBING`, 이륙지점 기준 거리 감시 |
| **WP-C** | F-6, F-9 | **경계 부드러움** — `_step_hold` 슬루레이트, 천이 고도 계단 램프, TRANSITION_FW→STREAMING 스냅백 |
| **WP-D** | F-8, F-10, F-11 | **경로·고도** — `_cruise_alt` 스칼라화 해소, 짧은 경로 `_FW_LOOKAHEAD` 적응, `d_end_thresh` 기본값 |
| **WP-E** | F-5, F-7 | **선회 품질** — 선회 중 고도 침하, 헤딩 정렬 마진 |
| **WP-F** | F-13, F-14, F-16 | **정리·진단** — 미사용 속도 프로파일, ENTRY의 FOLLOWING 오로그, 재요청 WARN 10Hz |
| **WP-G** | F-12 + 미실측 | **플래너 블로킹**, F-15(정렬구간 OFFBOARD 이탈)·`param_set` 주입·C9 |
| **WP-H** | — | **실기체 배포·검증** |

---

## 3. 세션 계획 R1~R7

> 공통 금지: **SITL 실행 세션은 항상 1개만**(gz 서버·UDP 14540/14580 점유).
> 공통 규율: 커밋 태그 `[sitl]`(코드 수정 포함 시 `[main]` 병기), push는 오케스트레이터가 관리.
> **모든 세션은 실기체 PX4 빌드(`/root/PX4-vehicle`)를 기본으로 쓴다.**

### R1 — AUTO_PATH 정밀 조사 (SITL 실행 최소, 구현 전 필수)

**왜 먼저:** S3 프로브는 **정동 800m 직선 한 레그, 55초**만 확인했다. 미션 통합·곡선·고도·종점
거동은 전혀 모른다. 구현 방향이 여기서 갈린다.

**반드시 답할 것 (PX4 소스 `control_auto_path()` 정독 + 짧은 프로브):**
1. **`velocity` 방향의 의미** — `control_auto_path`가 정의하는 경로는 "position setpoint를 지나고
   velocity 방향을 갖는 직선"인가? 그렇다면 우리가 넣을 방향은 **기체→목표점 베어링(`chi_cmd`)이
   아니라 경로 접선**이어야 한다. cte가 클 때 둘은 크게 다르다. **이걸 확정하지 않고 구현하면 안 된다.**
2. **고도(`position.z`) 추종** — AUTO_PATH가 z를 어떻게 쓰는가. 현행 대비 고도 응답이 바뀌는가
   (F-5 선회 침하·F-10 flower-pattern의 재측정 기준)
3. **종점 처리** — 경로 끝에서 position+velocity를 계속 주면 지나쳐 직진하는가, loiter로 도는가.
   `d_end_thresh` 판정 시점과의 상호작용
4. **선회 능력** — 목표점이 옆으로 크게 벗어났을 때 선회반경·수렴 거동 (현행 L1 대비)
5. **`type_mask=3520`이 yaw를 무시한다** — 현행 코드는 2026-07-21 사고 대응으로 yaw를 명시 발행한다.
   FW는 yaw를 직접 못 쓰지만 **TRANSITION_FW P3는 MC 상태에서 시작**한다. 그 구간에서 yaw 무시가
   안전한가(= PX4 MC가 현재 yaw를 유지하는가), 아니면 그 구간만 현행 유지해야 하는가
6. **AUTO_PATH 진입/이탈 조건** — `:389-392`의 `ISFINITE(lat)&&ISFINITE(lon)&&ISFINITE(vx)&&ISFINITE(vy)`.
   한 틱이라도 velocity가 NaN이면 `FW_POSCTRL_MODE_AUTO`(버그 경로)로 떨어지는가? **떨어진다면
   발행 누락 한 틱이 폭주로 이어진다** — 이게 A안의 최대 리스크다. 반드시 확인
7. 가드 있는 빌드(`c890d9db0a`)에서도 `:389-392`가 동일한지 소스로 확인

**프로브 방법:** 전체 미션 말고 S3 방식(수동 FW 진입 후 setpoint 직접 발행) + 짧은 ulog.
`logs/2026-07-27_s3_fw_offboard_probe/`의 프로브 스크립트 재사용.

**산출물:** `docs/sitl_vtol_auto_path_spec.md` — 위 7개 답 + 구현 규약 확정본(소스 줄번호 인용).
**게이트:** 오케스트레이터가 6번(velocity NaN 한 틱 → 모드 이탈 여부)을 직접 재현 확인.
**금지:** 비행 코드 수정.

---

### R2 — A안 구현 + 단위 테스트 (SITL 미사용)

**작업:**
1. `fc_ros/fc_ros/adapters/setpoint_publisher.py` — `PositionTarget` 발행 어댑터 신설
   (현행 `SetpointPublisher`는 `TwistStamped` 전용이라 그대로 두고 별도 클래스 권장).
   ENU 변환·`type_mask`·`coordinate_frame`을 한 곳에 가둘 것.
2. `fc_bridge/utils/` 에 **순수 함수**로 분리: 경로 접선 방향 산출, NED→ENU 변환,
   `type_mask` 상수. rclpy 없이 테스트 가능해야 한다(이 저장소 관례).
3. `offboard_node.py` — **FW 구간만** 교체. R1의 5번 답에 따라 TRANSITION_FW P3 처리를 결정.
   MC 구간(STREAMING-MC / FOLLOWING-MC / HOLD)은 **현행 위치+yaw 유지**.
4. `v_cruise`를 setpoint로 전달하는 코드가 있으면 정리(§1-3 6번). 대기속도는 `FW_AIRSPD_TRIM` 문서화.
5. **`velocity` 발행 누락 방지 가드** — R1 6번이 "한 틱 NaN이면 버그 경로로 떨어진다"면,
   velocity가 유효하지 않은 틱에는 **아예 발행하지 않는** 편이 안전하다(스트림 gap이 1초 미만이면
   `COM_OF_LOSS_T` 미달). 설계를 R1 결과에 맞춰 명시할 것.

**단위 테스트:** 접선 방향(직선/코너/종점), ENU 변환 왕복, `type_mask` 값, NaN 보장,
기존 `test_rotation.py`의 왕복검증 방식을 따를 것.

**게이트:** `pytest fc_bridge/tests fc_ros/test` 전건 통과 + **오케스트레이터가 테스트를 일부러
깨뜨려 red를 확인하고 원복해 green 재확인**(자기보고 신뢰 금지 규약).
**금지:** SITL 실행(다음 세션 몫), MC 경로 변경.

---

### R3 — A안 SITL 검증 (**AUTO_PATH 비행 검증의 본체**)

**합격 기준: 두 PX4 빌드 양쪽에서 통과.** A안의 존재 이유가 "가드 유무 무관"이므로
**기존 취약 빌드(`sitl7-orig-head`)에서도 통과해야 A안이 성립한다.**

**실행 (실기체 빌드 → 취약 빌드 순, 각 `--run-id`에 빌드 태그):**

| 순서 | 시나리오 | 확인 |
|---|---|---|
| 1 | A1 (직선 300m) | 기준선 회귀 |
| 2 | **A3 (L자)** | 현행 cte 7.2m 대비 |
| 3 | **B3 (직각 90°)** | 코너 오버슈트·WP 최근접·선회 침하 |
| 4 | **B4 (U턴 135°)** | 최악 선회. 현행 cte 13.9m·침하 7.45m 대비 |
| 5 | B5 (폐곡선) | 다중 코너 누적 |
| 6 | **B7 (단거리 40m)** | 종점 처리(R1 3번)가 실제로 어떻게 나타나는가 |
| 7 | **C2 (동쪽 90°)** | 현행 cte 19.6m·침하 6.76m 대비. **비-정북 필수 항목** |
| 8 | B8 (후방 180°) | 비-정북 |
| 9 | **C4 (바람 8m/s)** | 정렬 허용대 재이탈이 A안에서 바뀌는가 |
| 10 | 취약 빌드로 A3 + C2 재실행 | **가드 없어도 되는가 — A안의 핵심 주장** |

**추가로 반드시 볼 것:**
- `fixed_wing_lateral_setpoint` 발행 공백(속도-only에서 52.96초 공백이 났던 그 신호) — **0이어야 한다**
- `FW_POSCTRL_MODE`가 전 구간 `AUTO_PATH`로 유지되는가, 한 번이라도 `AUTO`로 떨어지는가
- §1-4 기준선 표 전 항목 재측정 → **개선/악화/불변**을 표로

**게이트:** 오케스트레이터가 ①취약 빌드 통과 ②모드 이탈 0건을 ulog로 직접 확인.
**중단 조건:** 취약 빌드에서 폭주가 재현되면 A안의 전제가 깨진 것 — **즉시 중단·보고.**

---

### R4 — 안전 (WP-B): 타임아웃 + 거리 상한

**대상:** F-1(ENTRY 무한대기 5.85km 실측) · F-2(역천이 무한) · F-3(정천이 무한) · F-15(정렬구간 이탈)

**설계 제안 (§4 결정 2·3 확정 후):**
- `TRANSITION_FW`/`TRANSITION_MC`/`ENTRY`/`CLIMBING`에 타임아웃 파라미터 신설.
  **규약은 `hold_timeout`·`mc_wp_timeout`과 동일** — 초과 시 강제 진행이 아니라 **안전 폴백**.
- 폴백 목적지: **이미 검증된 `_request_override()` 경로 재사용**(manual 시도 → AUTO.LOITER).
  새 경로를 만들지 말 것 — S7에서 3종 안전경로가 실증됐다.
- **거리 상한**: 이륙지점 기준 수평거리 파라미터 초과 시 OVERRIDE. 하니스 `RangeGuard`가
  C7에서 1564m에 실제 발동해 7km 비행을 막은 전례가 있다 — **같은 감시를 노드에 넣는다.**
- F-15: 정렬 구간에서 OFFBOARD를 잃으면 `_fw_offboard_requested`가 이미 True라 재요청하지 않는다.
  재요청 경로를 열되 **조종사 인계 가드(`is_pilot_takeover`)를 반드시 통과**시킬 것.

**검증 (주입):** C10 재현(ENTRY 무한대기 → 타임아웃에 걸리는가) · 정렬구간 OFFBOARD 이탈 주입(F-15) ·
거리 상한 발동 · **기존 안전경로 3종 회귀**(OVERRIDE-FW/MC, PILOT_TAKEOVER).
주입은 `on_log` 트리거 + in-process rclpy로(CLI는 4.04초 지연이라 짧은 이벤트를 못 노린다).

**게이트:** 오케스트레이터가 ①C10이 이제 타임아웃에 걸려 안전 상태로 떨어지는지 ②기존 안전경로
3종이 그대로인지 직접 확인.

---

### R5 — 부드러움·경로·고도 (WP-C + WP-D)

**A안 이후 수치가 바뀌므로 R3 결과를 기준선으로 삼는다.**

1. **`_step_hold` 슬루레이트** (F-6, 2026-07-21 수정후보 ②로 남아 있던 것). MC 분기의
   `_mc_pos_ramp`와 같은 방식.
2. **천이 고도 계단** (F-9) — `transition_alt ≠ waypoints[-1].z`일 때 첫 틱 ±30/−70m 계단.
   램프로 완화.
3. **TRANSITION_FW→STREAMING 스냅백** (F-6) — A안 적용 후에도 남아 있으면 목표점 산출을 일치시킨다.
4. **`_cruise_alt` 스칼라화 해소** (F-8) — 플래너가 이미 만드는 `alt_arr`/`gamma_ref`를 쓴다.
   A4로 실증(현재 중간 WP z=80m가 완전히 버려짐).
5. **짧은 경로 `_FW_LOOKAHEAD` 적응** (F-11) — 70m 고정이 40m 경로에서 추종 구간을 없앤다.
   경로 전장에 대한 상한을 두는 정도로.
6. **`d_end_thresh` 기본값** (§4 결정 5) — 통과량 ≈ `57 − thresh`.

**검증:** A1·A4·B7·C1a·C1b + R3에서 악화된 항목 재측정.

---

### R6 — 선회 품질·정리 (WP-E + WP-F) + 최종 회귀

1. **선회 중 고도 침하** (F-5) — R3에서 A안이 얼마나 개선했는지 먼저 보고, 남으면 원인 규명
   (현재 현상과 용량-반응만 확인됨). 침하는 코너 급격도가 아니라 **실제 선회량**에 붙는다.
2. **헤딩 정렬 마진** (F-7) — `wp0_heading_tol` 확대 또는 정착 조건 개선.
   근거: 잔류오차가 허용대 폭에 붙어 있고, 바람 8m/s에서 재이탈 12.5°.
3. **정리** — 미사용 속도 프로파일(F-13, `v_terminal`/`decel_dist`/`apply_terminal_decel`),
   ENTRY에서 FOLLOWING 오로그(F-14), 재요청 WARN 10Hz(F-16, **로그만 보면 2026-07-25 사고와 구별 불가**).
4. **플래너 블로킹** (F-12) — `__init__` 동기 실행. 최소 조치는 진행 로그 + 예상시간 출력,
   근본 조치는 비동기화. `v_cruise` 160배 민감성도 함께.
5. **미실측 보강** — `param_set` 주입, C9.
6. **최종 전 시나리오 회귀** — `scenarios.yaml` 26런 전량, 실기체 빌드.

---

### R7 — 실기체 배포 (WP-H)

**CLAUDE.md 규칙: FC 코드를 고쳤으면 즉시 실기체까지 반영한다 — 묻지 말고.**
유일한 예외는 사용자가 "현재 비행중"이라고 말한 경우.

절차는 `docs/rpi_deploy.md`. 요지:
`git push` → RPi `git pull` → 컨테이너 안에서 `colcon build --packages-select fc_ros` → 검증.
- **stale colcon build가 실비행 8건의 근본원인이었다**(`4dc30f9`) — 소스↔install md5 대조 필수
- 컨테이너 `fc`가 꺼져 있으면 `docker start fc` 선행
- 미커밋 로그 폴더가 `git pull`을 막으면 `rpi_deploy.md` §3 백업 절차
- **RPi5 fc_bridge venv numpy 버전 확인**(2.x면 `np.trapz` 계열 이슈 이력 있음)
- **PX4는 절대 업그레이드하지 않는다** (§1-2)

**배포 후 실기체 검증 계획도 이 세션에서 세울 것** — 실기체 FW+OFFBOARD 실적이 0건이므로
첫 비행은 보수적으로(짧은 직선 레그, 조종사 즉시 인계 대기).

---

## 4. 굴리기 전에 사용자에게 받을 결정 5건

> 답이 없으면 **제안값으로 진행**하고 그 사실을 명시한다(전부 되돌릴 수 있는 선택이다).

| # | 결정 | 제안 |
|---|---|---|
| 1 | **MC 구간도 A안으로 통일할 것인가** | **아니오 — FW만.** `mc_pos_control`은 `course`와 무관하고, `type_mask=3520`은 yaw를 무시해 2026-07-21 yaw 사고 대응이 무력화된다. 단 "MC 테스트기체도 최종기체와 동일 제어로직" 원칙과는 충돌하므로 사용자 확인 권장 |
| 2 | **타임아웃 초과 시 어디로 갈 것인가** | **`_request_override()` 재사용** (manual 시도 → AUTO.LOITER 폴백). S7에서 실증된 유일한 안전경로다. 새 경로 신설 금지 |
| 3 | **거리 상한 기본값** | **경기장 규격을 모른다.** 파라미터화하고 기본값은 보수적으로(예: 500m). 대회 경기장 크기를 알려주면 그에 맞춘다 |
| 4 | **yaml `v_cruise: 20.0`("임시" 주석)의 정식값** | 플래너 경로 형상과 계산시간(160배 민감)만 좌우하고 **실제 대기속도는 `FW_AIRSPD_TRIM`이 정한다**(오프보드에서 순항속도 지정 불가). 이 사실 위에서 재결정 필요 |
| 5 | **`d_end_thresh` 기본값** | 현재 10 → 종점을 46.8m 지나친다. 통과량 ≈ `57 − thresh`이므로 **오버슈트 0을 원하면 ≈57**. 다만 A안에서 이 관계가 바뀔 수 있으니 **R3 이후 확정** |

---

## 5. 검증 게이트 공통 규약 (오케스트레이터용)

- **세션 자기보고를 그대로 믿지 않는다.** 매 세션 종료 후 오케스트레이터가 직접 재현한다 —
  `git log`/`pytest` 재실행은 최소선, 수치 주장은 **원본 로그(node.log/ulog)에서 직접 교차확인**.
  실제로 이번 캠페인에서 요약과 원본이 어긋난 사례가 있었다(C4 "정렬 잔류오차 15.3°" →
  실제로는 정렬 완료 오차 불변, **허용대 재이탈 12.5°**가 진짜 신호였다).
- **"이 테스트가 진짜 버그를 잡는다"는 주장은 코드를 일부러 깨뜨려 red 확인 → 원복 green 재확인.**
- **경고의 무해성은 세션이 아니라 오케스트레이터가 게이트에서 판단한다.**
- **null 결과는 기능부재를 결론내기 전에 측정방법부터 의심.**
- 세션이 idle 루프("백그라운드 대기합니다"만 남기고 턴 종료)에 2회 빠지면 즉시 `TaskStop`하고
  오케스트레이터가 폴링을 넘겨받는다.

---

## 6. 환경 함정 (그대로 이어짐)

- `wsl.exe -d ... -- bash -lc '...'` 는 복잡한 셸 구문이 깨진다 → **로직은 스크립트 파일, 한 줄 호출**
- **PX4 콘솔(`pxh>`) 파일 리다이렉트 금지** — 분 단위 GB급 폭주
- 시나리오 사이 **`wsl.exe --terminate Ubuntu-22.04`** (개별 kill로는 gz 서버가 남아 중복 인스턴스)
- SITL arm: `CBRK_SUPPLY_CHK=894281`, `NAV_DLL_ACT=0` — **실기체 파라미터엔 절대 미적용**
- `fc_bridge`는 `pip install -e .` 금지(`.pth` 방식)
- WSL 클론에서 **`git clean` 금지** — gitignore된 `.ulg` 산출물이 지워진다
- 산출물 이송: WSL 클론 → `/mnt/c/sitl7_xfer/` → 호스트 워크트리. `.ulg`는 WSL 안에만
- 시뮬 정지 판별: `metrics.json`의 `ulog_duration_s / elapsed_s` (정상 0.92~0.99).
  벗어나면 PX4/코드 탓이 아니다 — `_simstall`로 보존 후 재실행
- 하니스 한계: `trajectory_setpoint` ulog 5Hz(발행 10Hz)라 **"계단 없음"을 주장하려면
  `resumption_gaps`까지 볼 것** / 시각정렬 잔차 ±1s 초과 런은 "경계 위반 건수" 신뢰 불가 /
  `setpoint 점프` 임계 1.5m는 FW 순항 lookahead 전진량보다 작아 상시 초과(경계 위반만 볼 것) /
  수직가속은 접지 충격이 지배(`excl_touchdown` 필드)

---

## 7. 회귀 시나리오 필수 요건 (이번 캠페인이 실증한 것)

1. **비-정북 레그를 반드시 포함시킬 것.** 정북 직선만으로는 "정북으로만 나는" 급의 버그가
   5건 연속 통과했다.
2. **런마다 PX4 빌드 커밋을 기록할 것** (`PX4_BUILDS.md`, `meta.json.px4_head`).
   어느 PX4에서 돈 결과인지 모르면 해석이 불가능하다.
3. **완주(PASS)를 성공으로 읽지 말 것.** B5는 4변을 다 돌았지만 종점 마진이 15m로 아슬아슬했고,
   B7은 "완주"했지만 추종 구간이 0.9초였다. **WP별 최근접 거리를 함께 볼 것.**
