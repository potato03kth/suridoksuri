---
doc_type: session_status
project: suridoksuri-1
scope: FC 세션 유일 진입점 — 트랙 보드(병행 작업 상태) + 환경 절차
last_updated: 2026-07-29
---

# FC 세션 진입 상태 문서

> **새 세션 진입:** 아래 트랙 보드에서 **재개할 트랙 블록 하나만** 읽고, 그 블록의 참조 문서만 필요 섹션 위주로 연다.
> 사용자가 "○○ 트랙 재개"라고 하면 해당 트랙, 지시가 없으면 ▶ 활성 트랙이 기본이다.
> `/session-log`는 세션이 건드린 트랙 블록**만** 갱신한다 — 다른 트랙의 상태는 보존된다.

---

## 공통 상태 (2026-07-11 갱신)

- **브랜치:** `dev--vision-computing-module` (전 트랙 공용 단일 브랜치. main 병합은 SITL-5 안정화 후 결정)
- **커밋 규율:** 트랙 전환 전 반드시 커밋(WIP 허용, 메시지에 `[main]`/`[mc-hw]`/`[sitl]`/`[vtol-hw]` 태그)
- **파라미터 규율:** 테스트 임시값은 yaml 수정 금지 — `phase2.launch.py v_cruise:=18.0 waypoints:="[...]"` launch 인자로만
- **미커밋:** `docs/session_log.md`·`docs/session_status.md`·`docs/archive/session_log_2026-06.md` (이번 세션-로그 갱신 — 다음 커밋에 포함하면 됨). 코드 수정(작업 H-2)은 `9451861`로 커밋 완료
- **vision 도메인은 별도 진입점:** `docs/vision_status.md` (트랙 보드) + `docs/vision_plan.md` (설계), 커밋 태그 `[vision]`. **FC 세션은 vision 문서를 읽지 않는다**(컨텍스트 격리).
- **하드웨어 (2026-07-09 갱신):** MC 테스트기체 **해체됨** → VTOL 테스트기체 **조립됨**. 코드에서 VTOL 천이를 진행하지 않으면 여전히 MC처럼 사용 가능하나, **PX6C의 PX4 파라미터로는 현재 조립이 MC인지 VTOL인지 구분할 방법이 없다** — PX4 설정값이 물리적 기체 형상을 반영하지 않으므로 어느 기체가 실제로 붙어 있는지는 이 문서(트랙 보드)로만 추적한다. 새 세션 진입 시 실기체 관련 가정("PX4 보고 mc/vtol 확인")을 하지 말 것. VTOL 기체에 **미진단 결함**이 있어 현재 비행 자체가 불가능 — 아래 ✈ vtol-실기체 트랙 참조.

---

## 트랙 보드

### 🎯 fc-정밀착륙 (F2) — ▶ **재개 가능** (플래시 선행조건 해소. 단 **실비행 검증은 아직 불가** — 아래 새 선행조건 2건)

> # 이 트랙을 재개한다면 → **`docs/fc_precision_land_handoff.md` 하나만 읽으면 된다.**
> 계약·붙는 자리·함정·**구현된 것과 안 된 것**·재개 순서가 전부 거기 있고, **vision 도메인
> 문서를 읽지 않아도 완주하도록** 썼다(도메인 컨텍스트 격리 유지).

- **🔴 2026-07-29 — 블로커 2건 + 노출·관측 2건 수정. `vision_landing:=true` 는 그 전까지 통째로 죽어 있었다.**
  ① **`_enter_vision_search()` 가 `self._sm` 을 세팅하지 않았다**(`8cb0861` 원본부터. 대조군
  `_enter_precision_land` 에는 있었다). `_exit_hold` → 진입해도 상태가 `HOLD` 로 남아 다음 틱에
  `_step_hold` 가 다시 안정조건을 만족 → **`_exit_hold` 10Hz 무한 루프**(SITL 실측 진입 1687회,
  `stable=2205/10`, 450.8s 타임아웃). **실기체였다면 WP1 상공에서 영원히 호버**했다.
  ② **래치가 vision 프레임이 아니라 제어틱을 셌다** — `vt.valid` 가 `vision_stale_timeout`(1.0s)
  동안 True 라 **같은 메시지 하나가 최대 10번 계수**됐고, setpoint 1건만 오고 침묵해도 0.2s 만에
  래치가 서면서 버퍼가 같은 좌표의 사본이라 **산포 필터까지 무력화**됐다(핸드오프 §6-1 #5·코드
  자신의 docstring과 정면 배치). → `VisionTargetBridge.setpoints_rx` 증가분 게이트로 수정,
  파라미터도 `vision_latch_ticks` → **`vision_latch_frames`** 개명(이름이 단위를 거짓말하면 같은
  버그가 다시 난다).
  ③ **launch/yaml 노출이 0개**여서 위생검사가 미선언 인자를 거부 → `vision_landing:=true` 는
  **launch 단계에서 실패**했다(= 실기체에서 F2 를 켤 방법이 아예 없었다). 18종 전부 노출,
  **기본 `vision_landing: false` 유지**.
  ④ **하니스가 LANDING 을 못 보고 있었다** — `_exit_hold` 단일 분기 도입으로 로그 문구가
  바뀌었는데 정규식이 그대로라 `state_timeline` 에서 LANDING 이 통째로 누락(실측 `R1_base`).
  → 문구는 안 되돌리고(단일 분기는 "한쪽만 고치는 사고"를 막는 구조다) **구·신 합집합**으로
  받고, `VISION_SEARCH`/`PRECISION_LAND` 2종 + GPS 착륙 폴백 3종을 추가.
  **🔴 왜 440건 전건 green 이면서 살아남았나:** `fc_ros/test/` 에 F2 노드측 상태기계 테스트가
  **0건**이었다(순수 함수 테스트는 "`_sm` 을 세팅했는가"를 원리적으로 못 본다).
  → **`fc_ros/test/test_offboard_f2_state.py` 신설** — rclpy 스텁 + `__new__` 껍데기로 **노드
  메서드를 실제로 실행**해 전이·연속틱 거동을 단언한다. 파라미터 3중 일치(YAML↔declare↔launch)
  테스트와 합쳐 **565건**(+125). 파괴검증 4종 전부 red→green 확인.
  **SITL 재검증**(격리 클론 `/root/ws_f2`, `PX4_DIR=/root/PX4-vehicle`, 바이너리 재빌드 확인):
  `false` 회귀 `G1_base` exit=done 152.9s(HOLD 67.89 → **LANDING 78.30** → DONE 121.76,
  비전 로그 0건) / `true` 해피패스 `G2_vision`(아래 로그 폴더). 산출물 `logs/2026-07-29_f2_fix/`.
  SITL 합성 vision 발행기를 저장소로 들여왔다 → **`tools/sitl/fake_vision.py`**(§6-1 장애주입 6종 지원).
- **✅ 보류 해제 (2026-07-29):** 보류 사유였던 **"선행: 🛩 sitl-vtol 실기체 플래시"가 해소됐다.**
  2026-07-28~29 F-17/F-4 패치본이 실기체에 올라갔고(CRSF RC 두절 사고 1건 발생·해소),
  ulog 로 패치 정상작동까지 확인됐다 — 상세는 🛩 sitl-vtol 트랙 최신 블록 ·
  `docs/px4_v6c_patch_build.md` §11.
  - (원 보류 사유, 이력 보존) F2는 `HOLD` **이후**에 발동하는데 거기 도달하려면 FW 순항 →
    역천이 → HOLD를 거쳐야 하므로, **천이가 안 고쳐지면 F2 코드가 실행되는 지점까지 기체가
    가지 못한다.** 또 `sitl_vtol_remediation_plan.md` §4-1 4번이 "첫 실비행에 미검증 변수를 둘로
    만들지 말라"를 명문화하고 있어 천이 패치와 정밀착륙을 같은 비행에서 처음 켜는 것은 금지다.
- **🔴 그러나 선행 조건이 새로 생겼다 — 실비행 검증은 아직 못 한다 (2026-07-29):**
  - **자기계 헤딩 의존 결함 미해결 (`5d55b3f`)** — 재캘리브레이션은 적용됐으나 지표가 악화됐다
    (`test_ratio` 평균 1.97→2.62, `cs_mag_fault` ON 0%→92.7%). 원인은 전류 간섭이 아니라
    **캘리 방향 커버리지**이며, 통과 기준은 **"재캘리 후 기수를 남쪽에 두고 `test_ratio<1` 확인"**.
    이게 흔들리면 `_step_transition_fw` Phase 2 의 "헤딩 정렬 완료" 판정 근거 자체가 흔들린다.
  - **배터리 게이트 부재 (`f8e951f`)** — flight02 는 `Emergency battery level` 이 t=8.64s(고도 약 4m)
    부터 떠 있는 채로 50m 까지 올라가 천이를 시도했다. `offboard_node` 상태기계에 배터리 게이트가
    없다. **F2 는 임무 끝단**(탐색 최악 171s 소요, 아래 시간예산 참조)이라 이 결함의 직격 대상이다.
  - ⇒ **지금 재개해서 할 수 있는 것은 아래 "미검증 ①SITL 장애주입"과 잠정값 실측이다.**
    `vision_landing:=true` 실비행은 위 2건 해소 후.
- **✅ 구현·배포 완료 (`8cb0861`)** — `vision_landing:=false`(기본)면 종전 `HOLD → LANDING`과
  **완전히 동일**하다(구독조차 만들지 않는다). RPi 배포·검증까지 끝냈다(소스↔install md5 일치,
  import 통과, **QoS BEST_EFFORT 실기체 확인**). 즉 **지금 기체에 얹혀 있어도 무해**하다.
  - 상태 2종: `VISION_SEARCH`(탐색고도 정렬 → 제자리 확인 → 아르키메데스 나선 확대, 실패 시
    15m/18m 재탐색 1회 → GPS 착륙 폴백) · `PRECISION_LAND`(수평은 vision, **수직은 FC가 스케줄**).
  - 신규 모듈: `fc_bridge/execution/search_pattern.py`(나선·풋프린트·검출고도 상한) ·
    `fc_bridge/execution/precision_land.py`(래치·하강게이트·인계판정) ·
    `fc_ros/adapters/vision_target_bridge.py`(두 토픽 구독·계약 파싱).
  - **인수인계 D-a~D-d 전부 처리**: `vision_veto_timeout`(10s) 신설 / `_RANGE_GUARDED_STATES`에
    `VISION_SEARCH` 포함·`PRECISION_LAND` 제외 / `listen_lt`는 열지 않음 / AGL은 이륙지점 지면
    기준 근사(라이다 배선 여전히 없음).
- **🔴 미검증 — 재개 시 여기서 시작한다:** ①**SITL 장애주입**(핸드오프 §6-1: vision SIGKILL →
  `link_dead` → 폴백 / veto → 하강 중단 / setpoint 침묵 시 마지막 값을 붙들지 않는지) ②`vision_landing:=true`
  실비행 **0건** ③실촬영 검출 검증 전무(vision 골든셋이 전부 합성).
- **구현 중 실측으로 잡은 것 3건:** 최대반경 기준 선회속도를 나선 전체에 쓰면 r=5m에서 tilt 27°
  → 25m 고도 시선오프셋 12.75m로 **링 간격(12.2m)과 같은 자릿수**라 커버리지가 통째로 틀어진다
  (매 틱 반경별 재산출로 수정, 최대 tilt 6.0°) / `vision_search_timeout` 90s가 1회차 97s를 완주
  직전에 잘랐다(→120s 역산) / **`DiagnosticStatus.level`이 1바이트 `bytes`로 온다**
  (`shim_node.py:221`) — 핸드오프 §2-2가 "OK=0/WARN=1/ERROR=2"라고만 적어둬 그대로 믿고
  `level == 1`로 짜면 **테스트 전건 green인 채 실기체에서 거부권만 조용히 사라진다.**
- **탐색 시간 예산(실측):** 1회차 97s / 재탐색 74s / **최악 합계 171s(2.9분)**. 임무 끝단이라
  배터리 여유를 보고 반경·타임아웃을 줄일지 판단할 것.
- **잠정값 주의:** 기체 최외곽 반경 `R` **미측정**(2026-07-29 실측 예정)이라 vision이 내는 **착륙점
  좌표 자체가 잠정값**이다. 카메라 마운트 요각 ψ_m도 미측정 — **틀리면 착륙 오프셋의 *방향 자체*가
  틀린다**(수정할수록 멀어지는 증상이면 이걸 먼저 의심).

### 🚁 mc-실기체 — ▶ 활성 (신규 기체로 부활, 2026-07-18 확인. **2026-07-27: WP 정착 도입 — SITL 완주 + RPi5 배포·검증 완료**)

- **내용:** RPi5(Ubuntu 24.04) + Pixhawk 6C 순수 MC 테스트기체 브링업 (SITL-5 변형, `vehicle_type:=mc`). 2026-07-09 물리적 해체됐던 것과 별개로(또는 그것을 재조립한 것인지 미확인) **2026-07-18 "부활한 MC 테스트기체"로 실비행 진행 중임을 로그로 확인.**
- **마지막:** **(2026-07-27, MC 웨이포인트 "정착" 도입 — fly-by 통과 폐기, 커밋 `4e8e378`)** 사용자 지적: "MC는 호버가 되는데 WP에 정착하지 않을 이유가 있는가, 그게 훨씬 자연스러운 비행이다." **종전 결함:** `mc_wp_advance()`가 거리 조건 하나(`dist < mc_end_thresh`)만 봐서, 반경 경계를 스치는 순간 목표가 다음 WP로 바뀌었다 — 기체는 WP 위에 가보지도 않고 코너를 자르며 지나갔고, 2026-07-25 flight04에서 WP를 **1.8~1.9m 지점에서 "통과" 판정**한 게 이 동작이다(커밋 `3f389b6`이 "미완"으로 남겨둔 항목). **수정:** 정착 판정 도입 — 반경 안 + 수평속도 < `mc_wp_settle_speed`가 `mc_wp_settle_time` 동안 연속 유지돼야 다음 WP로 전진, 조건이 깨지면 카운터 0 리셋, 마지막 WP도 동일하게 정착해야 경로완료. 반환값 `(idx, settled_ticks, done)` 3튜플. 기본값(`settle_req=0`)은 종전 fly-by와 완전 동일(`climbing_reached()`의 `vz_down`과 같은 하위호환 규약). 위치 setpoint 발행 경로는 미변경 — `_mc_pos_ramp`가 목표 도달 후 고정되므로 기체가 그 점 위에 호버한다(MC도 위치 setpoint를 쓰는 최종기체 동일 제어로직 원칙 유지). 정착 실패로 갇히는 경우는 `mc_wp_timeout`(20s)이 강제 진행(`hold_timeout`과 같은 규약). **파라미터:** `mc_wp_settle_speed` 0.3m/s · `mc_wp_settle_time` 1.0s · `mc_wp_timeout` 20.0s 신설, `mc_end_thresh` 2.0→**1.0m**(정착 방식에선 WP 위 수렴을 기다리므로 반경을 조여도 놓칠 위험 없음). 현장 튜닝용으로 `mc_end_thresh`·`mc_wp_settle_time`을 launch 인자로 노출. **검증:** pytest 297 통과(신규 6건). **SITL 회귀검증 PASS**(이 노트북 `Ubuntu-22.04`, `gz_x500`, 5m 사각형 5점 경로) — WP0~WP4 **전부 정착**했고 정착 시점 WP 오차가 **0.02/0.17/0.05/0.14/0.10m**(종전 1.8~1.9m 통과와 대조), ARM→CLIMBING→STREAMING→OFFBOARD→FOLLOWING→HOLD→LANDING→disarm 완주, 타임아웃·경고 없음. **RPi5 배포·검증 완료**(`git pull`+컨테이너 `colcon build --packages-select fc_ros`, 소스↔install md5 일치·yaml 신규 파라미터 반영·import 통과). 배포 시 컨테이너 `fc`가 꺼져 있어(`Exited (255) 5 hours ago`) `docker start fc` 선행 — pull은 미커밋 로그 폴더(`logs/2026-07-25_flight14~19`, `2026-07-26_manual`) 때문에 막혀 `rpi_deploy.md` §3 백업 절차로 `~/drone_ws/_pull_backup_20260727_005448`에 옮기고 진행했다. **백업본 대조 완료 — 삭제해도 안전:** rosbag(`rosbag_0.db3`/`metadata.yaml`)은 전부 크기 일치, `2026-07-26_manual/`의 ulog 8건은 커밋 `6b699b3`이 각 비행 폴더로 이동시킨 것들로 이동처에서 크기 일치(158→flight05, 170→flight08, 174→flight10, 190→flight14, 196→flight16, 201→flight17, 203→flight19, 199는 제자리), `notes.md`만 백업본이 분석 전 구버전이다(커밋본이 상위집합). 삭제는 사용자 판단으로 남겨둠.
  그 전 **(2026-07-25, 문서 미반영분 3건 — 이번 세션 자가복구로 확인)** 트랙 보드가 07-24에서 멈춰 있었고 그 뒤 커밋 4건이 기록 없이 쌓여 있었다: ①`c251323`/`0c45bc7` 2026-07-25 비행로그 분석 — offboard가 EKF 로컬원점 절대좌표를 지령하던 결함 확정, 지오이드 수정은 배포 정상이었고 3축 중 1축만 보정된 게 원인 ②`3f389b6` **경로 기준계를 이륙지점 기준으로 수정**(`waypoint_frame` 파라미터 신설, 기본 `takeoff`) + **조종사 인계 시 노드 정지**(`_State.PILOT_TAKEOVER` 신설 — 종전엔 `_step_following`이 조종사의 POSCTL 인계를 0.9초간 10회 재요청으로 뒤집어 조종사가 KILL 스위치를 써야 했다) ③`cd5fda9` 컨벤션·토픽 의미론 감사 수정 6건 ④`6b699b3` **2026-07-25 15비행 로그 회수·분석** — flight14·16이 조종사 개입 0으로 완주(경로이탈 최대 0.94/2.36m, WP 최근접 0.10~0.64m)해 `waypoint_frame=takeoff`의 실기체 유효성 입증(H16 종결), "사각형이 안 그려진다"는 지령 waypoints가 꼭짓점 3개였던 것(기체는 그 삼각형을 정확히 그림), flight17 VTOL 천이 이상방향은 자기계 기각으로 헤딩 65° 오차(H17 신설), 고도 침하는 `thrust_z` 포화·최저 셀전압 2.46V·`MPC_THR_HOVER` 0.5 vs 실측 호버 0.66~0.88. 상세는 각 커밋 메시지와 `logs/2026-07-25_review.md`.
  그 전 **(2026-07-24, 작업 H-2 SITL 재현 — 근본원인 확정+수정, 커밋 `568fbe5`)** 바로 아래 "그 전" 항목이 "미확정"으로 남긴 STREAMING 오버슈트 근본원인을, `docs/sitl_verification_log.md` "작업 H-2" 체크리스트를 `PX4_HOME_LAT/LON/ALT`(실측 지면 19.2m AMSL) 통제조건으로 그대로 실행해 확정. **②geoid 정합 FAIL로 재현:** `/mavros/home_position/home.geo.altitude`가 19.2가 아니라 43.98로 수신(오차 24.78m, 한국 geoid separation과 거의 정확히 일치) — 원인은 애초 의심했던 "PX4 자체 버그"가 아니라 **MAVROS가 ROS REP-103 관례(NavSatFix.altitude=WGS84 타원체고) 때문에 HomePosition.geo.altitude에도 EGM96 보정을 적용**하는 것이었음(원인 위치 PX4→MAVROS로 정정). 이 오염값을 그대로 쓰던 home_amsl 계산이 4m 상승명령을 실측 ~28.8m 상승으로 실행시킴 — 체크리스트가 이미 예견해뒀던 "+25~40m 과상승 → geoid/ellipsoid 혼동" 시나리오 그대로. **수정:** `offboard_node.py::_cb_home`의 home_amsl 소스를 `/mavros/home_position/home`(`geo.altitude`)에서 `/mavros/altitude`(`amsl` 필드, `GLOBAL_POSITION_INT.alt`를 보정 없이 relay)로 교체 — 나머지(수렴판정·신선도검사·`takeoff_request_fields`)는 미변경. **같은 통제조건으로 재검증 PASS:** home_amsl≈19.4(실측 19.2와 일치) → CommandTOL 4m 정상 요청 → CLIMBING→STREAMING→OFFBOARD→FOLLOWING→HOLD→LANDING→disarm 전체 미션 정상 완주. pytest fc_bridge+fc_ros 172 전부 통과(회귀 없음). 이 경로는 `_is_mc` 분기가 없어 **MC/VTOL 동일 적용**(오늘 VTOL 비행에도 동일 유효). 상세는 `docs/sitl_verification_log.md` "작업 H-2" 배너·체크리스트 참조. **⚠ 다음 실비행 전 필수: RPi5 fc_ros가 이 커밋(`568fbe5`)을 반영했는지 확인 — 아직 `git pull`+재빌드 안 됐다면 이 버그가 그대로 살아있는 상태다.**
  그 전 **(2026-07-24, STREAMING 오버슈트 조사 — 브리프 1차 가설 정정 + `_home_amsl` 세션 재사용 버그 수정 + `climbing_reached()` 속도조건 추가 + SITL 회귀검증)** 직전 세션이 남긴 `docs/mc_hw_next_session_brief.md`(가설: `climbing_reached()`가 속도 무시하고 STREAMING 조기전환)를 이어받아 조사. 사용자 힌트("MC 테스트기체에서도 오버슈트 발생한 적 있다")로 2026-07-20 flight01 사고(아래 "그 전" 항목, 당시 "고도 오버슈트 근본원인 미해결"로 남겨진 채 방치)와 동일 계열임을 먼저 확인. **원본 ulog(`logs/2026-07-24_sitl_streaming_overshoot/05_07_03.ulg`) 재디코드 결과 브리프 가설은 틀렸음이 드러남:** PX4 내부 위치제어기 목표고도(`vehicle_local_position_setpoint.z`)가 3.0m에서 전혀 감속하지 않고 실측 AGL과 나란히 5.53m까지 계속 상승 — 즉 우리 노드가 아니라 **PX4 자체가 애초에 3m가 아닌 다른 고도를 목표로 이륙 중이었음.** 결정적 물증: `vehicle_command`(NAV_TAKEOFF) 요청고도 = 50.47023m AMSL, 그런데 이 ulog의 실제 `home_position.alt` = 0.25093m AMSL — 요청 AGL이 3.0m가 아니라 **약 50.2m**였다는 뜻. 50.47023−3.0=47.47023은 **바로 앞 세션(2026-07-24, home_amsl 회귀검증 비행)에서 확정된 "지면 47.5m"와 사실상 동일값** — 그 비행에서 확정된 `_home_amsl`이 SITL 인스턴스가 바뀐(실제 홈 0.25m) 이번 비행에 그대로 재사용된 것. 코드로 확인: `offboard_node.py`의 `self._home_amsl`/`_home_amsl_samples`는 `__init__`에서 한 번만 초기화되고 ARM/이륙 사이클마다 리셋되는 지점이 전혀 없어, 한 번 수렴 확정되면 프로세스 생애주기 내내(같은 세션의 여러 비행에 걸쳐) 재검증 없이 재사용됨 — PR #4(`home_amsl_confirmed()`)가 막는 "최초 GPS 미수렴 스냅샷" 시나리오와는 다른, PR #4로 못 막는 하위 시나리오. **수정 2건 (사용자 승인, 옵션3 "둘 다 수정"):** ①`_cb_home`에 메시지 신선도 검사 추가(`home_amsl_sample_fresh()` 신규 순수함수, `msg.header.stamp` 나이가 `home_amsl_max_age`(기본 1.0s)를 넘으면 수렴 표본에서 제외) — 세션 내 재사용/래치 경로 원천 차단. ②`climbing_reached()`(브리프 원 가설이었으나 부차 요인으로 재평가)에 `vz_down`/`vz_tol`(기본 0.3m/s) 파라미터 추가 — 위치+수직속도 안정 둘 다 요구, 기본값은 기존 호출과 동일 동작 보존. `_step_climbing()`이 `state.vel_ned[2]`를 전달하도록 갱신. **pytest:** `fc_bridge/tests/test_state_logic.py` 신규(이 모듈 테스트 전무했음, 10건) 포함 `fc_bridge` 65 전부 통과(기존 55+신규 10, 회귀 없음). **SITL 회귀검증:** 이 노트북 WSL(`Ubuntu-22.04`) 환경을 이 세션 내에서 직접 재사용(재구축 불필요, `wsl.exe` 상호운용으로 사람 개입 없이 이 세션에서 직접 접근 가능함을 새로 확인). **⚠ 그런데 검증 도중 이 두 수정으로도 안 풀리는 훨씬 근본적인 별개 문제를 발견함(미해결) — 완전히 신선한(재사용 아닌) PX4 SITL에서도 `home_amsl`(47.44)·CommandTOL 계산(50.4 AMSL) 둘 다 정상인데 실제 상승고도가 3.0m가 아니라 ~50.2m(AGL)에서 멈춤, 이 50.2가 목표 AGL도 목표 AMSL도 아니라 "요청 AMSL 숫자값(50.4) 자체"와 거의 정확히 일치 — PX4/MAVROS가 AMSL 절대고도를 로컬 상대고도인 것처럼 그대로 소비하는 것으로 강하게 의심됨(미확정). 이게 맞다면 지금까지의 모든 오버슈트 사고(flight01·오늘 두 재현 전부)의 공통 근본원인일 가능성.** 이번 2건 수정은 안전하게 유지하되 "오버슈트 해결됨"으로 보고하지 않음 — 사용자 확인 후 범위 재설정 필요. 상세는 `docs/session_log.md` 2026-07-24 "STREAMING 오버슈트 조사" 항목 "⚠ SITL 회귀검증에서 훨씬 더 근본적인 별개 문제 발견" 참조. `docs/mc_hw_next_session_brief.md`는 완료 표시·정정 배너 추가해 보존(아카이브용 원본) — 단 이 새 발견은 아직 그 문서엔 반영 안 됨, session_log가 최신.
  그 전 **(2026-07-24, 2026-07-23 저녁 실비행 사고분석 + `_cb_home` 수렴판정 수정 + SITL 회귀검증 완료 — "3m 상승명령이 30m로 실행됨 + 오프보드 미이행" 사용자 질의 응답 → 재발방지 코드수정 → SITL로 실증)** 사용자가 지상국 Windows(E:\Downloads)에서 회수한 ulog 2건(`10_42_23.ulg`=`logs/2026-07-23_manual/`, `11_32_15.ulg`=`logs/2026-07-23_flight01/`)을 RPi(`100.67.27.83`) journalctl·`fc` 컨테이너(bash_history/launch.log/rosbag, docker cp로 회수)와 교차분석. **① "3m 대신 30m 상승":** PX4 결함 아님 — `offboard_node.py::_cb_home`이 `/mavros/home_position/home`을 단발 스냅샷하는 기존 설계 갭(아래 ③ "잔여 리스크" 항목, 종전엔 EKF 드리프트 3~5m 규모)이 이번엔 **26.7m**로 재현(`_home_amsl`=393.6 AMSL 스냅샷 vs 실제 EKF 홈고도 366.93 AMSL, launch.log에 `CommandTOL 이륙 요청 alt=396.6m AMSL (지면 393.6+3.0)` 명시적으로 남음, AGL 최대 29.72m로 계산치와 정확히 일치) — `fc` 컨테이너/MAVROS가 arm 수분 전(20:28~20:32 KST)에 막 재시작된 반면 PX4는 23분 앞서(20:09:29) 부팅돼 있어, 새 MAVROS 구독자가 PX4 부팅 초기(GPS 수직정확도 미수렴 시점)에 래치된 오래된 home_position을 그대로 받았을 가능성 유력. **② 오프보드 명령 이행 여부: 이행 안 됨.** `nav_state`가 POSCTL→AUTO_TAKEOFF(t=1.11s)→AUTO_LOITER(t=21.46s, PX4 자동전환)→POSCTL(t=55.88s, 조종사 개입)→MANUAL — **OFFBOARD 진입 자체가 한 번도 없었음.** `vehicle_command`는 최초 이륙요청(t=1.09s) 이후 전무 — ulog `t=24.19s Connection to mission computer lost`가 결정적 원인. **③ 근본원인 — RPi 전원 불안정:** journalctl에 이 세션 동안 `hwmon hwmon4: Undervoltage detected!`를 동반한 연쇄 재부팅 3회(20:03/20:17/20:24~20:31 KST) 확인, arm(20:32:15) 불과 48초 전(20:31:27)에도 RPi가 다운돼 이후 다음날 00:17까지 완전히 꺼져 있었음(`project_rpi5_usbc_power_psu_max_current` 메모리의 비-PD 전원 한계와 같은 계열 증상 추정) — 이게 미션컴퓨터 연결 끊김의 직접 원인. rosbag도 `metadata.yaml` 없이 비정상 종료(정상 Ctrl-C 경로 아님)로 정황 뒷받침. 배터리는 부하중 최저 10.92V로 정상(전압붕괴형 사고 아님, 07-18/07-20 세션과는 별개 원인 계열). 같은 날 앞선 `10_42_23.ulg`(19:42 KST)는 RPi가 완전히 다운돼 있던 시간대의 별개 수동 POSCTL 비행(오프보드 무관, `vehicle_command` 토픽 자체 없음). **④ 후속 사용자 요청("버그 재발을 막아라")으로 코드수정 완료:** `fc_bridge/execution/state_logic.py::home_amsl_confirmed(samples, tol=0.5, min_samples=3)` 신규(순수함수) — 최근 `min_samples`개 home_position AMSL 샘플이 `tol` 이내로 수렴해야 신뢰. `offboard_node.py::_cb_home`이 수신값을 단발 대입하지 않고 샘플 리스트(`_home_amsl_samples`, 최근 20개 유지)에 누적 후 `home_amsl_confirmed()`로 확정, `_step_arm_takeoff`는 미확정 시 CommandTOL을 계속 보류(수렴 대기 로그로 최근 샘플·tol 노출)하도록 변경 — 새 ROS 파라미터 `home_amsl_tol`(기본 0.5)·`home_amsl_min_samples`(기본 3) 추가. `docs/session_status.md` 이 "잔여 리스크" ③ 항목이 권고했던 대책 (a)를 그대로 구현. pytest 신규 9건 포함 fc_ros+fc_bridge 162 전부 통과(순수 로직). **✅ SITL 회귀검증 완료(2026-07-24, 같은 세션 — 사용자가 "이 노트북에 SITL 새로 구축" 지시).** 이 노트북(24.04)엔 SITL이 없어 신규로 별도 WSL 배포판(E드라이브, Ubuntu-22.04) 구축 — 상세 절차·트러블슈팅(`fc_bridge` pip install이 `ros2` CLI를 깨뜨리는 문제→`.pth` 방식으로 해결, SITL 벤치 arm에 필요한 `CBRK_SUPPLY_CHK`/`NAV_DLL_ACT` 프리플라이트 우회, PX4 콘솔을 파일로 리다이렉트하면 로그가 수 분 만에 GB급으로 폭주하는 문제 등)은 `docs/wsl_dev_env_setup.md` 섹션 F에 신규 기록. **핵심 실증 두 가지:** ①PX4 SITL disarm 상태에서 60초간 `home_position`을 관찰한 결과 **약 34회 재발행**됐고, 도중 실제로 47.46m→48.97m로 값이 바뀌는 드리프트를 실측(우려했던 "PX4가 단발만 보내 무기한 대기"는 기우로 확인, `min_samples=3` 안전). ②사고와 동일한 launch 인자(`vehicle_type:=mc transition_alt:=3.0 waypoints:=[0,0,3, -3,-1,3, 0,0,3]`)로 `offboard_node`를 실행 → `home_position AMSL 미수렴(최근 1개→2개)` 경고 후 3개 수렴 시점에 `CommandTOL 이륙 요청 alt=50.5m AMSL (지면 47.5+3.0)`로 **정확한 고도**가 계산됐고(26.7m류 오차 재현 안 됨), 이어 CLIMBING→STREAMING→OFFBOARD→FOLLOWING→HOLD→LANDING→disarm까지 **미션을 끝까지 정상 완주**(사고 당일처럼 AUTO_LOITER에 갇히지 않음). STREAMING 진입 직후 AGL이 순간 5m대로 오버슈트했다가 정상화된 것은 관측됐으나 이번 수정과 무관한 별개의(기존에 "잔여 리스크"로 이미 알려진) 현상으로 판단, 별도 이슈로 남김(코드 미수정). rosbag 손상(`database disk image is malformed`)으로 `/mavros/global_position/global` 이력을 직접 대조해 원래 사고의 메커니즘(26.7m 오차의 정확한 유래)을 확정하지는 못했다(정황 기반 가설로 남음 — SITL 검증은 "고쳤다"를 확인한 것이지 "원래 왜 그랬는지"를 확정한 것은 아님). **후속 권고(아직 미구현):** (b) 다음 실비행 전 RPi 전원계통 점검을 체크리스트화 (c) `record_flight.sh`가 rosbag 비정상종료를 감지·경고하게. 상세는 `logs/2026-07-23_flight01/notes.md`(+`analysis_auto.md`)·`logs/2026-07-23_manual/notes.md`·`docs/wsl_dev_env_setup.md` 섹션 F 참조. **STREAMING 오버슈트는 자기완결 브리프 `docs/mc_hw_next_session_brief.md` 신설(원본 데이터: `logs/2026-07-24_sitl_streaming_overshoot/`) — 다음 세션은 그 문서로 진입.**
  그 전 **(2026-07-22, "PID 튜닝을 한번도 안 한 게 원인 아니냐" 사용자 질의 → H14 신설, 분석만·코드 미수정)** `docs/mc_hw_open_hypotheses.md`에 H14(제어: PX4 자세/각속도 PID 게인이 이 커스텀 기체용으로 한 번도 튜닝 안 됐을 가능성)를 신설. 기존 재분석(roll 오차가 27~33°로 커질 때까지 얼로케이터가 포화되지 않았다는 관측)을 "게인이 충분했다면 초기 오차부터 강하게 반응해 잡혔어야 하는데, 느린 초기반응 자체가 게인 부족의 시그니처"로 재해석하면 강한 정황증거가 됨 — 세션로그·트랙보드 어디에도 이 기체의 PID 튜닝 수행 기록이 없다는 점도 뒷받침. H12(이함 트랜지언트)·H13(CW/CCW 추력비대칭)과 배타적이지 않음(H13=교란원, H14=왜 못 버텼는가의 보완관계). **검증비용 0인 것부터:** 현재 `MC_ROLLRATE_P/I/D`·`MC_PITCHRATE_P/I/D`·`MC_ROLL_P`/`MC_PITCH_P`를 PX4 v1.14 기본값과 대조(우선순위 0번, 5분·비행/벤치 불필요). 그다음 단계로 **PX4 온보드 오토튠**(`mc_autotune_attitude_control`) 실행을 제안 — 이 기체는 지상국 텔레메트리 스트리밍 없이도 RPi5↔FC 기존 MAVLink(mavros) 링크만으로 트리거·모니터링·게인 커밋까지 가능(시스템식별·게인계산 자체가 FC 온보드에서 완결되는 기능이라 별도 지상 컴퓨터 불필요) — 단 이 기체는 롤폭주 사고이력이 있어 **반드시 테더(안전줄) 상태로 roll 축 단독부터** 실행할 것. 우선순위 목록에 0번·7번으로 추가, 상세 근거는 그 문서 H14 항목·이력(2026-07-22) 참조.
  그 전 **(2026-07-21, flight04 yaw 스핀 코드결함 수정 — `_publish_pos_setpoint()` 헤딩 명시화)** 아래 flight04 분석에서 특정된 수정후보 ①을 적용. `_publish_pos_setpoint(pos_ned, yaw_ned)`로 시그니처 변경(yaw 필수 인자화) + `fc_bridge/utils/rotation.py::yaw_ned_to_quat_enu()` 신설(순수함수, `vehicle_state_bridge.py`의 디코드 공식과 역변환 관계, pytest 신규 11개로 왕복검증). 8개 호출부 전부 갱신 — STREAMING(MC/FW)·HOLD·FOLLOWING(MC)은 `state.yaw`(현재 헤딩 그대로 미러링, 위치를 현재값으로 스트리밍하는 기존 설계와 동일 원리로 순간점프 원천 차단), TRANSITION_FW phase3(둘 다)는 정렬 완료된 `chi_wp`, TRANSITION_MC keepalive는 `self._end_dir` 베어링, FOLLOWING(FW)은 기존에 버려지던 `L1Guidance.compute()`의 `chi_cmd`(첫 반환값)를 재사용 — 전부 이미 계산되어 있던 값이라 새 추정치 도입 없음. **수정후보 ②(`_step_hold()` 슬루레이트 미적용, 위치 순간점프 1.6m/0.2s)는 이번 범위 밖 — 아직 미수정.** pytest: fc_bridge 55 전부 통과(신규 rotation 11개 포함, flight04 실측 헤딩 -1.704rad 케이스 포함), fc_ros(순수 로직) 65 전부 통과 — `offboard_node.py` 자체는 여전히 rclpy 의존이라 이 WSL엔 실행 검증 불가, **다음 실비행 전 SITL 회귀검증 필수**(기존 관례). 코드 리뷰만 거치고 실비행 미검증 상태로 커밋.
  그 전 **(2026-07-21, "배터리 말고 다른 원인 있나" 사용자 질의 → H13 신설·최유력)** flight04·flight08·flight09 세 비행(서로 다른 배터리 팩)의 `analyze_flight.py` 모터커맨드를 축별로 재비교한 결과 **CW 대각선(motor0 전우+motor1 후좌) 평균 0.45~0.51 vs CCW 대각선(motor2 전좌+motor3 후우) 평균 0.67~0.76로 세 비행 모두 동일방향·유사배수 재현** — 배터리 팩과 무관하게 나타나므로 그동안의 전류과다·고도미달을 배터리보다 이 기계적 비대칭(CCW쌍이 이미 포화근접)이 더 유력하게 설명. `docs/mc_hw_open_hypotheses.md`에 H13으로 신설·최유력 등재, H5(좌/우 프레이밍)는 부정확했던 것으로 정정해 흡수. 다음 지상점검 최우선순위를 "CW쌍 vs CCW쌍 분리 추력벤치"로 변경. 상세는 그 문서 이력(4차) 참조.
  그 전 **(2026-07-21, flight09 — 배터리를 보유 2팩 중 나머지 하나로 교체 후 재비행, "신품 대조군" 해석은 오류로 정정)** flight04 분석 직후 사용자가 배터리를 바꾸고 바로 비행한 flight09(`log_41`, 37.5s) 회수·분석. 최초엔 이를 "신품 배터리"로 오해해 H1/H2(배터리 노후/용량부족)를 반증하는 증거로 기록했으나, **사용자가 정정: 원래 배터리 2개 중 하나를 다 쓰고 기존에 같이 쓰던 나머지 하나로 바꾼 것 — 신품이 아님.** 그래서 실측치(휴지 12.36V, 부하중 8.61V/39.3A로 flight04와 동급 이상 붕괴)는 그대로 유효하나 **"배터리 교체로도 안 고쳐졌다"는 인과적 결론은 철회** — `docs/mc_hw_open_hypotheses.md` H1/H2/H3 상태를 정정 이전으로 되돌림(진짜 신품 팩 실측 전까지 미검증 유지). 비행 자체 관측(목표고도 4.0m 근처도 못 감, dist_bottom 최대 0.99m, t=6.42s 배터리 페일세이프로 AUTO_LOITER 강제진입 재현, 이함 실패 후 지상에서 스틱입력 없이 roll -17.5°까지 요동, 조종사 POSCTL 개입으로 착지)은 그대로 유효. 상세는 `logs/2026-07-21_flight09/notes.md`·`docs/mc_hw_open_hypotheses.md` 이력(3차 정정) 참조.
  그 전 **(2026-07-21, flight04 방금비행 — "2.5m 정지 후 상승, 4m에서 순간 yaw스핀 후 착륙" 사용자 보고 분석 완료, 신규 코드결함 특정)** `tools/flight_logs/collect_new_logs.py`로 오늘자 4개 플라이트(flight01~04, 그중 01·02는 ARM 전 조기종료로 데이터 없음) + 미회수 ulog 5개(id30~34) 회수. flight04(`log_34`, 34s)가 사용자 증언과 정확히 일치 — pyulog 직접 디코드로 **두 현상 모두 원인 특정, 서로 무관한 별개 원인:** ①**2.5m 정지=PX4 배터리 페일세이프.** t=7.24s `battery_warning`+`battery_low_remaining_time` 동시 세팅(전류 28~32A 지속, remaining이 0.81→0.00까지 붕괴) → `user_intention`은 AUTO_TAKEOFF 그대로인 채 `failsafe=1`로 AUTO_LOITER 강제진입(고도 2.6m→1.7m 주저앉아 5초 정체), t=12.04s 배터리 경고 해소와 함께 AUTO_TAKEOFF 재진입·재상승 — 기존 H1/H2(배터리 노후/용량부족) 가설과 정합. ②**4m 지점 yaw스핀=신규 코드결함(`_publish_pos_setpoint` 미설정 orientation).** `fc_ros/fc_ros/nodes/offboard_node.py::_publish_pos_setpoint()`(562~574행)가 `PoseStamped.pose.orientation`을 한 번도 설정 안 함 → ROS2 기본값(단위쿼터니언, ENU yaw=0=NED yaw=90°)이 그대로 발행돼, OFFBOARD 진입 첫 틱(t=18.36s)에 실제 기체 헤딩(-97.64°)과 무관하게 yaw=90° 커맨드가 순간점프로 나감 → 관측된 yaw 268~290° 회전(최대각속도 198.8°/s, t=19.19s), 이 구간 고도 3.85~4.4m로 "4m 지점"과 일치(quat_reset_counter 불변, EKF 리셋 아닌 실제 자세변화 확인). **2026-07-20 flight01 제어상실 사고(`8ea5e35`)의 재발 변종** — 그때 고친 것은 위치(`_mc_pos_ramp` 슬루제한)뿐, yaw는 애초에 어떤 상태에서도 설정된 적이 없어 이 경로가 미수정으로 남아 있었음. 부가로 `_step_hold()`(688~697행)도 `_step_following()`과 달리 슬루제한 없이 목표점을 직접 발행해 위치 순간점프(1.6m/0.2s)도 동반. **이번 세션은 분석만, 코드 미수정** — 실비행 코드라 사용자 확인 후 진행(SITL 회귀검증 관례). 수정후보: ①`_publish_pos_setpoint()`에 목표/현재 헤딩 명시 설정 ②`_step_hold()`에도 슬루레이트 적용. flight03(`log_33`, 8.4s)도 ulog 매칭 완료(미분석). 상세 수치는 `logs/2026-07-21_flight04/notes.md` 참조.
  그 전 **(2026-07-21) `docs/mc_hw_open_hypotheses.md` 신설 — 롤 폭주/추력부족 가설 추적 문서.** "비행→로그수집→분석→평가→비행" 사이클 반복 시 매번 가설공간을 새로 만들지 않기 위해, 아래 재분석에서 나온 H1~H12(전기/기계/제어/결합 4계열)를 표 하나로 고정. **현재 최유력은 H12(이함 트랜지언트가 트리거, 배터리 약화가 회복여유를 깎음).** 다음 비행 분석 후엔 이 문서의 상태 칸만 갱신할 것 — 상세 서술은 여기(트랙보드)에 늘어놓지 말고 그 문서로. 지상검증 우선순위 3개(모터벤치/배터리IR/이함트랜지언트 촬영)도 그 문서에 정리됨.
  그 전 **(2026-07-21, flight02/03 롤 폭주 재분석 — 트리거 시점 특정 + 이전 결론 일부 정정)** 사용자가 이전 세션 결론에 두 가지 재질의: ①flight03 t=4.4s LOITER 진입이 본인 조종기 조작이었다는 주장 ②"`torque_setpoint_achieved`가 낮게 나온 게 롤을 잡으려고 반대쪽 모터를 끄는 현상 자체의 부산물 아니냐"는 의혹. 둘 다 로그로 검증:
  - **①정정: t=4.4s는 조종기 아님.** `vehicle_status.nav_state_user_intention`(사람이 마지막 선택한 모드)+`failsafe` 대조 결과 flight02·flight03 둘 다 초기 LOITER 진입(flight03 t=4.41s, flight02 t=5.31s)은 `user_intention`이 TAKEOFF(17)에 그대로 머문 채 `failsafe`=1로 강제된 **PX4 자체 페일세이프**임을 재확인. **실제 조종기 개입(user_intention이 nav_state와 동시에 POSCTL=2로 전환, failsafe=0)은 flight03 t=9.35s·flight02 t=21.38s** — 둘 다 이미 롤이 60~80°대로 폭주한 **이후** 시점. 즉 최초 폭주 자체는 두 비행 모두 순수 소프트웨어 제어 구간(AUTO_TAKEOFF+페일세이프-LOITER)에서 일어났고 조종기 개입은 원인이 아니라 뒤늦은 반응 — 이 결론(조종기가 폭주를 유발하지 않았다는 것)은 유지, 다만 t=4.4s를 조종기로 지목했던 것 자체가 오류였음.
  - **②확인됨, 이전 결론 인과관계 과장 정정:** `torque_setpoint_achieved`는 roll/pitch/yaw/thrust 통합 플래그라 roll 전용 판단엔 부적합했음 — roll 전용 `unallocated_torque[0]`로 재검증한 결과 **두 비행 모두 roll 오차가 27~33°를 넘어서기 전까지는 얼로케이터가 전혀 포화되지 않고 요구 토크를 100% 달성**(flight03: t=0.5~5.16s roll -5.6°→+28.7° 구간 unallocated≈0, t=5.37s 이후 비로소 벌어짐 / flight02: t=18.5~20.24s roll ±13° 이내 구간 unallocated≈0, t=20.9s 이후 벌어짐). **즉 "얼로케이터가 최대 차동까지 다 썼는데도 못 이김"은 오차가 이미 30° 안팎으로 커진 뒤의 결과이지, 폭주를 일으킨 독립적 증거가 아니었음 — 이전 세션의 프레이밍(포화=하드웨어 이상의 증거) 정정.**
  - **트리거는 더 이른 시점으로 이동, 여전히 미확정:** flight03에서 `vehicle_angular_velocity`(roll rate)가 t≈3.63s(이함 t=2.3s로부터 약 1.3초 뒤)에 잡음 수준(±2°/s)에서 갑자기 16~22°/s로 튀는 게 최초 이상 징후 — 이 순간 요구 토크는 정상 트림 수준이고 포화도 없었음. **정적인 좌우 추력편차라면 이함 즉시부터 트림 이상이 드러났어야 하는데 1.3초 지연 후 발생** — "상시 존재하는 고정 편차"보다 "이함 트랜지언트(다리 비대칭 이탈 또는 RPM 상승에 따른 프롭 진동 급증)"에 무게가 실림(미확정).
  - **수동비행 이함 순간 재검증(사용자 증언 — "이함시 항상 우측, 왼쪽 스틱으로 카운터함" 확인):** `ground_contact` 1→0 전이로 실제 이함 순간 특정 3건 중 log_28은 이함+1.5초 뒤 우측(+5.8°)으로 전환되며 동시에 stick_roll -0.19(왼쪽 카운터) 투입 확인 — 증언과 일치. log_29(log_28 38초 뒤)는 이함 직후부터 왼쪽 스틱(-0.28~-0.35) 선제 유지로 롤이 안정적이었던 것으로 추정(스틱 안 놓았을 때 자연 거동은 미확인). log_26은 뚜렷한 우측 편향 없음(3초 관찰창 밖에서 나타났을 가능성 배제 못함). **이전에 "log_26/29 최대|roll| -20°/-17.1°"로 인용했던 사례는 이함 순간이 아니라 별개의 저고도(AGL -2.4~+0.8m) 스틱중립 흔들림 사건(좌우 둘 다 발생)으로 재분류 — 이함시 우측편향 패턴과 혼동하지 말 것.**
  - **다음 세션 우선순위 갱신:** 기존 (a)모터 벤치 테스트·(c)배터리 교체검증은 유지. **신규 추가: 이함 트랜지언트 자체를 저속촬영/테더로 재현 관찰**(다리 이탈 비대칭 여부, RPM별 `imbalanced_prop_metric` 급증 시점) — 트리거가 정적 편차가 아니라 이함 순간의 동적 사건일 가능성이 이번 재분석으로 커짐. 세부 수치는 각 flight `notes.md` 참조.
  그 전 **(2026-07-20, flight02·flight03·수동비행 5건 — pyulog 정밀분석 완료, 위치기반 슬루레이트 수정과는 무관·별개 원인 확정)** 바로 아래 "그 전" 항목이 로그수집만 하고 미룬 분석을 수행. **결론: STREAMING/FOLLOWING 세트포인트 수정과 무관** — `vehicle_attitude_setpoint`(`q_d`) 디코드 결과 목표 roll은 전 구간 0° 부근(±7°)으로 정상 명령, 세트포인트 순간점프성 결함 없음(flight01류 버그 재현 아님). **실제 원인은 두 갈래 증거로 수렴:** ①**배터리 전압붕괴** — flight02(부하중 최저 8.53V/3S, 셀당2.84V, 전류34.3A)·flight03(9.76V, 26.3A)·수동5건 중 고전류 3건(8.5~9.9V, 17.8~37.7A)까지 이 세션 내내 반복 — 07-18 세션에서 이미 확립된 "전압붕괴→실추력이 명령치 못 미침" 패턴과 동일. ②**flight03에서 목격된 롤 폭주:** t=3.7~6.1s 사이 roll 0°→84°로 폭주(우측이 바닥 쪽), t≈12s 완전전복(-179.8°→+178.8°) — 얼로케이터는 정상 대응해 우측모터(FR/RR)를 최대(1.00/0.85)로, 좌측모터(RL/FL)를 0.00까지 깎아 되돌리려 했으나 **가용한 최대 차동을 다 쓰고도 롤레이트가 178°/s까지 가속, 못 이김**. **가설(미확정, 물리점검 필요): 좌우 모터 실추력 편차 또는 편심 CG가 상시 존재하는데, 정상전압에선 얼로케이터 여유로 보정되다가 전압붕괴로 전 모터 최대추력이 낮아지면 그 여유가 사라져 못 잡는 것** — 수동비행 5건도 고전류(전압붕괴)일수록 roll 흔들림이 큼(저전류 2건은 ±7° 안정, 고전류 3건은 14~20°)이 이 가설과 정합. flight02는 목표AGL 5.0m 중 최대 2.20m 도달 후 정체(AMSL/home_position 기준은 정합적이라 좌표버그 아님, 순수 상승력 부족), t=21.4s POSCTL 전환 직후 roll 84.6°까지 순간 치솟았다 74~76°로 유지(완전전복까지는 안 감). **다음 세션 필수 확인:** (a) 지상에서 모터별 정적 스로틀 스윕으로 좌/우 실추력 편차 실측(프롭 손상·ESC열화 배제/확정) (b) flight02→flight03 사이 `battery_status.remaining`이 0.00→0.64로 뛴 것(팩 교체 여부 미확인 — 사용자 확인 필요) (c) 배터리를 완전 충전·정상 상태로 교체한 뒤 동일 재현 여부(전압가설 검증). notes.md(`logs/2026-07-20_flight02·flight03·manual/`)에 세부 수치 전부 기록. ulog 11개(id19~29) 커밋·push는 아래 "그 전" 항목에서 이미 완료.
  그 전 **(2026-07-20, flight02·flight03·수동비행 5건 — 위치기반 슬루레이트 수정 배포 직후 재비행, 로그수집만 완료·분석 전)** RPi 재빌드 배포 직후 사용자가 재비행: **오프보드 2건**(`flight02`: `transition_alt:=5.0`, notes "thtrotle loss, cant incline" / `flight03`: 동일 조건, notes "throthle loss, cant incline" → "flip over") + **수동비행 5건**(RC only, ROS/record_flight.sh 미사용 → launch.log/rosbag 없음, `logs/2026-07-20_manual/`에 ulog만 보관). **사용자 보고 특이사항(미분석, 원문 그대로 기록):** ① 추력이 부족한 것처럼 잘 못 날고 빌빌김 ② 바닥에서 뜰랑 말랑할 때 항상 기체 우측이 바닥에 닿아있다가 뒤집힘(flip). **이 재비행은 바로 아래 항목(STREAMING/FOLLOWING 위치기반 슬루레이트 수정) 배포 직후 진행된 것** — 증상이 그 수정과 관련 있는지, 아니면 별개의 하드웨어 문제(추력·좌우 불균형·이전 flight01 사고로 인한 물리적 손상 가능성 등)인지는 **전혀 분석하지 않음, 다음 세션 몫**. ulog 11개(id19~29, FC에서 직접 회수) 전부 `logs/2026-07-20_flight02/`·`logs/2026-07-20_flight03/`·`logs/2026-07-20_manual/`에 저장해 git 커밋·push 완료. launch.log상 두 비행 모두 `pos_ned[2]`(h_up)가 이륙 내내 -9~-12 부근(원점 기준 크게 마이너스)으로 목표 5.0m에 전혀 못 미침 — "빌빌긴다"는 관찰과 raw 텔레메트리 수치가 정합적(원인 미분석).
  그 전 **(2026-07-20, flight01 착륙 직후 WiFi 장기끊김 — brcmfmac 커널버그로 근본 메커니즘 확정 + 완화조치 적용)** 사용자가 "일전 비행에도 연결끊김이 있었다, 원인분석하라"고 요청 → `wifi_watch.log` 대조 결과 착륙(16:19:29) 약 1분 뒤부터 wlan0 `carrier=0`가 **8분 25초**(16:20:32~16:28:57) 지속(재부팅 없이, 시스템은 계속 살아있었음). 이 구간 커널로그에 `brcmfmac: brcmf_set_channel: set chanspec 0x____ fail, reason -52`가 11초 간격으로 4개 채널을 순환하며 **164회** 반복 — 웹서치로 확인한 결과 **RPi5 브로드컴 WiFi 드라이버의 알려진 미해결 커널 버그로 확정**([raspberrypi/linux#6049](https://github.com/raspberrypi/linux/issues/6049)). 기술적 원인(이슈 코멘트 r41k0u의 GDB/SWD 디버깅): disconnect-reconnect 사이클 중 regulatory-domain 플래그가 `restore_custom_reg_settings`에서 stale한 `orig_flags`로 복원되며 채널설정이 계속 거부됨 — **일단 한번 끊기면 이 버그 때문에 재연결이 평소(수십 초)보다 훨씬 오래 걸리는 것이 실비행 중 "안 붙는다"는 체감의 핵심 메커니즘.** 최초 트리거(왜 16:20:32에 처음 끊겼는지)는 여전히 미확정(신호거리/RC 2.4GHz 간섭/전원 sag 후보 경합, `vcgencmd get_throttled`=0x0이라 하드웨어 감지 언더볼트는 아님). **완화조치 2건 적용(사용자 실행):** ①`sudo iw dev wlan0 set power_save off`+udev 영구화 — 적용 확인됨(재부팅 로그에 `power save disabled`) ②`/etc/modprobe.d/brcmfmac.conf`에 `options brcmfmac roamoff=1 feature_disable=0x282000`(이슈에서 여러 사용자가 이 조합으로 증상 해소 보고) — 파일 반영·재부팅 완료, 부팅로그에 "Unknown parameter" 경고 없어 파라미터 자체는 유효하게 받아들여진 것으로 추정되나 `feature_disable`은 이 드라이버 빌드에서 sysfs 비노출이라 값 직접 확인은 못함(root 권한 필요). **이 버그 자체는 업스트림 미해결**(정식 패치 없음) — 적용한 두 조치는 커뮤니티에서 검증된 우회책. 다음 비행 후 `~/wifi_watch.log`로 장기(수 분 단위) `carrier=0` 재발 여부가 최종 검증. 상세는 Claude 메모리(`project_rpi5_tailscale_wifi_drops.md`)·`docs/session_log.md` 2026-07-20(WiFi 근본원인) 항목 참조.
  그 전 **(2026-07-20, flight01 제어상실 사고 — 근본원인 규명+수정, 아래 "기록 전용" 세션의 후속)** 직전 실비행(`logs/2026-07-20_flight01/`) 중 상승 직후 기체가 순간 제어를 잃고 롤/피치/요가 급변, 조종사가 RC로 수동 회수한 사고의 원인을 아래 "기록 전용" 세션이 회수해둔 ulog로 규명. `log_18` 분석 결과 **STREAMING(321행)/`_step_following()`(775행)이 FW 전용 70m lookahead(flower-pattern 회피용 pursuit 유도)를 MC에도 그대로 적용**해, 짧은 경로(~12m)에서 lookahead가 항상 경로 끝점(WP1)으로 클램프 → OFFBOARD 진입 첫 순간 **실제 위치(클라이밍 중 수평드리프트+고도오버슈트로 (-4.4,1.2,7.3m))와 무관한 절대좌표 `(0,0,4.0)`·yaw 90°로 순간점프 발행** → PX4가 급격한 자세보정을 시도해 roll -16°/pitch -30.8°/yaw rate 최대186°/s(t=11.3~13.0s, 아래 "기록 전용" 항목의 16:19:03 KST 추정 시각과 일치). EKF `quat_reset_counter` 불변 확인 — 센서/EKF 결함 아님. **수정(1차, 이후 사용자 지적으로 정정됨):** 처음엔 MC를 속도 세트포인트로 전환했으나, **사용자가 "최종 VTOL 기체는 위치기반으로 동작하고 이 MC 테스트기체는 그 최종기체 제어로직을 검증하기 위한 것이니 MC만 속도기반으로 바꿀 이유가 없다"고 지적** — 타당한 지적으로 즉시 위치기반 유지로 재수정. **최종 수정:** STREAMING은 MC도 위치 setpoint를 발행하되 매 틱 **현재위치를 그대로** 스트리밍(OFFBOARD 확정 순간 PX4가 이어받는 값이 항상 실제위치와 일치). FOLLOWING은 기존 FW lookahead 목표점 계산은 그대로 두되, 그 목표로 향하는 위치 setpoint를 `self._mc_pos_ramp`로 슬루레이트 제한(≤`v_approach`=5.0m/s, 기존 파라미터 재사용)해 점진 접근시켜 순간점프를 제거 — FW 경로 자체는 완전히 미변경. `fc_bridge`(rclpy 비의존) 순수 로직으로 사고 시점 실측오차(4.5m+3.3m)를 넣어 시뮬레이션 — 틱당 최대 이동량 0.5m로 약 1.2초에 걸쳐 수렴함을 확인(수정 전엔 즉시 4.54m+ 순간점프). **`offboard_node.py` 자체는 rclpy 의존이라 이 WSL엔 실행 검증 불가 — 다음 실비행 전 SITL(`gz_x500`) 회귀검증 필수.** **커밋(`8ea5e35`)·push·RPi 반영 완료** — RPi5 `git pull` 후 `docker exec fc colcon build --packages-select fc_ros` 재빌드 성공, 설치본이 소스와 diff 일치 확인(2026-07-18의 "빌드 미반영" 재발 없음). 아래 "기록 전용" 항목의 미확인사항 중 (a)id18 분석은 완료, (b)id16/17 정체·(c)RC 오버라이드 정확시각·(d)`MIS_TAKEOFF_ALT`·(e)vtol-실기체 연관성은 **제어상실의 직접원인이 소프트웨어 세트포인트 버그로 확정된 이상 우선순위 낮음**(참고용으로만 남김). 상세는 `docs/session_log.md` 2026-07-20(flight01 제어상실 — 근본원인 규명+수정) 항목.
  그 전 **(2026-07-20, flight01 — 오프보드 전환 직후 제어상실 사고, 기록 전용)** `climbing_reached` 허용오차 수정 후 실비행(`transition_alt:=4.0`, 삼각 왕복). launch.log만 보면 ARM(16:18:50)→이륙→CLIMBING(4.0m, 16:19:00)→**OFFBOARD 확인(16:19:03)**→FOLLOWING→WP1 도달→LANDING→착륙완료(16:19:29, 총 39초)로 "정상 완주"처럼 보여 Claude가 대화 중 **"첫 오프보드 성공"으로 잘못 판단**했으나, **조종사(사용자) 직접 증언으로 정정: 수직 상승 완료 직후(오프보드 전환 시점 전후로 추정, 16:19:03 부근) 기체가 순간 제어를 잃고 북서쪽으로 픽 쓰러지며 roll, 즉시 조종사가 RC로 조종권 회수.** 즉 **16:19:03 이후 log의 FOLLOWING→WP1→LANDING 시퀀스(26초)는 실제 자율비행 수행 기록이 아닐 가능성이 높음**(위 "마지막" 항목에서 원인 규명 완료 — 위치 세트포인트 순간점프에 대한 급격한 자세보정이 실체였음). 데이터 가용성: ①**ulog 회수 완료(갱신: 최초엔 실패했으나 사용자가 Pixhawk 전원을 재연결해줘서 재시도 성공)** — FC 로그 목록에 오늘자 3건 확인: id16(07:18:28 UTC, 156,843B)·id17(07:18:28 UTC, 156,277B, id16과 같은 초에 별도 로그 — 재시동/재arm 추정, 원인 미확인)·**id18(07:19:30 UTC, 1,729,984B — 이 비행의 본 로그, 위 "마지막" 항목에서 분석 완료)**, UTC+9=KST로 정확히 이 비행 시각대(16:18~16:19 KST)와 일치. ②**rosbag엔 `/mavros/imu/data` 포함**(자세 쿼터니언·각속도) ③**`/fc_ros/override` 토픽이 이번 rosbag엔 기록 안 됨**(설정된 11개 중 10개만 실제 기록 — RC 오버라이드 개입 정확 시각은 여전히 미특정, 낮은 우선순위) ④**wifi_watch.log 참고자료(인과관계 없음으로 사실상 정리):** 게이트웨이 ping 무응답이 비행 시작 전 16:15경부터 이미 간헐 발생 중이었고, wlan0 `carrier=0`은 16:20:38~16:26:14로 **사고 시점(16:19경)보다 약 1분 뒤에 시작** — 타이밍상 사고와 직접 겹치지 않음. flight01 폴더(`logs/2026-07-20_flight01/`, ulog 3개 포함)·`notes.md`에 원본 전부 보존.
  그 전 **(2026-07-20, climbing_reached 허용오차 도입)** `fc_bridge/execution/state_logic.py::climbing_reached()`의 천이고도 도달판정을 단측 임계값(`AGL >= transition_alt`)에서 ±0.5m 허용구간(`abs(AGL-transition_alt)<=alt_tol`)으로 변경 — 목표고도 바로 아래(예 -0.1m)에 정착하면 CLIMBING이 무한대기하던 문제 대응(사용자 실비행 보고). 아래 flight09 진단에서 실측된 3.63m/목표4.0m 갭(0.37m)도 이 허용오차 안에 들어가 그 케이스를 구제할 가능성이 있음. **N,E(수평)은 의도적으로 제외** — CLIMBING 중 수평은 PX4 AUTO.TAKEOFF가 자체 관리해 목표 N,E가 없고, 비-RTK GPS 수평오차가 통상 0.5m를 넘어 수평까지 조건화하면 CLIMBING 영구대기라는 더 심각한 회귀를 유발할 위험이 컸음. `fc_ros/test/test_offboard_node.py` 경계값 케이스(하한/상한/직전값) 추가·기존 케이스 갱신 완료(WSL에 pytest 미설치라 순수 스크립트로 수동 재현 검증). **PX4가 목표고도 도달 전 스스로 `AUTO.LOITER`로 복귀하는 근본원인(아래 flight09 진단)은 미해결** — 이번 수정은 그 위에서 벌어지는 증상(허용오차 내 소폭 미달)만 완화, 더 크게 미달하면 여전히 무한대기. **다음 실비행에서 CLIMBING→STREAMING 정상 트리거 확인 필요.** 상세는 `docs/session_log.md` 2026-07-20 항목.
  그 전 **(2026-07-18, 사용자 현장 요청 — 원격 리빌드 + flight09 진행 + 신규 고도미달 진단)** 사용자가 비행장에서 RPi5 터미널 접근 불가 상태로 "리빌드해달라" 요청 → Claude가 SSH로 직접 처리. `sudo docker exec`가 비밀번호 요구로 막혔다가 사용자가 `suri`를 `docker` 그룹에 추가해줘서 해결, `docker exec fc colcon build --packages-select fc_ros` 성공 실행·`install/`이 최신 소스와 diff 일치 확인. **이 리빌드가 바로 위 "실비행 8건 전수분석"이 지목한 근본원인(colcon 빌드가 2026-07-07 이후 정체돼 구버전 AMSL-미적용 코드가 실려 있던 문제)을 해소한 것과 동일 조치** — 별도 백그라운드 세션이 flight01~08을 분석하는 동안 이 세션은 그 원인을 실제로 고친 셈(교차참조, 시간 순서상 이 세션의 리빌드가 먼저 실행됨). 리빌드 직후 사용자가 `flight09` 비행(`transition_alt:=4.0`, 삼각 왕복) 진행 → **rosbag+launch.log 수집 완료**(`logs/2026-07-18_flight09/`, RPi 로컬에만 존재, 아직 git 미커밋 — 다음에 로그 커밋할 때 포함할 것). ulog(FC상 id=13, 09:49:06 UTC, 3.3MB)는 최초 `--list`엔 잡혔으나 실제 다운로드 2회 실패(heartbeat 없음) → 재확인 결과 `/dev/ttyACM0`에서 3초간 0바이트 수신(포트 점유 프로세스 없음, 컨테이너 안팎 다 확인) — **FC가 USB enumeration은 유지한 채 완전히 꺼진 상태로 판단**(배터리 분리 후 USB 5V만으로는 FMU 미기동 가능성). 사용자 확인 후 재시도 보류(바쁨) — **ulog 미회수 상태로 세션 종료, 필요시 `python3 tools/flight_logs/pull_ulog.py --log-id 13 --out logs/2026-07-18_flight09/`로 나중에 회수**. **flight09 진단(rosbag 기반, `/mavros/state`+`/mavros/local_position/pose` 직접 디코드):** 사용자가 "목표 4.0m인데 3.6m에서 천이 명령이 안 나갔다" 보고 → 분석 결과 `climbing_reached()` 판정 로직·`home_amsl` 계산(51.8m AMSL = 47.8+4.0, 정확) 둘 다 문제 없음(AMSL 버그는 이 비행에서 이미 해소돼 있었음, 위 리빌드 덕분). **실제 원인은 PX4 자체 거동:** `AUTO.TAKEOFF` 진입(t=1.68s) 후 **겨우 4.66초 뒤(t=6.34s) PX4가 스스로 `AUTO.LOITER`로 복귀**(`pose.z` 전체 비행 최댓값 3.63m, t=44.00s) — offboard_node 재요청 로그 없이 t=10.68s에 `AUTO.TAKEOFF` 재진입(조종사 RC 수동 재이륙 추정) → t=14.68s 다시 `AUTO.LOITER`로 복귀 후 **착륙까지 42초간 OFFBOARD 모드 진입 자체가 한 번도 없었음**. `_step_climbing()`(`offboard_node.py:418-429`)은 `pos_ned[2]` 순수 폴링만 하고 PX4가 AUTO.TAKEOFF를 이탈했는지는 감지·대응하지 않는 설계 갭 — 그래서 소프트웨어는 조용히 CLIMBING에 무한 대기, "천이 명령 미하달"로 관측됨. **근본원인 미확정(후속 필요):** PX4 파라미터 `MIS_TAKEOFF_ALT`가 CommandTOL 요청 고도와 별개로 이륙완료 기준을 정하고 있을 가능성이 유력하나, FC 전원이 꺼져 파라미터 확인 못함 — 위 ④ 배터리 전압붕괴(같은 팩이 계속 쓰였다면 flight09에도 재현 가능)도 배제 못함(flight09 rosbag엔 `/mavros/battery` 미기록이라 이 비행 자체로는 확인 불가, ulog 회수해야 확정). **다음 확인 시:** FC 전원 재연결 후 ① ulog id=13 회수 ② `MIS_TAKEOFF_ALT` 파라미터 조회 ③ **(2026-07-20 갱신)** `climbing_reached()` 허용오차(±0.5m) 도입으로 이 비행의 갭(0.37m)은 구제되나, `_step_climbing()`에 AUTO.TAKEOFF 이탈 자체를 감지·대응하는 로직 추가 여부는 여전히 미결정(사용자 판단 필요) — 위 "마지막" 항목 참조.
  그 전 **(2026-07-18, 실비행 8건 ulog 전수분석 — "왜 arm이 한번에 안 되는지/climb 명령-실제 괴리/다른 문제" 진단 요청 답변)** `flight01~08` 전부 목표고도(4.9~6.0m) 미달로 종료된 원인을 pyulog로 전수분석. **①근본원인(전체 8건 공통):** 8건 전부 `vehicle_command`(NAV_TAKEOFF) `param7`이 `transition_alt` 원값(4.9~6.0m) 그대로였고 `home_position.alt`(실측 AMSL)는 18.5~22.2m — 이는 `9451861`(작업 H-2, 2026-07-12) 이전의 구버전 버그와 동일 증상. RPi SSH 확인 결과 소스(`~/drone_ws/src/suridoksuri`, HEAD=`07681d3`)는 `9451861` 이후로 최신인데, **설치본** `~/drone_ws/install/fc_ros/lib/python3.10/site-packages/fc_ros/nodes/offboard_node.py`는 여전히 구버전(`takeoff_request_fields(self._transition_alt)` 1-인자 호출, AMSL 미적용 로그 문구) — 8건 launch.log와 문구 일치. 빌드 산출물 `install/fc_ros/lib/fc_ros/offboard_node` mtime=1783389881(**2026-07-07 11:04 KST**, `7414c1d` 직후 — `000f478`·`9451861`보다도 이전) → **`colcon build --packages-select fc_ros`가 2026-07-07 이후 RPi에서 한 번도 재실행 안 됨**(소스는 두 번 git pull됐지만 빌드 미반영). fc_bridge는 PYTHONPATH 참조라 무관, colcon 빌드되는 fc_ros install/만 stale. **결과경로 A(6/8: 01,03,04,05,06,07):** navigator가 arm 직후 "Already higher than takeoff altitude" → `actuator_motors` control[0..3] 비행 내내(flight01 약 11초) ≤0.002로 사실상 무추력 → `COM_DISARM_PRFLT`(10.0s)로 t≈11.05s "Disarmed by auto preflight disarming". **결과경로 B(2/8: 02,08):** 동일한 잘못된 명령에도 `takeoff_state`가 FLIGHT까지 진행·모터 램프(0.4~1.0)됐으나 최고 AGL 겨우 1.55~1.57m(flight02/01, 4.9~6.0m 목표 대비), flight08은 순증가 0.00m(요잉만 2.1→2.7→3.0→-3.1rad 스핀)로 종료 — 둘 다 이륙 5~6초 후 "Low battery" 경고 + 이후 반복 "Disarming denied: not landed"(조종사 RC 수동회수 정황), flight02="Disarmed by landing", flight08="Disarmed by RC switch". 8건 전부 `vehicle_control_mode.flag_control_offboard_enabled`가 단 한 번도 True 안 됨 — `fc_bridge/execution/state_logic.py::climbing_reached`(AGL 기준, AMSL 버그와 무관)가 AGL 미달로 계속 불충족돼 OFFBOARD 자체가 요청된 적이 없음. 사용자가 별도로 언급한 "고도 도달 후 offboard 실패"는 이 8건 중에는 재현되지 않음(전부 목표고도 미도달) — 이 데이터셋 밖의 다른 비행 사례로 추정. **⚠안전 경고(중요):** RPi가 현재 분열 상태 — fc_bridge 소스는 `home_amsl` 필수 2-인자로 바뀌었는데 fc_ros install/은 여전히 1-인자로 호출 중. 지금 재비행하면 컨트롤 타이머 콜백에서 `TypeError`(암 상태에서 발생 가능)로 기존 버그보다 더 위험한 상황이 될 수 있음 — **다음 비행 전 RPi `colcon build --packages-select fc_ros` 필수.**
  **②arm 자체는 8건 모두 정상("arm이 한번에 안 됨" 미재현):** launch.log(ROS) ↔ ulog `vehicle_command_ack`(cmd=400/ARM, result=0=ACCEPTED, 8건 전부)를 대조 — 8건 모두 단 1회 시도로 즉시 성공(launch 후 1~3초: 01=1.0s, 02=1.0s, 03=3.0s, 04=1.0s, 05=1.0s, 06=1.0s, 07=3.0s, 08=1.0s), arm 시점 GPS 전부 fix_type=4/DGPS·위성 15~17개·eph 0.7~1.2m로 양호. 즉 이 8건(같은 세션 내 FCU 재부팅 없는 연속 재실행)에서는 사용자가 말한 "arm이 한번에 안 됨/40~60초 후 arm" 증상이 전혀 재현 안 됨. 다만 `fc_ros/fc_ros/nodes/offboard_node.py::_step_arm_takeoff`(약 371~414행)에 실제 설계 갭 존재 — (a) 384행 `self._arm_cli.call_async(req)` 호출 후 future 결과를 확인하지 않고 `_arm_sent=True`로 원샷 래치(383~387행) → PX4가 실제로 거부(콜드스타트 preflight/EKF/GPS 미준비 등)하면 재시도·에러진단 없이 영구 대기, 전체 ROS2 스택 재기동만이 유일한 탈출구(`arm_sent` 리셋). (b) 392행 `if self._home_amsl is None:` 이후 `/mavros/home_position/home` 수신을 무기한 대기(2초 throttle 경고만, 392~397행) — GPS+EKF 수렴 필요. (참고: flight03/04/07에서 `/mavros/cmd/arming 서비스 없음` 경고 있었으나 1~4초 내 자연 해소, 이번 8건에서는 원인 아님). **최유력 가설(이 데이터로 확정 불가, 후속 필요):** 40~60초 지연은 FCU 전원 재투입 직후 **콜드스타트**(MAVROS 미연결·GPS/EKF 홈포지션 미확정 상태)에서만 나타날 가능성 — 이번 8건은 전부 FCU 재부팅 없는 warm 재실행이라 이 케이스가 없음. **후속 권고(미구현, 기록만):** arm 서비스 호출에 결과확인+bounded retry/backoff 추가, home_position 대기 중 주기적 로깅 추가(콜드스타트 스톨을 셸에서 바로 보이게).
  **②-보정(사용자 재질의로 정정): "arm 성공 여부가 climb 성공을 가른다"는 최초 설명은 틀렸음 — 데이터로 반박됨.** flight01·05·06은 flight02·08과 arm 소요시간이 동일(1.0s, 서비스경고 없음)인데도 climb 실패. 실제 판별자는 PX4 내부 `vehicle_constraints.want_takeoff` 플래그: 실패 6건(01,03,04,05,06,07)은 t≈1.1~1.3s에 `False`로 고정된 뒤 비행 끝까지 단 한 번도 안 바뀜(모터 0.000~0.002 그대로, 완전 무추력). 반면 02는 t=3.13~4.13s, 08은 t=5.12~6.12s에 정확히 **1.0초간만** `True`로 바뀌었고, 그 창 안에서 실제 모터 출력이 급상승(flight02: t=2.874s [0,0.05,0.02,0.03]→t=3.979s [0.17,0.41,0.35,0.23])하며 "Takeoff detected"가 그 직후 찍힘. 이 1초 펄스는 두 번째 arm/CommandTOL 명령이 아니고(`vehicle_command`/`_ack`에 건별 정확히 1회씩만 존재 확인) 조종스틱 입력도 아님(flight02는 `manual_control_setpoint.throttle`이 비행 내내 -1.00 고정, `sticks_moving=False`) — PX4 이륙/착륙감지 상태머신 내부에서 벌어진 일이며 대응하는 로그 메시지가 전혀 없어 ulog만으로는 정확한 트리거를 특정 못함(펄스 시각이 3.13s/5.12s로 고정 타이머와도 안 맞음). **결론: "가끔 climb 성공"은 운(랜덤)이 아니라 이 근본버그 위에서 벌어지는 PX4 내부의 좁은 레이스 상태이며, arm 소요시간·서비스경고 스팸과는 무관.** 원인 규명에는 이 PX4 정확한 펌웨어 버전의 takeoff/land_detector 소스 대조가 필요(후속, 미착수) — 다만 실용적으로는 AMSL 수정이 배포되면 "이미 도달함" 오판 자체가 없어지므로 이 레이스 조건도 함께 사라질 가능성이 높음.
  **③AMSL 수정이 배포돼도 남는 잔여 리스크(과거 VTOL 비행에서 수정 적용 후에도 문제 있었던 이유 설명):** `home_position.alt`(home_amsl로 쓰임)는 안정적이지 않음 — 실측: flight02는 arm 후 20초간 18.47→19.27→20.31→21.34m(+2.9m 드리프트), flight08은 22.25→21.84→**17.56**(t=5.85s, −4.7m 급변)→21.24m(t=33s)로 요동. `EKF2_HGT_REF=1`(GPS 우선)+GPS 수직정확도(`epv`≈1.2~2.2m) 조합이 arm 직후 10~30초간 이 정도 드리프트를 정상적으로 만들어냄. `_step_arm_takeoff`가 CommandTOL 발사 순간 `self._home_amsl`을 **단발 스냅샷**으로 읽는데, 벤치테스트 특성상 `transition_alt`를 낮게(4.9~6.0m) 유지 중이라 이 3~5m 드리프트가 목표치의 60~100%를 잠식 — **AMSL 수정이 정확히 배포돼도 EKF/GPS 높이기준 노이즈만으로 오차 여유가 소진될 수 있음**(과거 VTOL 비행에서 수정 적용 후에도 climb/고도 문제가 있었다는 사용자 언급을 뒷받침하는 근거). **후속 권고(미구현, 기록만):** (a) home_position이 N회 연속 허용오차 이내로 안정될 때까지 CommandTOL 보류, 또는 (b) 실기체 조건 허용 시 `transition_alt`를 더 크게 잡아 이 잔여노이즈 비중을 낮춤.
  **④추가 기여 요인 — 배터리 전 세션 미교체:** `battery_status.remaining`이 flight01=0.51→flight08 시작=0.38→flight08 종료=**0.00**로 단조감소(3S팩, `BAT1_N_CELLS=3`, 휴지전압 11.1~11.4V — flight01 시점부터 이미 상당히 소모됨). `BAT_LOW_THR`=0.15/`BAT_CRIT_THR`=0.07이고 이륙에 성공한 두 비행(02,08) 모두 이륙 5~6초 후 "Low battery" 경고 발생. 이륙에 성공한 두 비행의 실측 상승률은 0.1~0.15m/s로 `MPC_TKO_SPEED`=1.5m/s 설정 대비 **약 10배 느림** — AMSL 버그와 별개로(버그 하나만으로는 10배 저하 설명 안 됨) 8회 연속 비행으로 소모된 배터리가 추력을 갉아먹은 것과 부합. **정량 확인(사용자 재질의로 추가분석): 전압강하가 직접 원인.** `battery_status`(3S팩) 실측 — flight02: 휴지전압 11.13V → 부하 중 최저 **8.40V**(t=15.6s, 2.80V/cell), 최대전류 31.4A. flight08: 휴지전압 11.39V → 최저 **7.50V**(t=18.4s, **2.50V/cell**), 최대전류 33.5A. 두 비행 모두 전류 15A 초과 구간이 전체 샘플의 28~65%(flight02: 68/241, flight08: 110/170)를 차지 — 순간 스파이크가 아니라 비행 대부분에 걸친 지속적 전압붕괴. flight02는 전류 4→29A로 치솟는 t=4~14s 구간 동안 오히려 고도가 -2.69→-3.15m로 **하강**(모터는 실제로 구동 중인데 추력이 중량을 못 이김) → 전류가 3~5A로 내려가고 전압이 10.5~11V로 회복된 이후(t≈20s+)에야 느리게 순상승. 즉 커맨드는 정상적으로 실추력을 요구했으나(control 출력 0.4~1.0 확인) **배터리가 그 전류를 전압붕괴 없이 공급 못해 실제 전달 추력이 명령치에 크게 못 미친 것** — 소프트웨어 로직 문제가 아니라 배터리 내부저항/노화 또는 기체 대비 팩 용량 부족 가능성. **운영 권고:** 실비행 세션 중에도(세션 사이뿐 아니라) 배터리를 교체/충전할 것. 팩 노후도·내부저항 점검 권장.
  그 전 **(2026-07-18)** 사용자 요청으로 Claude가 RPi5(Tailscale `100.67.27.83`, hostname `doksuri`, 계정 `suri`)에 SSH로 직접 접속해 비행 로그를 조사 → **문서에 기록 없이 07-17(6회)·07-18(8회) 총 14회 실비행이 이미 진행돼 있었음** 발견(`vehicle_type:=mc`, `transition_alt` 5.0~6.0m, 삼각 왕복 웨이포인트). `record_flight.sh` 사용으로 rosbag+launch.log는 14회 전부 존재. FC(Pixhawk)에서 직접 회수한 ulog는 **11개 전부 오늘(07-18) 새벽~오전 것뿐**(UTC 03:06~04:17) — 이 FC에 07-17 이전 로그가 전혀 없어 **SD카드가 새것이거나 다른 Pixhawk 유닛일 가능성**(미확정). 그중 8개(id3~10)는 오늘 `flight01~08`과 시각이 정확히 1:1 매칭돼 각 폴더에 편입 완료, 앞선 3개(id0~2, KST 12:06/12:09/12:59)는 `record_flight.sh` 쓰기 전 로그라 대응 rosbag/launch.log 없음 → `logs/2026-07-18_unlogged/`에 "비행기록 부족함"으로 별도 보관. `flight08` launch.log 확인상 ARM→CommandTOL 이륙(6.0m)→CLIMBING 정상 진입, 텔레메트리 정상 수신. **notes.md(비행조건 외 관찰/결론)는 14회 전부 비어있음** — 조종사가 아직 안 채움.
- **확인됨 (2026-07-18, 사용자):** 이 "부활한 MC 기체"는 ✈ vtol-실기체 트랙의 결함 기체와 **다른 물리 개체** — Pixhawk·ESC 모두 신규 유닛으로 교체, 외형 프레임만 이전과 동일해 보이는 것뿐. 따라서 ✈ vtol-실기체의 결함과 **무관**(해소 여부와 별개 사안). FC에 07-17 이전 ulog가 전혀 없던 이유도 이걸로 설명됨 — 신규 Pixhawk이라 로그 저장소 자체가 새것.
- **미확인 (다음 세션):** 07-17·07-18 14회 비행의 실제 결과(관찰/결론) — notes.md 채우기 필요(별도 작업으로 진행 중)
- **로그 인프라 버그 2건 발견 (2026-07-18, main-code 트랙에서 상세):** ulog 자동회수가 지금까지 한 번도 성공한 적 없었음 — (a) RPi 호스트에 pymavlink 자체가 미설치 (b) `record_flight.sh`를 컨테이너 `fc` 안 root로 실행해 `logs/` 하위 폴더가 root 소유가 되어 suri 계정 쓰기 불가. 상세·수정 상태는 🔧 main-code 트랙 참조.
- **실비행 중 RPi5 tailscale/WiFi 반복 끊김 진단 (2026-07-18~19):** 07-18 재비행에서도 수차례 끊김 재발 → 사용자가 비행 중단. SSH 원격 진단 결과 근본 원인은 **RC 2.4GHz ↔ WiFi 핫스팟 2.4GHz 동일대역 간섭**으로 유력 판단(5GHz는 GPS 간섭 우려로 문서 권고상 못 씀 — 대역 전환으로 회피 불가). 부수적으로 RPi5 브로드컴 WiFi 드라이버의 **전원관리(power save) 활성 상태**도 확인돼 있음(`sudo iw dev wlan0 set power_save off` 처방, 사용자 실행 확인 아직 안 됨). 다음 비행부터 대조 가능하게 **`~/wifi_watch.log`**(RPi5, 5초 간격 wlan0 carrier/ping 기록, `~/scripts/wifi_watch.sh` + crontab `@reboot`로 상시 실행) 배포 완료. 상세 진단 경과·오진단 이력은 Claude 메모리(`project_rpi5_tailscale_wifi_drops.md`) 참조.
- **RPi5 USB-C 전원 협상 완화 (2026-07-19):** 비행 중엔 공식 5V/5A PD 어댑터를 못 쓰고 BEC 등 비-PD 5V 전원을 쓰므로, EEPROM `PSU_MAX_CURRENT`를 기본값(5000, 미설정 시)에서 **1600으로 변경**(사용자 직접 실행·확인 완료) — 5A 강제 협상 없이 최소 협상으로 부팅 진행. 상세는 Claude 메모리(`project_rpi5_usbc_power_psu_max_current.md`) 참조.
- **주의:** AUTO.TAKEOFF는 GPS 락 필수(실내/벤치 불가) · 실기체 FC는 PX4인지 확인부터 · **비행 전 SD카드 삽입 확인 (2026-07-07 이걸로 비행 실패 이력)** · **비행 중 tailscale/WiFi 간헐적 끊김 — 근본 메커니즘은 brcmfmac 커널버그(#6049)로 확정, 완화조치 적용됨(위 "마지막" 참조) — RC/Pixhawk 텔레메트리 링크와는 무관, SSH/tailscale 원격접속 편의 채널만의 문제** · flight01(2026-07-20) 제어상실 사고는 원인 규명·수정·RPi 반영 완료 — **단 SITL 회귀검증 전까지 재비행 금지** · **flight02/03·수동비행 5건(2026-07-20) 분석 결과 롤 폭주(우측 접지 후 전복) 원인 미확정 — 모터별 실추력 편차 지상점검·배터리 교체검증 전 재비행 시 동일 전복 재현 위험, 위 "마지막" 참조**
- **참조:** **`docs/mc_hw_open_hypotheses.md`(반복되는 이상현상 열린 가설 목록 — 새 비행 분석 후 여기 상태만 갱신)** · **`docs/mc_flight_procedure.md`(비행 절차 전체 — 로깅 사용/미사용 둘 다, "절차는?" 질문엔 이 문서 그대로 출력)** · `flight_plan.md` 작업 H·SITL-5 섹션 · `pixhawk6c_rpi4_integration_guide.md` · `fc_ros/fc_ros/nodes/offboard_node.py`(`_step_arm_takeoff`/CLIMBING·OFFBOARD 상태머신) · `logs/2026-07-18_flight01~08`·`logs/2026-07-18_unlogged/`(이번에 회수한 원본)

### 🔧 main-code — ⏸ 대기

- **내용:** fc_ros/fc_bridge 기능 개발 및 공용 인프라. **작업 G(비행 로그 수집·분석) [코드] 완료·검증(V2)·커밋**
- **마지막:** **(2026-07-21) `tools/flight_logs/analyze_flight.py` 신설 — 비행 로그 표준 진단 스크립트.** flight02/03 사고분석 세션에서 매번 pyulog 코드를 새로 짜며 세션 컨텍스트를 소모하던 것을 대체. `<플라이트폴더>/` 하나 넘기면 고도/모드권한타임라인(조종기 vs 페일세이프 자동판별)/자세·각속도 이상조짐/축별 얼로케이터 포화시점/모터별 커맨드/배터리/실패감지기/이함순간을 `analysis_auto.md`(+`--json`)로 뽑아준다 — 해석은 안 하고 사실만. 다루지 않은 토픽은 목록으로 출력해 누락 방지. 순수 함수 22개 신규 pytest 케이스 추가, 전체 61개 통과(이 WSL에 pytest 설치함, 기존 "미설치" 메모는 아래 07-18 항목 기준 구버전 상태). 사용법은 `tools/flight_logs/README.md` §4. **다음:** 이 스크립트를 로그수집 서브에이전트/스킬과 엮어 "비행→수집→분석→평가" 사이클 자동화하는 게 목표(사용자 요청, 수십 회 반복 예정이라 세션 컨텍스트 보호 필요) — 로그수집 스킬은 아직 미착수.
  그 전 **(2026-07-18)** 🚁 mc-실기체 로그 조사 중 작업 G 인프라의 실사용 버그 2건 발견 — ① **ulog 자동회수가 지금까지 한 번도 성공한 적 없었음**: RPi 호스트에 pymavlink가 아예 미설치라 `pull_ulog.py`가 매번 조용히 실패(실패 메시지가 어디에도 저장 안 돼 발견이 늦어짐) — Claude가 임시로 `pip install --user --break-system-packages pymavlink`로 우회 설치·확인함(영구화 필요: 컨테이너 이미지 또는 RPi 셋업 스크립트/문서에 pymavlink 설치 단계 반영할 것). ② **`record_flight.sh`를 컨테이너 `fc` 안 root로 실행**해 `logs/<날짜>_flightNN/` 폴더가 root 소유가 됨 → `suri` 계정이 그 안에 쓰기 불가(ulog를 못 넣음, 향후 `fetch_logs.ps1`/scp도 root 소유 파일 자체는 읽기는 되지만 정리는 어려움) — **수정 완료**: `record_flight.sh` 종료 시 `$FLIGHT_DIR`을 `$LOG_ROOT` 소유자로 chown하도록 추가(best-effort, `2>/dev/null || true`로 실패해도 스크립트 안 죽음). `bash -n` 통과, 이 WSL에 `pytest`/`pymavlink` 미설치라 `test_flight_logs.py` 로컬 실행은 못함(그 테스트는 `pull_ulog.py` 순수함수 대상이라 이 변경과 무관 — 회귀 위험 낮음). **커밋 `07681d3` 후 origin push 완료, RPi `git pull` 재실행 완료(서브에이전트) — 다음 비행부터 적용됨.** 사용자가 RPi에서 직접 `sudo chown -R suri:suri ~/drone_ws/src/suridoksuri/logs`를 실행해 기존 root 소유 폴더도 정리됨(확인함) — 이 참에 스테이징에 있던 07-18 ulog 8개도 RPi 원본 위치(`flight01~08`)로 옮겨 RPi 쪽 사본도 완전해짐. **교차참조(2026-07-18, 실비행 8건 ulog 분석):** 🚁 mc-실기체 트랙에서 `flight01~08` 전수분석 결과 근본원인이 **배포/빌드 파이프라인 갭**으로 확인됨 — RPi 소스(`~/drone_ws/src/suridoksuri`)는 작업 H-2(`9451861`) 이후로 최신인데 colcon 빌드 산출물(`~/drone_ws/install/fc_ros/...`)이 2026-07-07(`7414c1d` 직후) 이후 한 번도 재빌드되지 않아 구버전 `takeoff_request_fields` 1-인자 호출이 실비행 8건 전부에 그대로 실렸음(AMSL 미적용). 상세·재발방지·잔여 리스크는 🚁 트랙 "마지막" 참조.
- **로그 git 커밋 방침 전환 (2026-07-18):** 기존 "GitHub 업로드 안 함" 결정을 사용자가 번복 — `logs/`를 `.gitignore`에서 제외하고 **일반 git 커밋**(LFS 아님)으로 다기기 공유. 트레이드오프(이력 영구 팽창, clone 속도 저하) 인지하고 감수. 상세는 `tools/flight_logs/README.md`·`flight_plan.md` 작업 G "업로드 방침" 참조. 오늘 07-18 로그(14회 중 rosbag/ulog 있는 것 전부, 53개 파일)를 이번에 커밋. 오늘 14회 비행분(`logs/2026-07-18_*`)은 수동으로 회수·정리 완료(🚁 트랙 참조).
- 그 전 2026-07-11 — **작업 H-2: 이륙 실패 실기체 ulog 진단 + 수정**(`9451861`). 2026-07-07 광주 비행 ulog(`02_17_49`) pyulog 분석 → `CommandTOL.altitude`(NAV_TAKEOFF param7)는 **AMSL 절대고도**인데 `transition_alt`(상대)를 그대로 실어 지면 AMSL(19.2m)보다 낮은 목표 → PX4 `Already higher than takeoff altitude`로 이륙취소·preflight disarm(배터리·GPS·SD 전부 정상, 무관). 수정: `takeoff_request_fields(transition_alt, home_amsl)`→`altitude=home_amsl+transition_alt`(+`/mavros/home_position/home` 구독·home 미수신 시 이륙 보류), CLIMBING 게이트를 지면기준 AGL로(`climbing_reached(…, ground_ref_up)`, 로컬 원점≠지면 2.11m 보정). pytest fc_ros 60/fc_bridge 44(신규 7). **SITL 재검증 대기** — `sitl_verification_log.md` 작업 H-2(재현엔 `PX4_HOME_ALT`로 지면 AMSL>transition_alt 필요 + geoid 확인). 그 전 2026-07-06 — ① **작업 H 완료·SITL PASS·커밋** — `offboard_node.py` `_step_arm_takeoff`를 `SetMode(AUTO.TAKEOFF)`→`CommandTOL(/mavros/cmd/takeoff, altitude=transition_alt)`로 교체(`7414c1d`). 요청 필드 조립은 순수함수 `fc_bridge/execution/state_logic.py::takeoff_request_fields()`로 분리(rclpy 없는 Windows에서도 pytest 가능). **1차 SITL 실패 → 원인 수정 → 재검증 PASS:** `latitude=0.0, longitude=0.0`을 "현재 위치"로 잘못 가정(MAVLink 관례는 **NaN**, `0.0/0.0`은 실좌표라 PX4가 고도 미상승 후 preflight disarm) → NaN으로 수정(`000f478`). WSL gz_standard_vtol `transition_alt:=50.0` 재검증 시 정상 상승·CLIMBING 통과 확인. 잔존 `guided_target`/"no origin" 경고는 MAVROS humble 알려진 QoS 코스메틱 이슈로 무해. pytest 130(fc_ros+fc_bridge) 전부 통과. 상세: `flight_plan.md`·`sitl_verification_log.md` "작업 H" ② 그 전: **planner 2종 본선 이식**(다른 계정 repo `suridouksuri`의 Fable 작업 회수): eta3 **v3.3**(2D 퇴화 WP NaN 근본수정)+**StraightLinePlanner**(신규)+`resolve_planner_name` 기체타입 자동선택 — `584cff3` ③ **transition_alt launch 오버라이드** `356ae5a` ④ 그 전: V2 검증·pull_ulog livelock 수정 `b580953`
- **배터리 텔레메트리 (2026-07-21, 사용자 요청) — [코드] 완료·단위테스트 통과, SITL/실기체 미검증:** `/mavros/battery`(`sensor_msgs/BatteryState`, MAVROS가 MAVLink `SYS_STATUS`/`BATTERY_STATUS`를 변환해 발행)를 `telemetry_node.py`가 구독하도록 추가(원안은 raw `MavlinkConn.recv_match()` 직접 구독이었으나, 이미 MAVROS 경유 파이프라인(pose/twist/state/extended_state와 동일 패턴)이 있어 그쪽으로 통일). `VehicleState`에 `battery_voltage`(V)·`battery_current`(A)·`battery_remaining`(0.0~1.0, 기본값 1.0) 필드 추가, `fc_ros/adapters/vehicle_state_bridge.py::update_from_battery()`(percentage NaN 시 이전값 유지)로 변환, `copy()`에도 반영. `package.xml`에 `sensor_msgs` 의존성 추가(그동안 빠져있었음 — geometry_msgs만 있고 sensor_msgs 없었음). 테스트 8개 신규(`fc_ros/test/test_telemetry_node.py`) — fc_ros+fc_bridge 전체 142개 통과(WSL, pytest). **미검증:** 실제 MAVROS가 이 토픽을 기대한 필드명/QoS로 발행하는지는 SITL 또는 RPi 실기체에서 `ros2 topic echo /mavros/battery` 확인 필요(WSL엔 mavros 미설치라 이 세션에서 직접 못함). 커밋 전.
- **다음(우선순위순, 2026-07-11 갱신 — ✈ vtol-실기체 결함으로 비행 보류 중):** ① **지금 가능(WSL SITL) — 작업 H-2 SITL 재검증** (`sitl_verification_log.md` 작업 H-2 체크리스트대로: `PX4_HOME_ALT`로 지면 AMSL>transition_alt 재현 후 AMSL 수정 확인 + geoid 정합) ② ~~실기체 검증~~ **결함 해소까지 보류**(🚁 하드웨어 해체, 이설된 ✈ vtol-실기체서 재개) ③ **지금 가능 — 작업 F**(임의 WP 견고성, [코드] Claude 단독) ④ **지금 가능 — V1·V3·V4**(하드웨어 불필요, Claude 단독) ⑤ **WSL SITL만 있으면 가능 — V2·V5** ⑥ RPi 배포 검증(pull_ulog 실측 속도)은 결함 해소 후
- **주의:** 최신 코드(작업 H 포함, `000f478`까지 커밋·푸시 완료)가 RPi에 **미전파** — RPi에서 `git pull` 필요(RPi 정본=호스트 `~/drone_ws/src/suridoksuri`, potato03kth). WSL(`~/suridoksuri-1`)은 이미 pull·재빌드 완료. `waypoints` 300 m·`v_cruise 20.0` 유지 결정(2026-06-30). V2/V5는 MAVROS 중지 필요(단독 링크). **작업 H가 실기체로 검증되기 전까지** 🚁 트랙의 "transition_alt를 낮게" 임시조치를 유지할 것 — SITL은 PASS했으나 실기체 미확인. **작업 H-2(AMSL 이륙고도 수정, `9451861`)는 단위테스트만 통과 — SITL 재검증 전이라 실비행 반영 금지.** geoid 리스크(MAVROS `geo.altitude`가 ellipsoid면 이륙 과상승) SITL 로그로 판별.
- **참조:** `fc_ros/fc_ros/nodes/offboard_node.py`(`_step_arm_takeoff`) · `fc_bridge/execution/state_logic.py`(`takeoff_request_fields`) · `fc_bridge/planning/planner_runner.py`(resolve_planner_name) · `vtol_sim/…/straight_line_planner.py`·`eta3clothoid_v3_1_planner.py`(v3.3) · `tools/flight_logs/VERIFY.md`(V1~V5) · `flight_plan.md`·`sitl_verification_log.md`(작업 H) · `docs/flight_plan.md`(작업 G)

### 🛩 sitl-vtol — ▶ 활성 (**2026-07-29: R5 F-9 ✅완료·SITL 검증. 남은 R5 항목은 F-8·F-10·F-11·F-6**)

- **✅ R5 F-9 (천이 고도 계단) 완료 — 코드·단위테스트·SITL A/B 전부 (2026-07-29, `94989b6`+`f6f8789`).**
  전문은 `docs/sitl_vtol_remediation_plan.md` R5 2항 · 런 산출물 `logs/2026-07-29_r5_f9/`(notes.md).
  - **원인:** `_step_transition_fw` Phase 1·2 는 속도 setpoint(hover)라 위치 setpoint 의 고도
    성분이 **아예 없다가**, 헤딩 정렬이 끝난 **Phase 3 첫 틱**에 `_cruise_alt`(=`wp[-1].z`+경로원점)
    **절대값**이 그대로 나간다. 계단 = `wp[-1].z − transition_alt`.
  - **수정:** `fw_setpoint_alt()` 순수함수(R2 의 `slew_setpoint` 재사용) + `_fw_alt()`.
    **FW 위치 setpoint 4곳 전부**가 이 하나를 쓴다(한 곳만 램프하면 계단이 다음 경계로 옮겨갈 뿐).
    파라미터 `alt_slew_rate` 기본 **3.0 m/s**, 0 이하 = 비활성(R1 타임아웃과 같은 규약).
    수평 성분은 안 건드린다(F-6 소관). 계단 5m 초과 시 WARN — flight02 의 `transition_alt`
    유실(U+00A0)을 잡는 두 번째 그물이다.
  - **SITL A/B (같은 빌드에서 `alt_slew_rate` 만 0.0↔3.0):** 첫 틱 고도계단
    C1a **+29.331→+7.628m(−74%)** / C1b **−70.631→−8.291m(−88%)**.
    **고도 수렴은 안 늦어졌다**(C1a +1.31s / C1b +0.19s) — 3.0 을 기체 실측 수직속도
    (1.4~1.8 m/s)보다 빠르게 잡은 근거가 실측으로 확인된 것. **회귀 0건.**
    잔여 7.6~8.3m 는 램프 결함이 아니라 `3.0 × (PX4 가 유한 position 을 싣기까지의 2.5~2.8s)` 다.
  - 🔴 **이 A/B 에서 드러난 별개 회귀 — R5 F-10 으로 이관:** C1b 가 **램프 on/off 양쪽 모두**
    FOLLOWING 을 469초 내내 못 벗어나고 cte 136~141m·`min_agl` −2.7~−3.5m(접지)로 끝난다.
    캠페인 `C1b_pxvehicle` 는 2바퀴 돌고 **완주**했었다(FOLLOWING 43.5s). **수정 전에서도 같으므로
    F-9 탓이 아니고**, 캠페인 이후 R1/R2/`v_cruise` 18 구간에서 생긴 회귀로 보인다.
  - **하니스:** `tools/sitl/f9_alt_probe.py` 신설(`setpoint_jump` 3D 노름은 F-6 에 가려 F-9 를
    못 본다). ⚠️ **시나리오 사이 `wsl --terminate Ubuntu-22.04` 는 권고가 아니라 필수** —
    건너뛰면 `set_preflight_bypass` 가 `readback=None` 으로 실패하고 offboard_node 가
    `/mavros/cmd/arming 서비스 없음` 을 10Hz 로 무한 반복해 ARM 조차 못 한다(이번에 실측).
  - **실기체 배포는 아직 안 했다** — 사용자가 기체 물리 점검 중(자기계 재캘리·pusher)이라
    `/dev/ttyACM0` 을 안 건드리는 조건이었고, 배포 자체는 시리얼과 무관하므로
    **점검이 끝나면 `docs/rpi_deploy.md` 절차로 바로 반영할 것**(`colcon build --packages-select fc_ros`).

- **✅ 실기체 플래시 완료 (2026-07-28~29 — 사고 1건 발생·해소, 전문 `docs/px4_v6c_patch_build.md` §11):**
  - **1차 플래시(사고):** 패치본 `…f17f4patch_20260728.px4` 는 교체 성공했으나 **ELRS 조종기가 완전히 두절**됐다.
    순정 `px4_fmu-v6c_default` config 에 `CONFIG_DRIVERS_RC*` 가 하나도 없어 CRSF 드라이버가 안 들어간다
    (`crsf_rc status` → command not found, `listener input_rc` → never published). 원 빌더가 바꾼 것은
    **컴파일러만이 아니라 보드 config** 였다는 뜻 — §9-4/§10-4 「남는 의문」의 진짜 답(§11-3).
  - **2차 플래시(해소, 현재 기체 탑재본):** `tools/px4/v6c_crsf_rc.patch` 로 `CONFIG_DRIVERS_RC_CRSF_RC=y`
    추가 후 재빌드한 `…f17f4patch_crsf_20260728.px4`. 실측 — `Build datetime Jul 28 2026 20:27:22`
    (**crsf 포함본 판별의 유일한 식별자**. Toolchain 10.3.1·git-hash `c890d9db0a` 는 사고본과도 같다) /
    `crsf_rc status` = `/dev/ttyS1`(TELEM3, `RC_CRSF_PRT_CFG=103` 일치), RX 1120B·유효 CRC 80·**Invalid 0**.
  - **파라미터 대조 결론: 잃은 것 없음.** 덤프 `logs/2026-07-28_px4_flash/px4_params_2026-07-28_final-crsf.json`(1438개).
    원본(02:38, 1437개) 대비 **MISSING 0건** / ADDED 1건 `UXRCE_DDS_CFG=0`(의도 — FLASH 여유가 남아 uxrce 를
    켠 채로 뒀다, 값 0·우리는 MAVROS 사용). 사고본(20:14, 1436개) 대비 ADDED 2건(`RC_CRSF_PRT_CFG`·`RC_CRSF_TEL_EN`)
    은 EEPROM 잔존값이 드라이버 복귀와 함께 **재설정 없이 되살아난 것**. 개수 1436+2=1438 정합. `SYS_AUTOSTART=13000`·
    스틱 1~4·ARM 12·KILL 11·FLTMODE 10 유지. **`RC_MAP_OFFB_SW 7→0`·`RC_MAP_TRANS_SW 8→7` 은 사용자 의도 변경(확인받음) — 결함 아님.**
  - **✅ F-17 패치가 실기체에서 정상 작동함이 ulog 로 확인됐다** — `142150a`(flight02 재현검증)에서
    `position_setpoint_triplet` 발행 **4건 전부 `course=NaN`**. 패치 전이라면 `course=0.0f`(= "정북 유지"
    유효명령)가 나왔어야 한다. `ver all` 이 패치 여부를 자기신고하지 않는다는 §4-2 한계를 비행 거동이 메꿨다.
  - **⚠ 다만 실비행 검증은 아직 못 한다 — 펌웨어 무관한 별개 미해결 2건:** 자기계 **헤딩 의존 오차**(`5d55b3f`,
    재캘리 후에도 `test_ratio` 1.97→2.62·`cs_mag_fault` ON 0%→92.7%. 통과 기준은 "기수 남쪽에서 `test_ratio<1`") ·
    **배터리 게이트 부재**(`f8e951f`, flight02 는 `Emergency battery level` 상태로 50m 까지 상승해 천이 시도).
  - **다음:** ①위 2건 해소 ②비-정북 레그 실비행(R7) ③R5 에서 MC HOLD 수직가속 확인.

- **✅ 그 전 (2026-07-28, PX4 패치 2단계 — 빌드·SITL 검증):** 재현절차 전문 **`docs/px4_v6c_patch_build.md`**.
  - **패치** `tools/px4/f17_f4_offboard_nan.patch` — `FixedWingModeManager.cpp` 오프보드 변환 블록에 `current.yaw = NAN` / `current.course = NAN` 2줄. **PX4 저장소엔 커밋하지 않는다**(우리 것이 아님) — 워킹트리에만 얹고 패치파일을 우리가 보관.
  - **빌드** `px4_fmu-v6c_default` 성공, 툴체인 `arm-none-eabi-gcc 10.3.1`(jammy universe, `ubuntu.sh --no-sim-tools`). **FLASH 98.65% = 1,939,520/1,966,080 B, 여유 25.9 KB, 패치비용 +8 B.** 경고 0건. `.px4` sha256 `f1c16e2b…`.
  - **🚨 펌웨어가 패치 여부를 자기신고하지 않는다** — PX4 버전헤더 생성이 `--dirty` 를 안 붙여 `git_identity` 가 순정과 **완전히 동일**하다(`v1.18.0-alpha1-592-gc890d9db0a`). **구별 수단은 sha256 뿐.** 플래시 직전 반드시 재확인.
  - **⚠ SITL 검증은 v6c 가 아니라 `px4_sitl_default` 로 돈다** — `make px4_sitl_default` 를 따로 안 하면 패치 전 바이너리로 검증하게 된다(이번에 실제로 stale 상태였음).
  - **F-17 해소 확정.** 천이중 PX4 지령 course: 순정 C2 **−0.00~−0.04°(정북, `atan2(−E,3000)` 과 소수 셋째자리까지 일치)** → 패치 후 정북 지령 소멸. B8 **177.71°** / A1 **2.52°** / A3 **2.35°** 로 **기체 실제 헤딩 추종**(정북 예측값 0.006~0.011° 와 명확히 구분) = `:549` `_yaw` 폴백이 살아난 직접 증거. C2 는 `course=NAN` 이 규약대로 "미사용" 처리돼 **천이중 토픽 발행 자체가 사라졌다**(60.648s > 천이종료 60.080s). 천이 종료 틱의 **90.7° 순간계단도 소멸**.
  - **C2 피해:** 북향이탈 **21.78 → 0.38 m** · 기하 cte **21.76 → 0.37 m** · yaw 최저 **43.6 → 88.2°** · 오버슈트 **129.2 → 92.2°** · 고도최저 **43.23 → 49.06 m** · 순항 고도편차 **FAIL(6.76) → PASS(2.06)**.
  - **회귀 0건** — A1·A3·B8·C2 4런 전건 완주(exit=0), **판정항목이 PASS→FAIL 로 뒤집힌 사례 없음**. B8·C2 는 오히려 FW cte WARN→PASS.
  - **⚠ 순수 A/B 는 아니다:** 기준선은 `3b52ac1`·`v_cruise 20`, 패치런은 `893a5eb`·`v_cruise 18`(R1·R2 포함). **A3 `node_log_cte` 7.2→14.6 은 R2 의 `_find_segment` 변경 탓**으로 본다(객관지표인 기하 cte 는 15.38→15.21 불변). **미규명 1건: MC HOLD 수직가속 상승**(A1 5.98→10.34, C2 6.71→11.73 m/s²; B8 은 불변) → R5 확인.
  - **하니스 개선 3건:** `run_scenario.py --launch-arg KEY=VALUE`(임시 파라미터로 yaml 안 고치게) · `meta.json` 에 `px4_dirty`/`px4_diff_sha256` 기록(커밋 해시만으론 워킹트리 패치를 구별 못 함) · `tools/sitl/f17_transition_probe.py` 정식 편입(종전 임시 스크립트 `/mnt/c/sitl7_xfer/f17_probe*.py` 는 소실돼 재현 불가였음. 기준선 21.78m/43.23m/43.6°/129.2° 를 정확히 재현하는 것으로 검증).
  - **⚠ WSL 클론 표류 원인 규명:** `/root/drone_ws/src/suridoksuri` 의 `remote.origin.fetch` 가 **mc-hw 브랜치 하나로 좁혀져** 있어 `git fetch origin` 이 조용히 성공하면서도 `origin/dev--vision-computing-module` 이 13커밋 뒤(`b1af926`)에 멈춰 있었다. 표준 refspec 으로 복구 후 `893a5eb` 정렬 완료(`.ulg` 135개 전량 보존, `git clean` 미사용).
  - ~~**다음:** ①실기체 플래시 — 사용자 승인 후 별도 세션(USB·QGC 필요)~~ → **①은 2026-07-28~29 완료**(위 블록·§11). ②비-정북 레그 실비행(R7) ③R5 에서 MC HOLD 수직가속 확인.

- **내용:** WSL SITL VTOL 검증. SITL-1~4 전부 PASS (2026-06-30) → SITL-7 전면 회귀 캠페인(`docs/sitl_vtol_campaign.md`)
- **마지막:** **(2026-07-27, S3 — A3 6.5km 폭주 근본원인 규명 완료)** **결론: 우리 코드는 정상, PX4 상류 회귀 버그다.** `FixedWingModeManager.cpp:2107` 이 오프보드 `trajectory_setpoint`→`_pos_sp_triplet` 변환 시 구조체를 제로초기화하면서 `course` 필드에 NaN 을 넣지 않아 `course=0.0f` 가 남는다. `PositionSetpoint.msg:36` 규약은 "NaN=미사용"이므로 0.0 은 **"코스 0 rad(정북) 유지" 유효명령**으로 해석되고, `:577`/`:781` 코스 분기가 발화해 `navigateBearing` 이 **기체 현재위치를 지나는 무한직선**을 만든다(`:2782`) → lat/lon 은 한 번도 안 읽히고, 횡오차가 구조적으로 항상 0 이라 **경보·페일세이프가 안 걸린다.** A3 ulog 실측: OFFBOARD 431초 4310샘플 전부 `course_setpoint=4.371138829e-08 rad`(정북)·`signed_track_error=0.000000`, OFFBOARD 이탈 직후 t=567.3 에 진짜 오차 −6442.7m 출현. **회귀 계보 확정:** `8b3ef1cf9e`(2026-05-27 병합, 삽입) → `2e59c98b7c`(05-28, GUIDED_COURSE 가드로 차단) → **실기체 `c890d9db0a`(07-06, 가드 포함 = 안전)** → `1499238f1c`(07-17 병합, 가드 revert = 재삽입) → **SITL `9bb0d365c4`(07-23, 취약)**. 실기체가 178커밋 뒤라 우연히 무사한 것 — **PX4 업그레이드 금지.** **⚠ Phase 1 "성공" 5건 전건 무효:** A1/A2/A4/B1/B6 waypoint 가 **전부 정북 직선**이라 "정북으로만 나는" 버그가 그대로 통과했다. FW 경로추종은 한 번도 검증된 적 없다. **⚠ 실기체 FW+OFFBOARD 비행 실적 0건**(ulog 78건 전수 스캔, FW 구간 자체가 2.4초 수동 테스트 1건뿐) — "실기체에선 된다"는 소스 추론일 뿐 미검증. **✅ 우회로 실측 확보:** `setpoint_raw/local` 로 **위치+속도 동시 발행** → `FW_POSCTRL_MODE_AUTO_PATH`(`:392`) 진입해 코스 분기를 원천 회피, SITL 실측 **횡오차 0.2m·정동 90.0° 정확 추종**. 속도-only 는 기각(`FW_POSCTRL_MODE_OTHER` 로 빠져 **setpoint 발행 자체가 52.96초 정지** — 실측). 프로브 `tools/sitl/fw_offboard_probe.py`(비행코드 무수정), 궤적 `logs/2026-07-27_s3_fw_offboard_probe/`.
  그 뒤 **(2026-07-27, S4~S7 — 캠페인 완료)** **S4: SITL을 실기체 PX4(`c890d9db0a`)로 정렬.** `/root/PX4-vehicle` 에 별도 worktree+빌드(기존 `/root/PX4-Autopilot` 은 태그 `sitl7-orig-head` 로 보존, 무변경). 가드 유무는 `FixedWingModeManager.cpp` `control_auto_position()` 의 `&& _vehicle_status.nav_state == NAVIGATION_STATE_GUIDED_COURSE` **한 줄 차이**. 그 빌드에서 **A3(L자) 완주** — cte 6292m→**7.2m**, FOLLOWING 472s 폭주→**26.4s 정상종료**, vtol 3→1→4→**2→3**. **S5: Phase 2 6건 전량 완주**(B8 후방/B2 곡선/B3 직각/B4 U턴/B5 폐곡선/B7 단거리), cte 전부 마지막 1m 이하 수렴, 90° 코너 오버슈트 0.085~0.915m(코너 안쪽 19~22m 절삭), 선회반경 초과는 135°(B4)에서만. **S6: 하니스 경고누락 수정**(`ros2 launch` stdout 중계 줄을 정규식이 버려 플래너 NR 잔차 경고 7건이 판정서에 누락돼 있었음 — 기존 15런 소급 재분석, **판정 변화 0건**) **+ Phase 3 전반부 7런**. **S7: 장애주입 4종 + 바람.** **✅ 안전 경로 전부 실증**(setpoint 중단을 ulog `offboard_control_mode` 절단으로 확인): OVERRIDE-FW(SITL-4와 동일 3줄)·**OVERRIDE-MC 최초 검증**·**PILOT_TAKEOVER 최초 검증**(POSCTL 실제 진입 101.5s, 인계 후 `vehicle_command` 0건)·C3 OFFBOARD 상실 0.97s 후 복구 완주(mode flapping 미관측). **바람 8m/s**(gz world `<wind>` 주입, EKF 검증 7.89m/s): cte 1.2→4.0m, 고도편차 2.24→**3.99m FAIL**, 그리고 **정렬 허용대 재이탈 0.0→0.2178rad(12.5°)** — 정렬 완료 오차는 −2.5°로 불변(그게 게이트값)이지만 **더 센 바람이면 20틱을 못 채워 `TRANSITION_FW` 영구 대기 가능**. 종합은 `logs/2026-07-27_sitl_vtol_campaign/campaign_report.md`.
- **결함 요약(심각도순, 상세는 campaign_report.md §4):** 치명 — `entry_mode=mid_flight` 시 ENTRY 무한대기+**5.85km 이탈**(C10 실측, 파라미터 하나로 발현) / `TRANSITION_MC`·`TRANSITION_FW` 타임아웃·재시도 부재(SITL 미재현, 코드 구조 위험). 높음 — PX4 상류 회귀에 안전 의존 / **선회 중 고도 침하 최대 7.45m**(선회량에 비례, 직선 경로 C2에서도 6.76m) / 상태경계 setpoint 계단 214~300m·60~117m(**PX4 무관, `offboard_node` 문제로 확정**) / 헤딩 정렬 마진 0.5°(구조적) / **정렬구간 OFFBOARD 이탈은 복구 경로 없음**(미실측). 중간 — `_cruise_alt` 스칼라화(A4 실증) / `transition_alt≠wp[-1].z` 시 고도계단 ±30·−70m 실측 / 고도오차→종점포착 실패→flower-pattern+2.111g(C1b) / 짧은 경로 추종구간 소멸+역천이 0.315g(B7) / 플래너 `__init__` 블로킹 최대 263.5s·`v_cruise`에 160배 민감 / 속도 프로파일 전량 미사용 / **재요청 WARN 로그가 throttle 밖 10Hz**(로그만으론 2026-07-25 사고와 구별 불가).
- **통과:** 전 경로 완주·`vtol_state` 3→1→4→2→3 균일(정천이 2.42~2.60s/역천이 4.92~6.09s) · **C8 geoid 회귀 VTOL 이관 합격**(`alt=53.0m AMSL (지면 3.0+50.0)` 정확) · `_find_segment` O(N) 우려 기각(N=501 p95 104ms) · STREAMING 1틱 통과(전 런 0.11s)로 **VTOL엔 MC의 STREAMING 오버슈트 문제가 구조적으로 없음**(CLIMBING 언더슛 −0.6~−3.0%) · `d_end_thresh` 스윕으로 **통과량 ≈ 57−thresh 선형** 확정(≈57에서 오버슈트 0).
- **▶ 다음 세션 진입점: `docs/sitl_vtol_remediation_plan.md`** — R1~R7 실행계획. **R1·R2는 완료·배포됐고 A안은 기각 확정이다(§4-2). 남은 것은 R5·R6뿐.** 그 문서의 §4(사용자 확정 결정)·§4-1~4-3(A안 심사방침/기각판정/F-17)을 먼저 읽을 것.
- **🔴 A안은 기각됐다 — 다시 도입하지 말 것 (2026-07-27, 사용자가 "매우 부정적으로 검토하라" 지시 → 적대적 심사 → 기각).** 근거(오케스트레이터가 PX4 소스로 직접 재현 확인, 전문은 `docs/sitl_vtol_auto_path_spec.md`):
  - `control_auto_path()` `:1085-1123`이 `.altitude = pos_sp_curr.alt` 로 **고도를 현행과 똑같이 넘긴다 ⇒ F-5(선회 중 고도침하 7.45m) 개선 근거 0.** 같은 함수에 **종점 처리 코드가 한 줄도 없어**(수용반경·LOITER 승격 전무) 현행의 *유계* 실패를 **무계 직선 이탈**로 바꾼다(R-c).
  - 모드분기 `:387-398`에 히스테리시스·래치 전무. **MC=PoseStamped / FW=PositionTarget 이라는 A안 설계 자체가 채널 전환 = `AUTO`↔`AUTO_PATH` 모드전이** ⇒ 채택기준 "모드 이탈 0건"은 **설계상 달성 불가**(R-a).
  - **이득의 전제가 우리 데이터로 반증됨:** C4 바람 cte 시계열이 `-0.2→-1.6→-4.0→-1.8→+0.7→+1.4→+1.7→-0.5→-0.2` 로 **부호 2회 반전하는 감쇠 진동**이다. "점 추적이라 오프셋이 남는다(규제 부재)"가 맞다면 단조 오프셋이어야 한다.
  - 유일 근거였던 "횡오차 0.2m"는 PX4 내부 `track_err`(자기 지령 대비 자기 오차)라 우리 cte와 **범주가 다르다.**
  - R-f(업그레이드 취약성)만 **반증**됐다(12개월간 `control_auto_path()` 변경 0건). 그러나 R-a·R-c·R-e 봉쇄 불가로 §4-1 1항 불충족.
  - **기각 시 조치(확정):** 현행 위치-only 유지 + PX4 `c890d9db0a` 핀 + 업그레이드 필요 시 상류 한 줄(`course = NAN`). **F-17 패치에 `course = NAN`이 포함돼 있어, 그 패치를 플래시하면 이 잔비용도 사라진다.**
- **다음(요약) — 2026-07-29 갱신:** ①**R5(경로·고도)**: ~~천이 고도계단(F-9)~~ **→ ✅ 완료·SITL 검증(위 최신 블록)**. 남은 것은 `_cruise_alt` 스칼라화(F-8)·짧은경로 `_FW_LOOKAHEAD`(F-11)·`d_end_thresh` 기본값(F-10, **C1b 회귀가 여기 물렸다 — 위 블록 참조**). **⚠ `d_end_thresh`를 ≈57로 키우면 폐회로가 깨진다** — `R2_closed` FOLLOWING 진입 시 종점까지 12.18m뿐이고 **런간 변동 포함 실제 최악 여유 0.6m**(관측대역 10.60~15.32m). 폐회로에서 반드시 함께 검증할 것. **미규명: MC HOLD 수직가속 상승**(A1 5.98→10.34, C2 6.71→11.73 m/s², B8 불변) ②**R6(선회 품질·정리)**: F-5(선회 중 고도침하 — **A안으로는 안 고쳐진다, 종방향 TECS 문제**)·F-7 헤딩정렬 마진·F-13 미사용 속도프로파일·**F-12 플래너 비동기화**(아래 참조) ③~~**F-17 실기체 플래시**(사용자 승인 대기, 준비 완료)~~ **→ ✅ 완료(2026-07-29, 위 최신 블록 참조)** ④미실측: `param_set` 주입·C9.
  - **완료·배포됨(재작업 금지):** 타임아웃 4종+거리상한 300m+F-15+F-16(R1, `ca8809d`) · `_find_segment` 창탐색+`_step_hold` 슬루+F-14+플래너 진행로그(R2, `2f024a7`) · `v_cruise` 정식값 **18.0**(`28a8701`). 전부 RPi5 반영·md5 대조 완료(`893a5eb`).
  - **`v_cruise`로 F-12를 못 고친다(실측 확정):** 18은 20의 **96~97%**다. "160배 민감"은 곡선이 아니라 **16↔17 계단**이고, ≤16이 싼 이유는 NR이 일찍 포기해 **경로가 변형**되기 때문이다(전장 L자 405.58m/폐회로 809.42m vs 이론 400/800). **F-12의 해법은 비동기화(R6)뿐.**
  - **회귀 시나리오 필수요건에 「비-정북 *천이*」 추가** — 캠페인 24런 중 비-정북 천이는 C2·B8 2건뿐이었다(A3/B4/B5는 코너가 있어도 첫 레그가 정북). §7-1의 "비-정북 *레그*"만으로는 F-17을 못 잡았다.
- **참조:** **`logs/2026-07-27_sitl_vtol_campaign/campaign_report.md`(종합)** · `sitl_vtol_fw_offboard_rootcause.md`(근본원인 전문) · `sitl_vtol_campaign.md` · `sitl_vtol_static_audit.md` · `logs/2026-07-27_sitl_vtol_campaign/PX4_BUILDS.md`(런↔빌드 대응) · `tools/sitl/README.md`(하니스) · `sitl_verification_log.md`

### ✈ vtol-실기체 — ▶ 활성 (수리한 VTOL 테스트기체로 오프보드 비행 예정, 2026-07-24 사용자 확인)

- **확인됨 (2026-07-18):** 🚁 mc-실기체 트랙의 "부활한 MC 테스트기체"(07-17·07-18 14회 실비행)는 이 VTOL 기체와 **무관한 별도 물리 개체** — Pixhawk·ESC 모두 다른 유닛, 외형 프레임만 동일해 보이는 것. 아래 결함과는 관계없음. 상세는 🚁 mc-실기체 트랙 "확인됨" 참조.
- **내용:** VTOL 실기체 전체 사이클 + RC override→POSCTL 실측(SITL-1 이월 항목). RPi5+Pixhawk6C 전자장치를 🚁 mc-실기체에서 이설.
- **마지막:** **(2026-07-24, 비행 전 코드 준비상태 점검 — CV 미사용 예정 비행, `planner`/`vehicle_type` 파라미터 동작 확인 + 크래시 버그 1건 발견·수정)** 사용자 질의("MC 테스트기체 코드수정이 있었는데 VTOL 비행해도 괜찮나") 계기로 `offboard_node.py`/`fc_bridge/planning/planner_runner.py`/`eta3clothoid_v3_1_planner.py`/`straight_line_planner.py` 코드 직접 확인. **결론: MC용 코드수정(yaw setpoint 수정 등, 07-21)은 경로계획 모듈과 무관 — 영향 없음.** 확인된 사실: ①`planner:="auto"`는 `vehicle_type=mc`→`straight`(NR/최적화 없는 단순 3D 직선, 완전중복 WP만 병합, 그 외 거부 없음), `vehicle_type=vtol`(기본값)→`eta3`로 자동분기(`resolve_planner_name()`). ②eta3는 짧은 경로/급선회에도 "거부"하지 않음 — WP가 2개뿐이면 NR 자체를 건너뛰고(N≤2 특수케이스), 3개 이상이면 NR 잔차가 커도 affine 보정으로 WP 통과를 강제하고 콘솔 WARNING만 출력(예외 없음). 실제 `raise`는 "병합 후 distinct WP가 2개 미만"(사실상 한 점) 경우뿐. ③명시적으로 `planner:="straight"`를 VTOL 기체에 줘도 됨(항상 auto보다 우선) — 단 곡률 불연속을 그대로 노출하므로 FW/천이구간엔 부적합, MC구간 검증용으로만. **✗ 그러나 점검 중 별개의 실제 크래시 버그 발견:** eta3 NR 해석 경로(WP 3개 이상)가 `np.trapz`를 호출하는데, 이 개발머신 numpy(2.5.1)에서 `np.trapz`가 완전히 제거되어 있어 `AttributeError`로 즉시 크래시함 — "경로가 짧아서 거부"가 아니라 numpy 버전 비호환 크래시. 리포 자체 회귀테스트(`vtol_sim/tests/test_eta3_v3_degenerate_wp.py::test_normal_wps_unaffected`)도 이 크래시로 실패 상태였음(`fc_bridge/tests/`만 pytest 대상이라 이 결함이 CI로 안 잡혀왔던 것으로 보임). **`_trapz()` 로컬 사다리꼴적분 함수로 교체해 수정, `fc_bridge/tests`(65)+`vtol_sim/tests/test_eta3_v3_degenerate_wp.py`(3) 전부 통과 확인, `dev--vision-computing-module`에 커밋·push 완료(`0785777`).** **⚠ 미확인 — 이 비행 전 반드시 확인 필요: RPi5 companion computer의 fc_bridge venv numpy 버전.** 그쪽 numpy가 2.x 계열(trapz 제거 버전)이면 실비행 중에도 동일 크래시 재현 가능 — 이번 커밋이 이미 반영됐는지, 혹은 numpy가 애초에 구버전(trapz 존재)인지 먼저 확인할 것. 관례대로 이 수정은 SITL 회귀검증 전 실비행 투입 비권장이나, WP를 2개로만 구성(예: 천이점→착륙점 단일 leg)하면 N≤2 특수케이스로 이 NR 경로 자체를 안 타므로 당장은 우회 가능. 그 전 2026-07-09 — VTOL 테스트기체 조립 완료했으나 기체 결함으로 비행 불가했던 이력, 이후 수리 경과는 사용자 구두 확인 기준(이 문서에 별도 수리완료 로그는 없음). **PX4 파라미터로 mc/vtol 물리 형상 구분 불가**(공통 상태 참조) — 실기체 세션 시작 전 어느 기체가 붙어 있는지 이 문서로 먼저 확인할 것.
- **결함 해결 전에도 진행 가능한 작업 (비행 불필요):**
  - **main-code 트랙**: 작업 F(임의 WP 경로 견고성 하니스, `fc_bridge/tests/test_arbitrary_wp.py`, [코드] Claude 단독) · `VERIFY.md` V1(pull_ulog 재조립 단위테스트)·V3(record_flight.sh 하니스)·V4(fetch_logs.ps1 증분복사) — 전부 하드웨어·비행 불필요, Claude 단독 완결
  - **sitl-vtol 트랙**: WSL SITL(`gz_standard_vtol`)로 `VERIFY.md` V2·V5, SITL-6(임의 WP 생성·추종) — 시뮬레이터만 있으면 되고 실기체 결함과 무관
  - **vision 도메인**: 별도 트랙으로 분리됨 → `docs/vision_status.md` (FC와 독립, 병행 가능)
- **진입 전 필수 (결함 해결 후 실비행 재개 시):** `flight_plan.md` "첫 비행 전 지상 안전 테스트" + "필수 조정 파라미터 체크리스트" 전 항목
- **참조:** `flight_plan.md` SITL-5·튜닝 가이드·안전 섹션 · `tools/flight_logs/VERIFY.md`(V1~V5) · `flight_plan.md` 작업 F 섹션

---

## 환경 참조 (절차 — 자주 바뀌지 않음)

### 실기체 (RPi5) — 🚁 트랙

| 항목 | 내용 |
|---|---|
| 하드웨어 | RPi5 (Ubuntu 24.04) + Pixhawk 6C (PX4 플래시됨), 순수 MC 테스트기체 |
| 원격접속 | Tailscale `100.67.27.83` (hostname `doksuri`, 계정 `suri`). **Claude용 SSH 키 등록됨**(2026-07-18, `claude-code-wsl-suridoksuri`, 이 WSL 개발컴 `~/.ssh/id_ed25519`) — 새 세션에서도 바로 `ssh suri@100.67.27.83` 가능, 비밀번호 불필요. `sudo`/`docker` 명령은 여전히 비밀번호 필요(그룹 미가입) — 안 되면 사용자에게 요청 |
| ROS2 | Docker `ros:humble` 컨테이너 (이름 `fc`, 항상 `sudo`). 네이티브 Jazzy 미채택 |
| 설치물 | MAVROS·numpy 설치됨. fc_ros는 colcon 빌드, fc_bridge+vtol_sim은 `PYTHONPATH=/drone_ws/src/suridoksuri` |
| 기동 | `phase2.launch.py vehicle_type:=mc` |
| 무선 환경 | 원격조종 RC **2.4GHz**, 인터넷 백홀은 폰 핫스팟(`DepartmentOfAgriculture`) **2.4GHz**(5GHz는 GPS 간섭 우려로 문서 권고상 미사용). tailscale/SSH 간헐적 끊김의 근본 메커니즘은 **brcmfmac 커널버그(raspberrypi/linux#6049, regdomain 플래그 손상으로 재연결 지연)로 확정** — `power_save off`+`roamoff=1 feature_disable=0x282000`(2026-07-20 적용) 완화조치 있음, 최초 트리거(신호거리/RC간섭/전원)는 미확정. 진단 경과는 Claude 메모리(`project_rpi5_tailscale_wifi_drops.md`) 참조 |
| 상시 모니터링 | `~/wifi_watch.log`(RPi5) — 5초 간격 wlan0 carrier/ping, `~/scripts/wifi_watch.sh` + crontab `@reboot`로 자동 재기동 (2026-07-19 배포) |
| 전원 | USB-C 급전, 비행 중 5V/5A PD 불가(BEC 등 사용) → EEPROM `PSU_MAX_CURRENT=1600`으로 완화 적용 (2026-07-19, 사용자 확인) |

> **개발컴은 22.04/Humble 유지** — 업그레이드하지 않는다 (검증된 환경 재현 우선).

### SITL (WSL, 개발컴) — 🛩 트랙

```bash
# T1 — PX4 SITL (VTOL. MC 검증은 gz_x500)
cd ~/PX4-Autopilot && make px4_sitl gz_standard_vtol

# T2 — MAVROS
ros2 launch mavros px4.launch fcu_url:=udp://:14540@localhost:14557

# T3 — fc_ros
cd ~/drone_ws && source install/setup.bash
ros2 launch fc_ros phase2.launch.py
```

**코드 동기화 (Windows 수정·커밋 후 WSL에서):**

```bash
cd ~/drone_ws
git pull
colcon build --packages-select fc_ros
source install/setup.bash   # 빌드 후 매번
```

> `fc_bridge`는 colcon 패키지가 아니라 순수 Python 라이브러리 — **`cd fc_bridge && pip install -e .`로 설치하지 말 것(2026-07-24 노트북 SITL에서 실측, `ros2` CLI가 깨짐)**, `docs/wsl_dev_env_setup.md` 섹션 E의 `.pth` 방식 사용.

### SITL (WSL, 이 노트북 — E드라이브, 2026-07-24 신설)

개발컴과 별개 머신. 이 노트북 기본 WSL(24.04)엔 ROS2 Humble을 못 깔아 **별도 WSL 배포판을
E드라이브에 설치**해뒀다(`wsl --import`, Canonical jammy rootfs — `wsl --install -d Ubuntu-22.04`는
목록에 없어 실패함). 최초 구축 절차·트러블슈팅(3건) 전체는 `docs/wsl_dev_env_setup.md` 섹션 F.

```bash
# 진입 (기본 사용자 root, sudo 불필요)
wsl -d Ubuntu-22.04 --cd ~

# T1 — PX4 SITL (MC는 gz_x500, VTOL은 gz_standard_vtol)
cd ~/PX4-Autopilot && HEADLESS=1 make px4_sitl gz_x500

# T2 — MAVROS (포트가 개발컴 문서의 14557이 아니라 14580 — PX4 v1.18 기준, 버전별 확인 필요)
ros2 launch mavros px4.launch fcu_url:=udp://:14540@localhost:14580

# T3 — 벤치 arm 테스트에만 필요(SITL이 전원/GCS를 시뮬레이션 안 함, 실기체엔 미적용)
ros2 param set /mavros/param CBRK_SUPPLY_CHK 894281
ros2 param set /mavros/param NAV_DLL_ACT 0

# T4 — fc_ros
cd ~/drone_ws && source install/setup.bash
ros2 launch fc_ros phase2.launch.py vehicle_type:=mc transition_alt:=3.0 waypoints:="[...]"
```

**설치 위치:** WSL 배포판 자체 `E:\wsl\Ubuntu-22.04\`, PX4-Autopilot 소스는 배포판 내부
`/root/PX4-Autopilot`(E드라이브 통과 아님 — 훨씬 빠름). `fc_bridge`는 위와 동일하게 `.pth` 방식.
**PX4 콘솔을 파일로 리다이렉트 금지**(또는 배포판 로컬 디스크로만) — `pxh>` 프롬프트가 비-TTY에서
폭주해 로그가 분 단위로 GB급이 됨. 상태확인은 `ros2 topic/service`·`ss -uln`으로.
**정리:** `wsl --unregister Ubuntu-22.04`(+`E:\wsl\Ubuntu-22.04` 수동삭제) — 재구축 비용 크므로
디스크 여유 있는 한 유지 권장.

### QGC ↔ WSL 연결 (PX4 재기동마다)

```bash
# Step 1 — IP 확인 (WSL)
WIN_IP=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'); echo "Windows IP: $WIN_IP"

# Step 2 — PX4 콘솔
pxh> mavlink start -x -u 14551 -r 4000000 -t <WIN_IP>

# Step 3 — QGC (Windows): Comm Links → Add → UDP 14551 → Connect
```

상세: `docs/sitl_verification_log.md` "Windows QGC ↔ WSL SITL 연결" 섹션.
