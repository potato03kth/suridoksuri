---
doc_type: session_log
project: suridoksuri-1
---

# 세션 로그

> 최신 세션이 위에 온다. `/session-log` 커맨드로 세션 종료 전 자동 작성.
> **최근 8개 세션만 유지** — 초과분은 `/session-log`가 `docs/archive/session_log_YYYY-MM.md`로 이동한다.
> 과거 기록: `docs/archive/session_log_2026-06.md` (2026-06-18 ~ 06-30) · `docs/archive/session_log_2026-07.md` (2026-07-03 ~)

---

## 2026-07-24 — [mc-hw] 2026-07-23 저녁 실비행 사고분석 — 오프보드 3m 상승명령이 30m로 실행 + 오프보드 미이행 원인 규명 → `_cb_home` 수렴판정 수정 → SITL 회귀검증 완료

**브랜치:** `dev--vision-computing-module`
**목적:** 사용자가 전날(2026-07-23) 저녁 비행에서 "오프보드로 3m 상승/남3m 이동/복귀/하강을 명령했는데 30m까지 상승했고, 이후 시야 이탈로 나머지 명령 이행 여부를 모른다"고 보고 — 원인 규명 요청. 비행 중 네트워크(컴패니언 연결)가 끊겨 자동 ulog 회수가 실패했을 것으로 추정된다는 정황도 함께 전달됨. 원인 규명 후 사용자가 "버그 재발을 막아라"고 후속 요청 — 재발방지 코드수정까지 이어짐.

### 완료

- **ulog 확보:** 지상국 Windows(호스트 E:\Downloads, WSL `/mnt/e/Downloads`)의 최신 ulog 2건을 확인 — `10_42_23.ulg`(PX4 부팅 19:40:31 KST), `11_32_15.ulg`(PX4 부팅 20:09:29 KST). `pyulog`(`ulog_info`)로 시작시각·소요시간 확인.
- **RPi(컴패니언, `suri@100.67.27.83`) 원격 조사** — 사용자 지시대로 현재 연결된 Pixhawk(오늘 비행한 기체와 다른 개체)에서 ulog를 받으려 하지 않고, `journalctl`로 해당 시각대(19:30~20:40 KST) 기록만 확인. **핵심 발견:** 이 시간대 RPi가 `hwmon hwmon4: Undervoltage detected!`를 동반한 연쇄 재부팅을 최소 3회(20:03/20:17/20:24~20:31 KST) 겪었고, `11_32_15.ulg` 비행의 arm(20:32:15) 불과 48초 전(20:31:27)에도 다운돼 이후 다음날 00:17까지 완전히 꺼져 있었음. `docker cp`로 `fc` 컨테이너(중지 상태)의 `bash_history`·`/root/.ros/log`를 안전하게(컨테이너 재시작 없이) 추출, `scp`로 `logs/2026-07-23_flight01/`(rosbag·launch.log·notes.md) 회수 — rosbag는 `metadata.yaml`이 없어 정상 Ctrl-C 종료가 아닌 비정상 중단으로 확인됨.
- **pyulog 정밀분석 (`11_32_15.ulg`, 88.2초):** `nav_state`가 POSCTL→AUTO_TAKEOFF(t=1.11s)→AUTO_LOITER(t=21.46s, PX4 자동전환)→POSCTL(t=55.88s, 조종사 개입)→MANUAL(t=56.71s)로 **OFFBOARD 진입 자체가 한 번도 없었음** 확인. `vehicle_command`엔 최초 이륙요청(t=1.09s, `MAV_CMD_NAV_TAKEOFF` param7=396.581m AMSL) 단 1건뿐, 이후 어떤 커맨드도 없음. 로그텍스트 `t=24.19s Connection to mission computer lost`·`t=55.88s Pilot took over using sticks`·`t=84.01s Kill engaged`·`t=87.16s Disarmed by landing`. AGL 시계열로 climb이 t≈0~22s에 0→29.7m까지 진행되고 AUTO_LOITER 진입 후에도 그대로 유지됨을 확인(`analyze_flight.py` 결과: AGL 최대 29.72m, home_position AMSL=366.93m).
- **근본원인 특정 (코드 대조):** RPi에서 회수한 `launch.log`에 `offboard_node`가 직접 남긴 로그 `CommandTOL 이륙 요청 alt=396.6m AMSL (지면 393.6+3.0) -> CLIMBING` 발견 — 의도한 상승폭(+3.0m)은 정확했으나 노드가 "지면 AMSL"로 쓴 값(393.6)이 ulog에 기록된 실제 EKF 홈고도(366.93)보다 **26.7m** 높았음. `fc_ros/fc_ros/nodes/offboard_node.py::_cb_home`(219~305행)이 `/mavros/home_position/home` 수신값을 단발 스냅샷하는 기존에 문서화된 설계 갭(`docs/session_status.md` mc-실기체 트랙 "잔여 리스크" — 종전엔 EKF/GPS 드리프트로 3~5m 규모)이 이번엔 훨씬 큰 폭으로 재현된 사례로 결론. `fc` 컨테이너/MAVROS가 이 비행 arm 수분 전(20:28~20:32 KST)에 막 재시작된 반면 PX4는 23분 앞서(20:09:29) 부팅돼 있었다는 점에서, 새 MAVROS 구독자가 PX4 부팅 초기(GPS 수직정확도 미수렴 구간)에 래치된 오래된 home_position을 그대로 받았을 가능성을 유력 가설로 제시(확정은 다음 비행에서 `_home_amsl` 수신시각·GPS eph/epv를 같이 남겨야 함).
- **flight1(`10_42_23.ulg`) 정체 확인:** RPi가 완전히 다운돼 있던 시간대(18:23~20:03 KST)의 별개 비행 — `vehicle_command` 토픽 자체가 없어 오프보드/미션컴퓨터와 무관한 순수 수동 POSCTL 비행(컴퍼스/배터리 경고 반복 관측)으로 확인, 오늘 사고와 무관.
- **커밋 자료화:** 두 ulog + RPi에서 회수한 rosbag/launch.log를 `logs/2026-07-23_flight01/`(사고 비행)·`logs/2026-07-23_manual/`(무관 비행, 신규)에 배치, `analyze_flight.py`로 `analysis_auto.md` 생성, `notes.md` 관찰/결론 채움. `docs/session_status.md` 🚁 mc-실기체 트랙 "마지막" 갱신.
- **재발방지 코드수정 (사용자 요청 "버그 재발을 막아라"):** `fc_bridge/execution/state_logic.py`에 순수함수 `home_amsl_confirmed(samples, tol=0.5, min_samples=3)` 신규 — 최근 `min_samples`개 home_position AMSL 샘플이 `tol` 이내로 수렴해야 신뢰할 값을 반환, 아니면 `None`. `offboard_node.py::_cb_home`을 단발 대입에서 샘플 누적(`_home_amsl_samples`, 최근 20개 유지) + `home_amsl_confirmed()` 확정 판정으로 변경, `_step_arm_takeoff`는 미확정 상태면 CommandTOL을 계속 보류하며 최근 샘플·tol을 로그로 노출(기존 "home_position 미수신" 경고를 "미수신"/"미수렴" 두 경우로 구분). 새 ROS 파라미터 `home_amsl_tol`(0.5)·`home_amsl_min_samples`(3) 추가. `fc_ros/test/test_offboard_node.py`에 회귀테스트 9건 추가(단발 스테일 스냅샷 미신뢰·26.7m 규모 드리프트 거부·트레일링 윈도만 검사·경계값 등) — `pytest fc_ros/test/ fc_bridge/` **162 전부 통과**(기존 153 + 신규 9).
- **rosbag 손상 확인:** `_home_amsl`이 실제로 어떤 시점에 어떻게 393.6→366.9로(또는 애초에 393.6만) 수신됐는지 `/mavros/global_position/global` 이력으로 직접 대조하려 했으나, 회수한 `rosbag_0.db3`가 `database disk image is malformed`로 손상돼(비정상 종료의 추가 물증) 불가 — 메커니즘은 정황 기반 가설로 남음.
- **SITL 회귀검증 완료 (사용자 후속 지시 — 이 노트북엔 SITL이 없다는 지적 → "이 노트북에 새로 구축" 선택, E드라이브에 설치 지정):** 개발컴과 별개인 이 노트북(24.04)에 SITL이 전혀 없었음을 확인 → 새 WSL 배포판(`Ubuntu-22.04`, `wsl --import`로 E드라이브에 설치, Canonical jammy rootfs)에 ROS2 Humble+MAVROS+PX4-Autopilot(`gz_x500`)을 신규 구축. 과정에서 겪은 문제 3건과 해결책을 `docs/wsl_dev_env_setup.md` 섹션 F에 기록: ①`fc_bridge`를 `cd fc_bridge && pip install -e .`로 설치하면(기존 섹션 E 문구, 틀렸던 것으로 판명) 네임스페이스가 깨지고 `ros2` CLI 자체가 `PackageNotFoundError: ros2cli`로 죽음 → 저장소 루트를 가리키는 `.pth` 파일 방식으로 해결. ②SITL은 실제 전원/GCS 하드웨어를 시뮬레이션하지 않아 `CBRK_SUPPLY_CHK`/`NAV_DLL_ACT` 두 파라미터로 프리플라이트를 우회해야 arm 가능(실기체 파라미터에는 미적용). ③PX4 콘솔을 파일로 리다이렉트하면 `pxh>` 프롬프트가 비-TTY 재출력 루프에 빠져 로그가 수 분 만에 GB급으로 폭주(2.1GB까지 확인) — 상태확인은 `ros2 topic/service`로, 콘솔은 배포판 로컬 디스크에만. **실증 결과:** disarm 상태에서 60초간 `home_position` 관찰 → **약 34회 재발행 확인**(우려했던 "PX4가 단발만 보내 무기한 대기" 가능성 기각), 도중 47.46m→48.97m 드리프트도 실측. 사고와 동일한 launch 인자로 `offboard_node` 재현 실행 → `home_position AMSL 미수렴(1개→2개)` 경고 후 3번째 표본에서 수렴, `CommandTOL 이륙 요청 alt=50.5m AMSL (지면 47.5+3.0)`로 **정확한 고도** 계산 확인(26.7m류 오차 재현 안 됨) → CLIMBING→STREAMING→OFFBOARD→FOLLOWING→HOLD→LANDING→disarm까지 **미션 전체를 정상 완주**(AUTO_LOITER에 갇히지 않음, 사고와 대조적). STREAMING 진입 직후 AGL이 순간 5m대로 오버슈트했다가 정상화되는 현상을 관측했으나 이번 수정과 무관한 별개 사안으로 판단(코드 미수정, 기록만).

### 결정

- **`_cb_home` 수렴판정 방식(N회 연속 tol 이내)을 채택** — `docs/session_status.md`에 이미 기록돼 있던 권고안 (a)를 그대로 구현. 대안(CommandTOL 직전 능동 재조회)은 MAVROS/PX4 쪽에 새 요청-응답 프로토콜이 필요해 더 무거워 채택 안 함.
- **SITL 검증에서 드러난 STREAMING 오버슈트는 이번 커밋 범위에 포함하지 않음** — `home_amsl_confirmed()` 수정과 별개 메커니즘(OFFBOARD 전환 지연 중 상승 지속, 기존에 알려진 잔여 리스크)으로 판단, 별도 이슈로 분리.
- **RPi 쪽 조사는 journalctl/컨테이너 로그 열람만 수행** — 사용자 지시대로 현재 연결된(다른) Pixhawk에서 ulog를 받으려는 시도는 하지 않음. `fc` 컨테이너는 `docker cp`로만 접근(재시작 없이 정지 상태 그대로 유지).
- **SITL 배포판은 삭제하지 않고 유지** — 재구축 비용이 커서(ROS2 Humble+PX4 클론/빌드) 디스크 여유(184GB) 있는 한 다음 세션도 재사용 가능하게 남겨둠. 프로세스만 정리.

### 다음 세션

1. **RPi 전원계통(BEC/배터리 분배) 점검** — undervoltage 경고가 실비행 중 반복 재현됐으므로 다음 비행 전 전원 안정화(PD 트리거 or 별도 레귤레이터) 필요, `project_rpi5_usbc_power_psu_max_current` 메모리와 연계.
2. `record_flight.sh`가 rosbag 비정상종료(`metadata.yaml` 없음)를 감지·경고하도록 개선 검토(미구현 아이디어).
3. **STREAMING 진입 직후 AGL 오버슈트(5m대, 목표 3m) — 자기완결 브리프 준비됨.** 사용자 지시("머지 후 계속하겠다, 새 세션에서")로 원본 SITL ulog(`logs/2026-07-24_sitl_streaming_overshoot/05_07_03.ulg`)를 회수·1차 분석(`nav_state` 타임라인상 오버슈트 대부분이 `offboard_node`가 아직 OFFBOARD 권한 없는 PX4 AUTO_TAKEOFF 구간에서 발생 — `climbing_reached()`가 속도 무시하고 위치만으로 "도달" 판정하는 게 유력 원인 후보)까지 마치고 `docs/mc_hw_next_session_brief.md` 신설. **다음 세션은 이 문서로 진입.**

---

## 2026-07-22f — [mc-hw] "PID 튜닝 미실시" 가설 검토 + H14 신설 (분석만, 코드 미수정)

**브랜치:** `dev--vision-computing-module`
**목적:** 사용자가 "현재 비행문제가 PID 튜닝을 한 번도 안 한 데 있는 것 아니냐"는 가설을 검토하고, 텔레메트리 없이 자동 PID 튜닝하는 방법을 고민해달라고 요청.

### 완료

- **가설 타당성 검토:** `docs/mc_hw_open_hypotheses.md`의 기존 H1~H13을 재검토한 결과, "제어" 분류(H10 EKF 왜곡·H11 얼로케이터 기하)는 이미 조사되어 거의 기각된 상태라 PID 게인 자체는 한 번도 의심된 적이 없었음(맹점 확인). 2026-07-21 재분석에서 확립된 관측("roll 오차가 27~33°로 커질 때까지 `unallocated_torque[0]`≈0, 즉 얼로케이터가 전혀 포화되지 않음")을 새 관점으로 재해석: 게인이 이 기체의 관성/추력특성에 맞게 충분했다면 오차가 작을 때부터 비례+미분항이 강하게 반응해 초기에 잡혔어야 함 — 느린 초기반응 자체가 게인 부족의 전형적 시그니처. 이 기체가 비표준 모터/프롭 조합(H13에서 CW/CCW 비대칭 확인 — PX4 기본 게인이 가정하는 표준 프레임과 물리특성이 다름을 시사)인데도 PID 튜닝을 수행했다는 기록이 세션로그·트랙보드 어디에도 없음 — "한 번도 안 했다"는 전제 자체는 정황상 타당. 다만 H12(이함 트랜지언트)·H13(CW/CCW 추력비대칭)을 대체하는 게 아니라 보완하는 관계로 정리(H13=상시 교란원, H14=왜 그 교란을 못 버텼는가).
- **텔레메트리 없는 자동 PID 튜닝 방법 정리:** PX4 v1.14+(이 기체 펌웨어, `docs/pixhawk6c_rpi4_integration_guide.md` 확인)는 `mc_autotune_attitude_control` 모듈로 온보드 자동 오토튠을 지원 — 안전한 호버 중 축별로 소진폭 교란을 주입해 응답을 시스템식별하고 `MC_ROLLRATE_P/I/D`, `MC_PITCHRATE_P/I/D` 등을 자동 계산, 지상국으로 로그를 스트리밍해 오프라인 분석할 필요가 없음(식별+게인계산이 FC 온보드에서 완결). 트리거(`MAV_CMD_DO_AUTOTUNE_ENABLE`)·진행상태 모니터링·최종 커밋까지 전부 RPi5↔FC 기존 MAVLink 링크(mavros)로 처리 가능해, 지상국 실시간 텔레메트리 스트리밍이 불안정하거나 없어도(이 프로젝트는 RPi5 WiFi 불안정 이력 있음, `project_rpi5_tailscale_wifi_drops`) 무관하게 실행 가능하다는 점이 핵심.
- **문서 갱신:** `docs/mc_hw_open_hypotheses.md`에 H14 신규 행 추가(테이블·이력), 검증 우선순위에 "0번(비용 0, 게인값 대조)"·"7번(테더 상태 온보드 오토튠, roll 단독부터)" 추가. `docs/session_status.md` 🚁 mc-실기체 트랙 "마지막" 갱신.

### 결정

- **코드 수정 없음** — 이번 요청은 가설 검토·조사방법 정리이며 실비행 코드(`offboard_node.py` 등)에는 손대지 않음.
- **우선순위 0번(파라미터 대조)을 물리 벤치보다 먼저 하도록 배치** — H13의 모터별 정적 추력벤치(기존 1순위)는 물리 작업이 필요하지만, PID 게인 대조는 FC 파라미터 조회만으로 5분 내 끝나는 무료 확인이라 다음 세션 시작 즉시 먼저 처리하는 게 합리적.
- **오토튠은 반드시 테더(안전줄) 상태로, roll 축 단독부터** — 이 기체는 이함 직후 롤 폭주(제어상실) 사고 이력이 있어(H12/H13 문서 전체), 오토튠의 소진폭 교란 주입 중 예상 밖 반응이 나올 가능성에 대비.

### 다음 세션

1. **(비용 0, 최우선)** FC에서 `MC_ROLLRATE_P/I/D`, `MC_PITCHRATE_P/I/D`, `MC_ROLL_P`, `MC_PITCH_P` 조회 → PX4 v1.14 기본값과 대조. 동일하면 "한 번도 안 튜닝됨" 사실상 확정.
2. 위 확인 후 테더 상태에서 PX4 온보드 오토튠(roll 축부터 단독) 실행 — `docs/mc_hw_open_hypotheses.md` H14·우선순위 7번 참조.
3. 기존 H13 최우선 항목(모터별 정적 추력벤치, CW쌍 vs CCW쌍)은 그대로 유지 — H14와 순서 상충 없음(둘 다 지상 작업, 순서만 비용순으로 배치).

---

## 2026-07-22e — [vision] rpi_capture.py 수동 초점/노출/게인 제어 추가 — 체커보드 촬영 흐림+과다어둠 원인 대응

**브랜치:** `dev--vision-computing-module`
**목적:** 사용자가 체커보드를 40cm/210cm에서 촬영했는데 두 거리 모두 초점이 안 맞고 과하게 어두웠던 문제 대응. 원인: V4L2 raw 직접 캡처 경로가 libcamera를 완전히 우회해 연속 AF/AE가 전혀 없음 — 오토포커스 렌즈(dw9807 VCM)의 `focus_absolute`가 기본값(480)에서 한 번도 안 움직였고 `exposure`/`analogue_gain`도 방치돼 있었음.

### 완료

- **`set_focus_absolute()`/`set_exposure_gain()`(`vision/tools/rpi_capture.py`) 신규** — `--focus`(0~1023)/`--focus-settle-ms`(기본 200ms)/`--exposure`/`--gain` CLI로 노출. focus는 VCM 물리 이동 정착시간 확보를 위해 설정 후 지정된 시간만큼 대기.
- **HTTP 미리보기 페이지에 실시간 조정 UI 추가** — focus/exposure/gain 숫자입력폼 + focus -10/+10 빠른버튼(`GET /controls`, 303 리다이렉트, 새 JS 없이 기존 순수 HTML 폼 패턴 유지). `_CaptureSession`이 "원하는 값"과 "마지막 적용값"을 분리해 실제로 바뀐 경우에만 하드웨어에 적용.
- **`--focus-sweep START:END:STEP` 신규** — focus_absolute를 스윕하며 각각 촬영, 중앙 크롭(`--sweep-roi`, 기본 0.6) 기준 라플라시안 분산이 가장 높은 값을 보고. 순수 함수(`laplacian_sharpness`/`crop_center_fraction`/`pick_best_focus`/`parse_focus_sweep_spec`)로 분리해 하드웨어 없이 단위테스트 가능.
- **단위테스트 34개 신규(`vision/tests/test_rpi_capture.py`)** — 합성 체커보드로 라플라시안 선명/블러 비교, ROI 크롭 중앙정렬, 스윕 결과 최적값 선택(동점 처리 포함), `START:END:STEP` 파싱 에러케이스, `set_focus_absolute`/`set_exposure_gain` 범위검증·v4l2-ctl 명령 구성·정착시간 sleep 호출(모두 `_run`/`time.sleep` 몽키패치), `_CaptureSession`의 지연 적용 로직(값 안 바뀌면 하드웨어 재호출 안 함), `_render_page` 값 반영. `pytest vision/tests/` **189 passed**(기존 155 + 34).
- **실기체 조사(pseudo 테스트 아님) — 노출 개선을 실측으로 확인:** 방치된 기본값(`exposure=874,gain=112`)으로 어두운 실내 `--single-shot` → mean=25.23. 수동으로 `exposure=2400,gain=800` → **mean=126.74(약 5배)**. 노출 방치가 "과하게 어둡다" 증상의 실제 원인임을 확인.
- **실기체 조사 — 초점 스윕이 "평평하다"는 최초 결과는 측정 방법 문제였음을 규명:** 노출 개선 후에도 전체 프레임 기준 step=100 스윕(0~1023)은 평평했다(9~11 범위) — 오케스트레이터의 최초 관찰(150 vs 480 거의 동일)과 일치해 "VCM이 안 움직이는 게 아닌가" 의심됐으나, 원인은 카메라가 바닥에 낮게 고정돼 있어 프레임 하단 대부분이 렌즈 최단 초점거리보다 가까운 바닥이라 어떤 focus 값에서도 항상 흐린 것 — 이 영역이 전체 프레임 분산을 압도해 배경의 실제 초점 변화를 가렸다. **중앙/배경 ROI로 제한 + step을 20으로 좁히자** 배경 클러터(약 1.5~2m) 영역에서 뚜렷한 피크가 나타남: 1차 스윕 `focus=560`(선명도 18.20, baseline ~14.8), 2차 독립 스윕(다른 시각, 주변광 변화 후) `focus=580`(선명도 13.07, baseline ~10.3) — **두 번 모두 540~600 좁은 구간에 재현되는 피크**, 같은 값 반복 촬영은 편차 ±0.1 이내로 안정적. `--focus-sweep 460:660:40` 도구 실행 결과도 `focus=580`으로 수동 조사와 일치.
- **정착시간 실측:** 0~500ms(및 극단값 전환 0~2000ms) 범위에서 지연에 따른 결과 차이를 못 찾음 — VCM 정착이 이 범위보다 훨씬 빠르다는 뜻으로 해석. 기본값 200ms는 세션 지시 추정범위(100~300ms)의 중간값으로, 실측이 부정하지 않는 보수적 선택.
- **RPi 실배포 검증:** 로컬에서 커밋·푸시 후 RPi에서 `git checkout origin/... -- vision/tools/rpi_capture.py vision/tests/test_rpi_capture.py`로 갱신, `--single-shot --focus 560 --exposure 2400 --gain 800`, `--focus-sweep 460:660:40`, HTTP 서버(`/`, `/controls?focus=560`, `/preview.jpg`, `/capture`) 전부 실제로 기동/호출해 정상 동작 확인.
- `vision/CLAUDE.md`에 "rpi_capture.py 수동 초점/노출/게인 제어" 절 신규 — 위 조사 경과 전체(실측 수치·왜 최초 스윕이 평평했는지·선명도 지표 신뢰 전제조건 3가지·다음 캘리브레이션 촬영 가이드) 기록.

### 실제 테스트

```
pytest vision/tests/   # 189 passed (기존 155 + 신규 test_rpi_capture.py 34개)
```
RPi 실기체: `--single-shot`(focus/exposure/gain 지정) 성공, `--focus-sweep` 성공(피크 재현), HTTP `/`·`/controls`·`/preview.jpg`·`/capture` 전부 실제 curl로 응답 확인.

### 결정

- **exposure/gain 범위는 하드코딩하지 않기로 함** — focus_absolute(렌즈 드라이버 고유 범위, 센서 모드 무관)와 달리 exposure 최댓값은 센서 모드(vertical_blanking)에 따라 달라질 수 있어, 특정 모드에서 실측한 범위를 검증에 하드코딩하면 다른 모드에서 유효값을 잘못 거부할 위험이 있음. 값을 그대로 v4l2-ctl에 넘기고 범위 위반은 v4l2-ctl 자체가 거부하게 둠.
- **초점 스윕 ROI 기본값을 중앙 60% 크롭으로 정함** — 전체 프레임 기준 지표가 근접 배경(카메라 초점범위 밖)에 압도되는 문제를 실측으로 확인했고, 캘리브레이션 타겟은 보통 화면 중앙에 두고 촬영하므로 중앙 크롭이 일반적으로 합리적인 기본값이라 판단. `--sweep-roi`로 조정 가능하게 열어둠.
- **HTTP 컨트롤 UI는 최소 확장으로 제한** — 슬라이더/JS 없이 숫자입력폼 1개 + focus 빠른버튼 2개만 추가(세션 지시의 "과설계 금지"에 따름).

### 다음 세션

1. **40cm급 근접 거리에서의 초점 피크는 이번 세션에서 확인 못 함(원격 세션이라 체커보드/타겟을 직접 들거나 옮길 수 없었음)** — 다음 실촬영 세션(사람이 물리적으로 참여 가능할 때)에서 체커보드를 40cm와 210cm에 각각 들고 `--focus-sweep`를 두 거리에서 따로 돌려 실제로 다른 최적값이 나오는지 확인. 메커니즘(VCM이 실제로 움직이고 거리별로 다른 지점에서 피크가 남)은 이미 배경 물체로 증명됐으므로 남은 건 정량 확인뿐.
2. 위 1번이 확인되면 실제 체커보드 캘리브레이션 촬영(§7.9 항목2, `calibrateCamera`) 착수.
3. `LiveFrameSource`(`vision/utils/frame_source.py`) 재구현 — 여전히 대기 중(2026-07-22b부터 인계).
4. 골든셋 실촬영 데이터 교체 — 여전히 미착수.

### 주의

> 노출/게인 실측값(`exposure=2400~2602`, `gain=800~900`)은 **이번 세션 저녁 실내 조도 기준 참고값일 뿐, 고정값이 아니다** — 세션 내에서도 주변광이 바뀌며 같은 설정의 mean이 60~207 사이를 오갔다. 매 촬영 세션마다 HTTP 미리보기로 밝기를 재확인할 것. 초점 피크(540~600)도 카메라-배경 거리(~1.5~2m)에 종속된 값이라 다른 거리에 그대로 적용하면 안 됨 — 거리마다 `--focus-sweep` 재실행 필요.

---

## 2026-07-22d — [vision] rpi_capture.py `_MEDIA_DEVICE` 하드코딩 버그 수정(동적 탐색) + gray-world 화이트밸런스 실카메라 검증 마무리

**브랜치:** `dev--vision-computing-module`
**목적:** 오케스트레이터가 직전 세션 종료 직후 RPi를 직접 재확인하다 발견한 버그(`_MEDIA_DEVICE = "/dev/media1"` 하드코딩이 재부팅/재연결마다 바뀌어 `configure_pipeline()`이 `CalledProcessError`로 실패) 수정 + 2026-07-22c가 RPi 오프라인으로 못 끝냈던 gray-world 화이트밸런스 실카메라 검증 마무리.

### 완료

- **버그 재현:** RPi 원격 저장소를 `dev--vision-computing-module`(당시 `376aa3d`)로 갱신한 뒤 `--single-shot` 실행 → `media-ctl -d /dev/media1 -l ...`가 `CalledProcessError`(exit 1)로 실패하는 것을 실제로 재현. 원인 조사: RPi에서 `for d in /dev/media*; do media-ctl -d $d -p; done`으로 5개 media 디바이스를 전수 조사 — `/dev/media0`/`/dev/media1`=`pispbe`(ISP 백엔드, 무관), `/dev/media2`=`rp1-cfe`이지만 링크 0개·`imx708` 엔티티 없음(연결 안 된 CSI 포트), `/dev/media3`=`rp1-cfe`이고 `imx708` 센서 엔티티가 실제로 있음(진짜 카메라 파이프라인), `/dev/media4`=`rpivid`(무관). 즉 driver명(`rp1-cfe`) 일치만으로는 media2/media3 둘 다 걸려 판별이 안 되고, **센서 엔티티(`imx708`) 존재 여부까지 봐야 정확히 가려짐** — 이걸 직접 RPi에서 두 디바이스의 실제 `media-ctl -p` 출력을 비교해 확정.
- **`_find_cfe_media_device()` 신규(`vision/tools/rpi_capture.py`)** — `/dev/media*`를 정렬 순회하며 각각 `media-ctl -p`를 돌려 `_media_ctl_topology_has_camera()`(driver==`rp1-cfe` && `imx708` 엔티티 존재, 정규식 파싱)로 판별. **캐시하지 않음** — `configure_pipeline()`이 호출될 때마다(매 프레임 캡처마다) 새로 탐색해 항상 최신 상태를 봄. 못 찾으면 확인한 모든 디바이스+driver명을 담은 `RuntimeError`. `_iter_media_device_paths()`(`/dev/media*` 목록 취득)를 별도 함수로 분리해 테스트에서 몽키패치 가능하게 함. `configure_pipeline()`의 6개 `_MEDIA_DEVICE` 참조를 전부 `_find_cfe_media_device()` 호출로 교체. `unpack_raw10()`/`debayer_to_bgr8()`/화이트밸런스 로직은 손대지 않음(세션 지시대로 media 디바이스 탐색 부분만 수정).
- **실제 재검증(pseudo 테스트 아님):** 수정된 코드를 RPi 원격 저장소에 배포 후 `--single-shot` 재실행 → 성공(`/dev/media3`을 자동으로 찾아 사용, 4608x2592 정상 촬영). `--white-balance`/`--no-white-balance` 양쪽 다 재확인 — 아래 화이트밸런스 항목 참조.
- **단위테스트 13개 신규(`vision/tests/test_rpi_capture.py`)** — `_media_ctl_topology_has_camera`/`_media_ctl_driver_name`은 RPi에서 실제로 받은 `media-ctl -p` 출력 3종(연결된 rp1-cfe/연결 안 된 rp1-cfe/무관한 pispbe, 핵심 라인만 남긴 실측 텍스트 그대로 픽스처화)으로 검증. `_find_cfe_media_device`는 `_run`/`_iter_media_device_paths`를 몽키패치해 여러 디바이스 중 정답 선택·enumeration 순서 무관하게 찾음·전부 불일치 시 디바이스 목록 포함 에러·media 디바이스 자체가 없을 때 에러·일부 디바이스 조회 실패(`CalledProcessError`)해도 나머지로 계속 찾는 것까지 검증(subprocess 호출은 몽키패치, 파싱/탐색 로직 자체는 진짜 실행 — 하드웨어 없이 노트북에서도 통과).
- **`pytest vision/tests/` 155 passed**(기존 142 + 신규 13).
- **gray-world 화이트밸런스 실카메라 검증 마무리** — 위 수정이 적용된 코드로 같은 RPi 실행에서 재확인: **보정 끄기 B=64.96 G=101.45 R=69.31(spread 36.49)** / **보정 켜기 B=75.02 G=78.12 R=76.15(spread 3.10)**, spread 약 11.8배 감소. 코드(2026-07-22c에 이미 커밋)는 그대로, 실측 검증만 이번에 닫음.
- `vision/CLAUDE.md`에 "rpi_capture.py media 디바이스 동적 탐색" 절 신규(왜 번호가 바뀌는지·판별 기준·구현·테스트 요약) + "gray-world 화이트밸런스" 절에 실측 수치 반영 + 파일역할표 `tools/rpi_capture.py` 행을 최신 상태(V4L2 브링업 완료, 동적탐색 반영)로 갱신. `docs/vision_status.md` 트랙 블록 갱신(이번 완료 반영, "다음 세션 진입 시 실행할 명령" 블록은 완료 표시, `LiveFrameSource` 재구현이 여전히 다음 순번임을 명시).

### 실제 테스트

```
pytest vision/tests/   # 155 passed (기존 142 + 신규 test_rpi_capture.py 13개)
```
RPi 실기체: `--single-shot` 수정 전 실패 재현 → 수정 후 성공 재현(둘 다 실제 프로세스 실행, 로그 확인).

### 결정

- **media 디바이스 판별 기준은 driver명 단독이 아니라 "driver + 센서 엔티티 존재"로 정함** — RPi에 driver가 같은 `rp1-cfe` 인스턴스가 2개(연결된 CSI 포트/연결 안 된 CSI 포트) 있어 driver명만으로는 모호함을 실측으로 확인했기 때문. 힌트로 주어졌던 "csi2 링크 수 0 아님"도 동작하긴 하지만, 센서 엔티티 존재 여부가 더 직접적이고("이 디바이스에 우리가 쓰려는 카메라가 실제로 물려있는가") 이후 `configure_pipeline()`이 어차피 `imx708` 엔티티를 대상으로 명령을 보내므로 같은 이름을 판별에도 재사용하는 게 자연스럽다고 판단.
- **탐색 결과를 캐시하지 않기로 함** — media 디바이스 번호가 프로세스 실행 도중 바뀔 가능성은 낮지만, `configure_pipeline()`은 촬영마다(프리뷰 틱마다) 호출되므로 매번 재탐색해도 비용이 크지 않고(디바이스 5개 정도, RPi에서 체감 지연 없음) 정확성을 우선했다. `_VIDEO_DEVICE`/`_CSI2_SUBDEV`/`_SENSOR_SUBDEV`는 이번 조사에서 안정적으로 재현돼 손대지 않음(세션 스코프 밖이기도 함).
- **RPi 시스템 레벨 변경 없음** — 이번 세션은 순수 파이썬 코드 수정 + 원격 저장소 파일 갱신(`git checkout origin/... -- <경로>`로 개별 파일만)만 수행. media-ctl/v4l2-ctl 런타임 상태 변경은 스크립트 실행 중에만 발생하고 프로세스 종료 시 원래대로(기존 관례와 동일).

### 다음 세션

1. **`LiveFrameSource`(`vision/utils/frame_source.py`) 재구현** — 여전히 다음 순번(2026-07-22b부터 인계됨, 이번 세션이 끼어든 버그 수정으로 순서가 밀리지 않음). `cv2.VideoCapture` 기반 현재 구현이 V4L2 raw(`pRAA`) 경로와 비호환임이 실측 확인돼 있음 — `configure_pipeline()`/`capture_frame_bgr()` 재사용 방향 추천.
2. 체커보드 캘리브레이션(§7.9 항목2) — 여전히 미착수.
3. 골든셋 실촬영 데이터 교체 — 여전히 미착수.

### 주의

> `rpi_capture.py`를 다시 만질 때 `_MEDIA_DEVICE` 유사 패턴(디바이스 번호 하드코딩)을 새로 추가하지 말 것 — 이번에 실측으로 번호가 안 고정된다는 게 확인됨. `_find_cfe_media_device()` 패턴(driver+엔티티 존재로 판별, 매 호출 재탐색)을 재사용할 것.

---

## 2026-07-22c — [vision] rpi_capture.py gray-world 화이트밸런스 보정 추가 (실카메라 검증은 RPi 오프라인으로 미완)

**브랜치:** `dev--vision-computing-module`
**목적:** 직전 세션(2026-07-22b)의 실촬영 검증에서 발견된 raw 베이어 경로의 강한 초록 화이트밸런스 편향(ISP/libcamera 완전 우회라 화이트밸런스 미적용)을 고쳐, 이후 HSV 색상 탐지(`distress_coarse.yaml` 등) 실측 검증 시 왜곡 요인이 되지 않게 함. 체커보드 캘리브레이션은 사람이 실물 체커보드를 준비해야 해서 병행 대기 중이라 그 사이 처리.

### 완료

- **`vision/tools/rpi_capture.py`에 `apply_gray_world_white_balance(bgr8)` 신규** — `debayer_to_bgr8()`와 분리된 순수 함수. gray-world 가정("전체 이미지 R/G/B 채널 평균이 같아야 한다")에 따라 채널별 평균의 평균(회색 기준값)에 각 채널을 맞추는 게인(`gray/mean_c`)을 곱하고 0~255로 클립. 완전 검은 이미지는 0나눗셈 회피로 원본 반환. numpy만 사용.
- **`debayer_to_bgr8(..., white_balance=True)` 옵션 추가** — 기본 켜짐. `capture_frame_bgr()`/`_CaptureSession`/CLI 전부 관통. CLI에 `--white-balance`/`--no-white-balance`(`argparse.BooleanOptionalAction`) 추가, `--single-shot` 출력에 B/G/R 채널 평균 추가.
- **단위테스트 8개 신규**(`vision/tests/test_rpi_capture.py`) — 합성 초록편향 이미지 중화 검증·중립 이미지 near-noop·형상/dtype 보존·완전 검은 이미지 0나눗셈 회피·에러 케이스 2건·`debayer_to_bgr8` 기본값 검증·`white_balance` on/off 채널격차 비교. 기존 RGGB 채널순서 회귀테스트는 새 기본값(WB on)이 그 테스트의 의도적 채널 밝기차를 지워버려 실패하게 된 것을 `white_balance=False` 명시로 수정. `pytest vision/tests/` **142 passed**(기존 134 + 신규 8).
- **`vision/CLAUDE.md`에 "rpi_capture.py gray-world 화이트밸런스" 절 신규** — 채택 근거, §5.5 "흰 박스 화이트 앵커"와의 관계(다른 레이어), 구현 요약, CLI 플래그 기록.
- `docs/vision_status.md` 트랙 블록 갱신.

### 실제 테스트

```
pytest vision/tests/   # 142 passed (기존 134 + 신규 test_rpi_capture.py 8개)
```

### 미완료 — RPi 실카메라 검증 (블록됨)

**세션 내내 RPi(`doksuri-3`, 100.67.27.83)가 tailscale 오프라인이었음.** `tailscale status`에 `active; relay "tok"; offline, last seen 2h ago`로 일관 표시(수 분간 재확인해도 변화 없음, tx만 계속 증가 — 이쪽에서 보내는 패킷은 있으나 RPi 응답 없음), 직접 SSH도 `Connection timed out`. RPi5 WiFi 장기끊김 커널버그(brcmfmac, `project_rpi5_tailscale_wifi_drops.md`)와 증상이 다르고(그 이슈는 재연결까지 8분+ 걸리는 패턴이지 완전 무응답은 아니었음), 이번 세션 시간 내(수 분 대기)에도 복구 안 됨 — 원격으로 할 수 있는 진단이 없어 원인 규명은 하지 않음(범위 밖). 세션 지시("pseudo 테스트 금지, 실제 카메라로 찍은 실제 이미지로 검증")를 지키기 위해 실카메라 검증 없이 "완료"로 덮지 않고 명시적으로 미완료 처리.

- 순수 함수(그레이월드 게인 계산·적용) 자체는 합성 테스트로 완전히 검증됨 — 하드웨어 없이 가능한 부분은 다 함.
- 실측 검증(보정 전/후 실제 촬영 이미지의 B/G/R 채널 평균 비교)만 RPi 온라인 대기 중.
- 다음 세션에서 즉시 실행할 명령은 `docs/vision_status.md` 해당 항목에 그대로 복붙 가능하게 기록해 둠.

### 결정

- **RPi 오프라인 상태에서 실측 결과를 추정/생성해 보고하지 않음** — 세션 지시("pseudo 테스트 금지")를 지키기 위해 코드·단위테스트·문서까지만 완료하고 실측은 정직하게 미완료로 남김. 5분간 tailscale 재연결을 기다렸으나 상태 변화 없어 대기를 중단하고 나머지 작업(문서화·커밋)으로 전환.
- 체커보드 캘리브레이션(§7.9 항목2)과 `LiveFrameSource` 재구현은 이번에도 손대지 않음 — 이번 세션 스코프는 화이트밸런스로 한정.

### 다음 세션

1. **최우선:** RPi 온라인 확인 후 `docs/vision_status.md` "다음 세션 진입 시 실행할 명령"으로 gray-world 화이트밸런스 실카메라 검증(보정 전/후 B/G/R 채널 평균 비교) 완료.
2. RPi가 왜 이번엔 완전 무응답이었는지(기존 brcmfmac 이슈와 다른 패턴) 필요시 다음 실비행 전 재확인 — 이번 세션에서 원인 규명 안 함.
3. 이후 순서는 기존과 동일: 체커보드 캘리브레이션 → `LiveFrameSource` 재구현 → 골든셋 실촬영 데이터 교체.

### 주의

> RPi 하드웨어를 전혀 건드리지 못한 세션 — SSH 연결 자체가 안 됐으므로 시스템 레벨 변경 여부를 논할 대상 자체가 없었음(코드/문서 변경만 노트북/WSL 로컬에서 수행).

---

## 2026-07-22b — [vision] RPi5 카메라 V4L2 RAW 직접 캡처 브링업 완료 + 실촬영 검증

**브랜치:** `dev--vision-computing-module`
**목적:** 6세션 전(2026-07-21a) libcamera가 RPi5용 PiSP IPA 모듈 없이 빌드돼 있어 카메라 브링업이 막힌 채 중단된 상태였고, 그동안 사용자가 실비행을 이유로 RPi SSH/실카메라 작업을 전면 금지해 카메라 독립 대체 트랙(§7.9 4·5·6·7번)만 진행해 왔음. 이번 세션에서 그 금지가 해제되어 카메라 브링업을 재개, V4L2 RAW 직접 캡처(사용자 확정 방향)로 실제 완료.

### 완료

- **rp1-cfe 미디어 파이프라인 브링업 성공(libcamera 완전 우회).** `/dev/video0`가 MC-centric 캡처 노드라 단순 `v4l2-ctl --set-fmt-video`만으론 `VIDIOC_STREAMON`이 `-EPIPE`로 실패하던 것을, `dmesg` dynamic_debug로 커널 로그를 추적해 3개 원인(링크 비활성·field 불일치·임베디드 메타데이터 패드 폭 불일치)을 순서대로 확정·해결. 상세 명령·근거는 메모리 `project_rpi5_ubuntu_camera_stack.md` "✅ 브링업 완료" 절과 `vision/tools/rpi_capture.py` 모듈 docstring에 기록.
- **`vision/tools/rpi_capture.py` 전면 재작성** — 작동 불가였던 GStreamer `libcamerasrc` 버전을 media-ctl/v4l2-ctl 기반 파이프라인 구성 + `unpack_raw10()`(MIPI RAW10 언패킹) + `debayer_to_bgr8()`(수동 디베이어)로 교체. `--single-shot` 플래그 신규(원격 검증용). HTTP 미리보기/촬영 서버 구조는 유지.
- **실제 촬영 검증(pseudo 테스트 아님):** RPi에서 실제 실행해 4608x2592 실사진 획득 → 노트북으로 가져와 육안 확인(천장 조명·문·서랍장이 보이는 실제 방 사진). 두 번째 프레임과 바이트 비교해 52.8% 픽셀이 다름을 확인 — 캐시 아닌 라이브 캡처임을 통계로 증명. 640x480 요청 시 1536x864로 스냅되는 것도 실측 확인.
- **`vision/tests/test_rpi_capture.py` 신규(8개)** — `unpack_raw10`/`debayer_to_bgr8` 순수 함수 단위 테스트(왕복 검증·경계값·RGGB 채널 순서 검증). `pytest vision/tests/` **134 passed**(기존 126 + 신규 8).
- **`LiveFrameSource` 실장치 연결 최소 검증** — `configure_pipeline()` 이후 `cv2.VideoCapture('/dev/video0', cv2.CAP_V4L2)`를 시도해 `isOpened()`는 성공하지만 `read()`가 실패함을 확인. 즉 현재 `LiveFrameSource` 구현은 V4L2 raw+수동 디베이어 경로와 인터페이스가 안 맞음 — 무리하게 통합하지 않고 다음 세션 과제로 명확히 남김(`docs/vision_status.md` "다음" 2번).
- **RPi 원격 저장소 동기화:** `/home/suri/drone_ws/src/suridoksuri`가 FC 도메인의 미추적 비행로그(`logs/2026-07-21_*`)로 `git pull`이 막혀 있어(다른 도메인 소관, 건드리지 않음) `git checkout origin/dev--vision-computing-module -- vision/tools/rpi_capture.py vision/tests/test_rpi_capture.py`로 필요한 두 파일만 갱신해 실기체에서 실제 커밋된 코드로 검증.
- **커널 dynamic_debug 정리** — 진단 중 켰던 `rp1_cfe`/`v4l2-subdev.c`/`mc-entity.c` dynamic_debug를 세션 종료 전 명시적으로 껐음(런타임 상태만 변경, 영구 설정 미변경).
- `docs/vision_status.md` 트랙 블록 갱신(FC 트랙 `docs/session_status.md`는 건드리지 않음 — 도메인 격리 유지), 메모리 `project_rpi5_ubuntu_camera_stack.md` 갱신(브링업 완료 기록).

### 실제 테스트

```
python3 vision/tools/rpi_capture.py --single-shot --out /tmp/... --main-size 4608x2592
  # [단발촬영] .../single_shot.png shape=(2592, 4608, 3) dtype=uint8 mean=23.13 std=25.32 min=16 max=255
# 두 번째 독립 촬영과 비교: byte-identical=False, 평균 절대차 0.28, 52.8% 픽셀 다름 (라이브 캡처 증명)
# 640x480 요청 → 실제 1536x864로 스냅되는 것도 재현 확인
pytest vision/tests/   # 134 passed (기존 126 + 신규 test_rpi_capture.py 8개)
```

### 결정

- **체커보드 실촬영 캘리브레이션(§7.9 항목2)은 이번 세션에서 하지 않음** — 세션 지시에 따라 "작은 단위로 끊어서 다음 세션에 넘기는 게 낫다"는 원칙 적용. V4L2 raw 캡처 자체의 브링업+검증까지가 이번 세션 스코프.
- **`LiveFrameSource` 어댑터 재구현도 이번 세션에서 하지 않음** — 인터페이스 비호환만 실측으로 확인하고 명시적으로 다음 세션 과제로 남김(무리한 확장 지양 원칙).

### 다음 세션

1. 체커보드 실물 준비 → `rpi_capture.py`로 여러 각도/거리 촬영 → OpenCV `calibrateCamera`로 인트린식/왜곡계수 산출
2. `LiveFrameSource`를 `configure_pipeline()`/`capture_frame_bgr()` 기반 어댑터로 재구현(또는 별도 클래스 분리)
3. 골든셋을 실촬영 데이터로 교체(`vision/tests/golden/README.md` 절차), 40m 티어 `known_limitation` 2건 실측 재검증
4. `MjpegStreamer`도 실제 RPi 네트워크 환경(대역폭/지연/Wi-Fi 끊김)에서 미검증 상태 — 여유 되면 실측

### 주의

> RPi는 공유 장치 — 이번 세션에서 변경한 것은 V4L2/media-controller **런타임 상태**(링크 활성화, 서브디바이스 포맷, 프로세스 종료 시 초기화됨)와 진단용 dynamic_debug(세션 종료 전 껐음)뿐. `/boot/firmware/config.txt` 등 영구 설정, WiFi 완화조치, 패스워드리스 sudo 등 기존 FC/mc 관련 설정은 손대지 않음.

---

## 2026-07-22a — [vision] ② 조난자 구역 실측 스펙(3.0m×3.0m×0.105m) 반영 — 계획서·distress_coarse.yaml min/max_area 재도출·골든셋 갱신

**브랜치:** `dev--vision-computing-module`
**목적:** 2026-07-21 방향전환("2차예선까지 빠께스 우선")이 하루 만에 정정되면서(메모리 `project_vision_2nd_qualifier_bucket_target.md` "[2026-07-22 정정]" 절), ② 조난자 구역 실제 확정 스펙(가로 3.0m×세로 3.0m×높이 0.105m, 초록 라이즈드 플랫폼)이 나옴 — 이를 계획서·검출 설정값·골든셋에 반영. 빠께스 트랙은 vision 파트 완료 후로 연기됨(이번 세션 스코프 아님).

### 완료

- **`docs/vision_plan.md` 갱신:** §2 타겟 표(②)·"초록 색/치수" 각주·§5.3·§10 열린 항목을 실측 스펙(3.0m×3.0m×0.105m 라이즈드 플랫폼, 물리적 구조물)으로 갱신. 버티포트 흰 필드와 동일 3m 풋프린트임을 명시해 §4.1 GSD 표의 기존 "3m 피처" 컬럼을 재사용하도록 각주 추가(신규 GSD 컬럼 안 만듦). 버티포트(①) 관련 내용·규격은 손대지 않음(이미 확정, 이번 세션 스코프 아님).
- **`vision/presets/distress_coarse.yaml` `min_area`/`max_area` 재도출:** 300/500000(직전 감사 세션이 "GSD 미확정 상태의 임의값, 물리적 근거 없음"으로 지적)에서 **8000/200000**으로 교체. 계산: 매트 실측 3m 풋프린트 + §4.1 GSD 표(3m 피처 컬럼) 기준이되, 계획서 가정 화각 102°가 아니라 **실측 화각 75°**(`docs/vision_status.md` 기존 기록 — RPi 장착 카메라가 클론이라 화각이 계획서 가정과 다름)로 재계산. 공식 `gw(h)=2h·tan(37.5°)≈1.535h`(지상폭, m), 다운스케일 1536px 기준 GSD로 매트 한 변 픽셀 길이 산출:
  - 10m → 한 변 ≈300px, 면적 ≈90,000px²
  - 20m → 한 변 ≈150px, 면적 ≈22,500px²
  - 40m → 한 변 ≈75px, 면적 ≈5,625px²
  - `min_area=8000`(40m 확실히 배제, 20m는 검출 면적이 명목값의 ~35%까지 열화돼도 통과할 마진) / `max_area=200000`(10m 명목값의 약 2.2배 여유, 배경 오탐만 배제). 계산 전 과정은 yaml 헤더 주석 + `vision/CLAUDE.md` 신설 절 "distress_coarse.yaml min_area/max_area 도출 근거"에 기록(코드 인라인 주석이 아니라 문서 근거 기록 — 기존 "라이브 스트림 어댑터 기본값" 패턴 재사용). **검출 알고리즘(`modules/*.py`)은 변경하지 않음** — 파라미터 값만 교체.
- **골든셋 갱신 (`vision/tests/golden/distress/{10m,20m,40m}/`):** `generate_synthetic.py`가 그리는 매트 픽셀 크기를 위 계산값(한 변 300px/150px/75px)으로 교체 — 더 이상 "가까움/중간/멂"의 임의 placeholder가 아니라 "실측 스펙에서 GSD로 역산한 크기". `labels.json`의 `altitude_label`/`note`도 "스키마 자리표시자" 문구를 실제 계산값 설명으로 교체. 10m/20m는 새 `min_area`(8000) 이상이라 검출, 40m(~5,625px²)는 미만이라 미검출 — 기존과 동일한 방향의 일관된 스토리(정합성 확인됨). **vertiport 골든셋은 손대지 않음**(여전히 스키마 placeholder — §4.1 GSD 표 자체〈102° 가정〉가 재검증 대기라 이번 세션 범위 밖). `vision/tests/golden/README.md` distress 행도 갱신.
- **`docs/vision_status.md` 트랙 블록 갱신:** 공통 상태의 대회 규정 요약에 초록구역 실측 스펙 반영, "빠께스 트랙 연기" 사실을 별도 항목으로 명시(다음 세션이 헷갈리지 않도록), 트랙 보드에 이번 세션 완료 내역 추가, "주의" 절의 화각 불일치 관련 문구를 vertiport(여전히 placeholder)/distress(더 이상 placeholder 아님)로 구분.

### 실제 테스트

```
pytest vision/tests/test_golden_regression.py -v   # 18 passed — 신규 프레임으로 실제 replay.run_replay() 재생, 몽키패치 없음
pytest vision/tests/                                # 126 passed
```

### 결정

- **빠께스(소형 단일물체) 탐색 코드/추상화는 이번 세션에서 만들지 않음** — 2026-07-22 정정 지시에 따라 vision 파트(초록구역 포함) 완료 후로 연기. 다음 vision 세션도 트랙 보드(정밀착륙)를 그대로 이어가면 됨.
- **40m 티어를 "물리적으로 타당한 미검출"로 유지하기로 결정한 근거:** 실측 화각 75°(102°보다 좁음)로 재계산하면 3m 매트가 40m에서도 명목상 결코 작지 않게(~5,625px²) 보이지만, 색상/윤곽 열화에 대한 보수적 안전마진을 반영한 `min_area=8000` 기준으로는 여전히 미달 — "판별 불가능할 만큼 작다"가 아니라 "coarse 대역(40~15m) 원거리 경계에서 안전마진을 우선한 보수적 임계값"이라는 의미로 §5.3/yaml 주석에 명시.
- **버티포트 골든셋·규격은 건드리지 않음** — 이미 확정, 이번 세션 스코프 아님(사용자 지시).

### 다음 세션

1. 트랙 보드(`docs/vision_status.md` 👁 vision-정밀착륙)를 그대로 이어간다 — RPi 카메라 브링업은 여전히 RPi 작업 허가 대기 중, 카메라 독립 대체 트랙(§7.9 4·5·6·7번)은 이미 소진됨.
2. 빠께스 트랙은 **vision 파트(트랙 보드) 전체 완료 후에만** 착수 — 메모리 `project_vision_2nd_qualifier_bucket_target.md` "[2026-07-22 정정]" 절 참조, 먼저 진입하지 말 것.
3. RPi 작업 허가가 떨어지면: 카메라 브링업 선택지 확정 → 캡처 도구 완성 → 골든셋(vertiport+distress 둘 다) 실촬영 데이터로 교체, 이때 distress의 이번 계산값(75° 화각 가정)도 실측으로 재검증.

---

## 2026-07-21f — [vision] 품질 감사 결함 수정 (리소스 leak 2건 + 거짓 로그 1건 + 골든셋 커버리지 갭 1건)

**브랜치:** `dev--vision-computing-module`
**목적:** 새 기능 아님 — 오케스트레이터가 독립 감사 서브에이전트 2개를 돌려 `main.py`/`replay.py`/`tools/jsonl_view.py`의 diff를 라인 단위로 정독시키고 그중 2건은 코드 재확인까지 거쳐 확정한 진짜 결함 3건 + 커버리지 갭 1건을 이번 세션에서 TDD(레드→그린)로 수정. RPi/실카메라 작업은 이번에도 금지, 노트북(WSL) 로컬만 사용.

### 완료

- **리소스 leak (`main.py`/`replay.py`):** `blackbox = BlackBoxLogger(...)` 생성 이후 `streamer.start()`(실 소켓 bind, 포트충돌 시 `OSError` 가능)가 뒤이은 `try:...finally: blackbox.close(); streamer.stop()` **밖**에서 호출되고 있었음 — `start()`가 예외를 던지면 `try`에 진입도 못 해 `finally`가 안 돌고 `blackbox`가 열어놓은 큐스레드/파일핸들이 leak됨. `streamer` 생성/`start()`를 `try` 블록 안으로 옮겨 실패해도 `finally`가 항상 실행되게 수정. 감사가 `main.py` 기준으로 지적했지만 `replay.py`도 직접 확인해보니 동일 구조(`writer`/`frame_count` 초기화만 `try` 밖에 있고 `streamer.start()`도 `try` 밖)였어서 같이 수정. 회귀: `MjpegStreamer.start`를 몽키패치로 `OSError` 나게 만들고 `blackbox.close()`가 실제로 호출되는지 검증 — `test_main.py::test_streamer_start_failure_still_closes_blackbox`, `test_replay.py::test_streamer_start_failure_still_closes_blackbox`(수정 전 레드 확인 → 수정 후 그린 확인).
- **거짓 "저장" 로그 (`replay.py:134-135`):** `if output: logger.info("저장: %s", output)`가 `output` 인자 존재 여부로만 게이팅해, 실제 `cv2.VideoWriter`는 프레임 루프 안에서 첫 프레임 처리 시에만 생성되므로 0프레임 재생이면 파일 자체가 안 만들어지는데도 "저장" 로그가 찍히던 문제. `main.py:165-167`(`if writer: writer.release(); print(...)`)와 동일하게 실제 `writer` 존재 여부로 게이팅하도록 수정. 회귀: `open_dir_or_bag`를 0프레임 가짜 소스로 몽키패치해 "저장" 로그가 안 찍히는지(`test_zero_frames_with_output_does_not_log_saved`), 정상적으로 프레임이 처리되는 대조군에선 찍히는지(`test_nonzero_frames_with_output_logs_saved`) 둘 다 검증. (실제 0프레임 mp4는 컨테이너/코덱 사정으로 `cv2.VideoCapture`가 아예 못 여는 경우가 있어 `open_dir_or_bag`를 가짜 빈 소스로 바꿔치기해 결정론적으로 재현.)
- **x축 스케일 혼용 (`tools/jsonl_view.py:124-127`):** `_x()`가 `x_field="ts"`인데 해당 행의 `ts`가 없으면 `frame_id`(정수, 보통 0~N)로 새 나머지 `ts-t0`(경과초, 보통 훨씬 작은 소수) 스케일과 섞여 시간축이 행마다 뒤로/앞으로 튀어 보이던 문제. 이 파일에 이미 있던 score/latency의 nan-gap 패턴(결측은 nan으로 채워 라인만 끊고 스케일을 안 섞음)과 같은 철학으로 `x_field="ts"`인데 `row.ts is None`이면 x좌표도 `nan`으로 처리하도록 수정. `t0 is None`(전체 행에 ts가 하나도 없는 경우)은 원래대로 `frame_id` 폴백 유지 — 그 경우는 전체가 일관되게 frame_id 축이라 혼용이 아님. 회귀: 일부 행만 `ts=None`인 실제 `BlackBoxLogger` 산출 JSONL(수기 JSON 아님)로 x좌표 배열이 nan-gap 처리되는지 검증 — `test_jsonl_view.py::test_x_axis_ts_gap_does_not_mix_scale_with_frame_id`.
- **골든셋 커버리지 갭 — no_target × distress_coarse.yaml:** `tests/golden/no_target/`은 `vertiport_coarse.yaml`로만 오탐 회귀됐고, 필터 기준이 전혀 다른(무채색 사각형 vs 초록 HSV) `distress_coarse.yaml`에 대한 오탐 방지 회귀가 없었음. 기존 `no_target/labels.json` 스키마는 손대지 않고(과설계 금지) `no_target/distress_coarse/` 리프 디렉터리를 새로 추가(같은 `frame_000.png` 재사용, `preset`만 `distress_coarse.yaml`) — "리프 디렉터리 하나당 labels.json 하나" 기존 관례 그대로. 실제로 파이프라인을 돌려 `expect_num_detections=0`을 확인 후 기록(허위 없음). 기존 파라미터화 테스트가 새 리프를 자동 발견해 검증하며, 커버리지 자체가 앞으로도 지켜지는지 확인하는 명시적 테스트 `test_no_target_has_distress_coarse_regression_case`도 추가. `tests/golden/README.md`에 "타겟/프리셋변형" 리프 패턴(고도 계층이 없는 타겟용) 문서화.
- **검출 로직(color.py/detector.py/vertiport_*.py) 전혀 건드리지 않음** — 전부 CLI 진입점(`main.py`/`replay.py`)/뷰어(`tools/jsonl_view.py`)/골든셋 스캐폴드 레벨 수정.
- `pytest vision/tests/` **126 passed**(기존 118 + 신규 8: `test_main.py` 1, `test_replay.py` 3, `test_jsonl_view.py` 1, `test_golden_regression.py` 명시 회귀 1 + 새 골든 리프로 인한 자동 파라미터화 2).

### 커밋

- `78ed2c2` — main.py/replay.py 리소스 leak 수정(streamer.start()를 try/finally 안으로) + 회귀 테스트
- `15d8e58` — jsonl_view.py x축 스케일 혼용 수정(nan-gap) + 회귀 테스트
- `ed83d3e` — no_target 골든셋에 distress_coarse.yaml 케이스 추가 + 커버리지 회귀 테스트

(참고: `78ed2c2`는 replay.py의 리소스 leak 수정과 "저장" 로그 게이팅 수정이 같은 파일 인접 영역이라 한 커밋에 같이 들어갔음 — 커밋 메시지는 leak 수정만 언급하지만 diff에 두 수정 모두 포함됨.)

### 다음 세션

- 이번 세션은 감사 결과 수정 전용이라 vision 트랙 자체의 다음 단계는 `docs/vision_status.md`의 "다음" 항목(전부 RPi 작업 허가 필요) 그대로 유효 — 변경 없음.

---

## 2026-07-21e — [vision] 라이브 스트림 어댑터(MJPEG-over-HTTP, compute_tap VGA)

**브랜치:** `dev--vision-computing-module`
**목적:** 사용자가 실비행 중이라 이번 세션도 RPi SSH 접속·실카메라 작업 전면 금지 유지. `docs/vision_plan.md` §7.9 "지금 당장 할 일" 5번(라이브 스트림 어댑터, 카메라 독립적으로 진행 가능)을 노트북(WSL) 로컬에서만 진행

### 완료

- **`vision_plan.md` §7.9 정독** — "작동영상 피드백 3경로" 표 (b)행("라이브 저해상 스트림 — compute_tap(VGA급) → MJPEG-over-HTTP 또는 ROS2 image_transport, 현장 랩탑 실시간")과 "비침습 전제: dev 계측이 제어루프를 지연시키면 안 된다" 확인. 세션 지시에 따라 MJPEG-over-HTTP만 구현(ROS2 경로 제외 — 도메인 간 import 금지 원칙, `rclpy`/`fc_ros` 미사용)
- **`vision/utils/stream.py` 신규 — `MjpegStreamer`.** `push_frame()`은 bounded queue(기본 길이 2)+drop-oldest로 절대 블로킹하지 않음 — `vision/utils/blackbox.py`의 `_DropOldestQueueHandler`와 동일 패턴 재사용(새 패턴 발명 안 함, 세션 지시 준수). 다운스케일(종횡비 유지, 640x480 VGA 박스 안에 맞춤, 업스케일 없음)·JPEG 인코딩·HTTP 서빙은 전부 별도 스레드. 표준 라이브러리 `http.server.ThreadingHTTPServer`만 사용(새 무거운 의존성 없음). `/stream`(MJPEG multipart)과 `/`(브라우저 미리보기 `<img>` 페이지) 두 경로 제공
- **동시성 버그 발견·수정** — `push_frame`을 여러 producer 스레드가 동시에 부르는 실제 테스트(느린 소비자 시나리오)에서 "가득 찬 큐에서 가장 오래된 항목 제거 후 삽입"이 락 없이 이뤄져 `queue.Full`이 새는 레이스가 실제로 재현됨(`PytestUnhandledThreadExceptionWarning`으로 발견). `threading.Lock`으로 evict+insert를 원자화해 수정 — 재현 즉시 경고 사라짐 확인. `blackbox.py`의 동일 패턴은 로거 호출 경로가 사실상 단일 producer라 이 레이스가 안 드러나 그대로 둠(근거를 `vision/CLAUDE.md`에 기록)
- **`vision/main.py`/`vision/replay.py`에 opt-in 연결** — `--display stream`(+ `--stream-host`/`--stream-port`, 기본 `0.0.0.0:8080`). main.py에 있던 기존 "stream=미구현" placeholder(에러 종료)를 실제 구현으로 대체. 스트림을 켜지 않으면 `MjpegStreamer`가 아예 인스턴스화되지 않음(오버헤드 없음, 회귀 테스트로 고정)
- **진짜 테스트(pseudo 아님) — `vision/tests/test_stream.py` 신규 10개:** 실제 HTTP 서버 기동(임시 포트) → 골든셋(`DirFrameSource`로 조달, `vision/tests/golden/vertiport/10m`, RPi/실카메라 미사용) 프레임을 실제로 push → 진짜 `http.client`로 `/stream` 접속 → 진짜 MJPEG 바이트를 `cv2.imdecode`로 디코드 성공 확인. VGA 박스 다운스케일(1280x960→정확히 640x480, 240x320은 업스케일 없이 그대로) 실측. 비차단 큐 자체(200회 push에도 큐 길이 상한 유지) + **실제 느린 컨슈머**(응답 연결만 하고 body를 절대 안 읽는 진짜 소켓) 붙여놓은 채로 300회 push 시간 실측(<1초) — §7.9 비침습 전제의 핵심 요구사항 실증
- **`test_main.py`/`test_replay.py`에도 CLI 경로 통합 테스트 추가** — 기존 `test_display_stream_not_implemented`(더 이상 사실이 아님)를 실제 구현 검증 테스트로 교체. `test_replay.py`는 재생 도중 실제 HTTP GET으로 접속해 실제 프레임 디코드까지 확인(타이밍 결정론 확보용으로 `Pipeline.run`에 테스트 전용 소폭 지연 monkeypatch — 스트리밍 배관 자체는 실동작)
- **수동 스모크(pytest 밖)** — 300프레임 재생 폴더로 `python -m vision.replay ... --display stream --stream-port 8099`를 백그라운드 프로세스로 실제 실행 → `curl`로 실제 5초간 ~1.85MB MJPEG 수신 → 프레임 디코드 성공(500x500 원본이 640x480 박스에 640x480 미만인 480x480으로 축소, 종횡비 유지 확인) → 재생 종료 후 프로세스가 스트리머 `.stop()`으로 정상 종료(좀비/행 없음) 확인
- **`vision/CLAUDE.md` 갱신** — 파일역할표에 `utils/stream.py` 행, "라이브 스트림 어댑터 기본값" 절 신설(해상도 640x480 박스/포트 8080/바인딩 0.0.0.0/JPEG quality 80/큐 길이 2 — §7.9가 정확한 수치를 안 못박아 세션 지시에 따라 합리적 기본값으로 확정 + 근거), 테스트 규칙표에 `utils/stream` 행
- **`docs/vision_status.md` 갱신** — §7.9 카메라 독립 대체 트랙(4·5·6·7번) 전부 완료 명시, 남은 항목(1·2·3번)이 전부 RPi 허가 필요임을 트랙 헤더/본문에 명확히 기록
- **`pytest vision/tests/` 118 passed** (기존 106 + 신규 10 `test_stream.py` + `test_replay.py` 1건 + `test_main.py` 순증 1건)

### 결정

- **ROS2 image_transport 경로는 이번 세션에서 하지 않음** — §7.9 (b)행이 MJPEG-over-HTTP와 ROS2 둘을 언급하지만, 세션 지시가 MJPEG-over-HTTP만 명시적으로 요구했고 vision 도메인은 `rclpy`/`fc_ros`를 import하지 않는다는 루트 `CLAUDE.md` 원칙과도 부합. ROS2 경로가 필요해지면 별도 세션에서 도메인 간 의존 관계를 루트 `CLAUDE.md`에 먼저 기록하고 논의
- **다운스케일은 letterbox 패딩 없이 종횡비 유지 축소** — 관찰용 브라우저 `<img>`가 알아서 맞추므로 정확히 640x480으로 패딩하는 복잡도는 불필요하다고 판단(과설계 금지 지시 준수)
- **큐 길이 기본값 2, JPEG quality 80** — 관찰이 목적(최신 프레임 우선, 지연 누적 회피)이지 무손실 기록이 목적이 아님. 상시 기록은 기존 `--output`/mp4 덤프 경로가 담당

### 다음 세션

1. **[RPi 작업 허가 필요]** 카메라 브링업 재개 — `docs/vision_status.md` "🟡" 블록의 4개 선택지 확인부터
2. **[RPi 작업 허가 필요]** 캡처 도구 완성 → 실촬영 체커보드 → 카메라 인트린식 캘리브레이션
3. **[RPi 작업 허가 필요]** 골든셋을 실촬영 데이터로 교체 + `MjpegStreamer`를 실제 RPi↔랩탑 네트워크 환경에서 실측(지금까지는 로컬 WSL HTTP 왕복만 검증됨)
4. **카메라 독립 대체 트랙 완전 소진** — §7.9 "지금 당장 할 일" 1~7번 중 카메라 독립 항목(4·5·6·7)이 전부 끝남. RPi 허가 없이 다음 vision 세션이 들어오면 이 트랙에서 더 진행할 카메라 독립 작업이 없다는 걸 먼저 확인할 것(다시 §7.9를 훑을 필요 없음, 이미 확인됨)

### 주의

> `docs/vision_status.md` 트랙보드가 이미 갱신됨 — 다음 세션은 그 문서만 읽으면 됨.
> `MjpegStreamer`는 로컬(WSL) HTTP 왕복만 검증됨 — 실제 RPi 네트워크(대역폭/지연/Wi-Fi 끊김, `project_rpi5_tailscale_wifi_drops.md` 참조) 조건에서의 동작은 미검증.
> `vision/utils/blackbox.py`의 `_DropOldestQueueHandler`에도 이론상 동일한 동시성 레이스가 있을 수 있음(다중 producer 스레드에서 `queue.Full` 누수) — 이번 세션에서 발견한 건 `stream.py` 쪽뿐이라 `blackbox.py`는 손대지 않았지만, 향후 blackbox를 여러 스레드에서 동시에 호출하는 상황이 생기면 재검토 필요.

---

## 2026-07-21d — [vision] 골든셋 폴더 스캐폴드 + 재생 회귀 assert

**브랜치:** `dev--vision-computing-module`
**목적:** 사용자가 실비행 중이라 이번 세션도 RPi SSH 접속·실카메라 작업 전면 금지 유지. `docs/vision_status.md` "다음" 4번(§7.9 항목7 — 골든셋 폴더 스캐폴드 + 재생 회귀 assert, 카메라 독립적으로 진행 가능)을 노트북(WSL) 로컬에서만 진행

### 완료

- **`vision_plan.md` §7.9/§7.5/§2 정독** — "골든셋 회귀 테스트: 라벨된 프레임 폴더(고도별·타겟별) 검출 유지 assert" 요구 확인. 실기체 데이터가 없어(§ vision_status.md 주의) 이번 세션은 **폴더 구조/스키마 스캐폴드**를 확정하고 합성 프레임으로 동작을 증명하는 것으로 범위를 좁힘(진짜 라벨링 데이터셋은 카메라 브링업 이후)
- **골든셋 스키마 확정** — `vision/tests/golden/<타겟>/<고도라벨>/frame_NNN.png` + 리프 디렉터리별 `labels.json`(target/altitude_label/preset/note/frames[expect_num_detections/expect_stage_meta/known_limitation]). `DirFrameSource`가 이미지만 골라 읽으므로 `labels.json`이 같은 디렉터리에 있어도 재생에 영향 없음을 `vision/utils/frame_source.py` 확인으로 검증. 구조/스키마/재생성법/실기체 데이터 교체 절차는 `vision/tests/golden/README.md`에 문서화(§ 요구사항 4번)
- **타겟 종류 선정 — "3종" 예시를 문자 그대로 따르지 않고 실제 구현된 검출 경로에 맞춤:** ① `vertiport`(기존 `vertiport_coarse.yaml` 3단 캐스케이드 그대로 사용) ② `distress`(전용 모듈이 없어 신규 `vision/presets/distress_coarse.yaml` 프리셋 추가 — **기존 `ColorFilter`+`RectDetector` 조합일 뿐 신규 검출 로직 없음**, §5.3 "HSV 초록+사각" 그대로) ③ 하기구역은 전용 형상판별(빨간 십자) 검출기가 없어 **의도적으로 제외**(제네릭 rect_detector로 "십자"라고 우기면 허위 검증이라 판단) + `no_target`(④ 단순착륙과 동일 조건 — 피듀셜 없는 평지에서 가장 정교한 캐스케이드조차 오탐하지 않는지, 오탐방지 회귀). 사유는 `vision/tests/golden/README.md` "빠진 것" 절에 명시
- **고도 티어(10m/20m/40m) 픽셀 스케일을 GSD 표(§4.1)에서 정밀하게 역산하지 않기로 결정** — §4.1 GSD 표는 화각 102° 가정인데 실장착 카메라는 75°로 확인돼(`docs/vision_status.md`) 표 자체가 재검증 대기 상태. 정밀 역산은 거짓 정밀도가 되므로, "가까움/중간/멂" 스키마 자리표시자로만 고도 라벨을 씀 — `generate_synthetic.py` docstring과 README에 이 판단 근거를 명시
- **실측으로 실제 파이프라인 스케일 민감성을 발견** — `test_vertiport_cascade.py`와 동일 비율의 합성 도형을 다양한 픽셀 스케일로 실제 `vertiport_coarse.yaml` 파이프라인에 돌려본 결과, 큰 스케일(예 흰 필드 지름 220px 이상)에서는 3단 캐스케이드 전부 확인되지만 작은 스케일(94px)에서는 `white_field`는 후보를 내고도 `black_v` 형상매칭이 탈락(rejected)해 최종 검출 0건이 됨 — 저해상에서 `MorphologyModule`의 고정 `kernel_size=5`가 V 노치 주변 화이트필드 연결성을 깨는 것이 원인으로 보임. **검출기/프리셋 파라미터는 튜닝하지 않고(이번 세션 범위 밖)**, 실측된 그대로를 `known_limitation: true`로 골든셋에 고정. `distress` 40m 티어도 유사하게 매트 픽셀 면적이 `min_area`(300) 미만이 되는, 물리적으로 타당한 미검출 케이스로 고정
- **`vision/tests/golden/generate_synthetic.py` 신규** — 위 결정을 코드화한 합성 프레임 생성기(pytest 대상 아님, 수동 재생성 도구). 실행해 7개 리프 디렉터리(`vertiport`×3, `distress`×3, `no_target`×1)의 `frame_000.png`+`labels.json` 생성
- **`vision/tests/test_golden_regression.py` 신규(15개 테스트)** — §요구사항의 핵심: **실제** `vision.replay.run_replay()`(`DirFrameSource`+실제 `Pipeline`, 몽키패치 없음)로 골든 폴더를 재생시켜 JSONL 블랙박스에 찍힌 실제 검출 개수를 `labels.json`과 비교. 보조로 `Pipeline.run()`을 직접 호출(이것도 실제 호출)해 캐스케이드 단계별 `state.meta`까지 검증. **진짜로 회귀를 잡는지 수동 검증** — `RedRingDetector.min_points`를 999999로 일시 변경(원복 전제)해 관련 4개 테스트가 실제로 실패하는 것 확인 후 원복
- **`vision/CLAUDE.md` 갱신** — 파일역할표에 `presets/distress_coarse.yaml`·`tests/golden/` 행 추가, 테스트 규칙표에 골든 회귀 행 추가, 공통규칙 4번 "골든셋 회귀는 데이터 수집 후" 문구를 실제 상태(합성 데이터로 스캐폴드 시작)로 갱신
- **`pytest vision/tests/` 106 passed** (기존 91 + 신규 15)

### 결정

- **"타겟 3종×고도 2~3구간" 예시를 문자 그대로 따르지 않음** — 실제로 파이프라인이 지원하는 타겟(①버티포트, ②조난자구역〈신규 프리셋으로〉)에만 골든셋을 만들고, ③(하기구역)은 전용 검출기가 없어 가짜로 채우지 않기로 판단. 대신 `no_target`(오탐방지)을 세 번째로 넣어 개수는 맞추되 정직성을 우선함
- **고도 라벨을 GSD 표 기반 정밀 픽셀값으로 만들지 않음** — 카메라 화각 재검증 대기 상태에서 정밀 역산은 실제보다 더 신뢰도 높아 보이는 착시를 만든다고 판단, "근접/중간/원거리" 스키마 자리표시자임을 코드 docstring과 README 양쪽에 명시
- **40m 티어의 미검출을 "버그"로 보고 파라미터를 고치지 않음** — 세션 지시("검출기 파라미터를 바꾸지 마라, 실패하면 기대값을 현재 동작에 맞게 정하라")를 그대로 따름. 다만 원인 추정(고정 kernel_size의 스케일 민감성)은 README/status에 기록해 다음 세션이 "왜 이 값인지" 재추측하지 않도록 함

### 다음 세션

1. **[RPi 작업 허가 필요]** 카메라 브링업 재개 — `docs/vision_status.md` "🟡" 블록의 4개 선택지 확인부터
2. (카메라 독립, 대체 가능) §7.9 항목5 — 라이브 스트림 어댑터(`compute_tap` VGA → MJPEG/ROS image)
3. (카메라 독립, RPi 허가 후가 자연스러움) 골든셋을 실촬영 데이터로 교체 — `vision/tests/golden/README.md` "실기체 데이터가 들어오면" 절차대로. 이때 40m `known_limitation` 두 건도 실측 재검증(현재 동작이 실물에서도 재현되는지, 아니면 합성 도형만의 아티팩트인지 확인 필요)

### 주의

> `docs/vision_status.md` 트랙보드가 이미 갱신됨 — 다음 세션은 그 문서만 읽으면 됨.
> 골든셋(`vision/tests/golden/`)은 전부 합성 데이터 — 실촬영 아님. 고도 라벨(10m/20m/40m)은 GSD 정밀 매핑이 아니라 스키마 자리표시자.
> `vertiport_coarse.yaml`의 고정 `kernel_size=5` morphology가 저해상 스케일에서 흰 필드 연결성을 깨는 스케일 민감성을 이번 골든셋 작업 중 발견 — 검출기는 미변경, 실기체 데이터로 재검증 필요.

---

## 2026-07-21c — [vision] JSONL 뷰어/플롯 최소본(tools/jsonl_view.py)

**브랜치:** `dev--vision-computing-module`
**목적:** 사용자가 실비행 중이라 이번 세션도 RPi SSH 접속·실카메라 작업 전면 금지 유지. 직전 세션이 연결한 blackbox JSONL이 "쌓이기만 하고 안 보이는" 상태라, `docs/vision_status.md` "다음" 3번(§7.9 항목6 — JSONL 뷰어/플롯 최소본, 카메라 브링업과 독립적으로 진행 가능)을 노트북(WSL) 로컬에서만 진행

### 완료

- **`vision_plan.md` §7.9 정독** — "JSONL은 쌓기만 하면 안 보인다 → 뷰어가 필수", "시간축으로 검출점수·latency·state·alt를 플롯" 요구를 확인. §7.9 항목6은 "최소본"으로 명시 — Foxglove 연동 등 과설계는 범위 밖으로 확정
- **`vision/tools/jsonl_view.py` 신규** — `BlackBoxLogger`(vision/utils/blackbox.py)가 실제로 쓰는 JSONL 스키마를 그대로 읽는 3개 함수:
  - `load_records(path)` — `type=frame`/`type=rejection` 레코드를 `FrameRow` 리스트 + rejection ts 리스트로 분리. score는 `chosen.confidence` 우선, 없으면 그 프레임 `detections` 중 최고 confidence(`confidence`/`score` 키 둘 다 방어)
  - `build_figure(frame_rows, rejection_ts, x_field="ts"|"frame_id")` — score/latency/state 3단 subplot. 결측값은 필터링해 이어붙이지 않고 **nan으로 채워 라인을 끊는다** — "검출 0인 프레임이 옆 프레임과 매끄럽게 이어진 것처럼" 보이는 오해를 방지. 이 덕분에 각 라인의 포인트 수(`len(xdata)`)가 항상 JSONL의 `type=frame` 행 수와 정확히 같음. rejection은 score 서브플롯에 빨간 세로 점선. state가 전부 None(현재 상태머신 미구현이라 실사용 시 항상 이 케이스)이면 빈 플롯 대신 안내 텍스트
  - `save_figure(fig, path)` — PNG로 저장(`matplotlib` **Agg 백엔드 고정** — headless-safe, GUI 강제 호출 없음)
  - CLI: `python vision/tools/jsonl_view.py <jsonl> [--output out.png] [--x-axis ts|frame_id] [--title ...]`
- **`vision/requirements.txt`에 `matplotlib>=3.7` 추가** — `.venv`에 설치 완료(9개 의존 패키지 함께 설치됨)
- **텍스트를 영문으로** — 처음엔 플롯 안내문구를 한글로 썼다가 `matplotlib` 기본 폰트(DejaVu Sans)에 한글 글리프가 없어 `UserWarning`(글리프 깨짐)이 남 → `vision/utils/visualize.py`의 기존 관례(이미지 위 텍스트는 영문 "CONFIRMED")를 따라 플롯 내부 텍스트만 영문으로 전환
- **진짜 테스트 (모킹만으로 통과하는 pseudo 테스트 금지 요건 준수)** — `tests/test_jsonl_view.py`(8개):
  - `vision.main`을 실제로 1회 실행(비디오 4프레임, 그중 1프레임은 빈 화면)해 디스크에 진짜 `.jsonl` 생성 → `load_records()` 결과 행 수가 실제 JSONL의 `type=frame` 행 수(4)와 일치, `build_figure()`의 score/latency 라인 포인트 수도 4(결측 프레임 위치에 실제 `math.isnan()` 구멍 존재 확인), `save_figure()`가 실제 PNG 파일(size>0)을 남기는지 assert
  - 이 테스트용으로 `color_filter→rect_detector` 직결 임시 preset(tmp_path에만 존재, `presets/*.yaml` 미변경)을 썼음 — `single_frame.yaml`의 `edge_detector→morphology(open, kernel 5)` 조합이 얇은 Canny 엣지를 지워버려 합성 사각형 테스트 도형에서 검출이 0이 되는 걸 확인했기 때문(실제 착륙지점 규모용 튜닝이라 튜닝 로직 자체는 정상, 건드리지 않음)
  - rejection 세로선·다중 state 케이스는 `BlackBoxLogger.log_frame`/`log_rejection`을 직접 호출해 만든 실제 JSONL로 검증(수기로 JSON 문자열을 쓴 게 아님)
  - CLI 진입점(`main()`)도 subprocess가 아니라 함수 직접 호출로 실제 파일 경로 인자를 태워 종단 검증
- **`vision/CLAUDE.md` 갱신** — 파일역할표에 `tools/jsonl_view.py` 행 추가, import 규칙의 `tools/` 항목에 "하드웨어 비의존 CLI 도구는 예외(.venv 설치+pytest 대상)" 명시(기존 `rpi_capture.py` 전용 규칙과 구분), 테스트 규칙표에 행 추가
- **실행 결과 육안 확인** — scratchpad에서 `vision.main` → `jsonl_view.py` CLI를 수동으로도 1회 더 실행해 PNG를 실제로 열어봄. score 라인이 빈 프레임 위치에서 정확히 끊기는 것(nan 처리) 확인
- **`pytest vision/tests/` 91 passed** (기존 83 + 신규 8). 커밋 `2e02e29`, push 완료

### 결정

- **score 필드 정의: `chosen.confidence` 우선, 없으면 detections 중 최고 confidence** — JSONL 스키마에 명시적 "score" 필드가 없고(§7.4 원문도 `detections[점수·위치]`로만 서술), main.py/replay.py는 현재 `TemporalFusion`을 안 쓰는 preset에선 `chosen`이 항상 None이라 detections 최고값으로 폴백하는 경로가 실제로 자주 탐. 대회 규정과 얽힌 "어떤 점수를 반드시 봐야 하는가" 같은 판단은 아니라고 보고 진행(막힘 아님) — 상태머신(§5.1) 연결 후 `chosen`이 실제로 채워지면 그쪽이 우선되도록 이미 구현돼 있음
- **결측을 nan으로 끊기 vs 필터링해 이어붙이기** — 처음엔 필터링(있는 점만 이어붙임)으로 구현했다가 수동 PNG 확인 중 "검출 0 프레임이 옆 프레임과 매끄럽게 이어진 것처럼" 보이는 게 디버깅 뷰어로서 오해를 부른다고 판단해 nan 방식으로 바꿈. 부수효과로 "포인트 수 = JSONL 행 수" 불변식이 더 명확해짐
- **`tools/` 디렉터리에 배치하되 CLAUDE.md의 기존 "tools/=미테스트" 규칙에 예외 신설** — 세션 지시가 `vision/tools/jsonl_view.py` 위치를 권장했고, 기존 관례 문구("tools/는 RPi 하드웨어 전용, .venv 밖, pytest 대상 아님")는 `rpi_capture.py`(picamera2/GStreamer 의존)를 염두에 둔 것이었음. `jsonl_view.py`는 하드웨어 의존이 전혀 없어 그 근거가 적용 안 됨 → 위치는 유지하되 CLAUDE.md에 예외를 명문화(다음 세션이 혼동하지 않도록)
- **테스트용 임시 preset을 tmp_path에만 만듦** — `presets/*.yaml`이나 검출 로직(`edge.py`/`morphology.py`)을 건드리지 않는다는 이번 세션 제약을 지키기 위한 선택. `single_frame.yaml`의 edge+morphology 조합이 실제로 튜닝 버그가 있는 건 아님(합성 테스트 도형과 실제 착륙지점 텍스처가 다를 뿐)이라고 판단해 원본은 그대로 둠

### 다음 세션

1. **[RPi 작업 허가 필요]** 카메라 브링업 재개 — `docs/vision_status.md` "🟡" 블록의 4개 선택지 확인부터. 메모리 `project_rpi5_ubuntu_camera_stack.md`에 경과 다 있음
2. (카메라 독립, 대체 가능) §7.9 항목5 — 라이브 스트림 어댑터(`compute_tap` VGA → MJPEG/ROS image)
3. (카메라 독립, 대체 가능) §7.9 항목7 — 골든셋 폴더 스캐폴드(라벨 프레임, 고도·타겟별) + 재생 회귀 assert
4. 상태머신(§5.1)이 실제로 연결되면 `jsonl_view.py`의 state 서브플롯이 자동으로 실데이터를 보여주는지(현재는 코드상 대응만 돼 있고 실데이터로 확인은 못 함) 재확인

### 주의

> `docs/vision_status.md` 트랙보드가 이미 갱신됨 — 다음 세션은 그 문서만 읽으면 됨.
> `jsonl_view.py`의 state 서브플롯은 실기체 데이터로 검증된 적 없음 — main.py/replay.py가 `state`를 채우는 코드 경로 자체가 아직 없기 때문(항상 "no state data" 안내만 뜬다). 상태머신 연결은 다음 세션 이후 몫.

---

## 2026-07-21b — [vision] FrameSource(Live/Dir/Bag) 어댑터 + 재생CLI + blackbox/logger를 main.py에 연결

**브랜치:** `dev--vision-computing-module`
**목적:** 사용자가 실비행을 나가면서 이번 세션엔 RPi SSH 접속·실카메라 작업을 전면 금지 — 직전 세션에서 막힌 카메라 브링업 대신, `docs/vision_status.md` "다음" 3번(§7.9 3번 이후=`FrameSource`+재생CLI+관측성 연결, 카메라 브링업과 독립적으로 진행 가능하다고 명시된 대체 트랙)을 노트북(WSL) 로컬에서만 진행

### 완료

- **`vision_plan.md` §7.9 정독** — Live/Dir/Bag 세 모드 의미, 재생 CLI 요구사항(`python -m vision.replay <녹화폴더|bag> --preset ...`), §7.5 기록·재생 결정론 요구를 확인하고 설계에 반영
- **`vision/utils/frame_source.py` 신규** — `FrameRecord`(frame_id/ts/image/telemetry) + 세 어댑터:
  - `LiveFrameSource`: 장치(인덱스/V4L2 경로) 연결 시도, 실패 시 `retries`회 재시도 후 `ConnectionError`. 프레임 읽기 실패 시에도 `ConnectionError`. 실카메라 미보유라 `cv2.VideoCapture`만 몽키패치해 재시도/에러 계약을 검증(§7.9 "Live=보조" 인터페이스 계약만 이번 세션 범위)
  - `DirFrameSource`: 녹화 폴더(이미지 파일들, 파일명 정렬로 결정론적 frame_id) + 선택적 `telemetry.jsonl`(frame_id로 매칭) — §7.9 (a) "재생 오버레이 뷰어" 주력 입력
  - `BagFrameSource`: 단일 비디오 파일 + 선택적 사이드카 `<basename>.jsonl` — 이 코드베이스엔 rosbag 의존성이 없어 "bag"을 비디오+텔레메트리 사이드카로 구현(Dir보다 압축된 단일파일 재생 경로)
  - `open_dir_or_bag()` 팩토리 — 경로가 디렉터리면 Dir, 파일이면 Bag 자동판별(재생 CLI가 사용)
- **`vision/replay.py` 신규** — 오프라인 재생 CLI. `Pipeline.from_config` + `open_dir_or_bag`로 동일 파이프라인을 결정론적으로 재생하며 로거+블랙박스 기록, `--display window`/`--output mp4` 지원
- **`vision/main.py`에 로거/블랙박스 실연결** — 기존엔 `utils/logging.py`/`utils/blackbox.py`가 독립 유틸로만 존재. 이제 `main()`이 매 실행마다 이중싱크 로거(provenance 헤더=git해시+config) + JSONL 블랙박스를 생성해 이미지/영상 각 프레임의 detections/latency/confirmed를 실제로 기록. `--log-dir`/`--log-name` 인자 추가(기본 `vision/results/logs`), 항상 on(드론 배치 시에도 관측성 확보가 목적)
- **진짜 테스트 (모킹만으로 통과하는 pseudo 테스트 금지 요건 준수)** — `tests/test_frame_source.py`(18개, tmp_path에 실제 png/mp4 생성 후 실디코딩·순서·telemetry 매칭·결정론 검증 + Live 몽키패치 4개), `tests/test_replay.py`(4개, 실제 녹화폴더/bag을 재생시켜 디스크의 JSONL 내용까지 assert), `tests/test_main.py`에 2개 추가(main.py 실행 후 실제 `.jsonl`/`.log` 파일 존재·내용 검증). 기존 4개 테스트는 `--log-dir`를 `tmp_path`로 명시해 실제 저장소 `vision/results/`를 더럽히지 않게 함
- **`vision/CLAUDE.md` 갱신** — 파일역할표에 `frame_source.py`/`replay.py` 추가, `logging.py`/`blackbox.py` 행에 "main.py/replay.py에 연결됨" 명시, 테스트 규칙표 갱신, import 규칙에 `replay.py` 행 추가
- **`pytest vision/tests/` 83 passed** (기존 59 + frame_source 18 + replay 4 + main 2). 커밋 `6a241e3`, push 완료

### 결정

- **Live/Dir/Bag 재생 CLI를 main.py와 별도 파일(`replay.py`)로 분리** — main.py는 "온보드 실행 진입점"(라이브 배치용), replay.py는 "책상 재생 진입점"(§7.9 (a) 데스크 주력)이라는 역할이 달라 섞지 않음. 두 파일이 `_show_window` 같은 작은 헬퍼를 각자 얇게 중복 보유하는 쪽을 택함(과한 공유 추상화보다 단순함 우선, `vision/CLAUDE.md` "config-driven callable 패턴" 기존 철학 계승)
- **"Bag"을 rosbag이 아니라 비디오+사이드카 텔레메트리로 구현** — 이 코드베이스엔 ROS/rosbag 의존성이 전혀 없고(vision은 독립 도메인, ROS2는 fc_ros 쪽), §7.9 원문도 "재생 CLI 엔트리(예: `python -m vision.replay <녹화폴더|bag>`)"라고만 하고 포맷을 못박지 않음 — Dir(폴더)과 대비되는 "압축된 단일파일" 의미로 해석해 비디오 파일로 구현. 추후 실제 rosbag 도입 필요성이 생기면 재검토
- **RPi/실카메라 작업은 이번 세션에서 전면 배제** — `LiveFrameSource`는 인터페이스 계약(재시도/에러)만 구현·검증했고, 실장치 연결은 RPi 작업 허가가 떨어진 다음 세션에서

### 다음 세션

1. **[RPi 작업 허가 필요]** 카메라 브링업 재개 — `docs/vision_status.md` "🟡" 블록의 4개 선택지 확인부터. 메모리 `project_rpi5_ubuntu_camera_stack.md`에 경과 다 있음
2. 브링업 완료 후 `LiveFrameSource`를 실제 RPi 카메라(V4L2 장치경로 또는 GStreamer 파이프라인 문자열)로 검증
3. (카메라 독립, 대체 가능) §7.9 항목5 이후 — 라이브 스트림 어댑터·JSONL 뷰어·골든셋 스캐폴드

### 주의

> `docs/vision_status.md` 트랙보드가 이미 갱신됨 — 다음 세션은 그 문서만 읽으면 됨.
> `vision/results/logs`·`vision/results/replay_logs`는 기본 로그 출력 위치이나 git엔 포함 안 함(루트 `CLAUDE.md` 정책) — 테스트는 전부 `tmp_path`를 써서 실저장소를 더럽히지 않는다.

---

## 2026-07-21 — [vision] HSV 테스트·버티포트 coarse 캐스케이드·관측성 골격 → RPi 카메라 브링업 중 긴급 세션종료

**브랜치:** `dev--vision-computing-module`
**목적:** 트랙보드 순서대로 진행(HSV 단위테스트 → coarse 캐스케이드 → 관측성) 후 카메라 캘리브레이션 착수 → RPi 카메라가 전혀 안 잡히는 걸 발견해 브링업 디버깅으로 전환, 원인 특정까지 마쳤으나 사용자 긴급 요청으로 세션 강제 종료

### 완료

- **HSV 초록/빨강 단위테스트** — `test_color.py`에 7개 신규(모드별 캡처/거부/채도경계, 빨강 저/고 hue band, 빨강 단일range Hue랩어라운드 미지원 회귀테스트). 커밋 `b4f008a`
- **버티포트 coarse 3단 캐스케이드(§5.2)** — `WhiteFieldDetector`(mask→원형 blob)/`BlackVMatcher`(matchShapes 검은V 형상검증)/`RedRingDetector`(빨강 Hue 양끝 게이팅+최소외접원 피팅, ColorFilter의 Hue랩어라운드 한계를 자체 해결). `presets/vertiport_coarse.yaml` 조립. 설계 교훈: `ColorFilter`가 `current`를 자기 mask로 지워버려 뒤 단계는 `original`을 읽게 설계. 단위3+통합1 테스트. 커밋 `7cca1fc`
- **관측성 골격 §7.9 2번** — 이중싱크 사람로거(`utils/logging.py`, provenance 헤더=git해시+config+캘리브id) + JSONL 블랙박스(`utils/blackbox.py`, bounded queue+drop-oldest+`QueueListener` 비차단, 거절이유 로깅). 단위 12개. 커밋 `bf7fdab`. **여기까지 `pytest vision/tests/` 59 passed, 전부 push 완료, RPi 저장소도 이 지점까지 fast-forward 동기 완료**
- **RPi 카메라 브링업 디버깅 (미완, 다음 세션 최우선 인계사항)** — 카메라 캘리브레이션 착수하려고 RPi 촬영 도구(`vision/tools/rpi_capture.py`, picamera2 기반)를 만들었으나 RPi가 Raspberry Pi OS가 아니라 **Ubuntu**라 picamera2/rpicam-apps가 apt에 없어 실패. GStreamer(`libcamerasrc`) 기반으로 재작성했으나, 그 전에 카메라 자체가 커널에도 안 잡히는 걸 발견 → 사용자가 제공한 서드파티 카메라보드(**"CAM109-IMX708AF-75", 정품 CM3 아님**) 제조사 PDF에서 정확한 해법(`camera_auto_detect=0`+명시적 `dtoverlay=imx708,cam0/cam1`) 확인 후 적용 → **커널/V4L2 레벨은 인식 성공**(`/dev/video0`=`rp1-cfe-csi2_ch0`). 그러나 **상위 libcamera 라이브러리가 여전히 카메라를 못 봄** — 원인 특정: 이 Ubuntu의 `libcamera-ipa` 패키지가 RPi5용 **PiSP ISP IPA 모듈 없이 빌드됨**(구형 vc4 IPA만 존재). picamera2/GStreamer 둘 다 이 동일 라이브러리를 거치므로 **똑같이 막힘** — 즉 `rpi_capture.py`의 GStreamer 재작성분은 현재 이 하드웨어에서 작동 불가. 사용자에게 4개 대안(V4L2 RAW 직접캡처 우회/libcamera 소스 재빌드/RPi OS 재설치/보류)을 제시했으나 **답 받기 전 사용자가 긴급 세션종료 요청** → 여기서 끊김

### 결정

- **패스워드리스 sudo를 RPi(`suri` 계정)에 설정** — 사용자 명시 동의 하에(`/etc/sudoers.d/suri-nopasswd`). 앞으로 Claude가 SSH로 직접 sudo 작업 가능, 매번 사용자에게 명령 전달→실행→보고 왕복 안 해도 됨
- **장착 카메라가 정품 Raspberry Pi Camera Module 3가 아니라 서드파티 클론임을 확인** — `vision_plan.md`가 가정한 화각(102°)과 실제 스펙(75°)이 다름. coarse 캐스케이드 탐지거리 가정에 영향 줄 수 있어 재검토 여지 있음(아직 미반영)

### 다음 세션

1. **최우선: 카메라 브링업 4개 선택지 중 어느 걸로 갈지 사용자에게 확인** — 메모리 `project_rpi5_ubuntu_camera_stack.md`에 전체 경과·정확한 진단 명령·재현법 다 있음, 처음부터 재조사할 필요 없이 바로 이어서 진행
2. (권장 방향이면) V4L2 RAW 직접 캡처로 `rpi_capture.py` 재작성 → 실제 체커보드 촬영 → 카메라 인트린식/왜곡 캘리브레이션
3. 또는 카메라 이슈와 독립적으로 §7.9 3번(`FrameSource` 재생 어댑터, logger/blackbox를 main.py에 연결)으로 트랙 전환 가능

### 주의

> `docs/vision_status.md` 트랙보드에 이 모든 내용이 이미 반영되어 있음 — 다음 세션은 그 문서만 읽으면 됨(이 로그는 서술 상세용, 트랙보드가 실질 진입점).
> `vision/tools/rpi_capture.py`는 커밋은 됐지만 **현재 이 RPi에서 작동 안 함** — 카메라 병목 해소 전엔 이 스크립트로 뭘 시도해도 헛수고.
> RPi에 있는 저장소는 `/home/suri/drone_ws/src/suridoksuri`(dev--vision-computing-module) 하나뿐이고, `/home/suri/drone_ws/suridoksuri/suridoksuri`는 완전히 다른 별도 clone(다른 remote/브랜치)이라 건드리면 안 됨.

---

## 2026-07-20 — [mc-hw] RPi5 WiFi 장기끊김 근본원인 확정(brcmfmac 커널버그) + 완화조치 적용

**브랜치:** `dev--vision-computing-module`
**목적:** 사용자가 "flight01 비행 직후에도 연결끊김이 있었다, 원인분석하라"고 요청 → 실제 원인분석 수행 + 해결책 적용

### 완료

- **재발 확인** — `~/wifi_watch.log`에서 flight01 착륙(16:19:29) 약 1분 뒤부터 wlan0 `carrier=0`가 **8분 25초**(16:20:32~16:28:57) 지속 확인. `last reboot`/`journalctl --list-boots` 대조 결과 이 구간에 **재부팅 없음**(현재 부팅이 전날 00:07부터 계속 이어짐) — 시스템 자체는 살아있었고 WiFi 링크만 끊김
- **커널로그 정밀조사 → 근본 메커니즘 확정** — 이 구간 `journalctl -k`에 `brcmfmac: brcmf_set_channel: set chanspec 0x____ fail, reason -52`가 11초 간격으로 4개 채널(0xd022/0xd026/0xd02a/0xd090)을 순환하며 **164회** 반복. 같은 부팅 전체로는 총 164회, 16:20:36~16:28:57 사이에 집중. wpa_supplicant 로그는 이 구간에 전무(더 상위 계층은 시도조차 못함 — 드라이버/펌웨어 레벨에서 막힘)
- **웹서치로 원인 특정** — 동일 에러 시그니처가 **RPi5 브로드컴 WiFi 드라이버(brcmfmac)의 알려진 미해결 커널 버그**([raspberrypi/linux#6049](https://github.com/raspberrypi/linux/issues/6049))와 정확히 일치함을 확인. `gh issue view`로 코멘트 전체 검토: ① 메인테이너(pelwell)는 "이런 다중 서브시스템 동시 장애는 전형적 전원부족 증상"이라 언급(단, 우리 쪽은 `vcgencmd get_throttled`=0x0이라 하드웨어 감지 언더볼트는 아님) ② r41k0u가 GDB/SWD로 직접 디버깅한 결과: disconnect-reconnect 사이클 중 regulatory-domain 플래그가 `restore_custom_reg_settings`에서 stale한 `orig_flags`로 복원되며 채널설정이 계속 거부되는 구조적 버그 — 즉 **일단 한번 끊기면 이 버그 때문에 재연결이 정상(수십 초)보다 훨씬 오래(8분+) 걸리는 것**이 핵심 메커니즘 ③ 다른 사용자(bsdelf)는 country code 불일치+PMF 조합이 실제 원인이었고 `roamoff=1 feature_disable=0x282000` 모듈 파라미터로 해소 보고
- **우리 환경 점검** — regdomain은 `ieee80211w`아님`ieee80211_regdom`=KR(정상), `wireless-regdb` 패키지 설치돼있음(명백한 누락 원인 아님). `iw`가 미설치라 AP가 실제 방송하는 country IE와 정확히 일치하는지는 확인 못함(sudo 필요, 미해결)
- **완화조치 2건 적용 (사용자 직접 실행, 명령어 그대로 복붙 형태로 전달)**
  1. `sudo iw dev wlan0 set power_save off` + `/etc/udev/rules.d/70-wifi-powersave-off.rules`로 영구화 — **적용 확인됨**(`iw dev wlan0 get power_save` → "Power save: off", 재부팅 로그에 `power save disabled`)
  2. `/etc/modprobe.d/brcmfmac.conf`에 `options brcmfmac roamoff=1 feature_disable=0x282000` — 파일 반영·재부팅 완료. `/sys/module/brcmfmac/parameters/`엔 `roamoff`만 파일로 존재(root 전용 읽기라 값 미확인), `feature_disable`은 이 드라이버 빌드에서 sysfs 비노출이라 파일 자체가 없음 — 단 부팅로그에 "Unknown parameter" 경고가 없어 두 파라미터 다 모듈 로드 시 정상 인식된 것으로 추정
- **사용자 피드백 반영** — 처음엔 "원인분석 + 추가정보 필요"까지만 보고하고 실제 해결책(모듈 파라미터 등)을 빠뜨렸는데, 사용자가 "해결방안은 어디갔는가"라고 지적 → GitHub 이슈에 이미 있던 커뮤니티 검증 우회책을 다시 정리해 제공

### 결정

- **이 버그는 업스트림에 정식 패치 없음** — 적용한 두 조치는 커뮤니티에서 반복 검증된 우회책이지 근본 수정이 아님. 완전히 안 끊기게 보장하는 건 아니고, 다음 비행 결과로 실효성 검증 필요
- **최초 트리거(왜 16:20:32에 처음 끊겼는지)는 규명 범위 밖으로 남김** — 신호거리/RC 2.4GHz 간섭/전원 sag 후보가 경합 중이나 이번 조치로 "한번 끊겨도 금방 복구되게"는 만들었으므로 실용적 우선순위는 낮춤

### 다음 세션

1. **다음 비행 후 `~/wifi_watch.log`에서 장기(수 분 단위) `carrier=0` 재발 여부 확인** — 이번 조치의 실질 검증
2. 재발 시: `sudo cat /sys/module/brcmfmac/parameters/roamoff`로 실제 적용값 확인, `iw`로 AP의 실제 country IE와 커널 "KR" 일치 여부 대조, 필요 시 netplan 네트워크 블록에 `ieee80211w=0`(PMF 비활성화) 추가 시도
3. 최초 트리거 규명하려면 wpa_supplicant 로그레벨을 debug로 올려 다음 발생 시 실제 deauth/assoc-reject reason code 캡처 필요(현재 INFO라 이번 8분 구간엔 로그 자체가 없었음), 재발 시점에 RC 조작 여부도 육안 기록

### 주의

> 이 버그의 프록시먼트(진짜 첫 트리거)는 여전히 미해결 — 완화조치는 "한번 끊겨도 빨리 복구"를 목표로 한 것이지 "안 끊기게"가 아님. 실비행 중 tailscale 끊김이 짧게(수십 초 이내) 발생해도 더 이상 놀라지 않아도 되나, 길게(수 분) 지속되면 이번 조치가 안 먹힌 것이니 재보고할 것.

---

## 2026-07-20 — [mc-hw] flight01 제어상실 사고 — 근본원인 규명 + STREAMING/FOLLOWING 위치 setpoint 슬루레이트 제한 (아래 "기록 전용" 세션의 후속)

**브랜치:** `dev--vision-computing-module`
**목적:** 사용자가 직전 실비행에서 "기체가 제어를 잃어 수동 착륙했다"고 보고, 회수한 ulog로 원인 규명 요청

### 완료

- **사고 로그 위치·특정** — `mc-hw-rpi5-wifi-diag` worktree(다른 세션, lock 보유— 읽기만 함)의 `logs/2026-07-20_flight01/`에서 발견. `log_18_2026-07-20-07-19-30.ulg`(38.7초, 본비행)와 `log_16`/`log_17`(각 1초 미만, 무관)을 pyulog로 직접 분석
- **타임라인 재구성:** t=1.9s CommandTOL 이륙(목표 AMSL 52.31=지면48.3+4.0m) → t=9.3~9.9s `climbing_reached()`가 AGL≈3.5~4.0m에서 정상 판정(허용오차 수정 유효 확인) → **t=9.9~11.3s AUTO.TAKEOFF가 계속 상승해 실고도 최대 7.6m 도달(목표 4.0m의 거의 2배 오버슈트)** → t=11.3s nav_state AUTO.TAKEOFF→OFFBOARD 전환 **바로 그 순간 OFFBOARD 첫 세트포인트가 `(N,E,Z)=(0,0,-4.0)`, yaw=90°로 순간점프 발행 — 실제 위치는 `(-4.4,1.2,-7.3)`, yaw≈-80°(수평 4.5m+수직 3.3m+요 170° 불연속)** → t=11.5~13.0s 격렬한 자세급변(roll -16°, pitch -30.8°, yaw rate 최대 186°/s) → t=16.4s 조종사 스틱 입력 감지(수동 회수 시작) → t=37.7s disarm. EKF `quat_reset_counter`는 이 구간 내내 불변 — 센서/EKF 결함 아님, 세트포인트 자체가 원인임을 확인
- **근본원인 확정:** `offboard_node.py` STREAMING(321행)과 `_step_following()`(775행) 둘 다 `L1Guidance.target_point_ned(pos, _FW_LOOKAHEAD=70.0)` + 절대위치 PoseStamped 발행 방식을 MC/FW 구분 없이 공용 — 70m lookahead는 FW가 목표점 근처에서 flower-pattern으로 도는 것을 막기 위한 FW 전용 기법(목표점을 항상 선회반경 밖에 둬 "도착"을 안 일어나게 하는 pursuit 유도)인데, 이번 비행 경로 총길이(~12m)보다 훨씬 커서 항상 경로 끝점(WP1)을 그대로 반환 — 기체의 실제 현재위치와 무관한 고정 절대좌표가 됨. 여기에 클라이밍 중 고도 오버슈트(AUTO.TAKEOFF→OFFBOARD 모드전환 확정까지 수 초 지연되는 동안 계속 상승, `session_status.md` 기존 문서화된 "home_position.alt 드리프트 잔여리스크"가 실제로 재현된 것으로 추정)까지 겹쳐 OFFBOARD 진입 첫 순간의 실제 오차가 구조적으로 클 수밖에 없었음
- **수정 1차 (속도제어 전환) → 사용자 지적으로 정정:** 처음엔 "MC는 PX4 OFFBOARD 속도 세트포인트를 정상 추종한다"는 점에 근거해 STREAMING을 0속도 스트리밍, FOLLOWING을 `L1Guidance.ned_velocity_cmd()`(속도기반 인터페이스)로 전환했으나, **사용자가 즉시 지적: "최종기체(VTOL)는 위치기반으로 동작할 것이고, 이 MC 테스트기체는 최종기체의 동작을 검증하기 위한 것인데 MC만 속도기반으로 바꿀 이유가 없다."** 정곡을 찌르는 지적 — 실제로 최종 VTOL은 `vehicle_type:=vtol`(`is_mc=False`)로만 운용돼 STREAMING/FOLLOWING의 FW 위치기반 경로만 타므로, MC 전용 속도제어 분기를 만들어봤자 최종기체가 실행할 코드와 다른 코드를 검증하는 셈이 되어 이 테스트기체의 존재 이유(최종기체 avionics·제어로직의 벤치 검증)에 반함
- **수정 2차 (최종, 위치기반 유지):** `fc_ros/fc_ros/nodes/offboard_node.py` — MC도 FW와 동일하게 `/mavros/setpoint_position/local` 위치 setpoint를 계속 발행하되(제어로직 자체는 미분기), 실제 발행값만 불연속을 없애도록 수정. ① **STREAMING:** MC는 매 틱 **현재위치 그대로**를 위치 setpoint로 스트리밍(`self._mc_pos_ramp = state.pos_ned`) — OFFBOARD 확정 순간 PX4가 이어받는 값이 항상 그 순간의 실제 위치와 일치해 점프가 없음. FW는 기존 lookahead 로직 완전 미변경(이미 SITL 검증됨, 건드리지 않음) ② **`_step_following()`:** MC는 기존과 동일하게 `target_point_ned()`로 lookahead 목표를 계산하되, 그 목표로 즉시 점프하지 않고 `self._mc_pos_ramp`를 `v_approach`(기존 ENTRY 상태 파라미터 재사용, 5.0m/s)로 슬루레이트 제한해 점진 접근시킨 값을 발행 — FW는 여기도 미변경. `fc_bridge`(rclpy 비의존) 순수 로직으로 사고 시점 실측오차(수평4.5m+수직3.3m)를 대입해 시뮬레이션 — 틱당 최대 이동량 0.5m로 약 1.2초에 걸쳐 수렴함을 확인(수정 전엔 즉시 4.54m+ 순간점프)
- **미검증:** `offboard_node.py`는 rclpy 의존이라 이 WSL 샌드박스(pytest·rclpy 모두 미설치)에서 실행 단위테스트 불가 — 문법 검사(`py_compile`)와 `fc_bridge` 순수 로직 레벨 수치 검증만 수행. **다음 실비행 전 반드시 SITL(`gz_x500` MC) 회귀검증 필요**
- **커밋·push·RPi 반영 완료** — 위치기반 정정본을 `8ea5e35`로 커밋해 `dev--vision-computing-module`에 직접 push. RPi5(`doksuri`, SSH)에서 `git pull`(`2bb8455..8ea5e35` fast-forward) 후 `docker exec fc colcon build --packages-select fc_ros` 재빌드 성공(3.70s), 설치본(`install/fc_ros/lib/python3.10/site-packages/fc_ros/nodes/offboard_node.py`)이 소스와 diff 일치 확인 — 2026-07-18에 지적됐던 "빌드 미반영" 재발 없음. **다음 실비행에 이 수정이 실제로 실릴 준비 완료, 단 SITL 회귀검증 전까지는 비행 보류.**

### 결정

- **MC 테스트기체는 항상 최종 VTOL과 동일한 위치기반 세트포인트 경로를 타야 한다** — 사용자가 명시적으로 정정한 원칙. 앞으로 MC 전용 분기가 필요할 때도 "제어 신호의 종류(위치 vs 속도)"는 FW와 통일하고, 값 계산·슬루레이트 등 "얼마나/어떻게 접근하는가"만 MC 전용으로 조정할 것
- STREAMING/FOLLOWING의 FW lookahead *계산 로직* 자체는 그대로 두고(짧은 경로에서 경로끝점으로 클램프되는 것 자체가 문제가 아님), 그 계산결과를 실제로 발행하는 방식(즉시 vs 슬루레이트 제한)만 MC에서 조정 — "MC에서 lookahead 값만 줄이는" 임시조치나 "MC를 속도제어로 바꾸는" 방식 둘 다 채택하지 않음
- HOLD 상태(MC가 FOLLOWING 완료 후 거치는 마지막 착륙 대기)는 이번 수정 범위에서 제외 — 이미 WP1 끝점을 직접 위치 목표로 쓰는 MC 인지 코드였고(주석에 명시), FOLLOWING 종료조건(`d_end_thresh=10m`) 때문에 진입 시점 오차가 이번 사고 규모(4.5m+) 만큼 커질 구조가 아니라 위험도가 다름

### 다음 세션

1. **최우선 — 다음 MC 실비행 전 SITL(`gz_x500`) 회귀검증 필수.** STREAMING 진입~OFFBOARD 확정~FOLLOWING~HOLD 전 구간에서 세트포인트 불연속(점프) 없이 부드럽게 추종하는지, 특히 클라이밍 중 의도적으로 드리프트/오버슈트를 재현해 확인
2. **고도 오버슈트 자체의 근본원인 규명(미해결, 이번 수정과 별개)** — AUTO.TAKEOFF→OFFBOARD 모드전환 확정 지연 동안 계속 상승하는 구조 자체는 안 고쳐짐(이번 수정은 그 위에서 벌어지는 세트포인트 불연속만 제거). `_step_climbing()`에 AUTO.TAKEOFF 이탈 자체를 감지·대응하는 로직 추가 여부(기존 flight09 기록에도 남아있던 미결정 사항)와 함께 재검토 필요
3. `logs/2026-07-20_flight01/`는 이번 병합으로 git 커밋 완료(아래 "기록 전용" 세션이 push함) — 별도 조치 불필요

---

## 2026-07-20 — [mc-hw] flight01 오프보드 전환 직후 제어상실 사고 — 기록 전용(분석 별도 세션)

**브랜치:** `dev--vision-computing-module`
**목적:** climbing_reached 수정 후 실비행(flight01) 로그수집 요청 → 도중 사용자가 "실제로는 사고였다"고 정정 → 원인 분석 없이(별도 세션 예정) 놓치는 사실 없게 원본·타임라인·데이터 가용성 기록

### 완료

- **초기 오판 → 사용자 정정:** launch.log만 보고 "ARM→CLIMBING(4.0m)→OFFBOARD 확인→FOLLOWING→WP1→LANDING→disarmed"(16:18:50~16:19:29, 39초)를 **"첫 오프보드 성공"으로 assistant가 잘못 판단**. 사용자가 직접 정정: 수직 상승 완료 직후(오프보드 전환 전후로 추정) 기체가 순간 제어를 잃고 **북서쪽으로 픽 쓰러지며 roll**, 즉시 RC로 조종권 회수. **launch.log의 OFFBOARD 이후 기록은 실제 자율비행 수행이 아니라 조종사가 수동 회수한 기체 위치를 소프트웨어가 그대로 읽은 것일 가능성이 높음**(미확정) — 이 정정을 `docs/session_status.md` 🚁 트랙과 `logs/2026-07-20_flight01/notes.md`에 반영
- **launch.log 유닉스 타임스탬프 → KST 정밀 변환:** ARM 16:18:50 · CommandTOL 이륙(alt=52.3m AMSL) 16:18:53 · 운용고도 4.0m 도달 16:19:00 · OFFBOARD 전환 요청 16:19:02 · **OFFBOARD 확인→FOLLOWING 16:19:03(사고 발생 추정 시각)** · WP1 홀드 16:19:05 · WP1 도달→LANDING·AUTO.LAND 16:19:06 · 착륙완료(disarmed) 16:19:29. OFFBOARD 진입 후 전체 시퀀스가 26초 만에 끝나는 것도 실제 자율 경로추종치고는 지나치게 빠름 — 사고설과 정합적
- **ulog 회수 — 최초 실패 후 성공:** FC 최초 확인 시 `/dev/ttyACM0` 없음(flight09와 동일 패턴)으로 회수 실패 기록했으나, **사용자가 Pixhawk 전원을 재연결**해줘서 재시도 → 성공. FC 로그 목록에 오늘자 3건: id16(UTC 07:18:28, 156,843B)·id17(UTC 07:18:28, 156,277B — id16과 같은 초, 원인 미확인)·**id18(UTC 07:19:30, 1,729,984B — 이 비행의 본 로그로 추정)**. UTC+9=KST로 비행 시각대(16:18~16:19 KST)와 정확히 일치 확인 후 3개 전부 `logs/2026-07-20_flight01/`로 다운로드. **pyulog가 RPi에 미설치라 내용 분석은 하지 않음**(다음 세션 몫)
- **rosbag 토픽 점검:** 설정된 11개 중 10개만 실제 기록됨 — **`/fc_ros/override`가 이번엔 기록 안 됨**(RC 오버라이드 개입 시점을 이 토픽으로 직접 특정 불가). `/mavros/imu/data`는 기록됨(자세 쿼터니언·각속도 포함) — ulog 분석 전이라도 이걸로 roll 이벤트 자체는 먼저 확인 가능
- **wifi_watch.log 대조(참고자료, 인과관계 미확정):** 게이트웨이 ping 무응답이 비행 시작 전 16:15경부터 이미 간헐 발생 중이었음. wlan0 `carrier=0`(인터페이스 완전 다운)은 16:20:38~16:26:14(약 5분 36초)로 **사고 시점(16:19경)보다 약 1분 뒤에 시작** — 타이밍상 사고와 직접 겹치지 않으나, 사용자가 "중간에 끊겼다"고 보고한 사실과는 일치. 인과관계 주장은 하지 않고 사실만 기록
- **flight01 `notes.md` 갱신** — 조종사 증언(사고 경위)·ulog 회수 상태·assistant 오판 정정 사실을 모두 반영해 다음 분석 세션이 이 폴더만 봐도 전체 맥락을 알 수 있게 정리

### 결정

- **이번 세션에서는 사고 원인 분석을 하지 않음** — 사용자가 "분석은 다른 세션에서 진행할 테니 놓치는 점 없이 메모만 하라"고 명시적으로 범위를 제한함. 데이터 수집·원본 보존·정확한 타임라인 기록에만 집중
- **launch.log의 "정상완주" 겉모습을 향후에도 그대로 신뢰하지 않기** — 소프트웨어 상태머신 로그만으로 실제 비행 성패를 판단하면 안 된다는 걸 이번에 직접 겪음(assistant 본인의 오판 사례로 기록)

### 다음 세션

> **(2026-07-20 갱신) 아래 항목 중 1번은 위 "근본원인 규명" 세션에서 완료됨** — 나머지는 제어상실 직접원인이 소프트웨어 세트포인트 버그로 확정된 이상 우선순위 낮음, 참고용으로만 남김.

1. ~~id18 ulog를 pyulog로 분석~~ — **완료** (위 항목 참조: STREAMING/FOLLOWING의 FW lookahead 오적용이 원인)
2. id16·id17(같은 초 156KB 2건)이 무엇을 기록한 로그인지 확인 (미착수, 낮은 우선순위)
3. `vehicle_command`/`manual_control_setpoint`로 RC 오버라이드 개입 정확한 시각 특정 (미착수, 낮은 우선순위)
4. `MIS_TAKEOFF_ALT` 등 PX4 파라미터 조회(flight09 잔여 미확정 사항과 함께, 미착수)
5. 이 사고가 ✈vtol-실기체 트랙의 과거 결함과 연관 있는지 검토 (미착수 — 다만 원인이 소프트웨어 세트포인트 버그로 확정돼 무관할 가능성이 높아짐)
6. wifi_watch.log의 carrier=0 구간이 이 사고와 무관한 별개 이슈인지 판단 (미착수, 타이밍상 직접 겹치지 않음만 확인됨)

### 주의

> **이 사고 이후 다음 실비행 전 반드시 근본원인 분석부터 완료할 것** — 원인 미상인 채로 재비행하면 동일 사고 재현 위험. **(2026-07-20 갱신) 위 "근본원인 규명" 세션에서 원인 파악·수정 완료, 단 SITL 회귀검증 전까지는 여전히 재비행 금지.**
> launch.log·notes.md의 "정상완주"처럼 보이는 문구를 이후 세션에서 그대로 인용하지 말 것 — 이 기록의 정정 내용을 먼저 확인.

---

## 2026-07-20 — [mc-hw] climbing_reached 허용오차 도입 + 병렬 세션 정리·병합

**브랜치:** `dev--vision-computing-module`
**목적:** 실비행 중 "고도가 정확히 일치해야만 천이한다"는 사용자 보고 대응 + 그 시점 병렬로 진행 중이던 다른 세션들(worktree)의 로컬 작업을 확인·정리해 브랜치에 반영

### 완료

- **`climbing_reached()` 판정을 단측 임계값 → ±0.5m 허용구간으로 변경** — 기존 `AGL >= transition_alt`는 목표고도 바로 아래(예 -0.1m)에 정착하면 절대 만족되지 않아 CLIMBING이 무한 대기하는 문제가 있었음(사용자 보고, flight09 실측과도 일치 — 아래 참조). `abs(AGL - transition_alt) <= alt_tol`(기본 0.5m)로 변경. **N,E(수평)은 의도적으로 제외** — CLIMBING 중 수평은 PX4 AUTO.TAKEOFF가 자체 관리해 이 노드에 목표 N,E가 없고, 비-RTK GPS 수평오차가 통상 0.5m를 넘어 수평까지 조건에 넣으면 CLIMBING 영구대기라는 더 심각한 회귀를 유발할 위험이 컸음. `fc_ros/test/test_offboard_node.py`에 경계값(하한/상한/직전값) 테스트 추가·기존 케이스 갱신 — pytest 미설치 환경이라 동일 입력값으로 순수 스크립트 재현해 수동 검증(fc_bridge/execution/state_logic.py는 rclpy 의존 없음)
- **flight09 진단과의 교차검증** — 병합 도중 다른 세션이 이미 dev 브랜치에 올린 flight09 진단(PX4가 목표 4.0m 중 3.63m에서 자체적으로 `AUTO.LOITER` 복귀, OFFBOARD 진입 전무)을 발견. 4.0−3.63=0.37m로 새 허용오차(0.5m) 안에 들어가 이번 수정이 그 케이스를 실제로 구제할 가능성 확인 — 다만 "PX4가 왜 목표 전에 스스로 포기하는지"는 별도 미해결 원인(`MIS_TAKEOFF_ALT`·배터리 등 후보, 미확정)으로 남음
- **PR 대신 직접 병합** — 처음엔 별도 브랜치+draft PR로 진행했으나 사용자가 "이미 main에서 분리된 dev 브랜치인데 PR 왜 하냐, 머지해라" 지적 → PR 닫고 `dev--vision-computing-module`에 직접 fast-forward/병합 push로 전환(이 프로젝트는 전 트랙이 이 dev 브랜치를 공용하며 SITL-5 안정화 후에나 main 병합을 결정하는 구조라 PR 절차가 불필요했음)
- **병렬 worktree 세션 감사** — 병합 시점에 로컬에 worktree 5개가 동시 존재(`agent-ab2c62d6605ef80b6`, `mc-hw-rpi5-wifi-diag`〈다른 활성 세션이 lock 보유, 손대지 않음〉, `mc-hw-transition-alt-tol`〈이 세션〉, `serene-crunching-cocoa`, 메인 체크아웃) 확인. `agent-*`(27eb6d2, want_takeoff 판별자+배터리 정량화 진단)와 `serene-crunching-cocoa`(0779f3d, flight09 진단)의 커밋은 이미 다른 경로로 `dev--vision-computing-module`에 병합돼 있었음을 확인 — 로컬에만 있던 작업 없음. `mc-hw-rpi5-wifi-diag`(b725538, WiFi 진단+USB-C 전원 조치)도 그 세션이 직접 push해 이미 dev에 반영된 상태였음(확인만, 병합은 그 세션이 수행). 메인 체크아웃(`/home/suri/suridoksuri`)의 로컬 브랜치 ref가 origin 대비 6커밋 뒤처져 있어 fast-forward로 최신화(작업 내용 없음, 안전한 정리)

### 결정

- **PR 워크플로 사용 안 함** — 이 저장소의 `dev--vision-computing-module`은 이미 사실상의 통합 브랜치이고 전 트랙이 여기 직접 커밋·push하는 관례라, 앞으로 이 브랜치로 향하는 작업은 별도 PR 없이 직접 병합·push한다(main으로 향할 때만 필요시 재검토)
- **`mc-hw-rpi5-wifi-diag` worktree는 lock 보유 세션이 있어 건드리지 않음** — 다른 활성 세션과 충돌 방지가 우선

### 다음 세션

1. **다음 MC 오프보드 실비행에서 CLIMBING→STREAMING이 ±0.5m 허용구간으로 정상 트리거되는지 확인** — 이번 수정의 실질 검증
2. **PX4가 목표고도 도달 전 스스로 `AUTO.LOITER`로 복귀하는 근본원인 규명(flight09, 미해결)** — FC 전원 재연결 후 ulog id=13 회수 + `MIS_TAKEOFF_ALT` 파라미터 조회. 이번 허용오차 수정과 별개로 필요(허용오차 밖으로 크게 미달하면 여전히 무한대기)
3. **`_step_climbing()`에 AUTO.TAKEOFF 이탈 자체를 감지·대응하는 로직 추가 여부** — flight09 트랙 기록에 "코드 수정 보류, 사용자 판단 필요"로 남아있음, 아직 미결정
4. **실기체 pytest 환경 부재** — 이 개발컴(WSL)엔 pytest/venv 구성이 안 돼 있어 이번 테스트 갱신도 수동 재현으로만 확인함. 필요 시 최소 `python3-venv` 설치 여부 확인
5. 정리 후보(급하지 않음): 이미 dev에 반영된 `agent-ab2c62d6605ef80b6`·`serene-crunching-cocoa`·`mc-hw-transition-alt-tol`(이 세션) worktree/브랜치 정리, origin의 stale `mc-hw/transition-alt-tolerance` 원격 브랜치 삭제

---

## 2026-07-19 — [mc-hw] RPi5 tailscale/WiFi 끊김 진단 + USB-C 전원 협상 완화

**브랜치:** `dev--vision-computing-module`
**목적:** 실비행 중 tailscale SSH 연결이 자꾸 끊긴다는 사용자 보고 원인 규명 + 비행용 비-PD 전원에서도 안정 부팅되게 조치

### 완료

- **WiFi 끊김 진단 (SSH 원격, 여러 차례 재접속하며 반복 조사):** 초기엔 재부팅 루프·tailscale 노드 중복(`doksuri-3` 등)을 의심했으나 둘 다 사용자가 정정(각각 "방금 비행 위해 켠 것", "계정 재사용으로 인한 정상 현상") — 오진단으로 폐기. `journalctl -u systemd-networkd`에서 wlan0가 재부팅 없이도 `Lost carrier`→재연결을 반복하는 패턴 확인(같은 AP `DepartmentOfAgriculture`·같은 IP로 매번 재연결). `journalctl -k`에서 `brcmfmac: brcmf_cfg80211_set_power_mgmt: power save enabled` 확인 — RPi5 브로드컴 WiFi 칩의 절전모드 활성으로 인한 잘 알려진 결함 가능성. 이후 사용자가 재비행에서도 수차례 끊김 재현 보고 → **RC 2.4GHz + WiFi 핫스팟 2.4GHz 동일대역 간섭**이 더 유력한 근본원인으로 격상(사용자 확인: RC 수신기 2.4GHz, 핫스팟도 2.4GHz 사용 중 — 5GHz는 "GPS 간섭 우려" 문서 권고로 회피 중이라 대역 전환으로 해결 불가). 최종 확정은 못했고 복수 가설 공존 상태.
- **`sudo iw dev wlan0 set power_save off` 처방 전달** — sudo 비밀번호 필요 + EEPROM/드라이버 급 변경이라 자동 적용하지 않고 사용자 직접 실행용 명령 전달(적용 여부 미확인).
- **비-root 상시 모니터링 배포 (`~/wifi_watch.sh` → `~/wifi_watch.log`)** — 사용자 명시적 승인 후 RPi5에 배포. 5초 간격 wlan0 carrier/operstate/gateway ping 기록, nohup 백그라운드 + crontab `@reboot`로 재부팅 후에도 자동 재기동. 배포 중 SSH 세션이 두 차례 exit 255로 끊겨(원인 불명, 어쩌면 같은 WiFi 불안정성의 방증) `nohup ... </dev/null >/dev/null 2>&1 &` 형태로 재시도해 성공.
- **EEPROM `PSU_MAX_CURRENT=1600` 적용** — RPi5는 USB-C 급전인데 비행 중엔 5V/5A PD 어댑터를 못 쓰고 BEC 등 비-PD 전원을 씀. 기본값(5000, 미설정)은 5A negotiation을 요구해 이런 전원에서 부팅 불안정을 유발할 수 있음 — 공식 문서상 표준 완화값 1600으로 변경하는 명령을 전달, 사용자가 직접 실행 후 확인 완료.
- **Claude 메모리 갱신** — `project_rpi5_tailscale_wifi_drops.md`(WiFi 진단 경과 전체, 오진단 포함) 신규, `project_rpi5_usbc_power_psu_max_current.md`(PSU_MAX_CURRENT 조치) 신규. `docs/session_status.md` 🚁 mc-실기체 트랙 + "실기체(RPi5)" 환경참조 표에 반영.

### 결정

- **모니터링 스크립트 배포는 자동실행 차단됨(auto-mode 분류기)** — 실비행 컴퓨터에 백그라운드 상시 프로세스+crontab을 자율적으로 심는 건 위험도가 높은 작업으로 분류돼 1차 시도가 거부됨. 우회하지 않고 사용자에게 설명 후 명시적 승인을 받고서야 재시도해 배포함 — 이런 종류(실비행 하드웨어의 지속 상태 변경)는 앞으로도 먼저 설명하고 승인받을 것.
- **EEPROM 변경은 자동 적용하지 않음** — 부트로더 재굽기는 되돌리기 어려운 하드웨어급 변경이라 사용자가 직접 실행하는 방식으로 진행(WiFi power_save 처방도 동일 원칙 적용, sudo 비밀번호 벽도 겹침).
- WiFi 끊김은 **미해결 상태로 세션 종료** — 다음 비행 결과로 검증 필요.

### 다음 세션

1. 다음 비행 후 `~/wifi_watch.log`(carrier=0/ping=LOSS 구간)를 FC 텔레메트리(스로틀·자세·고도)와 대조해 끊김이 거리/시간 비례(전원관리·RF거리)인지 특정 기동·RC 활성 순간에 몰리는지(RC 간섭) 구분
2. 사용자가 `sudo iw dev wlan0 set power_save off` 실행했는지 미확인 — 재확인 필요
3. RC-WiFi 2.4GHz 간섭이 최종 확정되면 완화책(안테나 물리적 이격, RC 송신출력 하향, 차폐) 검토 필요 — 5GHz 전환은 GPS 간섭 우려로 불가

### 주의

> RPi5 EEPROM(`PSU_MAX_CURRENT`)·WiFi 드라이버 설정은 SSH로 원격 확인은 가능해도 변경 적용엔 sudo 비밀번호가 필요(그룹 미가입, 기존 기록과 일치) — 앞으로도 이런 처방은 사용자 직접 실행 명령으로 전달할 것.

