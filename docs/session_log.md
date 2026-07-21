---
doc_type: session_log
project: suridoksuri-1
---

# 세션 로그

> 최신 세션이 위에 온다. `/session-log` 커맨드로 세션 종료 전 자동 작성.
> **최근 8개 세션만 유지** — 초과분은 `/session-log`가 `docs/archive/session_log_YYYY-MM.md`로 이동한다.
> 과거 기록: `docs/archive/session_log_2026-06.md` (2026-06-18 ~ 06-30) · `docs/archive/session_log_2026-07.md` (2026-07-03 ~)

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

---

## 2026-07-18 — [mc-hw] 현장 원격 리빌드 + flight09 로그수집 + 고도미달 진단

**브랜치:** `dev--vision-computing-module`
**목적:** 사용자가 비행장에서 RPi5 터미널에 접근 못하는 상태로 "리빌드해달라" 요청 → 이후 비행(flight09) 로그 수집·고도미달("천이 명령 미하달") 원인 진단까지 이어서 처리

### 완료

- **원격 리빌드** — SSH로 RPi5 접속, `docker exec fc colcon build --packages-select fc_ros` 실행. 처음엔 `sudo docker exec`가 비밀번호 요구로 막혔으나 사용자가 `suri`를 `docker` 그룹에 추가해줘서 sudo 없이 해결. 빌드 성공 확인 + `install/`이 최신 소스와 `diff` 일치 확인. 이 리빌드가 별도 백그라운드 세션이 동시에 진단 중이던 "flight01~08 stale colcon build" 근본원인을 실제로 해소한 조치였음(교차참조, 아래 결정 참조)
- **flight09 로그 수집** — 리빌드 직후 사용자가 실비행(`transition_alt:=4.0`, 삼각 왕복 웨이포인트) 진행. `record_flight.sh`로 rosbag+launch.log는 정상 수집(`logs/2026-07-18_flight09/`). ulog는 FC(Pixhawk)가 비행 후 전원이 내려간 것으로 판단돼(USB enumeration은 유지, `/dev/ttyACM0` 3초간 0바이트 수신) 회수 실패 — 사용자가 바빠서 재시도 보류, 미회수 상태로 종료
- **고도미달 진단** — 사용자 보고("목표 4.0m인데 3.6m가 최대, 천이 명령 미하달")를 rosbag 직접 디코드(`rosbag2_py`+`deserialize_message`로 `/mavros/state`·`/mavros/local_position/pose` 추출)로 검증. AMSL 계산(`home_amsl+transition_alt`)은 정확했음(리빌드 덕분) — 실제 원인은 PX4가 `AUTO.TAKEOFF` 진입 4.66초 만에 자체적으로 `AUTO.LOITER`로 복귀(`pose.z` 최댓값 3.63m)하고, 이후 조종사 재이륙 시도도 동일 패턴 반복 후 42초간 LOITER 고착 — **OFFBOARD 모드 진입이 이 비행 내내 한 번도 없었음**. `offboard_node._step_climbing()`이 PX4 모드 이탈을 감지하지 않는 순수 폴링 설계라 소프트웨어가 이를 알아채지 못했던 것으로 결론
- **RPi5 네트워크 단절 대응** — 진단 도중 RPi5가 Tailscale에서 완전히 끊겨(ping 100% 손실) 약 한 턴 동안 재접속 대기. 사용자가 "지금 연결했다"고 알려온 뒤에도 실제 재접속까지 재시도 필요했음(즉시 복구 아님)
- **다른 백그라운드 세션과의 협업** — flight01~08을 전수분석 중이던 별도 세션이 `worktree-agent-*` 브랜치에 남긴 커밋(근본원인: stale colcon build)을 발견 → `dev--vision-computing-module`로 fast-forward 병합 후 이 세션의 flight09 발견과 교차참조해 `session_status.md`에 함께 기록

### 결정

- **ulog 재시도 중단** — 사용자가 "바쁘니까 받지 마라"고 명시적으로 중단 지시. rosbag 기반 분석만으로 원인 결론은 이미 확정적이라고 판단해 그대로 수용, ulog는 다음 기회로 미룸
- **코드 수정은 보류** — `_step_climbing()`의 PX4 모드 이탈 감지 부재를 발견했지만 이번 세션에선 진단까지만 하고 실제 코드 변경은 하지 않음(사용자 판단 대기)

### 다음 세션

1. FC 전원 재연결 시 ulog id=13 회수 + `MIS_TAKEOFF_ALT` 파라미터 확인으로 "PX4가 왜 3.6m대에서 자체 이륙완료 처리했는지" 근본원인 확정
2. flight09 rosbag엔 배터리 토픽이 없어 전압붕괴 가설(위 flight01~08 분석 ④) 재현 여부 확인 불가 — ulog 확보 후 대조
3. `_step_climbing()` AUTO.TAKEOFF 이탈 감지·재시도 로직 추가 여부 사용자와 논의
4. `logs/2026-07-18_flight09/`가 아직 git 미커밋 — 다음 로그 커밋 배치에 포함

---

## 2026-07-18 — [main][mc-hw] 라파5 원격 로그 조사 → 문서 뒤처짐·인프라 버그 2건 발견

**브랜치:** `dev--vision-computing-module`
**목적:** 사용자가 완료한 실비행의 로그 확인 요청 → SSH 원격 접속 체계 구축 → 실제 접속해 조사 → 발견 사항 정리

### 완료

- **Tailscale SSH 키 등록** — RPi5(`100.67.27.83`, hostname `doksuri`)에 이 WSL 개발컴용 ed25519 키(`claude-code-wsl-suridoksuri`) 등록. 이후 세션에서 비밀번호 없이 바로 SSH 가능(`sudo`/`docker`는 여전히 비밀번호 필요, 그룹 미가입)
- **원격 조사로 문서-현실 괴리 발견** — `docs/session_status.md`엔 "✈ vtol-실기체: 07-09 이후 기체 결함으로 비행 보류"로 남아있었으나, 실제 `logs/` 디렉터리엔 07-07·07-11·07-17(6회)·07-18(8회, 오늘) 비행 폴더가 존재. 07-17·07-18 14회는 문서에 전혀 기록되지 않은 채 진행됨(`vehicle_type:=mc`)
- **작업 G(로그 인프라) 실사용 버그 2건 확정** — ① RPi 호스트에 pymavlink 미설치로 `pull_ulog.py` 자동회수가 지금까지 한 번도 성공한 적 없었음(실패가 어디에도 기록 안 돼 발견이 늦어짐) ② `record_flight.sh`를 컨테이너 `fc` 안 root로 실행해 `logs/<날짜>_flightNN/`이 root 소유가 되어 `suri` 계정 쓰기 불가
- **오늘(07-18) 비행 11개 ulog 전량 회수** — RPi 호스트에 pip 부트스트랩(`--user --break-system-packages`)으로 pymavlink 설치 → FC에서 직접 `.ulg` 11개 다운로드. 8개(id3~10)는 기존 `flight01~08`(rosbag+launch.log) 폴더와 시각 매칭해 완전한 폴더로 합침, 3개(id0~2, `record_flight.sh` 쓰기 전 로그)는 대응 rosbag/launch.log 없이 `logs/2026-07-18_unlogged/`에 "비행기록 부족함"으로 보관. root 소유 폴더 문제로 RPi 쪽 직접 write는 실패해 staging 폴더 경유 → 이 개발컴으로 scp 후 로컬에서 재조립
- **`record_flight.sh` 수정** — 종료 시 `$FLIGHT_DIR`을 `$LOG_ROOT` 소유자로 chown(best-effort, 실패해도 스크립트 안 죽음)해 향후 비행부터 root 소유 문제 방지. `bash -n` 통과. (`test_flight_logs.py`는 이 WSL에 pytest/pymavlink 미설치라 로컬 실행 못함 — 대상이 `pull_ulog.py` 순수함수라 이 변경과 무관해 회귀 위험은 낮음)
- **`docs/flight_plan.md` 작업 G 표 최신화** — "계획 확정, 미착수" → "✅ 완료"로 수정 (실제로는 이미 완료·검증됨)
- **RPi `git pull` (서브에이전트 위임)** — 처음엔 `origin`에 `07681d3`가 미푸시 상태라 반영 안 됨을 서브에이전트가 정확히 진단·보고 → 사용자 승인으로 push 후 재실행, RPi에 chown 수정 반영 확인(`grep chown record_flight.sh`)
- **RPi 소유권 정리 확인 + 07-18 로그 RPi 원본도 완결** — 사용자가 RPi에서 `sudo chown -R suri:suri logs/` 직접 실행 → SSH로 확인. 로컬 스테이징에 있던 07-18 ulog 8개도 (이제 쓰기 가능해진) RPi 원본 `flight01~08` 폴더로 이동해 RPi 쪽 사본도 완전해짐
- **비행로그 git 커밋 방침 전환** — "GitHub 업로드 안 함"(2026-07-06 결정) 재검토를 사용자에게 요청 → **일반 git 커밋으로 전환**(LFS 아님, 트레이드오프 인지하고 승인) 결정. `.gitignore` 루트의 `logs/` 제외 규칙 제거(`*.log`는 유지하되 `!logs/**/*.log`로 예외 처리해 `launch.log`/`rosbag_record.log`도 추적되게), `tools/flight_logs/README.md`·`flight_plan.md` "업로드 방침" 갱신. 오늘 07-18 로그 53개 파일(rosbag+ulog+launch.log 등) 커밋

### 결정

- 서브에이전트에 `record_flight.sh` 수정을 위임 시도했으나 `isolation: worktree`가 오래된 브랜치에서 갈라진 고아 워크트리를 만들어 `tools/flight_logs/`가 아예 없는 상태로 실패 — **이 프로젝트는 세션이 in-place로 작업하도록 설정돼 있어 worktree 격리를 쓰면 안 됨**(에이전트는 이를 정확히 감지하고 파일을 지어내지 않은 채 보고했음 → 직접 적용). 향후 서브에이전트 위임 시 isolation 옵션 쓰지 말 것. 반대로 순수 SSH/git 원격 작업(RPi git pull)은 isolation 없이 위임해 문제없이 완결됨
- **비행 로그를 git에 그대로 커밋하기로 번복** — 2026-07-06 "GitHub 업로드 안 함"(대용량 바이너리 이력 팽창 우려) 결정을 사용자가 다기기 공유 편의를 우선해 뒤집음. git 이력이 로그만큼 영구히 커지고 clone이 느려지는 트레이드오프는 알고 승인한 것 — 되돌리려면 히스토리 재작성(rebase/filter-repo) 같은 파괴적 작업이 필요해짐을 유의

### 다음 세션

1. **RPi pymavlink 설치를 임시 우회(`~/.local`, `--break-system-packages`)에서 영구화** — 컨테이너 이미지 또는 셋업 스크립트/문서에 반영
2. 07-17·07-18 14회 비행 notes.md(관찰/결론) 전부 비어있음 — 조종사가 채워야 실제 비행 평가 가능
3. 앞으로 `record_flight.sh`로 생기는 새 플라이트 폴더는 평소 커밋 워크플로에 포함(잊지 말 것 — 더는 `.gitignore` 자동 제외가 아님)

> ✈ vtol-실기체 vs 🚁 mc-실기체 정체 확인은 **해결됨(2026-07-18, 사용자 확인)** — 별도 물리 기체(Pixhawk·ESC 모두 다름, 외형만 동일)로 두 트랙 블록 정리 완료.

### 주의

> `logs/` 방침이 바뀌어 이제 **비행 로그는 커밋 대상**이다 — 새 플라이트 폴더 생성 후 커밋을 잊지 말 것. 저장소 용량이 계속 늘어나는 것은 의도된 트레이드오프.
> RPi `sudo`/`docker` 권한이 이 세션엔 없음(비밀번호 필요) — 컨테이너 안쪽 작업이 필요하면 사용자에게 요청할 것.

