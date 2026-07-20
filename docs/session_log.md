---
doc_type: session_log
project: suridoksuri-1
---

# 세션 로그

> 최신 세션이 위에 온다. `/session-log` 커맨드로 세션 종료 전 자동 작성.
> **최근 8개 세션만 유지** — 초과분은 `/session-log`가 `docs/archive/session_log_YYYY-MM.md`로 이동한다.
> 과거 기록: `docs/archive/session_log_2026-06.md` (2026-06-18 ~ 06-30) · `docs/archive/session_log_2026-07.md` (2026-07-03 ~)

---

## 2026-07-20 — [mc-hw] flight01 제어상실 사고 분석 + STREAMING/FOLLOWING MC 속도제어 전환

**브랜치:** `dev--vision-computing-module`
**목적:** 사용자가 직전 실비행에서 "기체가 제어를 잃어 수동 착륙했다"고 보고, 회수한 ulog로 원인 규명 요청

### 완료

- **사고 로그 위치·특정** — `mc-hw-rpi5-wifi-diag` worktree(다른 세션, lock 보유— 읽기만 함)의 `logs/2026-07-20_flight01/`에서 발견. `log_18_2026-07-20-07-19-30.ulg`(38.7초, 본비행)와 `log_16`/`log_17`(각 1초 미만, 무관)을 pyulog로 직접 분석
- **타임라인 재구성:** t=1.9s CommandTOL 이륙(목표 AMSL 52.31=지면48.3+4.0m) → t=9.3~9.9s `climbing_reached()`가 AGL≈3.5~4.0m에서 정상 판정(허용오차 수정 유효 확인) → **t=9.9~11.3s AUTO.TAKEOFF가 계속 상승해 실고도 최대 7.6m 도달(목표 4.0m의 거의 2배 오버슈트)** → t=11.3s nav_state AUTO.TAKEOFF→OFFBOARD 전환 **바로 그 순간 OFFBOARD 첫 세트포인트가 `(N,E,Z)=(0,0,-4.0)`, yaw=90°로 순간점프 발행 — 실제 위치는 `(-4.4,1.2,-7.3)`, yaw≈-80°(수평 4.5m+수직 3.3m+요 170° 불연속)** → t=11.5~13.0s 격렬한 자세급변(roll -16°, pitch -30.8°, yaw rate 최대 186°/s) → t=16.4s 조종사 스틱 입력 감지(수동 회수 시작) → t=37.7s disarm. EKF `quat_reset_counter`는 이 구간 내내 불변 — 센서/EKF 결함 아님, 세트포인트 자체가 원인임을 확인
- **근본원인 확정:** `offboard_node.py` STREAMING(321행)과 `_step_following()`(775행) 둘 다 `L1Guidance.target_point_ned(pos, _FW_LOOKAHEAD=70.0)` + 절대위치 PoseStamped 발행 방식을 MC/FW 구분 없이 공용 — 70m lookahead는 FW가 목표점 근처에서 flower-pattern으로 도는 것을 막기 위한 FW 전용 기법(목표점을 항상 선회반경 밖에 둬 "도착"을 안 일어나게 하는 pursuit 유도)인데, 이번 비행 경로 총길이(~12m)보다 훨씬 커서 항상 경로 끝점(WP1)을 그대로 반환 — 기체의 실제 현재위치와 무관한 고정 절대좌표가 됨. 여기에 클라이밍 중 고도 오버슈트(AUTO.TAKEOFF→OFFBOARD 모드전환 확정까지 수 초 지연되는 동안 계속 상승, `session_status.md` 기존 문서화된 "home_position.alt 드리프트 잔여리스크"가 실제로 재현된 것으로 추정)까지 겹쳐 OFFBOARD 진입 첫 순간의 실제 오차가 구조적으로 클 수밖에 없었음
- **수정 (`fc_ros/fc_ros/nodes/offboard_node.py`):** MC는 PX4 OFFBOARD 속도 세트포인트를 정상 추종한다(FW와 달리 무시 안 함, 코드 기존 주석에도 명시)는 점에 근거해 MC 전용 분기 추가 — ① STREAMING: MC는 절대위치 대신 0속도 스트리밍(제자리 유지, 점프 없음) ② `_step_following()`: MC는 `L1Guidance.ned_velocity_cmd()`(기존에 이미 존재하던, 속도 기반 L1 유도 인터페이스)로 속도 세트포인트 발행, FW는 기존 lookahead 위치 방식 그대로 유지(FW엔 여전히 필요한 로직이므로 미변경). `fc_bridge`(rclpy 비의존)로 사고 시점 실측 좌표를 넣어 수정 전/후 명령값을 직접 비교 검증 — 수정 전엔 즉시 4.54m 절대변위 요구, 수정 후엔 유한 속도(≈2m/s, 경로 v_profile과 일치)만 명령됨을 확인
- **미검증:** `offboard_node.py`는 rclpy 의존이라 이 WSL 샌드박스(pytest·rclpy 모두 미설치)에서 실행 단위테스트 불가 — 문법 검사(`py_compile`)와 `fc_bridge` 순수함수 레벨 수치 검증만 수행. **다음 실비행 전 반드시 SITL(`gz_x500` MC) 회귀검증 필요**

### 결정

- STREAMING/FOLLOWING의 FW lookahead 로직은 FW 전용으로 명확히 분리하고 MC는 속도 세트포인트 경로로 전환 — "MC에서 lookahead 값만 줄이는" 식의 임시조치는 채택하지 않음(여전히 절대위치 점프 방식이라 근본해결 아님)
- HOLD 상태(MC가 FOLLOWING 완료 후 거치는 마지막 착륙 대기)는 이번 수정 범위에서 제외 — 이미 WP1 끝점을 직접 위치 목표로 쓰는 MC 인지 코드였고(주석에 명시), FOLLOWING 종료조건(`d_end_thresh=10m`) 때문에 진입 시점 오차가 이번 사고 규모(4.5m+) 만큼 커질 구조가 아니라 위험도가 다름

### 다음 세션

1. **최우선 — 다음 MC 실비행 전 SITL(`gz_x500`) 회귀검증 필수.** STREAMING 진입~OFFBOARD 확정~FOLLOWING~HOLD 전 구간에서 세트포인트 불연속(점프) 없이 부드럽게 추종하는지, 특히 클라이밍 중 의도적으로 드리프트/오버슈트를 재현해 확인
2. **고도 오버슈트 자체의 근본원인 규명(미해결, 이번 수정과 별개)** — AUTO.TAKEOFF→OFFBOARD 모드전환 확정 지연 동안 계속 상승하는 구조 자체는 안 고쳐짐(이번 수정은 그 위에서 벌어지는 세트포인트 불연속만 제거). `_step_climbing()`에 AUTO.TAKEOFF 이탈 자체를 감지·대응하는 로직 추가 여부(기존 flight09 기록에도 남아있던 미결정 사항)와 함께 재검토 필요
3. `logs/2026-07-20_flight01/`는 아직 다른 worktree(`mc-hw-rpi5-wifi-diag`)에만 있고 git 미커밋 — 다음에 로그 커밋할 때 포함할 것

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

---

## 2026-07-15 — [vision] 계획 갭 반영·headless main.py·테스트환경

**브랜치:** `dev--vision-computing-module`
**목적:** vision 트랙 재개 — 목적·구현범위 점검 → 계획서 갭 반영 → 개발단계 디버깅 착수(headless main.py) → 실테스트 환경 구축

### 완료

- **vision 트랙 이해·구현범위 점검 → 계획서(`vision_plan.md`) 갭 8건 반영** (커밋 `af32ccf`): ④단순착륙 전략공백/내부불일치(§2 표+§5.6 신설), TERMINAL 데드레코닝·blob 타겟 스케일 융합규칙(§5.1), 빨강 ①원↔③십자 혼동 방어(§5.4), `TargetEstimate` 좌표 프레임 계약 미확정(§7.1+§10), CC 명령 수신 시임 `CommandSource`(§7.2), **개발단계 디버깅 워크플로 §7.9 신설**, 성능/지연 예산 등 §10/§11
- **main.py headless-safe** — `--display {none|window|file|stream}`, 모든 GUI(imshow/waitKey)를 window 뒤로 격리, 기본 `none`=GUI 미호출(드론 헤드리스 크래시 원천 제거). `tests/test_main.py` 회귀 4종(none=imshow 0회 불변식)
- **테스트 규칙 정비** — `vision/CLAUDE.md`에 테스트 방법 + 단위별 필수 테스트 표(15단위, ✅4/TODO 다수/폐기 1) + 공통 규칙 4. `vision/requirements.txt`(ASCII) 신설, `.gitignore`에 `.venv/` 등
- **개발컴 실테스트 환경 구축·통과** — `.venv`(Python 3.10.11, opencv-python 5.0.0.93, numpy 2.2.6, PyYAML 6.0.3, pytest 9.1.1). `pytest vision/tests/` → **16 passed**

### 결정

- **실테스트 환경 4구분 기록·검증**(사용자 지시) — 개발컴(항상 필수, 이번에 설치완료)·개발노트북(실비행 휴대)·개발노트북의 wsl·rpi(headless=`opencv-python-headless`). 단계별 추가, **최종 단계엔 4환경 전부 검증**. 매트릭스는 메모리 `project_vision_dev_env.md`에
- `vision/requirements.txt`는 **ASCII 유지** — 개발컴 pip(cp949)가 한글 주석에 `UnicodeDecodeError`. 한글 안내는 `vision/CLAUDE.md`에
- `geo_project.pixel_to_gps`는 폐기 예정 → 신규 테스트 금지

### 다음 세션

1. **미커버 단위 테스트 채우기** — `color` HSV 초록/빨강 모드 우선(정밀착륙 직결) + edge/morphology/fusion 등. 대상·규칙은 `vision/CLAUDE.md` 단위테스트 표
2. **또는 관측성 골격 §7.9 다음 항목** — 이중싱크 로거 + provenance 헤더(config+git해시+캘리브id)
3. (선행 대기) 카메라 인트린식/왜곡 캘리브레이션 + 실기체 3타겟 데이터 — 골든셋·색 캘리브 착수 조건

### 주의

> 개발컴만 `.venv` 준비됨 — **개발노트북·그 wsl·rpi는 미설치**(필요 단계에서 `vision/requirements.txt`, rpi는 headless 변형).
> 대회 상세규정 여전히 대기(`vision_plan.md` §10: ArUco ID·③빨간십자·초록 스펙·성공판정·CC 인터페이스).

---

## 2026-07-11 — [main][mc-hw] 이륙실패 ulog 진단 + AMSL 이륙고도 수정

**브랜치:** `dev--vision-computing-module`
**목적:** MC 테스트기체 마지막 이륙 실패(사용자 제공 ulog) 원인 분석 → 수정

### 완료

- **마지막 MC 이륙 실패 근본원인 확정** — 2026-07-07 광주 실비행 ulog(`02_17_49`, `logs/2026-07-07_0217_last/`에 저장·notes.md 분석)를 pyulog 직접 파싱. ARM·CommandTOL(NAV_TAKEOFF param7=4.0, lat/lon=NaN) 모두 ACCEPTED됐으나 navigator `Already higher than takeoff altitude` → 모터 미가동(출력 0.002) → 10초 후 `Disarmed by auto preflight disarming`. 배터리(12.15V, 새그 없음)·GPS(3D 22위성)·SD·OFFBOARD 전부 정상이라 무관 — 이전 실패(07-03 전압새그, 07-07 SD)와 다른 새 원인
- **원인 = AMSL/relative 프레임 버그(07-06 열린 질문의 실측 종결)** — `CommandTOL.altitude`(→ NAV_TAKEOFF param7)는 AMSL 절대고도인데 `transition_alt`(4.0, 지면 상대)를 그대로 실어 지면 AMSL(19.2m)보다 낮은 목표가 됨. SITL은 `transition_alt:=50`>지면(≈0)이라 그동안 가려졌음
- **수정 커밋 `9451861`** — ① `takeoff_request_fields(transition_alt, home_amsl)` → `altitude=home_amsl+transition_alt`, `/mavros/home_position/home` 구독, home 미수신 시 이륙 보류 ② CLIMBING 게이트 `climbing_reached(…, ground_ref_up)` 지면기준 AGL 보정(로컬 원점≠지면 2.11m — 안 고치면 이륙해도 CLIMBING 무한대기) ③ pytest fc_ros 60/fc_bridge 44 pass(신규 7)
- **SITL 재검증 체크리스트 작성** — `sitl_verification_log.md` "작업 H-2". 재현엔 `PX4_HOME_ALT`로 지면 AMSL>transition_alt 세팅 필수 + geoid(geo.altitude가 AMSL인지) 확인

### 결정

- 이륙 목표고도는 항상 **AMSL 절대고도(home_amsl + transition_alt)** 로 전송 — transition_alt 직접 전달 금지
- CLIMBING 판정은 이륙 순간 캡처한 지면 높이 기준 AGL로

### 다음 세션

1. **SITL 재검증(작업 H-2)** — `sitl_verification_log.md` 체크리스트대로. `PX4_HOME_ALT=100` 등으로 버그 재현 조건을 만든 뒤 수정 확인, **geoid 정합 반드시 확인**
2. PASS 시 "transition_alt를 MIS_TAKEOFF_ALT 이하로" 임시조치 완전 제거
3. 실기체 검증은 ✈ vtol-실기체 결함 해소 후

### 주의

> 수정은 **단위테스트만 통과, SITL 재검증 전**이다 — 실비행 반영 금지. geoid 미확인 리스크(MAVROS `geo.altitude`가 ellipsoid면 과상승) 있음, SITL 로그로 판별
> ulog·분석 notes는 `logs/`(git 제외)에 있음 — GitHub엔 없다

---

## 2026-07-09 — [vision] 정밀착륙 계획 확정 + 트랙 분리

**브랜치:** `dev--vision-computing-module`
**목적:** 비전 객체인식 본격 개발 전 고려사항 컨설팅(블라인드스팟·unknown-unknowns 중심) → 계획 확정·문서화

### 완료

- **착수 전 컨설팅 완료** — 타겟 3종 확정(①버티포트 원형 3m+중앙 50cm ArUco ②초록 매트+흰 박스, 박스 옆 착륙 ③빨간 십자, 규정 대기), 물리 제약 정량화(GSD/고도·tilt 오차·GPS 한계), 검출 전략(고전 CV, 타겟별 coarse→fine 2단)
- **하드웨어 갈림길 해소** — 카메라 mono OV9285→컬러 필요 판명→**RPi Cam Module 3 Wide 표준(IR-cut)** 확정(롤링셔터 수용+완화·초점 무한대·수동노출), 짐벌 없음(나디르+고무댐핑)→**자세 de-rotation 필수**, 라이다 1D 40m급
- **변경내성/관측성 설계** — ports&adapters, 레이어드 config+현장 색 캘리브레이터, 구조적 로깅(터미널+파일 JSONL, 비차단), 기록/재생, 세 화면(전송/연산/연출)+격리 규칙, FPV 인코더 어댑터(Pi5 무-HW인코더→USB2 raw 대역폭 벽)
- **문서화** — `docs/vision_plan.md` 신규(계획 정본), 루트 `CLAUDE.md` 의존관계 `vision→fc_ros`(상대 pose)로 교체(`pixel_to_gps` 폐기), `vision/CLAUDE.md` 재설계 배너
- **vision 트랙 분리** — `docs/vision_status.md` 신규(vision 전용 진입점, FC와 컨텍스트 격리), `/session-log` 도메인 라우팅 + `[vision]` 태그 추가, 메모리 `project_vision_plan.md` 신규

### 결정

- **검출은 고전 CV(ML 없음)** — 타겟이 전부 피듀셜/고대비 색·형상, 결정론=신뢰도(대회 오인식 절대불가)
- **측위: GPS 접근 + 30cm는 비전 폐루프** — 일반 GPS 절대좌표 한계, `geo_project.pixel_to_gps` 폐기
- **통합: 독립 ROS2 노드 + offboard 정밀착륙 서브상태**, 출력=상대 pose(LANDING_TARGET 피벗 호환)
- **vision은 FC와 별도 트랙·진입점** — 콜드스타트 컨텍스트 격리 우선(상호 안 읽음), 서술 로그만 공용

### 다음 세션

1. **카메라 인트린식+왜곡 캘리브레이션** (102° 광각, 없으면 pose 거짓)
2. **관측성 골격 먼저** — `vision/main.py` headless-safe 수정 + 구조적 로깅/JSONL 스캐폴딩
3. 실기체 데이터 수집(고도별 3타겟) 착수

### 주의

> **대회 상세규정 미공개** — ArUco 딕셔너리/ID·③빨간십자 규정·초록 색·치수 스펙 대기(`vision_plan.md` §10). 하드웨어(카메라/Pi4 인코더/라이다)도 변경 가능 → 전부 어댑터로 흡수 설계.
> **이번 세션 변경 미커밋**(문서만, 코드 착수 전). `.claude/commands/session-log.md` 라우팅 편집은 git 무시라 로컬만 반영.

---

