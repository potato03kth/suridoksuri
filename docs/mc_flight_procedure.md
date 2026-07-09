---
doc_type: procedure
project: suridoksuri-1
scope: RPi5 + Pixhawk 6C MC 실기체 비행 절차 — "절차는?" 질문에 그대로 출력하는 참조 문서
last_updated: 2026-07-07
---

# MC 실기체 비행 절차 (RPi5 + Pixhawk 6C)

> 사용자가 "절차는?"이라 물으면 **로깅 사용(A) + 미사용(B) 절차를 전부** 출력한다. 하나만 고르지 않는다.

## 0. 비행 전 필수 확인 (매번, 예외 없음)

```
[ ] SD카드가 Pixhawk에 꽂혀 있는지 — 컴퓨터에 옮겨꽂아놓고 까먹으면 PX4 prearm check
    "Logging is enabled, but no SD card is detected"로 arming 자체가 거부된다 (2026-07-06 실제로 이걸로 비행 실패)
[ ] GPS 락 (AUTO.TAKEOFF는 GPS 필수 — 실내/벤치 불가)
[ ] RC Layer1 override 스위치 동작 확인 (RTL/Position)
[ ] transition_alt / waypoints 값 확정 (launch 인자로만 — yaml 수정 금지)
[ ] 배터리 완충 (2026-07-03 3개 비행 모두 6~8초 시점 전압 새그 페일세이프 재현 이력 — 안 풀렸으면 재확인)
```

## 1. [RPi5 호스트, 도커 밖] 코드 갱신

```bash
cd ~/drone_ws/src/suridoksuri   # 정본. 절대 ~/drone_ws/suridoksuri/suridoksuri 아님(다른 계정 repo)
git pull
```

## 2. [컨테이너 `fc`, 터미널 A] 빌드 + MAVROS 기동 (계속 띄워둠)

```bash
sudo docker exec -it fc bash     # 꺼져있으면: sudo docker start -ai fc
cd /drone_ws
colcon build --packages-select fc_ros
source install/setup.bash
source /opt/ros/humble/setup.bash
ros2 launch mavros px4.launch fcu_url:=/dev/ttyACM0:57600
```
`/mavros/state`에서 heartbeat 확인 후 다음 단계. **이 터미널은 비행 끝날 때까지 유지.**

---

## 절차 A — 로깅 프로그램(`record_flight.sh`) 사용 (권장)

### 3-A. [컨테이너 `fc`, 터미널 B] 실행

```bash
cd /drone_ws/src/suridoksuri
export PYTHONPATH=/drone_ws/src/suridoksuri:$PYTHONPATH
./tools/flight_logs/record_flight.sh vehicle_type:=mc \
  transition_alt:=4.0 \
  waypoints:="[0.0,0.0,4.0, 8.0,0.0,4.0]"
```
(값은 그날 비행 계획에 맞게 조정. `phase2.launch.py`를 직접 부르지 않고 이 스크립트로 대체.)

### 4-A. 비행 종료

1. **터미널 B** Ctrl-C → rosbag 정지 → "수집 단계 시작" 출력.
2. **곧바로 터미널 A(MAVROS)도 Ctrl-C** — MAVROS가 시리얼 포트를 물고 있으면 `pull_ulog.py` 자동회수가 실패한다. 타이밍 놓쳐 실패했으면 MAVROS 내린 뒤 수동 재실행:
   ```bash
   python3 tools/flight_logs/pull_ulog.py --out logs/<날짜>_flightNN/
   ```
3. `logs/<날짜>_flightNN/`에 `rosbag/`·`launch.log`·`notes.md`·`*.ulg` 있는지 확인.
4. `notes.md` 3줄(비행조건/관찰/결론) 채우기.

### 5-A. [개발컴, PowerShell] 회수

```powershell
.\tools\flight_logs\fetch_logs.ps1 -Remote suri@<RPi-IP>
```

> **주의(미검증):** 이 로깅 도구는 2026-07-06에 코드만 완성되고 아직 실기체로 성공적으로 써본 적이 없다(`tools/flight_logs/VERIFY.md` 미결 항목: 호스트마운트 경로, RPi pymavlink 설치 여부, 실측 다운로드 속도). 처음엔 `--no-ulog` 옵션으로 launch/rosbag만 먼저 검증해보는 것도 방법.

---

## 절차 B — 로깅 프로그램 미사용 (기존 방식)

### 3-B. [컨테이너 `fc`, 터미널 B] 실행

```bash
cd /drone_ws/src/suridoksuri
export PYTHONPATH=/drone_ws/src/suridoksuri:$PYTHONPATH
ros2 launch fc_ros phase2.launch.py vehicle_type:=mc \
  transition_alt:=4.0 \
  waypoints:="[0.0,0.0,4.0, 8.0,0.0,4.0]"
```

### 4-B. 비행 종료

1. **터미널 B** Ctrl-C.
2. ulog가 필요하면 **터미널 A(MAVROS)를 내린 뒤** 수동으로:
   ```bash
   python3 tools/flight_logs/pull_ulog.py --out <원하는 폴더>/
   ```
   또는 SD카드를 직접 빼서 `/fs/microsd/log/<날짜>/`의 최신 `.ulg`를 수동 회수.
3. rosbag/notes.md 없음 — 폴더 규약(비행 1회=폴더 1개)이 자동 적용 안 됨.

---

## 공통 — 모니터링 (터미널 C, 언제든 새로 열기)

```bash
sudo docker exec -it fc bash
source /drone_ws/install/setup.bash
ros2 topic echo /mavros/state              # OFFBOARD 진입 여부
ros2 topic echo /mavros/statustext/recv    # PX4 거부 사유
```

## 공통 — 긴급 수동 전환

- **Layer 1 (최우선):** RC 모드스위치 → RTL/Position. ROS2/RPi 죽어도 동작.
- **Layer 2:** `ros2 topic pub --once /fc_ros/override std_msgs/msg/Bool "{data: true}"` — 미리 타이핑해두고 대기 권장.

## 참조

- 세션 진입점: `docs/session_status.md` 🚁 mc-실기체 트랙
- 근본 설계·최근 수정 이력: `docs/flight_plan.md` (작업 H — AUTO.TAKEOFF 목표고도)
- 로깅 도구 상세: `tools/flight_logs/README.md` · `tools/flight_logs/VERIFY.md`
