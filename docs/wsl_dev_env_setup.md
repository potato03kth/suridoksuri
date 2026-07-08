---
doc_type: reference
project: suridoksuri-1
scope: 개발컴(Windows) WSL SITL 환경 — 껐다 켰을 때 복구용 명령어 전체 목록
last_updated: 2026-07-09
---

# WSL 개발 환경 복구 가이드

> "WSL 다시 키기 두렵다" — 세팅 날아가서 다시 못 할까봐 걱정될 때 보는 문서.
> 실제로는 WSL은 껐다 켜도 디스크 내용(설치된 패키지, PX4 소스, ROS2 등)은 그대로 남는다.
> 매번 바뀌는 건 **IP뿐**이라 그것만 다시 잡아주면 된다.
>
> 근거: `docs/sitl_verification_log.md` (WSL SITL 환경 구축 로그)

---

## A. 최초 1회만 (디스크에 남아있으므로 이미 설치돼 있으면 스킵)

문서에 원본 그대로 남아있진 않아 공식 표준 설치법으로 재구성함 (★).

```bash
# ★ ROS2 Humble + MAVROS (apt)
sudo apt update && sudo apt install ros-humble-desktop ros-humble-mavros ros-humble-mavros-extras ros-humble-foxglove-bridge

# ★ MAVROS용 GeographicLib 데이터셋
sudo bash /opt/ros/humble/lib/mavros/install_geographiclib_datasets.sh

# ★ PX4-Autopilot 소스 (WSL 로컬 빌드)
cd ~
git clone https://github.com/PX4/PX4-Autopilot.git --recursive
cd PX4-Autopilot
bash ./Tools/setup/ubuntu.sh
```

## B. 방화벽 규칙 (최초 1회, Windows PowerShell 관리자)

```powershell
New-NetFirewallRule -DisplayName "Foxglove Bridge 8765" -Direction Inbound -Protocol TCP -LocalPort 8765 -Action Allow
New-NetFirewallRule -DisplayName "PX4-QGC-UDP-14550" -Direction Inbound -Protocol UDP -LocalPort 14550 -Action Allow -Profile Any
```

---

## C. WSL 재시작할 때마다 매번 (IP가 바뀌므로)

**C-1. WSL 터미널에서 IP 확인**
```bash
WIN_IP=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}')
WSL_IP=$(hostname -I | awk '{print $1}')
echo "Windows: $WIN_IP  /  WSL: $WSL_IP"
```

**C-2. Windows cmd.exe에서 foxglove 포트포워딩 갱신**
```cmd
netsh interface portproxy delete v4tov4 listenport=8765 listenaddress=0.0.0.0
for /f "tokens=1" %i in ('wsl -d Ubuntu-22.04 hostname -I') do netsh interface portproxy add v4tov4 listenport=8765 listenaddress=0.0.0.0 connectport=8765 connectaddress=%i
```

**C-3. QGroundControl Comm Link (Windows GUI, 값만 갱신)**
- `Server Address` = `<WSL_IP>:14551`

---

## D. SITL 매번 띄우는 순서 (WSL 터미널, 각각 새 창)

```bash
# 터미널 1 — PX4 SITL
cd ~/PX4-Autopilot
make px4_sitl gz_x500        # 또는 VTOL: make px4_sitl gz_standard_vtol

# 터미널 2 — MAVROS
ros2 launch mavros px4.launch fcu_url:=udp://:14540@localhost:14557

# 터미널 3 — foxglove (원격 시각화 쓸 때만)
source /opt/ros/humble/setup.bash
ros2 launch foxglove_bridge foxglove_bridge_launch.xml port:=8765

# 터미널 4 — 이 저장소 워크스페이스
cd ~/drone_ws && source install/setup.bash
```

**PX4 콘솔(pxh>)에서, QGC 연결용 (재시작마다):**
```
pxh> mavlink start -x -u 14551 -r 4000000 -t <WIN_IP>
```

---

## E. 이 저장소(`suridoksuri-1`) 빌드/설치

```bash
# fc_bridge — colcon 아님, pip 라이브러리 (최초 1회, 코드 바뀔 때 재실행)
cd fc_bridge && pip install -e .

# fc_ros — 매번 코드 바뀔 때
colcon build --packages-select fc_ros
source install/setup.bash
```

---

## 참고

- 완전 종료가 두렵다면: WSL을 끄지 말고 컴을 절전(sleep)만 시켜도 됨 — Windows 절전은 WSL 프로세스를 죽이지 않는다.
- 근본 환경 스택/검증 이력: `docs/sitl_verification_log.md`
- RPi5 실기체 절차는 별개 문서: `docs/mc_flight_procedure.md`
