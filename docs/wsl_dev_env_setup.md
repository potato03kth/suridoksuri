---
doc_type: reference
project: suridoksuri-1
scope: WSL SITL 환경(개발컴 + 노트북 E드라이브 별도 배포판) — 껐다 켰을 때 복구용 명령어 전체 목록
last_updated: 2026-07-24
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
# fc_ros — 매번 코드 바뀔 때
colcon build --packages-select fc_ros
source install/setup.bash
```

**⚠️ fc_bridge는 `cd fc_bridge && pip install -e .`로 설치하지 말 것(2026-07-24 실측, 이 섹션 원래 문구가 틀렸음)** —
`fc_bridge/setup.py`가 `fc_bridge/` 디렉터리 안에 있어 `find_packages()`가 그 디렉터리를 기준으로
스캔, `fc_bridge.execution.state_logic` 형태(`offboard_node.py`가 실제로 쓰는 import)가 아니라
`execution`/`guidance`/`comm`/`utils` 등을 **네임스페이스 없이** 최상위 패키지로 설치해버려서
`import fc_bridge`가 실패한다. 게다가 저장소 루트를 `export PYTHONPATH=<repo_root>:$PYTHONPATH`로
라이브 주입하면(RPi에서 쓰는 방식) 원인 불명으로 `ros2`/`ros2 launch` 자체가
`importlib.metadata.PackageNotFoundError: ros2cli`로 깨지는 현상이 재현됨(단독 `python3 -c`로는
재현 안 됨 — `ros2` CLI 고유의 entry-point 스캔 경로에서만 발생). **대신 `.pth` 파일로 주입:**

```bash
echo '<repo_root 절대경로, 예: /root/drone_ws/src/suridoksuri>' \
    > "$(python3 -c 'import site; print(site.getusersitepackages())')/suridoksuri_repo.pth"
python3 -c 'from fc_bridge.execution.state_logic import home_amsl_confirmed'  # 확인
```

이렇게 하면 `import fc_bridge.xxx`가 정상 동작하고 `ros2 --help`/`ros2 launch`도 깨지지 않는다.

---

## F. 이 노트북(Ubuntu-22.04, E드라이브) — 개발컴과 별개의 두 번째 SITL 환경 (2026-07-24 신설)

**"개발컴"(위 A~E)과 물리적으로 다른 머신.** 이 노트북 WSL 기본 배포판은 Ubuntu 24.04(Noble,
ROS2 Jazyy 대상)라 `ros-humble-desktop`을 못 까므로, **별도 WSL 배포판을 E드라이브에 임포트**해
Humble 환경을 분리했다. 최초 1회만 필요.

```powershell
# Windows PowerShell — Canonical WSL rootfs(jammy)를 받아 E드라이브에 import
# ("wsl --install -d Ubuntu-22.04"는 --list --online에 안 나와서 실패함 → --import 사용)
curl.exe -o E:\wsl\ubuntu-jammy-rootfs.tar.gz `
    https://cloud-images.ubuntu.com/wsl/jammy/current/ubuntu-jammy-wsl-amd64-ubuntu22.04lts.rootfs.tar.gz
wsl --import Ubuntu-22.04 E:\wsl\Ubuntu-22.04 E:\wsl\ubuntu-jammy-rootfs.tar.gz --version 2
```

```bash
# 진입 (기본 사용자 = root, 별도 계정 생성 안 함)
wsl -d Ubuntu-22.04 --cd ~

# ROS2 apt 저장소 등록 + Humble + MAVROS (섹션 A와 동일, root라 sudo 불필요)
apt update && apt install -y software-properties-common curl gnupg lsb-release
add-apt-repository universe -y
mkdir -p /usr/share/keyrings
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" > /etc/apt/sources.list.d/ros2.list
apt update && apt install -y ros-humble-desktop ros-humble-mavros ros-humble-mavros-extras python3-colcon-common-extensions git
bash /opt/ros/humble/lib/mavros/install_geographiclib_datasets.sh

# PX4-Autopilot (E드라이브가 아니라 배포판 자체 ext4 안, 훨씬 빠름)
cd /root && git clone https://github.com/PX4/PX4-Autopilot.git --recursive
cd PX4-Autopilot && bash ./Tools/setup/ubuntu.sh --no-nuttx
make px4_sitl gz_x500   # 최초 1회 빌드 — 완료 후 Ctrl-C로 종료해도 빌드 산출물은 남음
```

**MC 검증은 `gz_x500`, VTOL은 `gz_standard_vtol`.** MAVROS 접속 포트는 팀 문서(섹션 D)의
`14540@localhost:14557`가 아니라 **`14540@localhost:14580`**이었다(PX4 v1.18.0-beta1 기준,
`ROMFS/px4fmu_common/init.d-posix/px4-rc.mavlink`: onboard 인스턴스 local=14580, remote=14540) —
버전에 따라 다를 수 있으니 안 붙으면 그 파일에서 실측할 것.

```bash
ros2 launch mavros px4.launch fcu_url:=udp://:14540@localhost:14580
```

**⚠️ 벤치 SITL(수동 arm 테스트)에서만 필요한 프리플라이트 우회 2종 — 실기체 파라미터에는 적용하지 말 것:**
SITL은 실제 전원/텔레메트리 하드웨어를 시뮬레이션하지 않아 두 체크가 항상 막힌다.

```bash
ros2 param set /mavros/param CBRK_SUPPLY_CHK 894281   # "Preflight Fail: system power unavailable" 우회
ros2 param set /mavros/param NAV_DLL_ACT 0             # "Preflight Fail: No connection to the GCS" 우회
```

**fc_bridge 설치는 위 "⚠️" 박스(섹션 E)의 `.pth` 방식을 그대로 따를 것** — 이 환경에서 원인 규명·재현됨.

**PX4 콘솔 출력을 파일로 리다이렉트하지 말 것(또는 배포판 로컬 디스크로만) — `pxh>` 프롬프트가
비-TTY 출력에서 고속 재출력 루프에 빠져 로그가 수 분 만에 수백MB~수GB로 불어난다** (2026-07-24
실측: E드라이브 경유로 20초 만에 195MB, 방치 시 2.1GB까지 확인). 상태 확인은 `ros2 topic`/`ros2
service`/`ss -uln`으로 하고, 콘솔 로그가 꼭 필요하면 배포판 로컬 디스크(`/root/...`, E드라이브
아님)에 남기고 주기적으로 정리할 것.

**정리(디스크 회수 필요시만, 기본은 유지 권장 — 재설치 시간 큼):**
```powershell
wsl --unregister Ubuntu-22.04   # E드라이브의 설치 디렉터리(E:\wsl\Ubuntu-22.04)도 수동 삭제 필요
```

---

## 참고

- 완전 종료가 두렵다면: WSL을 끄지 말고 컴을 절전(sleep)만 시켜도 됨 — Windows 절전은 WSL 프로세스를 죽이지 않는다.
- 근본 환경 스택/검증 이력: `docs/sitl_verification_log.md`
- RPi5 실기체 절차는 별개 문서: `docs/mc_flight_procedure.md`
