# flight_logs — 비행 로그 자동수집·분석 체계 (작업 G)

**규약: 비행 1회 = 폴더 1개.** 전 테스트 트랙(🚁 mc-실기체 / 🛩 sitl-vtol / ✈ vtol-실기체) 공용.

```
logs/2026-07-06_flight01/          ← git 커밋 대상 (2026-07-18부터, 아래 업로드 방침 참조)
├── rosbag/                        rosbag2 녹화 (토픽: topics.txt)
├── launch.log                     ros2 launch 터미널 출력 전체
├── rosbag_record.log              rosbag2 자체 stdout (진단용)
├── log_12_2026-07-06-05-12-30.ulg PX4 ulog (자동 회수)
└── notes.md                       비행 조건 / 관찰 / 결론 (사람이 채움)
```

**업로드 방침 (2026-07-06 결정 → 2026-07-18 번복):** 여러 기기(RPi·개발컴·노트북)에서
git만으로 로그를 공유·접속하기 위해 **일반 git 커밋**으로 전환(사용자 결정, LFS 아님) —
`.gitignore`에서 `logs/`를 제외, rosbag(`.db3`)·ulog(`.ulg`) 그대로 커밋한다.
**인지된 트레이드오프:** git 이력이 로그만큼 영구히 커지고(되돌려도 이력 크기는 안 줄어듦)
clone/fetch가 계속 느려진다 — 감수하기로 함. 새 플라이트 폴더가 생기면 평소 커밋 워크플로에
포함시킬 것(`[main]`/`[mc-hw]` 등 태그). 공유·웹 분석이 필요한 ulog는 선택적으로
[PX4 Flight Review](https://logs.px4.io)에도 업로드 가능.

---

## 1. 녹화 — `record_flight.sh` (RPi / WSL 겸용)

```bash
# RPi 실기체 (Docker 컨테이너 fc 안에서)
./tools/flight_logs/record_flight.sh vehicle_type:=mc

# WSL SITL (--sitl: 비행 후 SITL 로그 디렉터리에서 ulog 복사)
./tools/flight_logs/record_flight.sh --sitl

# 벤치 기동 (arm 안 하므로 ulog 생략)
./tools/flight_logs/record_flight.sh --no-ulog vehicle_type:=mc

# launch 인자는 그대로 통과된다 (파라미터 규율: yaml 수정 금지, launch 인자로만)
./tools/flight_logs/record_flight.sh --sitl v_cruise:=18.0 waypoints:="[...]"
```

동작: 플라이트 폴더 생성 → rosbag2 백그라운드 녹화 → `phase2.launch.py` 실행.
**Ctrl-C로 launch를 끝내면** rosbag 정지 → ulog 자동 회수 → 폴더 내용 출력.
끝나면 `notes.md` 3줄을 바로 채울 것.

환경변수: `FLIGHT_LOG_ROOT`(로그 루트), `PX4_SITL_LOG_DIR`, `PULL_ULOG_ARGS`(예: `--url /dev/ttyACM1`).

## 2. ulog 단독 회수 — `pull_ulog.py`

실기체에서 launch 없이 ulog만 받을 때. **반드시 MAVROS 종료 후** (시리얼 단독 점유,
비행 중 대용량 전송은 제어 링크 오염).

```bash
python3 tools/flight_logs/pull_ulog.py --list                    # FC 로그 목록
python3 tools/flight_logs/pull_ulog.py --out logs/<플라이트폴더>/  # 최신 로그 다운로드
python3 tools/flight_logs/pull_ulog.py --log-id 7 --out .        # 특정 로그
```

기본 연결: `/dev/ttyACM0` @ 921600 (RPi USB 직결). 필요 패키지: `pymavlink`.
실패 시 폴백: **SD 카드 수동 회수** → `/fs/microsd/log/<날짜>/` 최신 `.ulg`를 플라이트 폴더에 복사.

> 구현 노트: MAVLink **로그 전송 프로토콜**(LOG_REQUEST_LIST/DATA — QGC와 동일 방식) 사용.
> 계획서의 "MAVFTP" 표기 대비 같은 링크·같은 파일이지만 디렉터리 탐색이 불필요해 더 견고하다.

## 3. 개발컴 회수 — `fetch_logs.ps1` (Windows)

```powershell
# 신규 플라이트 폴더만 증분 복사 (RPi logs/ → 로컬 logs/)
.\tools\flight_logs\fetch_logs.ps1 -Remote pi@<RPi-IP>
# 기본 Remote는 $env:RPI_REMOTE 또는 pi@raspberrypi.local
```

`-RemotePath` 기본값 `~/drone_ws/src/suridoksuri/logs`는 **RPi 배포 검증 때 확정** —
컨테이너 `fc`의 경로가 호스트 볼륨 마운트인지에 따라 달라진다 (아래 미결).

## 4. 분석 (개발컴)

**가장 빠른 길: Claude에게 로그 폴더 경로를 주면 됨.** (`logs/2026-07-06_flight01/ 분석해줘`)

수동 분석:

```bash
pip install pyulog                       # 1회
ulog_info  logs/.../log_12_*.ulg         # 메타데이터·메시지 목록·드롭아웃
ulog2csv   logs/.../log_12_*.ulg -o csv/ # 토픽별 CSV 변환
ulog_messages logs/.../log_12_*.ulg      # PX4 내부 로그 메시지 (거부 사유 등)
```

- **Flight Review (웹):** https://logs.px4.io 에 `.ulg` 업로드 — 자세·모드전이·진동·전원 그래프 자동 생성. 공유 필요할 때만.
- **rosbag 대조:** `ros2 bag info logs/.../rosbag` (녹화 확인), `ros2 bag play`(재생) — FC 시점(ulog)과 노드 시점(rosbag `/fc_ros/override`, `/mavros/statustext/recv`)을 교차 검증.
- **launch.log:** 파이썬 노드 stdout — 상태머신 전이·예외는 여기.

## 테스트

```bash
cd tools/flight_logs && pytest test_flight_logs.py   # 순수 함수 (pymavlink 불필요)
bash -n record_flight.sh                              # 문법 검증
```

WSL SITL dry-run 1회(rosbag·launch.log 생성 확인)는 사람과 협업 항목 — `flight_plan.md` 작업 G 합격 기준 참조.

## 미결 (RPi 배포 검증 시 확인 — flight_plan.md 작업 G [배포] 체크리스트)

- [ ] 컨테이너 `fc`의 `/drone_ws/src/suridoksuri`가 호스트 마운트인지 → 아니면 `FLIGHT_LOG_ROOT`를 호스트 마운트 경로로 지정하고 `fetch_logs.ps1 -RemotePath` 맞출 것
- [ ] RPi에 pymavlink 설치 여부 (`pip3 install pymavlink`)
- [ ] MAVLink 다운로드 실측 속도 (USB 직결 기준) — 판정 기준: 대표 ulog(실측 크기, 없으면 15 MB) 다운로드가 5분(재배터리 간격) 초과면 ① 틈새 전송(작업 G-2: gcs_url 브리지+disarmed 게이팅) 등록·구현, 그래도 부족하면 ② SD 수동 회수를 기본으로 뒤집음. 5분 이내면 현행 유지 (VERIFY.md V2 판정 절차와 동일 기준)
- [ ] PX4 `SDLOG_PROFILE` 기본값으로 충분한지 (고빈도 디버깅 필요 시 조정)
