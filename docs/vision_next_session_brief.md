---
doc_type: orchestrator_brief
scope: ffmpeg/H.264 라이브 디버그 스트림(camera_bringup.md Phase 3) + §9 빌드순서 이어가기(상태머신·현장 색 캘리브레이터) — 전부 물리개입 불필요
status: ▶ 시작 대기 (2026-07-25 준비, 아직 미착수)
created: 2026-07-25
last_updated: 2026-07-25
---

# 다음 세션 오케스트레이터 브리프

> **다음 세션 진입:** "너는 오케스트레이터이다"로 시작하고 이 문서 하나만 읽으면 된다.
> `docs/vision_status.md`(트랙 보드)·`docs/vision_plan.md`는 필요 섹션만 열되, 이 작업의 지시는
> 여기에 자기완결적으로 있다.
> 프로토콜은 메모리 `feedback_orchestrator_protocol` 준수 — **각 트랙은 fg가 아니면 bg, 세션
> 자기보고는 직접 재현 검증 필수, 진행상황 확인 없이 한 프롬프트로 몰아던지지 말 것.**
> **서브에이전트는 소네트로 생성한다.**
> **RPi SSH 백그라운드 실행 함정(2026-07-24 실측):** 원격 명령을 셸 `&`로 백그라운드시키면
> bash job-control이 그 job의 SIGINT를 자동으로 무시하게 만든다 — Ctrl+C/그레이스풀 종료를
> 검증할 땐 로컬 Bash 툴 자체의 `run_in_background`로 `ssh`(원격에서는 foreground exec)를 띄우고,
> 별도 SSH 세션에서 실제 PID로 직접 `kill -INT`할 것. 상세는 메모리 `project_rpi5_ubuntu_camera_stack.md`.
> **auto-mode 분류기 함정(2026-07-24/25 실측):** RPi에 SSH로 오래 사는 백그라운드 서버(스트림 등)를
> 띄우는 Bash 호출이 auto-mode 분류기에 막힐 수 있다 — 막히면 사용자에게 그 명령을 `!`로 직접
> 실행해달라고 요청하는 우회가 통한다(이번 세션에 실제로 그렇게 진행함).

---

## 0. 지금까지 뭐가 끝났나 (한 문단, 상세는 링크만)

2026-07-24까지 ArUco 정밀착륙 브랜치(4 Phase 완료, `docs/vision_aruco_branch.md`)와 `LiveFrameSource`
picamera2 재구현+`main.py` 라이브 배선이 끝났다. 2026-07-24 후속 세션에서 `MjpegStreamer` 실네트워크
(tailscale) 검증 완료 + fps 튜닝(`--live-resolution 1536x864`로 **2fps→17fps** 개선, 코드 변경 없음).
`pytest vision/tests/` **330 passed**, 전부 push됨(`origin/dev--vision-computing-module`).

**그 fps 튜닝 세션 중 사용자가 중요한 걸 짚었다: `docs/vision_camera_bringup.md`의 원래 로드맵에
Phase 3(ffmpeg/H.264 영상)가 있었는데 두 번(2026-07-23, 2026-07-24) 문서 레벨에서 조용히
누락돼 임시 땜빵(MJPEG)에 계속 투자하게 됐다.** 이번 브리프는 그 정정 — **ffmpeg Phase 3를
최우선으로 착수**하고, 그 다음으로 **물리개입이 필요 없는 나머지 빌드순서(§9)를 이어간다.**
상세 경위는 `docs/vision_status.md` 2026-07-24 "fps 튜닝 + ffmpeg Phase 3 혼동 재확인" 블록,
재발방지 메모리 `feedback_ffmpeg_phase3_not_deferred`.

**이번 브리프가 명시적으로 제외하는 것(사용자가 "물리개입 필요없는 나머지"로 범위를 한정함):**
- **Phase B(골든셋 실촬영 교체)** — 여전히 사용자 물리적 개입 필수, 정지조건. 이 브리프 범위 밖.
  이전 브리프 내용은 `docs/session_log.md`/git 히스토리에 남아있으니 촬영 가능해지면 그때 다시 꺼낼 것.
- **§9 빌드순서 7번(offboard 정밀착륙 서브상태 연결)** — `fc_ros`/`fc_bridge` 도메인을 건드려야
  하는데, 루트 `CLAUDE.md`가 "vision 세션에서 fc_ros/fc_bridge를 건드리지 않는다"고 명시했다.
  **vision 세션의 스코프가 아니다** — FC 세션에서 별도로 다뤄야 함.
- **체커보드 실측 캘리브레이션 재개** — 여전히 보류 결정 유효(메모리 `project_vision_calibration_deferred`).
  **재제안 금지**(사용자가 먼저 꺼내기 전까지).

---

## 1. 우선순위 1 — ffmpeg/H.264 라이브 디버그 스트림 (`docs/vision_camera_bringup.md` Phase 3)

### 왜
MJPEG(`utils/stream.py`)은 애초에 임시 땜빵이었다 — 인터프레임 압축 없음, 비트레이트/프레임레이트
제어 개념 자체가 없음(2026-07-24 세션에서 실측: 4608×2592 기준 ~2fps, 1536×864로 낮춰야 겨우 17fps).
`docs/vision_camera_bringup.md` §Phase3가 원래 의도한 저지연 H.264 경로를 지금 만든다.

### 🔴 확정 전제 (2026-07-25 사용자 확인 완료, 재논의 불필요)
- **검출 박스(annotated) 없이 카메라 원본만 스트리밍한다.** 이유: `rpicam-vid`/picamera2 인코더는
  카메라에서 직접 H.264를 뽑아 비전 파이프라인(Python `VisionState`)을 거치지 않는다 — annotated
  프레임을 얹으려면 파이프라인 출력을 다시 ffmpeg로 파이프(stdin rawvideo→libx264)해야 해서
  구현이 복잡해지고 Pi5 CPU 실시간 인코딩 부담도 커진다. **검출 결과 확인은 이미 있는 재생 오버레이
  뷰어(§7.9 (a), `python -m vision.replay --display window`)가 데스크 주력으로 맡는다** — 이
  H.264 스트림은 그와 다른 역할(초점/노출/프레이밍 등 카메라 원본 상태를 저지연으로 보는 것)이다.
- 위 결정에 따라 **`MjpegStreamer`(annotated 프레임 스트림)는 폐기하지 않는다** — 서로 다른 용도로
  병행 운용(MJPEG=검출결과 관찰용 저해상 스트림, H.264=카메라 원본 저지연 디버그). `camera_bringup.md`가
  "MJPEG 폐기 후보"라고 썼던 건 이 역할 분리가 이번에 명확해지기 전 가정이었다 — 실제로 폐기할지는
  이번 세션에서 다시 판단해도 되지만, 무리해서 지우지 않는 게 기본값.

### 사전조사로 이미 확인된 것 (2026-07-25, RPi 실기체 — 재확인 불필요)
- **`rpicam-vid`/`libcamera-vid` 없음.** 로컬 libcamera 소스빌드(`/home/suri/local-libcamera-src/libcamera`,
  `docs/vision_camera_bringup.md` Phase 1)는 `cam` 데모 앱만 만든다 — `rpicam-vid`는 **별도 저장소**
  (`raspberrypi/rpicam-apps`, 예전 이름 `libcamera-apps`)라 이 브링업만으로는 안 딸려온다.
  **`rpicam-apps`를 새로 소스빌드할 필요 없다** — 아래 대안이 있음.
- **`ffmpeg` CLI 없음(RPi에도, 이 노트북/WSL에도).** `sudo apt install ffmpeg`로 양쪽 다 설치 필요
  (RPi는 패스워드리스 sudo 이미 설정됨, 메모리 `project_rpi5_ubuntu_camera_stack.md`).
- **picamera2에 `H264Encoder`가 이미 있다** — `picamera2.encoders.H264Encoder`는 실제로
  `LibavH264Encoder`(libav, 즉 ffmpeg 라이브러리 기반 **소프트웨어** 인코딩 — `vision_plan.md`
  §7.7 "Pi5는 HW 인코더가 없다"와 일치, 별도 HW 인코더 불필요)다. `picam-venv`에서
  `from picamera2.encoders import H264Encoder`로 바로 import 확인됨.
- **`picamera2.outputs.FfmpegOutput`도 있다** — 내부적으로 `ffmpeg` subprocess를 띄워 RTSP/UDP
  등으로 muxing해준다(생성자: `FfmpegOutput(output_filename, ...)` — `output_filename`에
  `'-f mpegts udp://<ip>:<port>'` 같은 ffmpeg 출력 스펙을 그대로 줄 수 있음, picamera2 공식 문서
  패턴). 단 이 경로는 RPi에 `ffmpeg` CLI 설치가 필요하다.

### 두 가지 구현 경로 (다음 세션이 고를 것, 둘 다 확인된 사실 기반)
- **경로 A — `picamera2.outputs.FfmpegOutput` 사용(RPi에 `ffmpeg` 설치 필요).** `Picamera2` +
  `H264Encoder()` + `FfmpegOutput('-f mpegts udp://<노트북IP>:<port>')`로 UDP 송출, 노트북에서
  `ffplay udp://0.0.0.0:<port>` 또는 `mpv udp://@:<port>`로 수신. **구현 예제가 picamera2 공식
  문서에 있어 가장 빠르게 갈 수 있는 길** — 우선 추천.
- **경로 B — ffmpeg 없이 raw H.264 소켓 출력.** `H264Encoder`가 뱉는 raw NAL 유닛을 커스텀
  `Output`(picamera2가 지원하는 패턴, 소켓 `.makefile('wb')`에 직접 write)으로 UDP 소켓에 바로
  써서 RPi 쪽에 `ffmpeg` 설치 자체를 안 해도 되게 하는 방식. 노트북 쪽은 여전히
  `ffplay -f h264 udp://0.0.0.0:<port>` 필요(포맷 힌트 `-f h264` 필수 — 컨테이너 없는 raw
  elementary stream이라 자동 감지가 안 될 수 있음, 실측 확인 필요). UDP 프레임 조각화(MTU)로
  고해상도에서 끊길 가능성 — **실측으로 확인해야 하는 리스크**(단정 아님).
- **경로 A로 먼저 시도 권장** — 구현이 단순하고 실패 시 원인 좁히기 쉽다. 막히면 경로 B로 전환.

### 할 일
1. RPi/노트북 양쪽에 `sudo apt install ffmpeg` (경로 A 기준. 경로 B면 노트북만).
2. RPi에서 `Picamera2`+`H264Encoder`+`FfmpegOutput`으로 최소 스크립트 작성(새 파일 —
   `vision/tools/` 아래, `rpi_capture.py`/`calib_capture.py`와 같은 "하드웨어 전용 운영스크립트"
   범주, import 규칙표의 `tools/` 예외 적용). **해상도는 처음엔 낮게(예: 1536x864) 잡고 fps 실측
   후 필요하면 올릴 것** — MJPEG 튜닝에서 해상도가 병목이었던 교훈 재사용.
3. 노트북에서 `ffplay`/`mpv`로 실제 수신 확인(pseudo 아님, 실제 프레임이 화면에 뜨는지 — 이 환경엔
   디스플레이가 없으니 **오케스트레이터 자신은 화면을 못 본다**, 사용자에게 확인을 요청하거나
   `ffprobe`로 스트림 메타데이터/프레임 도착을 정량 확인하는 방식으로 대체).
4. 지연시간 실측(카메라→화면, 대략치라도) — MJPEG 대비 개선 여부를 수치로 기록.
5. `vision/CLAUDE.md`에 새 도구 항목 추가, `docs/vision_camera_bringup.md` Phase 3를 완료로 갱신,
   `docs/vision_status.md` 트랙 보드 갱신.
6. 단위테스트: 순수 로직(예: UDP 출력 스펙 문자열 조립, 해상도 파싱 등)이 있다면 분리해 테스트 —
   단 이 도구 자체는 하드웨어 전용이라 `tools/`의 "CI/pytest 대상 아님" 예외가 기본 적용됨
   (`jsonl_view.py`/`calib_analyze.py`처럼 하드웨어 비의존 부분이 생기면 그 부분만 예외적으로 테스트).

### 검증
- RPi/노트북 양쪽 실제 프로세스 기동 → 실제 네트워크(tailscale) 경유 → 실제 프레임 수신/디코드
  확인(ffprobe 메타데이터 또는 사용자 육안 확인). pytest는 330 유지 확인(순수 로직 분리분 있으면 소폭 증가).
- 정리: 프로세스 종료 후 카메라 release(`fuser`) 확인 — 기존 패턴 그대로.

---

## 2. 우선순위 2 — §9 빌드순서 이어가기 (물리개입 불필요, 소프트웨어만)

`docs/vision_plan.md` §9 빌드순서는 1~4번이 이미 완료됐다(nominal intrinsics·실기체 데이터수집
경로·관측성 골격·ArUco 브랜치). **5번(부분)·6번이 다음 차례이고 둘 다 순수 소프트웨어다:**

### 2a. §9 6번 — 공통 상태머신 + 안전 폴백 (`vision_plan.md` §5.1)
- 현재 **미구현** — `vision/CLAUDE.md`가 이미 "state 필드는 main.py/replay.py가 아직 안 채운다"고
  기록해뒀고(`jsonl_view.py`의 state 서브플롯이 항상 "no state data"), 코드베이스에
  `state_machine`/`StateMachine` 관련 파일이 전혀 없음(확인 완료).
- §5.1(및 §2 성공판정 정성기준 — "마지막 WP 도달→MC 스캔→목표 포착·확인→정중앙 상공 이동→
  부드러운 착륙"이 이미 상태머신 설계와 부합한다고 §10에 기록돼 있음)을 읽고 상태 enum·전이
  조건·`VisionState.meta`/신규 필드 배선을 설계·구현. **타겟 종류 무관 공통 골격**이라는 게
  핵심 요구 — 버티포트/조난자 전용 분기를 얹지 말 것.
- 순수 로직이라 하드웨어/실측 데이터 전혀 불필요 — 골든셋(합성) 프레임으로도 충분히 TDD 가능.

### 2b. §9 5번(부분) — 현장 색 캘리브레이터 (`vision_plan.md` §5.5)
- coarse 색 파이프라인(`ColorFilter` 등)은 이미 있지만, "라이브 ROI 샘플→HSV 임계값 자동설정"
  하는 **현장 캘리브레이터 자체는 미구현**(코드베이스에 관련 파일 없음, 확인 완료).
- §5.5·§7.9 "인터랙티브 튜닝 루프"(재생 화면 위 trackbar 또는 config hot-reload) 설계를 참조해
  최소 버전 구현 — 예: 재생/라이브 프레임에서 ROI 지정 → 그 영역 HSV 통계로 임계값 산출 →
  config(yaml) 갱신 제안. **실측 데이터 없이도(합성/골든셋 프레임으로) 로직 자체는 만들고
  테스트할 수 있다** — 실제 현장 조명 조건 튜닝은 나중(비행장에서), 이번엔 도구를 준비해두는 것.
- 우선순위는 6번(상태머신)보다 낮게 잡아도 됨 — 시간이 부족하면 상태머신을 먼저 끝낼 것.

### 검증 (2a, 2b 공통)
- 새 로직은 반드시 같은 커밋에 `tests/test_<모듈>.py` 추가(공통 규칙, `vision/CLAUDE.md` 참조).
- `pytest vision/tests/` 통과 확인 + 통과 개수 갱신.

---

## 3. 선택적 필러 (시간 남으면, 우선순위 1·2보다 낮음)

- **라이브 스트리밍 경로 초점(AF) 제어 미배선** — 2026-07-24 fps 튜닝 세션에서 발견: `LiveFrameSource`가
  `create_still_configuration()`만 호출하고 AF모드/렌즈위치를 명시적으로 세팅 안 함(카메라 드라이버
  기본 동작에 방치). 연속 AF 모드를 명시적으로 켜는 정도의 작은 배선이면 물리개입 없이도 가능
  (기본 동작이 이미 되는지부터 코드로 확인).
- **테스트 커버리지 TODO** — `vision/CLAUDE.md` 테스트 규칙표에 `registry`/`illumination`/`denoise`/
  `edge`/`morphology`/`background`/`tracker`/`fusion`/`image_loader`/`video_reader`/`visualize`가
  전부 `❌ TODO`로 남아있다. 위 1·2번을 먼저 끝내고도 시간이 남으면 이 갭을 메운다(우선순위 낮음 —
  기존 코드 품질/회귀망 강화일 뿐 새 기능 아님).

---

## 4. 참조

- `docs/vision_camera_bringup.md` — 카메라 브링업 로드맵(Phase 1~4), §Phase3가 이번 우선순위 1의 원본
- `docs/vision_plan.md` §5.1(상태머신)·§5.5(현장 색 캘리브레이터)·§7.7(인코더 어댑터/USB2 제약)·§9(빌드순서)
- `vision/CLAUDE.md` — 파일역할표·테스트 규칙표(TODO 현황)·"라이브 스트림 어댑터 기본값" 절
- 메모리 `feedback_ffmpeg_phase3_not_deferred` — 이번 우선순위 1이 왜 최우선인지의 배경
- 메모리 `project_rpi5_ubuntu_camera_stack.md` — libcamera/picamera2 브링업 전체 경과
- 메모리 `feedback_orchestrator_protocol` — 세션 실행 프로토콜
- `docs/vision_status.md` — 트랙 보드(2026-07-24 최신 블록에 이번 브리프로 이어지는 경위 전부 기록됨)
