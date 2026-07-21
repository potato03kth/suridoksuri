---
doc_type: session_status
project: suridoksuri-1
scope: vision 세션 유일 진입점 — 트랙 보드 + 설계 포인터
last_updated: 2026-07-21e
---

# vision 세션 진입 상태 문서

> **새 세션 진입:** 아래 vision 트랙 블록을 읽고, 설계 상세는 `docs/vision_plan.md`의 **필요 섹션만** 연다.
> FC 트랙은 `docs/session_status.md`에 별도로 있으며 **vision 세션은 그걸 읽지 않는다**(도메인 간 컨텍스트 격리).
> `/session-log`는 vision을 건드린 세션에서 **이 문서**의 트랙 블록을 갱신한다 (서술 로그는 공용 `docs/session_log.md`).

---

## 공통 상태 (2026-07-21)

- **브랜치:** `dev--vision-computing-module` (현재 FC 트랙과 공용. vision 전용 브랜치 분리는 미결정)
- **개발/테스트 환경:** 메인 개발주체 = **노트북(WSL)**, `.venv` 준비 완료(`source .venv/bin/activate`, 저장소 루트) — `pytest vision/tests/` **118 passed** 확인됨(2026-07-21e). 개발컴 `.venv`도 기존대로 유효. RPi는 headless지만 **SSH 상시 접근 가능**(tailscale `100.67.27.83`, 계정 `suri`, ed25519 키 등록됨, **패스워드리스 sudo 설정 완료** 2026-07-21) — Claude가 직접 SSH로 RPi 작업 가능. **장착된 카메라는 정품 Camera Module 3가 아니라 서드파티 클론 "CAM109-IMX708AF-75"**(IMX708 센서 동일, 화각은 대각 75°로 계획서 가정 102°보다 좁음). 상세 → 메모리 `project_vision_dev_env.md`, `project_rpi5_ubuntu_camera_stack.md`
- **커밋 규율:** vision 커밋은 메시지에 **`[vision]`** 태그
- **설계 정본:** `docs/vision_plan.md` — 확정 결정·물리 제약·검출 전략·변경내성/관측성·빌드 순서·블라인드스팟. **이 문서는 라이브 진척만** 담는다.
- **대회 규정(2026-07-21 대부분 확정):** ArUco=`DICT_4X4_50` ID23 **50cm×50cm**(원래 계획 가정과 일치 확인) · 버티포트 하기안전구역(빨간 원)=직경2m·선굵기5cm 고리 신규확정 · ③빨간십자·초록색/치수는 **비공개로 확정**(버티포트 유사크기+안전마진 가정) · 성공판정="매끄럽게 보이면"(정성) 확정 · ④단순착륙=GPS+라이다만 확정 · CC 인터페이스·나무조각상 판단기준만 여전히 대기 → 상세 `vision_plan.md` §10.

---

## 트랙 보드

### 👁 vision-정밀착륙 — ▶ 활성 (카메라 캘리브레이션 브링업은 RPi 작업 허가 대기 중 — §7.9 카메라 독립 대체 트랙 4·5·6·7번 전부 완료, 남은 건 전부 RPi 허가 대기 항목)

- **내용:** 착륙지점 인식·정밀착륙 시스템(RPi5 온보드). 고전 CV, 타겟별 coarse→fine 2단, 비전 폐루프 <30cm. 설계 정본 `docs/vision_plan.md`.
- **직전 완료(2026-07-21e, 노트북/WSL 로컬 세션 — RPi/실카메라 작업 금지 하에 진행):** §7.9 "지금 당장 할 일" 5번 — 라이브 스트림 어댑터(`compute_tap` VGA → MJPEG-over-HTTP). 상세:
  - `vision/utils/stream.py`: `MjpegStreamer` 신설. `push_frame()`은 bounded queue(기본 길이2)+drop-oldest로 **절대 블로킹하지 않음** — `vision/utils/blackbox.py`의 `_DropOldestQueueHandler`와 동일 패턴 재사용(새 패턴 발명 안 함). 다운스케일(종횡비 유지, VGA 640x480 박스 안에 맞춤, 업스케일 없음)·JPEG 인코딩·HTTP 서빙은 전부 별도 스레드(인코더 스레드 1개 + 클라이언트당 핸들러 스레드, 표준 라이브러리 `http.server.ThreadingHTTPServer`만 사용 — 새 무거운 의존성 없음). `/stream`(MJPEG multipart) + `/`(미리보기 `<img>` 페이지) 두 경로.
  - **버그 하나 실제로 발견·수정:** 초기 구현에서 큐가 가득 찼을 때 "가장 오래된 항목 제거 후 삽입"이 락 없이 이뤄져, 여러 producer 스레드가 동시에 `push_frame`을 호출하는 테스트(`test_push_frame_non_blocking_with_real_slow_consumer_connected`)에서 `queue.Full`이 새는 레이스가 실제로 재현됨 → `threading.Lock`으로 evict+insert를 원자화해 수정. (`blackbox.py`의 동일 패턴은 단일 로거 호출 경로라 이 레이스가 실질적으로 안 드러나 그대로 둠 — `vision/CLAUDE.md`에 근거 기록.)
  - `vision/main.py`/`vision/replay.py`: `--display stream`으로 opt-in 연결(`--stream-host`/`--stream-port`, 기본 `0.0.0.0:8080`). main.py의 기존 "stream=미구현" placeholder를 실제 구현으로 대체. 켜지 않으면 스트리머가 아예 안 뜬다(오버헤드 없음, `test_display_none_never_starts_streamer`로 회귀 방지).
  - **실제 검증(pseudo 테스트 아님):** `vision/tests/test_stream.py`(10개) — 진짜 HTTP 서버 기동(포트 0=임시 포트) → 골든셋(`DirFrameSource`로 조달, RPi/실카메라 미사용) 프레임을 실제로 push → 진짜 `http.client`로 `/stream` 접속 → 진짜 MJPEG 바이트를 `cv2.imdecode`로 디코드 성공 확인. VGA 박스 다운스케일(종횡비 유지, 1280x960→정확히 640x480 / 240x320은 업스케일 안 됨 그대로) 검증. 비차단 큐 자체(화이트박스, 200회 push에도 큐 길이 상한 유지) + **실제 느린 컨슈머**(응답을 연결만 하고 body를 절대 안 읽는 진짜 소켓) 붙여놓은 채로 300회 push 시간 실측(<1s) 검증. `vision/tests/test_main.py`/`test_replay.py`에도 실제 CLI(`--display stream`) 경로 통합 테스트 추가(`test_replay.py`는 재생 도중 실제 HTTP GET으로 접속해 실제 프레임 디코드까지 확인, 타이밍 결정론 확보를 위해 `Pipeline.run`에 테스트 전용 소폭 지연 monkeypatch 사용 — 스트리밍 배관 자체는 실동작).
  - **수동 스모크(pytest 밖, 실제 프로세스+curl):** `python -m vision.replay <300프레임 폴더> --display stream --stream-host 127.0.0.1 --stream-port 8099` 백그라운드 실행 → `curl http://127.0.0.1:8099/stream`로 실제 5초간 ~1.85MB MJPEG 수신 → 프레임 디코드 성공(500x500 원본이 640x480 박스에 맞춰 480x480으로 축소됨, 종횡비 유지 확인) → 재생 종료 후 프로세스 정상 종료(스트리머 `.stop()`이 프로세스를 안 붙잡음) 확인.
  - `vision/CLAUDE.md`: 파일역할표에 `utils/stream.py` 행 추가, "라이브 스트림 어댑터 기본값" 절 신설(해상도/포트/바인딩주소/JPEG quality/큐길이 근거 기록 — §7.9가 정확한 수치를 안 못박아 세션 지시에 따라 합리적 기본값으로 확정), 테스트 규칙표에 `utils/stream` 행 추가
  - `pytest vision/tests/` **118 passed**(기존 106 + 신규 10 `test_stream.py` + `test_replay.py` 1건 + `test_main.py` 순증 1건[구 "stream 미구현" 테스트 1개 대체·신규 2개 추가])
  - **ROS2 경로는 이번 세션에서 하지 않음**(세션 제약) — §7.9 (b)행이 "MJPEG-over-HTTP 또는 ROS2 image_transport" 둘을 언급하지만 이번엔 MJPEG-over-HTTP만 구현. `rclpy`/`fc_ros` 관련 import 전혀 없음(도메인 간 의존 없음 원칙 유지). ROS2 경로가 필요해지면 별도 세션에서 논의.
- **직전 완료(2026-07-21d, 노트북/WSL 로컬 세션 — RPi/실카메라 작업 금지 하에 진행):** §7.9 "지금 당장 할 일" 7번 — 골든셋 폴더 스캐폴드 + 재생 회귀 assert. 상세:
  - `vision/tests/golden/`: `<타겟>/<고도>/frame_NNN.png`+`labels.json` 스키마. **전부 합성(synthetic) 데이터** — 실촬영 아님(카메라 브링업 전이라 불가능, README에 명시). `vertiport`(①, 10m/20m/40m, `vertiport_coarse.yaml`) · `distress`(②, 10m/20m/40m, 신규 `presets/distress_coarse.yaml` — 전용 모듈 없이 기존 `ColorFilter`+`RectDetector` 조합, 검출 로직 신규 아님) · `no_target`(④ 단순착륙과 동일 조건 — 피듀셜 없는 평지에서 오탐 안 하는지). ③ 하기구역은 전용 형상판별 검출기가 없어 의도적으로 제외(README에 사유 명시). 구조/스키마/재생성법/실기체 데이터 교체 절차는 `vision/tests/golden/README.md`. 생성 소스는 `vision/tests/golden/generate_synthetic.py`(pytest 대상 아님, 수동 재생성 도구)
  - **실측으로 발견한 것(검출기 미변경, 사실 그대로 골든셋에 고정):** `vertiport` 40m 티어는 흰 필드는 후보로 잡히지만(`white_field.candidates=1`) `black_v` 형상매칭이 탈락(`rejected=1`)해 최종 검출 0건 — 저해상 스케일에서 `vertiport_coarse.yaml`의 고정 `kernel_size=5` morphology가 V 노치 주변 연결성을 깨는 게 원인으로 보임. `distress` 40m 티어는 매트 픽셀 면적이 `rect_detector.min_area`(300)보다 작아져 물리적으로 타당한 미검출. 둘 다 `known_limitation: true`로 라벨링, 검출기 파라미터는 손대지 않음(이번 세션 범위 밖) — 실기체 데이터로 재검증 대상.
  - `vision/tests/test_golden_regression.py`: **실제** `vision.replay.run_replay()`(DirFrameSource+실제 Pipeline, 몽키패치 없음)로 골든 폴더를 재생 → JSONL 검출 개수를 `labels.json`과 비교(15개 테스트: 스캐폴드 비어있지 않음 확인 1 + 리프 디렉터리 7개 × 2종 테스트). 검증으로 `RedRingDetector.min_points`를 일시적으로 999999로 바꿔 관련 4개 테스트가 실제로 실패하는 것 확인 후 원복 — 진짜 파이프라인을 물고 있음을 확인.
  - `vision/CLAUDE.md`: 파일역할표에 `presets/distress_coarse.yaml`·`tests/golden/` 행, 테스트 규칙표에 골든 회귀 행, 공통규칙 4번의 "골든셋 회귀는 데이터 수집 후" 문구를 "합성 데이터로 스캐폴드 시작됨" 으로 갱신
  - `pytest vision/tests/` **106 passed**(기존 91 + 신규 15)
- **🟡 카메라 인트린식 캘리브레이션 브링업 — RPi 작업 허가 대기 중 (이번 세션도 미착수, 의도적 보류):** 직전 세션(2026-07-21a)에서 libcamera가 RPi5용 PiSP IPA 모듈 없이 빌드돼 있어 카메라 브링업이 막힌 채 중단됨. **사용자가 실비행 나가면서 RPi SSH 접속·실카메라 작업을 계속 전면 금지**하고 있음 — 그래서 이번 세션도 그 대신 §7.9 5번(카메라 독립 트랙, 라이브 스트림 어댑터)을 진행했다. RPi 작업 재개 허가가 떨어지면 아래 "다음"의 1번부터 이어간다. 상세 경과·진단명령·4개 선택지는 메모리 `project_rpi5_ubuntu_camera_stack.md`에 그대로 보존됨(재조사 불필요).
- **⚠️ 카메라 독립 대체 트랙 소진됨 — §7.9 "지금 당장 할 일" 1~7번 중 4·5·6·7번(카메라 독립 항목) 전부 완료.** 남은 1·2·3번(1: 카메라 브링업 선택지 확정, 2: 캡처 도구+실촬영 캘리브레이션)은 전부 **RPi 실카메라 하드웨어가 있어야만** 진행 가능 — 이번 세션 제약(RPi SSH/카메라 금지)이 걸려있는 한 이 트랙에서 더 할 수 있는 카메라 독립 작업이 없다. 다음 vision 세션이 RPi 허가 없이 들어온다면, 이 트랙 진행보다 다른 작업(예: 검출 파라미터 튜닝 논의, 문서 정리 등)이나 대기가 맞다 — 새 카메라 독립 서브트랙을 찾으려 §7.9를 다시 훑을 필요 없음(이미 확인됨).
- **다음 (진입하면 이 순서, 전부 RPi 허가 필요):**
  1. **[RPi 작업 허가 필요]** 카메라 브링업 4개 선택지(V4L2 RAW 직접캡처 권장/libcamera 재빌드/RPi OS 재설치/보류) 중 어느 걸로 갈지 사용자에게 확인 → 메모리 `project_rpi5_ubuntu_camera_stack.md` 먼저 읽고 진입
  2. **[RPi 작업 허가 필요]** 선택된 방향으로 카메라 캡처 도구(`vision/tools/rpi_capture.py`) 완성 → 실제 체커보드 촬영 → 카메라 인트린식/왜곡 캘리브레이션
  3. **[RPi 작업 허가 필요]** 골든셋을 실촬영 데이터로 교체 — 절차는 `vision/tests/golden/README.md` "실기체 데이터가 들어오면" 참조. 이때 40m 티어의 `known_limitation` 두 건도 실측 재검증. `MjpegStreamer`도 실제 RPi 네트워크 환경에서 브라우저 접속 실측 필요(`LiveFrameSource`와 동일하게 지금까지는 인터페이스 계약만 검증됨)
- **주의:** Pi4 인코더/라이다 40m급 미확정 · 기존 `vision/` 틀은 폐기 아님(§12) · `geo_project.pixel_to_gps` 폐기 예정 · **버티포트 V 형상매칭은 실물 규격 미확인 상태의 합성테스트로만 검증됨** — 실기체 데이터 확보 후 `BlackVMatcher` 참조 V 템플릿(두께/종횡비)·`max_match_distance` 재검증 필요 · **카메라 화각이 계획서 가정(102°)과 다름(75°) — coarse 캐스케이드 탐지거리 가정 재검토 여지 있음, §4.1 GSD 표 자체가 재검증 대기라 골든셋 고도 라벨도 정밀 GSD 매핑이 아니라 스키마 자리표시자로 명시함(§7.9 항목7 완료 노트 참조)** · `LiveFrameSource`는 인터페이스 계약만 검증됨, 실장치 연결은 RPi 허가 후 검증 필요 · `MjpegStreamer`(§7.9 항목5)도 마찬가지로 로컬(WSL) HTTP 왕복만 검증됨, 실제 RPi↔랩탑 네트워크 환경(대역폭/지연/Wi-Fi 끊김 — `project_rpi5_tailscale_wifi_drops.md` 참조)에서의 실측은 RPi 허가 후 검증 필요 · **`jsonl_view.py`의 state 서브플롯은 실기체 데이터 없음** — main.py/replay.py가 아직 `state`를 채우지 않아(§5.1 상태머신 미구현) 실사용 시엔 항상 "no state data" 안내만 뜬다. 상태머신 연결되면 자동으로 실데이터가 나타남(코드 변경 불필요, 이미 대응돼 있음) · **골든셋(`tests/golden/`)은 전부 합성 데이터** — 실기체 데이터 확보 후 교체 필요(§7.9 항목7 완료 노트) · **`vertiport_coarse.yaml`의 고정 `kernel_size=5` morphology가 저해상(작은 픽셀) 스케일에서 흰 필드 연결성을 깨는 스케일 민감성이 골든셋으로 새로 드러남** — 검출기 튜닝은 이번 세션 범위 밖, 실기체 데이터/저고도 재검증 필요 · 세부 정정 이력·논의는 `docs/session_log.md` 참조
- **참조:** `docs/vision_plan.md` §2(타겟 스펙)/§5.2(버티포트 coarse 캐스케이드)/§5.5(색 항상성)/§7.5(기록·재생)/§7.9(관측성 워크플로) · `vision/CLAUDE.md`(파일역할표·테스트 규칙표) · 메모리 `project_rpi5_ubuntu_camera_stack.md`(카메라 브링업 전체 경과·진단명령·재현법)
