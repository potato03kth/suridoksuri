---
doc_type: orchestrator_brief
scope: ArUco 브랜치·LiveFrameSource 배선 완료 이후 다음 두 트랙 — MjpegStreamer 실네트워크 검증(자율 가능) + 골든셋 실촬영 교체(사용자 물리적 촬영 필요)
status: Phase A 완료(2026-07-24 후속 세션), Phase B만 남음 — 진입 전 사용자 확인 필수
created: 2026-07-24
last_updated: 2026-07-24b
---

# 다음 세션 오케스트레이터 브리프

> **다음 세션 진입:** "너는 오케스트레이터이다"로 시작하고 이 문서 하나만 읽으면 된다.
> `docs/vision_status.md`(트랙 보드)·`docs/vision_plan.md`는 필요 섹션만 열되, 이 작업의 지시는
> 여기에 자기완결적으로 있다.
> 프로토콜은 메모리 `feedback_orchestrator_protocol` 준수 — **각 Phase는 fg가 아니면 bg, 세션
> 자기보고는 직접 재현 검증 필수, 진행상황 확인 없이 한 프롬프트로 몰아던지지 말 것.**
> **서브에이전트는 소네트로 생성한다.**
> **RPi SSH 백그라운드 실행 함정(2026-07-24 실측):** 원격 명령을 셸 `&`로 백그라운드시키면
> bash job-control이 그 job의 SIGINT를 자동으로 무시하게 만든다(nohup 유무 무관, `trap`으로
> 직접 재현 확인됨) — Ctrl+C/그레이스풀 종료를 검증할 때 이 함정에 걸리면 "코드가 안 죽는다"고
> 오판할 수 있다. 검증할 땐 로컬 Bash 툴 자체의 `run_in_background`로 `ssh`(원격에서는 foreground
> exec)를 띄우고, 별도 SSH 세션에서 실제 PID로 직접 `kill -INT`할 것. 상세는 메모리
> `project_rpi5_ubuntu_camera_stack.md` "picam-venv opencv 업그레이드..." 절.

---

## 0. 지금까지 뭐가 끝났나 (한 문단, 상세는 링크만)

2026-07-24 하룻밤 세션에서 ArUco 정밀착륙 브랜치(nominal intrinsics→ArUco 디코드→solvePnP→
TargetEstimate→`main.py` 파이프라인 배선, `docs/vision_aruco_branch.md` 참조, 4 Phase 전부 완료)와
`LiveFrameSource`의 picamera2 재구현+`main.py` 라이브 모드 배선(`live`/`live:<camera_num>` 입력
스펙)이 전부 끝났다. **둘 다 RPi 실기체로 오케스트레이터가 직접 종단간 검증**(카메라 오픈→프레임
캡처→파이프라인→JSONL 기록→Ctrl+C 정상종료)까지 마쳤고, 그 과정에서 picam-venv의 구버전 cv2
(4.6.0, `ArucoDetector` API 없음)를 4.13.0.92로 업그레이드해 실기체 크래시를 고쳤다. `pytest
vision/tests/` **330 passed**. 전부 push됨(`origin/dev--vision-computing-module`). 상세 경과는
`docs/vision_status.md` 2026-07-24 트랙 블록 + 메모리 `project_rpi5_ubuntu_camera_stack.md`.

**이번 브리프의 범위는 그 다음 두 트랙이다 — 성격이 완전히 다르니 구분해서 읽을 것:**
- **Phase A(자율 가능):** `MjpegStreamer` 실제 RPi↔랩탑 네트워크 검증. 물리적 타겟/고도 불필요,
  RPi가 켜져 있고 카메라가 연결돼 있기만 하면 오케스트레이터 혼자 끝낼 수 있다.
- **Phase B(사용자 물리적 개입 필수):** 골든셋(`vision/tests/golden/`)을 실촬영 데이터로 교체.
  버티포트/조난자 타겟을 실제 고도(10m/20m/40m 상당)에서 촬영해야 하는데, 이건 드론 비행이나
  최소한 사람이 카메라를 들고 여러 거리에서 촬영해야 하는 일이라 **오케스트레이터가 원격으로
  대신할 수 없다.** 아래 "Phase B" 절의 지시대로, 진입 전에 반드시 사용자에게 일정을 먼저 확인할 것.

---

## 1. Phase A — MjpegStreamer 실네트워크 검증 (✅ 완료, 2026-07-24 후속 세션)

**결과: RPi→랩탑 tailscale 경로로 실제 12/12 프레임 정상 디코드 확인, 그레이스풀 종료·카메라
release 확인.** 상세는 `docs/vision_status.md` 2026-07-24 "Phase A 완료" 블록 참조. 아래는 이번에
실행한 내용(참고용, 재실행 불필요).

### 왜
`vision/utils/stream.py`의 `MjpegStreamer`는 로컬(WSL) HTTP 왕복만 검증됐다(`test_stream.py`,
같은 머신 loopback). **실제 RPi→랩탑(또는 오케스트레이터 환경)의 tailscale 네트워크 경로**로
브라우저/`curl`이 `/stream`에 접속해 진짜 MJPEG 프레임을 받는지는 아직 미검증 —
`docs/vision_status.md` "주의" 절에 명시된 마지막 남은 미검증 항목 중 하나다.

### 확정 전제 (재논의 불필요)
- 포트/바인딩 기본값(`0.0.0.0:8080`)·다운스케일(VGA 박스, 종횡비 유지)·JPEG quality 80 등은
  전부 `vision/CLAUDE.md` "라이브 스트림 어댑터 기본값" 절에 이미 확정돼 있다 — 바꾸지 마라.
- 타겟/고도 무관 — 카메라 앞에 아무것도 없어도 된다(그냥 지금 보이는 아무 장면).

### 할 일
1. RPi에서 `python -m vision.main live --preset vision/presets/single_frame.yaml --display stream
   --stream-host 0.0.0.0 --stream-port 8080`(또는 임의 preset)을 **`nohup setsid` 등으로 제대로
   분리해서** 백그라운드 실행(위 "RPi SSH 백그라운드 실행 함정" 절 — 이번엔 종료 검증이 아니라
   그냥 오래 띄워두는 것뿐이라 job-control 문제는 없지만, SSH 연결이 끊겨도 살아있게 하려면
   여전히 `nohup`/`setsid` 필요).
2. **오케스트레이터 자신의 환경(이 세션이 돌고 있는 랩탑/컨테이너)에서** `curl
   http://100.67.27.83:8080/stream`(tailscale IP, 실제 네트워크 경유 — loopback 아님)로 실제
   MJPEG 바이트를 몇 초간 받아보고, `cv2.imdecode`로 실제 프레임 디코드가 성공하는지 확인(기존
   `docs/session_log.md`/과거 세션이 loopback으로 했던 것과 같은 방식, 이번엔 진짜 네트워크).
   대역폭/지연(예: 몇 초에 몇 MB 받았는지)도 같이 기록.
3. 확인 끝나면 RPi의 스트리밍 프로세스를 정리하고(PID 확인 후 kill, 카메라 release까지 fuser로
   확인 — 위 함정 절 패턴 그대로) `docs/vision_status.md` "주의" 절에서 이 항목을 완료로 갱신.
4. **이 Phase는 코드 변경이 없을 가능성이 높다**(순수 검증) — 문제가 발견되면(예: 실네트워크에서
   프레임 드롭이 심함, Wi-Fi 끊김과 상호작용 등) 그 때 대응 범위를 다시 판단한다(무리하게 같은
   세션에서 고치려 하지 말고, 발견된 사실만 정확히 기록하고 다음으로 넘겨도 됨 — 프로토콜
   "정지조건"의 "인간 개입 요구" 케이스는 아니지만 새로 발견된 문제의 해법이 트레이드오프가
   있으면 사용자에게 먼저 확인).

### 검증
- 실제 `curl`/`cv2.imdecode` 성공 여부, 수신 바이트/시간, 디코드된 프레임 shape을 보고에 포함.
- pytest는 이 Phase로 바뀔 게 없으면 그대로 330 passed 유지 확인만.

---

## 2. Phase B — 골든셋 실촬영 데이터 교체 (사용자 물리적 개입 필수 — 정지조건)

### ⚠️ 진입 전 필수: 사용자에게 먼저 확인할 것

이 트랙은 **오케스트레이터 혼자 끝낼 수 없다.** 버티포트(흰 원+검은 V+빨간 고리, 직경 2m 고리)와
조난자 구역(3.0m×3.0m×0.105m 초록 매트)을 실제 물리 크기로 배치하고, 카메라를 10m/20m/40m
상당의 고도(또는 그에 준하는 GSD를 만드는 촬영 거리)에서 촬영해야 한다 — 드론 비행이거나, 최소
사람이 사다리/장대 등으로 카메라를 들고 여러 거리에서 촬영해야 하는 일이다. **RPi에 SSH로
접속하는 것만으로는 절대 할 수 없다.**

**다음 오케스트레이터 세션은 이 Phase에 진입하기 전에 반드시 사용자에게 다음을 확인한다:**
1. 실촬영이 가능한 시점(장비/장소/날씨 등)이 언제인지.
2. 그 전까지는 Phase A만 하고 대기하거나, 다른 트랙(빠께스 탐색 설계 등 — 단 이건
   `project_vision_2nd_qualifier_bucket_target` 메모리가 "vision 파트 완료 후"라 명시했으니
   시작 전 재확인 필요)으로 돌릴지.

**사용자가 먼저 촬영 세션을 언급하거나 준비됐다고 하기 전까지, 이 Phase를 무리하게 시작하거나
대체 방법(예: 임의 물체로 눈속임 촬영)을 제안하지 마라** — 골든셋의 신뢰성 자체가 실측 스펙과의
정합에 있다.

### 확정 전제 (재논의 불필요, 촬영이 실제로 가능해지면 그대로 따를 것)
- 절차는 `vision/tests/golden/README.md` "실기체 데이터가 들어오면" 절에 이미 5단계로 정리돼
  있다 — **새 절차를 만들지 말고 그대로 따를 것.**
- 촬영 도구는 이번에 새로 완성된 `python -m vision.main live --preset <프리셋> --display none
  --output <파일>`(라이브 모드, `docs/vision_status.md` 2026-07-24 항목 참조)를 쓸 수 있다 —
  `vision/tools/calib_capture.py`(체커보드 전용, 캘리브레이션 보류 중이라 여전히 동결)와는
  다른 도구이니 혼동하지 말 것.
- 재검증 대상으로 이미 알려진 것(README/트랙보드에 이미 적혀 있음, 여기서 재복사만):
  - `vertiport` 40m 티어의 `known_limitation`(흰 필드는 잡히나 `black_v` 형상매칭 탈락) —
    실측으로 여전히 재현되는지, 아니면 실촬영에서는 다르게 나오는지 확인.
  - `distress` 40m 티어(매트 픽셀 면적이 `min_area` 미만이라 미검출) — 물리적으로 타당한
    미검출인지 실측 재확인.
  - ArUco fine 단계(`vertiport_fine.yaml`)도 실제 50cm×50cm ID=23 마커를 촬영해 재검증 대상
    (nominal intrinsics 기반이라 pose 절대 정밀도는 여전히 "미검증" 상태 그대로 유지 — 이
    Phase가 그 플래그를 지우는 게 아님, 검출 자체의 실측 재현만 확인).
  - `BlackVMatcher` 참조 V 템플릿(두께/종횡비)·`max_match_distance`도 실측 규격으로 재검증 필요
    (`vision/CLAUDE.md` "주의" 절 참조).

### 할 일 (촬영이 실제로 가능해진 뒤)
1. `vision/tests/golden/README.md` "실기체 데이터가 들어오면" 5단계 그대로 수행.
2. `labels.json`의 기대값은 **사람이 눈으로 확인한 정답**이어야 한다 — 파이프라인 출력을 그대로
   베끼면 회귀테스트가 항상 통과하는 무의미한 테스트가 된다(README에 이미 명시된 원칙, 재확인만).
3. `pytest vision/tests/test_golden_regression.py -v`로 새 라벨이 실제와 맞는지 확인.
4. `docs/vision_status.md`/`vision/CLAUDE.md`의 관련 "골든셋은 전부 합성" 문구를 실측 반영으로
   갱신.

### 하지 말 것
- 촬영 없이 "그럴듯한" 라벨값을 추정해서 채우지 마라.
- 실측 캘리브레이션(체커보드) 재개를 이 김에 같이 하자고 제안하지 마라 — 여전히 보류 결정
  유효(메모리 `project_vision_calibration_deferred`), 예선 통과 후·§9 8번 진입 직전까지 금지.

---

## 3. 참조

- `docs/vision_status.md` 트랙 보드 — 2026-07-24 최신 블록(ArUco 브랜치·LiveFrameSource 완료 기록)
- `docs/vision_aruco_branch.md` — 완료된 ArUco 브랜치 브리프(참고용, 재진입 아님)
- `vision/tests/golden/README.md` — Phase B 절차의 원본
- 메모리 `project_rpi5_ubuntu_camera_stack.md` — picam-venv opencv 업그레이드·SIGINT job-control
  함정·라이브모드 실측 수치(프레임당 ~150ms @ 4608×2592)
- 메모리 `feedback_orchestrator_protocol` — 세션 실행 프로토콜
- 메모리 `project_vision_calibration_deferred` / `project_vision_2nd_qualifier_bucket_target` —
  Phase B 진입 시 재확인해야 할 우선순위/보류 결정들
