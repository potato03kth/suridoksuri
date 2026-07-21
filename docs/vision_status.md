---
doc_type: session_status
project: suridoksuri-1
scope: vision 세션 유일 진입점 — 트랙 보드 + 설계 포인터
last_updated: 2026-07-21d
---

# vision 세션 진입 상태 문서

> **새 세션 진입:** 아래 vision 트랙 블록을 읽고, 설계 상세는 `docs/vision_plan.md`의 **필요 섹션만** 연다.
> FC 트랙은 `docs/session_status.md`에 별도로 있으며 **vision 세션은 그걸 읽지 않는다**(도메인 간 컨텍스트 격리).
> `/session-log`는 vision을 건드린 세션에서 **이 문서**의 트랙 블록을 갱신한다 (서술 로그는 공용 `docs/session_log.md`).

---

## 공통 상태 (2026-07-21)

- **브랜치:** `dev--vision-computing-module` (현재 FC 트랙과 공용. vision 전용 브랜치 분리는 미결정)
- **개발/테스트 환경:** 메인 개발주체 = **노트북(WSL)**, `.venv` 준비 완료(`source .venv/bin/activate`, 저장소 루트) — `pytest vision/tests/` **106 passed** 확인됨(2026-07-21d). 개발컴 `.venv`도 기존대로 유효. RPi는 headless지만 **SSH 상시 접근 가능**(tailscale `100.67.27.83`, 계정 `suri`, ed25519 키 등록됨, **패스워드리스 sudo 설정 완료** 2026-07-21) — Claude가 직접 SSH로 RPi 작업 가능. **장착된 카메라는 정품 Camera Module 3가 아니라 서드파티 클론 "CAM109-IMX708AF-75"**(IMX708 센서 동일, 화각은 대각 75°로 계획서 가정 102°보다 좁음). 상세 → 메모리 `project_vision_dev_env.md`, `project_rpi5_ubuntu_camera_stack.md`
- **커밋 규율:** vision 커밋은 메시지에 **`[vision]`** 태그
- **설계 정본:** `docs/vision_plan.md` — 확정 결정·물리 제약·검출 전략·변경내성/관측성·빌드 순서·블라인드스팟. **이 문서는 라이브 진척만** 담는다.
- **대회 규정(2026-07-21 대부분 확정):** ArUco=`DICT_4X4_50` ID23 **50cm×50cm**(원래 계획 가정과 일치 확인) · 버티포트 하기안전구역(빨간 원)=직경2m·선굵기5cm 고리 신규확정 · ③빨간십자·초록색/치수는 **비공개로 확정**(버티포트 유사크기+안전마진 가정) · 성공판정="매끄럽게 보이면"(정성) 확정 · ④단순착륙=GPS+라이다만 확정 · CC 인터페이스·나무조각상 판단기준만 여전히 대기 → 상세 `vision_plan.md` §10.

---

## 트랙 보드

### 👁 vision-정밀착륙 — ▶ 활성 (카메라 캘리브레이션 브링업은 RPi 작업 허가 대기 중 — 그 사이 §7.9 4·6·7번 대체 트랙 완료)

- **내용:** 착륙지점 인식·정밀착륙 시스템(RPi5 온보드). 고전 CV, 타겟별 coarse→fine 2단, 비전 폐루프 <30cm. 설계 정본 `docs/vision_plan.md`.
- **직전 완료(2026-07-21d, 노트북/WSL 로컬 세션 — RPi/실카메라 작업 금지 하에 진행):** §7.9 "지금 당장 할 일" 7번 — 골든셋 폴더 스캐폴드 + 재생 회귀 assert. 상세:
  - `vision/tests/golden/`: `<타겟>/<고도>/frame_NNN.png`+`labels.json` 스키마. **전부 합성(synthetic) 데이터** — 실촬영 아님(카메라 브링업 전이라 불가능, README에 명시). `vertiport`(①, 10m/20m/40m, `vertiport_coarse.yaml`) · `distress`(②, 10m/20m/40m, 신규 `presets/distress_coarse.yaml` — 전용 모듈 없이 기존 `ColorFilter`+`RectDetector` 조합, 검출 로직 신규 아님) · `no_target`(④ 단순착륙과 동일 조건 — 피듀셜 없는 평지에서 오탐 안 하는지). ③ 하기구역은 전용 형상판별 검출기가 없어 의도적으로 제외(README에 사유 명시). 구조/스키마/재생성법/실기체 데이터 교체 절차는 `vision/tests/golden/README.md`. 생성 소스는 `vision/tests/golden/generate_synthetic.py`(pytest 대상 아님, 수동 재생성 도구)
  - **실측으로 발견한 것(검출기 미변경, 사실 그대로 골든셋에 고정):** `vertiport` 40m 티어는 흰 필드는 후보로 잡히지만(`white_field.candidates=1`) `black_v` 형상매칭이 탈락(`rejected=1`)해 최종 검출 0건 — 저해상 스케일에서 `vertiport_coarse.yaml`의 고정 `kernel_size=5` morphology가 V 노치 주변 연결성을 깨는 게 원인으로 보임. `distress` 40m 티어는 매트 픽셀 면적이 `rect_detector.min_area`(300)보다 작아져 물리적으로 타당한 미검출. 둘 다 `known_limitation: true`로 라벨링, 검출기 파라미터는 손대지 않음(이번 세션 범위 밖) — 실기체 데이터로 재검증 대상.
  - `vision/tests/test_golden_regression.py`: **실제** `vision.replay.run_replay()`(DirFrameSource+실제 Pipeline, 몽키패치 없음)로 골든 폴더를 재생 → JSONL 검출 개수를 `labels.json`과 비교(15개 테스트: 스캐폴드 비어있지 않음 확인 1 + 리프 디렉터리 7개 × 2종 테스트). 검증으로 `RedRingDetector.min_points`를 일시적으로 999999로 바꿔 관련 4개 테스트가 실제로 실패하는 것 확인 후 원복 — 진짜 파이프라인을 물고 있음을 확인.
  - `vision/CLAUDE.md`: 파일역할표에 `presets/distress_coarse.yaml`·`tests/golden/` 행, 테스트 규칙표에 골든 회귀 행, 공통규칙 4번의 "골든셋 회귀는 데이터 수집 후" 문구를 "합성 데이터로 스캐폴드 시작됨" 으로 갱신
  - `pytest vision/tests/` **106 passed**(기존 91 + 신규 15)
- **🟡 카메라 인트린식 캘리브레이션 브링업 — RPi 작업 허가 대기 중 (이번 세션도 미착수, 의도적 보류):** 직전 세션(2026-07-21a)에서 libcamera가 RPi5용 PiSP IPA 모듈 없이 빌드돼 있어 카메라 브링업이 막힌 채 중단됨. **사용자가 실비행 나가면서 RPi SSH 접속·실카메라 작업을 계속 전면 금지**하고 있음 — 그래서 이번 세션도 그 대신 §7.9 7번(카메라 독립 트랙)을 진행했다. RPi 작업 재개 허가가 떨어지면 아래 "다음"의 1번부터 이어간다. 상세 경과·진단명령·4개 선택지는 메모리 `project_rpi5_ubuntu_camera_stack.md`에 그대로 보존됨(재조사 불필요).
- **다음 (진입하면 이 순서):**
  1. **[RPi 작업 허가 필요]** 카메라 브링업 4개 선택지(V4L2 RAW 직접캡처 권장/libcamera 재빌드/RPi OS 재설치/보류) 중 어느 걸로 갈지 사용자에게 확인 → 메모리 `project_rpi5_ubuntu_camera_stack.md` 먼저 읽고 진입
  2. 선택된 방향으로 카메라 캡처 도구(`vision/tools/rpi_capture.py`) 완성 → 실제 체커보드 촬영 → 카메라 인트린식/왜곡 캘리브레이션
  3. (카메라 독립, 대체 가능) §7.9 항목5 — 라이브 스트림 어댑터(`compute_tap` VGA → MJPEG/ROS image)
  4. (카메라 독립, RPi 허가 후가 자연스러움) 골든셋을 실촬영 데이터로 교체 — 절차는 `vision/tests/golden/README.md` "실기체 데이터가 들어오면" 참조. 이때 40m 티어의 `known_limitation` 두 건도 실측 재검증
- **주의:** Pi4 인코더/라이다 40m급 미확정 · 기존 `vision/` 틀은 폐기 아님(§12) · `geo_project.pixel_to_gps` 폐기 예정 · **버티포트 V 형상매칭은 실물 규격 미확인 상태의 합성테스트로만 검증됨** — 실기체 데이터 확보 후 `BlackVMatcher` 참조 V 템플릿(두께/종횡비)·`max_match_distance` 재검증 필요 · **카메라 화각이 계획서 가정(102°)과 다름(75°) — coarse 캐스케이드 탐지거리 가정 재검토 여지 있음, §4.1 GSD 표 자체가 재검증 대기라 골든셋 고도 라벨도 정밀 GSD 매핑이 아니라 스키마 자리표시자로 명시함(§7.9 항목7 완료 노트 참조)** · `LiveFrameSource`는 인터페이스 계약만 검증됨, 실장치 연결은 RPi 허가 후 검증 필요 · **`jsonl_view.py`의 state 서브플롯은 실기체 데이터 없음** — main.py/replay.py가 아직 `state`를 채우지 않아(§5.1 상태머신 미구현) 실사용 시엔 항상 "no state data" 안내만 뜬다. 상태머신 연결되면 자동으로 실데이터가 나타남(코드 변경 불필요, 이미 대응돼 있음) · **골든셋(`tests/golden/`)은 전부 합성 데이터** — 실기체 데이터 확보 후 교체 필요(§7.9 항목7 완료 노트) · **`vertiport_coarse.yaml`의 고정 `kernel_size=5` morphology가 저해상(작은 픽셀) 스케일에서 흰 필드 연결성을 깨는 스케일 민감성이 골든셋으로 새로 드러남** — 검출기 튜닝은 이번 세션 범위 밖, 실기체 데이터/저고도 재검증 필요 · 세부 정정 이력·논의는 `docs/session_log.md` 참조
- **참조:** `docs/vision_plan.md` §2(타겟 스펙)/§5.2(버티포트 coarse 캐스케이드)/§5.5(색 항상성)/§7.5(기록·재생)/§7.9(관측성 워크플로) · `vision/CLAUDE.md`(파일역할표·테스트 규칙표) · 메모리 `project_rpi5_ubuntu_camera_stack.md`(카메라 브링업 전체 경과·진단명령·재현법)
