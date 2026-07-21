---
doc_type: session_status
project: suridoksuri-1
scope: vision 세션 유일 진입점 — 트랙 보드 + 설계 포인터
last_updated: 2026-07-21c
---

# vision 세션 진입 상태 문서

> **새 세션 진입:** 아래 vision 트랙 블록을 읽고, 설계 상세는 `docs/vision_plan.md`의 **필요 섹션만** 연다.
> FC 트랙은 `docs/session_status.md`에 별도로 있으며 **vision 세션은 그걸 읽지 않는다**(도메인 간 컨텍스트 격리).
> `/session-log`는 vision을 건드린 세션에서 **이 문서**의 트랙 블록을 갱신한다 (서술 로그는 공용 `docs/session_log.md`).

---

## 공통 상태 (2026-07-21)

- **브랜치:** `dev--vision-computing-module` (현재 FC 트랙과 공용. vision 전용 브랜치 분리는 미결정)
- **개발/테스트 환경:** 메인 개발주체 = **노트북(WSL)**, `.venv` 준비 완료(`source .venv/bin/activate`, 저장소 루트) — `pytest vision/tests/` 83 passed 확인됨(2026-07-21b). 개발컴 `.venv`도 기존대로 유효. RPi는 headless지만 **SSH 상시 접근 가능**(tailscale `100.67.27.83`, 계정 `suri`, ed25519 키 등록됨, **패스워드리스 sudo 설정 완료** 2026-07-21) — Claude가 직접 SSH로 RPi 작업 가능. **장착된 카메라는 정품 Camera Module 3가 아니라 서드파티 클론 "CAM109-IMX708AF-75"**(IMX708 센서 동일, 화각은 대각 75°로 계획서 가정 102°보다 좁음). 상세 → 메모리 `project_vision_dev_env.md`, `project_rpi5_ubuntu_camera_stack.md`
- **커밋 규율:** vision 커밋은 메시지에 **`[vision]`** 태그
- **설계 정본:** `docs/vision_plan.md` — 확정 결정·물리 제약·검출 전략·변경내성/관측성·빌드 순서·블라인드스팟. **이 문서는 라이브 진척만** 담는다.
- **대회 규정(2026-07-21 대부분 확정):** ArUco=`DICT_4X4_50` ID23 **50cm×50cm**(원래 계획 가정과 일치 확인) · 버티포트 하기안전구역(빨간 원)=직경2m·선굵기5cm 고리 신규확정 · ③빨간십자·초록색/치수는 **비공개로 확정**(버티포트 유사크기+안전마진 가정) · 성공판정="매끄럽게 보이면"(정성) 확정 · ④단순착륙=GPS+라이다만 확정 · CC 인터페이스·나무조각상 판단기준만 여전히 대기 → 상세 `vision_plan.md` §10.

---

## 트랙 보드

### 👁 vision-정밀착륙 — ▶ 활성 (카메라 캘리브레이션 브링업은 RPi 작업 허가 대기 중 — 그 사이 §7.9 4·6번 대체 트랙 완료)

- **내용:** 착륙지점 인식·정밀착륙 시스템(RPi5 온보드). 고전 CV, 타겟별 coarse→fine 2단, 비전 폐루프 <30cm. 설계 정본 `docs/vision_plan.md`.
- **직전 완료(2026-07-21c, 노트북/WSL 로컬 세션 — RPi/실카메라 작업 금지 하에 진행):** §7.9 "지금 당장 할 일" 6번 — JSONL 뷰어/플롯 최소본. 커밋 `2e02e29`, push 완료. 상세:
  - `vision/tools/jsonl_view.py`: `BlackBoxLogger`가 쌓는 JSONL(새 스키마 아님, 그대로 읽음)을 시간축 score/latency/state 3단 플롯(PNG)으로. score=`chosen.confidence` 우선, 없으면 그 프레임 `detections` 중 최고 `confidence`. rejection 레코드는 score 서브플롯에 세로 점선. state가 전부 None(상태머신 미구현)이면 안내 텍스트만 표시. 결측은 라인을 nan으로 끊어 옆 프레임과 잘못 이어붙이지 않음(포인트 수 = JSONL type=frame 행 수와 항상 일치). `matplotlib` Agg 백엔드로 headless-safe(GUI 강제 호출 없음). CLI: `python vision/tools/jsonl_view.py <jsonl> [--output out.png] [--x-axis ts|frame_id]`
  - `vision/requirements.txt`에 `matplotlib` 추가, `.venv`에 설치 완료
  - `vision/CLAUDE.md`: 파일역할표에 `tools/jsonl_view.py` 행 추가 + tools/ import 규칙에 "하드웨어 비의존 CLI 도구는 예외(.venv 설치+pytest 대상)" 명시 + 테스트 규칙표에 행 추가
  - **진짜 테스트로 검증**: `vision.main`을 실제 실행(색상 필터+rect_detector 직결 임시 preset, edge/morphology 조합은 합성 테스트 도형엔 안 맞아 우회 — 기존 presets/*.yaml·검출 로직은 미변경)해 진짜 JSONL 생성 → 그 파일을 뷰어에 먹여 `load_records()` 행 수=JSONL 행 수, `build_figure()` 각 라인 포인트 수=행 수, 결측 위치에 실제 nan 구멍 존재, PNG 실파일 생성(size>0)까지 assert. rejection/다중 state 경계 케이스는 `BlackBoxLogger`를 직접 호출해 만든 실제 JSONL로 검증(수기 JSON 아님). `pytest vision/tests/` **91 passed**(기존 83 + 신규 8)
- **🟡 카메라 인트린식 캘리브레이션 브링업 — RPi 작업 허가 대기 중 (이번 세션도 미착수, 의도적 보류):** 직전 세션(2026-07-21a)에서 libcamera가 RPi5용 PiSP IPA 모듈 없이 빌드돼 있어 카메라 브링업이 막힌 채 중단됨. **사용자가 실비행 나가면서 RPi SSH 접속·실카메라 작업을 계속 전면 금지**하고 있음 — 그래서 이번 세션도 그 대신 §7.9 6번(카메라 독립 트랙)을 진행했다. RPi 작업 재개 허가가 떨어지면 아래 "다음"의 1번부터 이어간다. 상세 경과·진단명령·4개 선택지는 메모리 `project_rpi5_ubuntu_camera_stack.md`에 그대로 보존됨(재조사 불필요).
- **다음 (진입하면 이 순서):**
  1. **[RPi 작업 허가 필요]** 카메라 브링업 4개 선택지(V4L2 RAW 직접캡처 권장/libcamera 재빌드/RPi OS 재설치/보류) 중 어느 걸로 갈지 사용자에게 확인 → 메모리 `project_rpi5_ubuntu_camera_stack.md` 먼저 읽고 진입
  2. 선택된 방향으로 카메라 캡처 도구(`vision/tools/rpi_capture.py`) 완성 → 실제 체커보드 촬영 → 카메라 인트린식/왜곡 캘리브레이션
  3. (카메라 독립, 대체 가능) §7.9 항목5 — 라이브 스트림 어댑터(`compute_tap` VGA → MJPEG/ROS image)
  4. (카메라 독립, 대체 가능) §7.9 항목7 — 골든셋 폴더 스캐폴드(라벨 프레임, 고도·타겟별) + 재생 회귀 assert
- **주의:** Pi4 인코더/라이다 40m급 미확정 · 기존 `vision/` 틀은 폐기 아님(§12) · `geo_project.pixel_to_gps` 폐기 예정 · **버티포트 V 형상매칭은 실물 규격 미확인 상태의 합성테스트로만 검증됨** — 실기체 데이터 확보 후 `BlackVMatcher` 참조 V 템플릿(두께/종횡비)·`max_match_distance` 재검증 필요 · **카메라 화각이 계획서 가정(102°)과 다름(75°) — coarse 캐스케이드 탐지거리 가정 재검토 여지 있음** · `LiveFrameSource`는 인터페이스 계약만 검증됨, 실장치 연결은 RPi 허가 후 검증 필요 · **`jsonl_view.py`의 state 서브플롯은 실기체 데이터 없음** — main.py/replay.py가 아직 `state`를 채우지 않아(§5.1 상태머신 미구현) 실사용 시엔 항상 "no state data" 안내만 뜬다. 상태머신 연결되면 자동으로 실데이터가 나타남(코드 변경 불필요, 이미 대응돼 있음) · 세부 정정 이력·논의는 `docs/session_log.md` 참조
- **참조:** `docs/vision_plan.md` §2(타겟 스펙)/§5.2(버티포트 coarse 캐스케이드)/§5.5(색 항상성)/§7.5(기록·재생)/§7.9(관측성 워크플로) · `vision/CLAUDE.md`(파일역할표·테스트 규칙표) · 메모리 `project_rpi5_ubuntu_camera_stack.md`(카메라 브링업 전체 경과·진단명령·재현법)
