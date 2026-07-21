---
doc_type: session_status
project: suridoksuri-1
scope: vision 세션 유일 진입점 — 트랙 보드 + 설계 포인터
last_updated: 2026-07-21b
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

### 👁 vision-정밀착륙 — ▶ 활성 (카메라 캘리브레이션 브링업은 RPi 작업 허가 대기 중 — 그 사이 §7.9 4번 대체 트랙 완료)

- **내용:** 착륙지점 인식·정밀착륙 시스템(RPi5 온보드). 고전 CV, 타겟별 coarse→fine 2단, 비전 폐루프 <30cm. 설계 정본 `docs/vision_plan.md`.
- **직전 완료(2026-07-21b, 노트북/WSL 로컬 세션 — RPi/실카메라 작업 금지 하에 진행):** §7.9 "지금 당장 할 일" 4번 — `FrameSource` 어댑터(Live/Dir/Bag) + 재생 CLI + blackbox/logger를 main.py에 실연결. 커밋 `6a241e3`, push 완료. 상세:
  - `vision/utils/frame_source.py`: `FrameRecord` + `LiveFrameSource`(실카메라 연결 재시도→명확한 `ConnectionError`, 실장치 검증은 RPi 허가 후)/`DirFrameSource`(녹화폴더=프레임파일+선택적 `telemetry.jsonl`)/`BagFrameSource`(단일 비디오+선택적 사이드카 `<basename>.jsonl`) + `open_dir_or_bag` 팩토리(디렉터리/파일 자동판별)
  - `vision/replay.py`: `python -m vision.replay <녹화폴더|bag> --preset ...` 오프라인 재생 CLI(§7.5/§7.9 (a)) — 동일 `Pipeline`으로 결정론적 재생, 로거+블랙박스 기록, `--output`으로 주석 mp4 저장
  - `vision/main.py`: 이중싱크 로거(provenance 헤더=git해시+config)+JSONL 블랙박스를 실제 파이프라인에 연결(`--log-dir`/`--log-name`, 항상 on) — 이제 `python -m vision.main` 실행마다 실제 `.log`/`.jsonl`이 남는다
  - **진짜 테스트로 검증**: Dir/Bag은 실제 png/mp4 파일을 실제로 디코딩(순서·telemetry 매칭·결정론 확인), Live는 실카메라 없어 `cv2.VideoCapture`만 몽키패치해 재시도/에러 계약 검증, main.py/replay.py는 실제 파이프라인 1회 실행 후 디스크의 JSONL 내용까지 assert. `pytest vision/tests/` **83 passed**(기존 59 + 신규 24)
- **🟡 카메라 인트린식 캘리브레이션 브링업 — RPi 작업 허가 대기 중 (이번 세션 미착수, 의도적 보류):** 직전 세션(2026-07-21a)에서 libcamera가 RPi5용 PiSP IPA 모듈 없이 빌드돼 있어 카메라 브링업이 막힌 채 중단됨. **사용자가 실비행 나가면서 이번 세션엔 RPi SSH 접속·실카메라 작업을 전면 금지**했음 — 그래서 이번 세션은 그 대신 위 §7.9 4번(카메라 독립 트랙)을 진행했다. RPi 작업 재개 허가가 떨어지면 아래 "다음"의 1번부터 이어간다. 상세 경과·진단명령·4개 선택지는 메모리 `project_rpi5_ubuntu_camera_stack.md`에 그대로 보존됨(재조사 불필요).
- **다음 (진입하면 이 순서):**
  1. **[RPi 작업 허가 필요]** 카메라 브링업 4개 선택지(V4L2 RAW 직접캡처 권장/libcamera 재빌드/RPi OS 재설치/보류) 중 어느 걸로 갈지 사용자에게 확인 → 메모리 `project_rpi5_ubuntu_camera_stack.md` 먼저 읽고 진입
  2. 선택된 방향으로 카메라 캡처 도구(`vision/tools/rpi_capture.py`) 완성 → 실제 체커보드 촬영 → 카메라 인트린식/왜곡 캘리브레이션
  3. (카메라 독립, 대체 가능) §7.9 항목5 이후 — 라이브 스트림 어댑터(`compute_tap` VGA → MJPEG/ROS image)·JSONL 뷰어/플롯·골든셋 폴더 스캐폴드
- **주의:** Pi4 인코더/라이다 40m급 미확정 · 기존 `vision/` 틀은 폐기 아님(§12) · `geo_project.pixel_to_gps` 폐기 예정 · **버티포트 V 형상매칭은 실물 규격 미확인 상태의 합성테스트로만 검증됨** — 실기체 데이터 확보 후 `BlackVMatcher` 참조 V 템플릿(두께/종횡비)·`max_match_distance` 재검증 필요 · **카메라 화각이 계획서 가정(102°)과 다름(75°) — coarse 캐스케이드 탐지거리 가정 재검토 여지 있음** · `LiveFrameSource`는 인터페이스 계약만 검증됨, 실장치 연결은 RPi 허가 후 검증 필요 · 세부 정정 이력·논의는 `docs/session_log.md` 참조
- **참조:** `docs/vision_plan.md` §2(타겟 스펙)/§5.2(버티포트 coarse 캐스케이드)/§5.5(색 항상성)/§7.5(기록·재생)/§7.9(관측성 워크플로) · `vision/CLAUDE.md`(파일역할표·테스트 규칙표) · 메모리 `project_rpi5_ubuntu_camera_stack.md`(카메라 브링업 전체 경과·진단명령·재현법)
