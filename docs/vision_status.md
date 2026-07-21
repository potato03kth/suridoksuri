---
doc_type: session_status
project: suridoksuri-1
scope: vision 세션 유일 진입점 — 트랙 보드 + 설계 포인터
last_updated: 2026-07-21
---

# vision 세션 진입 상태 문서

> **새 세션 진입:** 아래 vision 트랙 블록을 읽고, 설계 상세는 `docs/vision_plan.md`의 **필요 섹션만** 연다.
> FC 트랙은 `docs/session_status.md`에 별도로 있으며 **vision 세션은 그걸 읽지 않는다**(도메인 간 컨텍스트 격리).
> `/session-log`는 vision을 건드린 세션에서 **이 문서**의 트랙 블록을 갱신한다 (서술 로그는 공용 `docs/session_log.md`).

---

## 공통 상태 (2026-07-21)

- **브랜치:** `dev--vision-computing-module` (현재 FC 트랙과 공용. vision 전용 브랜치 분리는 미결정)
- **개발/테스트 환경:** 메인 개발주체 = **노트북(WSL)**, `.venv` 준비 완료(`source .venv/bin/activate`, 저장소 루트) — `pytest vision/tests/` 59 passed 확인됨. 개발컴 `.venv`도 기존대로 유효. RPi는 headless지만 **SSH 상시 접근 가능**(tailscale `100.67.27.83`, 계정 `suri`, ed25519 키 등록됨, **패스워드리스 sudo 설정 완료** 2026-07-21) — Claude가 직접 SSH로 RPi 작업 가능. **장착된 카메라는 정품 Camera Module 3가 아니라 서드파티 클론 "CAM109-IMX708AF-75"**(IMX708 센서 동일, 화각은 대각 75°로 계획서 가정 102°보다 좁음). 상세 → 메모리 `project_vision_dev_env.md`, `project_rpi5_ubuntu_camera_stack.md`
- **커밋 규율:** vision 커밋은 메시지에 **`[vision]`** 태그
- **설계 정본:** `docs/vision_plan.md` — 확정 결정·물리 제약·검출 전략·변경내성/관측성·빌드 순서·블라인드스팟. **이 문서는 라이브 진척만** 담는다.
- **대회 규정(2026-07-21 대부분 확정):** ArUco=`DICT_4X4_50` ID23 **50cm×50cm**(원래 계획 가정과 일치 확인) · 버티포트 하기안전구역(빨간 원)=직경2m·선굵기5cm 고리 신규확정 · ③빨간십자·초록색/치수는 **비공개로 확정**(버티포트 유사크기+안전마진 가정) · 성공판정="매끄럽게 보이면"(정성) 확정 · ④단순착륙=GPS+라이다만 확정 · CC 인터페이스·나무조각상 판단기준만 여전히 대기 → 상세 `vision_plan.md` §10.

---

## 트랙 보드

### 👁 vision-정밀착륙 — ▶ 활성 (카메라 캘리브레이션 브링업 중 — 긴급 세션종료로 미완, 아래서 바로 이어갈 것)

- **내용:** 착륙지점 인식·정밀착륙 시스템(RPi5 온보드). 고전 CV, 타겟별 coarse→fine 2단, 비전 폐루프 <30cm. 설계 정본 `docs/vision_plan.md`.
- **직전 완료(2026-07-21):** 관측성 골격 §7.9 "지금 당장 할 일" 2번(이중싱크 로거+JSONL 블랙박스+provenance 헤더, `pytest vision/tests/` 59 passed) 완료·커밋·푸시됨(`bf7fdab`). RPi 저장소(`/home/suri/drone_ws/src/suridoksuri`)도 이 시점까지 fast-forward 동기 완료.
- **🔴 지금 하던 일 (카메라 인트린식 캘리브레이션 착수 중 발견한 큰 사고):** RPi 카메라 촬영 도구(`vision/tools/rpi_capture.py`)를 만들다가 **RPi5 카메라가 소프트웨어적으로 전혀 안 잡히는 걸 발견 → 브링업 디버깅으로 전환, 커널 레벨까지는 뚫었으나 libcamera 라이브러리 자체의 구조적 결손에 막혀 중단.** 상세 경과·명령어는 메모리 `project_rpi5_ubuntu_camera_stack.md`에 전부 기록됨. 요약:
  1. RPi가 Raspberry Pi OS가 아니라 **Ubuntu**라 `picamera2`/`rpicam-apps`가 apt에 없음을 발견.
  2. 장착된 카메라가 정품 CM3가 아니라 **서드파티 클론 "CAM109-IMX708AF-75"**임을 사용자 제공 PDF로 확인 — `camera_auto_detect`가 이 보드를 못 알아봄.
  3. 제조사 PDF 해법대로 `/boot/firmware/config.txt`에 `camera_auto_detect=0` + `dtoverlay=imx708,cam0/cam1` 추가 → **재부팅 후 커널/V4L2 레벨에서 카메라 정상 인식 확인**(`/dev/video0` = `rp1-cfe-csi2_ch0`, dmesg에 `imx708 camera module ID` 로그).
  4. **그런데 그 위 libcamera 라이브러리가 여전히 카메라를 못 봄**(`CameraManager.cameras`가 빈 리스트) — 원인 특정 완료: 이 Ubuntu의 `libcamera-ipa` 패키지가 **RPi5용 PiSP ISP IPA 모듈(`ipa_rpi_pisp.so`) 없이 빌드됨**(구형 vc4 IPA만 있음). **이건 picamera2로 가든 GStreamer(`libcamerasrc`)로 가든 똑같이 막히는 근본 병목** — 그래서 처음 만들었던 picamera2 버전 스크립트를 GStreamer 버전으로 재작성했었는데, 그것도 결국 동일 벽에 막힌다는 게 뒤늦게 밝혀짐. **`vision/tools/rpi_capture.py`의 GStreamer 재작성분은 커밋은 했지만 현재 이 하드웨어에서 작동 불가 상태** — 다음 세션에서 코드 자체를 다시 손볼 필요 있음(아래 선택지 1번 방향).
  5. 사용자에게 다음 4개 선택지를 제시했으나 **사용자가 답하기 전 긴급 세션종료** — 다음 세션 최우선 할 일은 이 질문에 답 받고 진행하는 것:
     - **(권장) V4L2 RAW 직접 캡처 + 수동 디베이어** — libcamera 우회, 커널 V4L2는 이미 검증됨, 가장 빠름
     - libcamera를 PiSP 지원 포함해서 소스 재빌드 — 정공법, 느리고 리스크 큼
     - SD카드 정품 Raspberry Pi OS로 재설치 — 확실하지만 이 RPi5의 기존 FC/비행 설정 전부 다시 해야 함(공유 장치)
     - 일단 보류, 다른 트랙(§7.9 3번 등)으로 이동
- **다음 (진입하면 이 순서):**
  1. **위 4개 선택지 사용자에게 확인** — 메모리 `project_rpi5_ubuntu_camera_stack.md` 전체를 먼저 읽고 진입(진단 명령·정확한 원인 다 있음, 처음부터 재조사할 필요 없음)
  2. 선택된 방향으로 카메라 캡처 도구 완성 → 실제 체커보드 촬영 → 카메라 인트린식/왜곡 캘리브레이션
  3. **§7.9 3번 이후** — `FrameSource` 어댑터(Live/Dir/Bag) + 재생 CLI, blackbox/logger를 실제 파이프라인(main.py)에 연결 (카메라 브링업과 독립적으로 진행 가능한 대체 트랙)
- **주의:** Pi4 인코더/라이다 40m급 미확정 · 기존 `vision/` 틀은 폐기 아님(§12) · `geo_project.pixel_to_gps` 폐기 예정 · **버티포트 V 형상매칭은 실물 규격 미확인 상태의 합성테스트로만 검증됨** — 실기체 데이터 확보 후 `BlackVMatcher` 참조 V 템플릿(두께/종횡비)·`max_match_distance` 재검증 필요 · `logging.py`/`blackbox.py`는 아직 독립 유틸일 뿐 main.py/파이프라인에 미연결 · **카메라 화각이 계획서 가정(102°)과 다름(75°) — coarse 캐스케이드 탐지거리 가정 재검토 여지 있음** · 세부 정정 이력·논의는 `docs/session_log.md` 참조
- **참조:** `docs/vision_plan.md` §2(타겟 스펙)/§5.2(버티포트 coarse 캐스케이드)/§5.5(색 항상성)/§7.9(관측성) · `vision/CLAUDE.md`(테스트 규칙표) · 메모리 `project_rpi5_ubuntu_camera_stack.md`(카메라 브링업 전체 경과·진단명령·재현법)
