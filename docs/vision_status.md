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
- **개발/테스트 환경:** **메인 개발·디버깅 주체 = 노트북(WSL)** — 2026-07-21 **정식 `.venv` 준비 완료.** 사용자가 `sudo apt install -y python3.12-venv` 실행 → `python3 -m venv .venv`(저장소 루트) → `pip install -r vision/requirements.txt` → `pytest vision/tests/` **16 passed** 확인. (중간에 sudo가 안 돼 `pip3 --break-system-packages`로 시스템에 임시 설치했던 opencv-python은 venv 완성 후 제거함 — 잔재 없음.) 개발컴 `.venv`는 기존 준비 이력 유지. rpi 미설치(headless). **RPi에 Camera Module 3 물리 장착 완료**(2026-07-21) — 카메라 인트린식/왜곡 캘리브레이션 착수 가능. 4환경 매트릭스·설치법 → 메모리 `project_vision_dev_env.md` / `vision/requirements.txt` / `vision/CLAUDE.md` 테스트 섹션
- **커밋 규율:** vision 커밋은 메시지에 **`[vision]`** 태그
- **설계 정본:** `docs/vision_plan.md` — 확정 결정·물리 제약·검출 전략·변경내성/관측성·빌드 순서·블라인드스팟. **이 문서는 라이브 진척만** 담는다.
- **대회 규정(2026-07-21 대부분 확정):** ArUco=`DICT_4X4_50` ID23 **50cm×50cm**(원래 계획 가정과 일치 확인) · 버티포트 하기안전구역(빨간 원)=직경2m·선굵기5cm 고리 신규확정 · ③빨간십자·초록색/치수는 **비공개로 확정**(버티포트 유사크기+안전마진 가정) · 성공판정="매끄럽게 보이면"(정성) 확정 · ④단순착륙=GPS+라이다만 확정 · CC 인터페이스·나무조각상 판단기준만 여전히 대기 → 상세 `vision_plan.md` §10.

---

## 트랙 보드

### 👁 vision-정밀착륙 — ▶ 활성 (관측성 골격 착수 / 규정 확정 반영)

- **내용:** 착륙지점 인식·정밀착륙 시스템(RPi5 온보드). 고전 CV, 타겟별 coarse→fine 2단, 비전 폐루프 <30cm. 설계 정본 `docs/vision_plan.md`.
- **마지막:** 2026-07-21 — **종합점검 + 대회규정 다수 확정 반영.**
  - 확정: ArUco `DICT_4X4_50`/ID23/**50cm×50cm**(원 가정과 일치), 버티포트 하기안전구역=직경2m·5cm 고리, ③빨간십자·초록색 스펙 비공개 확정(버티포트 유사크기+안전마진 가정), 성공판정 정성기준("매끄럽게"), **④단순착륙=GPS+라이다만**(비전 개발범위 밖), 카메라 CM3 정식채용 방향(OIS 실증 조건부), TargetEstimate 프레임 잠정 광학기준, Pi4 인코더는 "일단 Pi5로, 필요시 전환" 방침.
  - (세션 중 한때 ArUco를 100mm로 오인해 "fine-lock 고도 ~5m로 축소" 경보를 냈으나 사용자 확인으로 정정 — **§4.1 GSD 표 원래대로 유효**, 새로운 문제 아님.)
  - **버티포트 실제 구조 확정(재정정):** 원 내부는 자갈이 아니라 **흰색 채워진 필드(직경3m)** — 검은 띠(테두리)+검은 V(solid, 실물 확인)+빨간 고리(직경2m·5cm)+ArUco(50cm) 순. coarse 전략을 3단 캐스케이드로 재설계(1차 흰필드 면적→2차 검은V 형상→3차 빨간고리 색게이팅+원피팅). §2/§5.2. 커밋 `116c3f6`.
  - 미해결: CC 인터페이스, 나무조각상 판정기준, 성능/지연 예산, TargetEstimate 단위·불확실성 필드, 검은띠 바깥 경계 처리(현수막 여분 등).
  - (이전 진척: 2026-07-15 계획서 갭 반영 + `main.py` headless-safe. 커밋 `af32ccf`. 2026-07-21 개발주체 노트북 전환 + RPi CM3 물리장착. 커밋 `84850b9`.)
- **다음:** ① **미커버 단위 테스트** — `color` HSV 초록/빨강 우선 + edge/morphology/fusion 등(`vision/CLAUDE.md` 표) ② **버티포트 coarse 캐스케이드 구현(§5.2)** — 흰필드(기존 gray-mode 재사용)→검은V 형상매칭(신규)→빨간고리 색게이팅+원피팅(신규) ③ **관측성 골격 §7.9 다음** — 이중싱크 로거 + provenance 헤더 ④ 카메라 인트린식+왜곡 캘리브레이션 + 실기체 3타겟 데이터(CM3 장착 완료로 착수 가능, OIS 실증도 이때 — **사용자의 물리 작업(체커보드 촬영) 필요**)
- **주의:** 노트북(WSL) `.venv` 준비 완료(4환경 매트릭스 중 개발컴에 이어 두 번째로 venv 준비됨) · rpi 미설치(headless) · Pi4 인코더/라이다 40m급 여전히 미확정 · 기존 `vision/` 틀은 폐기 아님 — ②조난자 구역 색 파이프라인 + ArUco 모듈 컨테이너로 재편(§12) · `geo_project.pixel_to_gps`는 폐기 예정
- **참조:** `docs/vision_plan.md` §2/§4.1a/§5.2/§5.6/§7.1/§7.7/§10 · `vision/CLAUDE.md`
