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
- **개발/테스트 환경:** 메인 개발주체 = **노트북(WSL)**, `.venv` 준비 완료(`source .venv/bin/activate`, 저장소 루트) — `pytest vision/tests/` 16 passed 확인됨. 개발컴 `.venv`도 기존대로 유효. rpi 미설치(headless). RPi에 **Camera Module 3 물리 장착 완료** — 캘리브레이션 착수 가능. 상세 → 메모리 `project_vision_dev_env.md`
- **커밋 규율:** vision 커밋은 메시지에 **`[vision]`** 태그
- **설계 정본:** `docs/vision_plan.md` — 확정 결정·물리 제약·검출 전략·변경내성/관측성·빌드 순서·블라인드스팟. **이 문서는 라이브 진척만** 담는다.
- **대회 규정(2026-07-21 대부분 확정):** ArUco=`DICT_4X4_50` ID23 **50cm×50cm**(원래 계획 가정과 일치 확인) · 버티포트 하기안전구역(빨간 원)=직경2m·선굵기5cm 고리 신규확정 · ③빨간십자·초록색/치수는 **비공개로 확정**(버티포트 유사크기+안전마진 가정) · 성공판정="매끄럽게 보이면"(정성) 확정 · ④단순착륙=GPS+라이다만 확정 · CC 인터페이스·나무조각상 판단기준만 여전히 대기 → 상세 `vision_plan.md` §10.

---

## 트랙 보드

### 👁 vision-정밀착륙 — ▶ 활성 (구현 착수: 단위테스트 → coarse 캐스케이드)

- **내용:** 착륙지점 인식·정밀착륙 시스템(RPi5 온보드). 고전 CV, 타겟별 coarse→fine 2단, 비전 폐루프 <30cm. 설계 정본 `docs/vision_plan.md`.
- **마지막(2026-07-21):** `ColorFilter(mode="color")` HSV 초록/빨강 단위테스트 추가 — `vision/tests/test_color.py`에 7개 신규(초록 캡처/거부/채도경계, 빨강 저/고 hue band, 빨강 단일range Hue랩어라운드 미지원 회귀테스트, meta) `pytest vision/tests/` 23 passed. `vision/CLAUDE.md` 테스트표 갱신. 커밋 예정.
- **다음 (이 순서로 진행, 사용자 확정):**
  1. **버티포트 coarse 캐스케이드 구현(§5.2)** — 흰필드(기존 `ColorFilter(mode="gray")` 재사용) → 검은V 형상매칭(신규) → 빨간고리 색게이팅+원피팅(신규)
  2. **관측성 골격 §7.9 다음 항목** — 이중싱크 로거 + JSONL + provenance 헤더
  3. **카메라 인트린식/왜곡 캘리브레이션** — CM3 장착 완료로 착수 가능하나 **체커보드 촬영은 사용자 물리 작업 필요**
- **주의:** rpi 미설치(headless) · Pi4 인코더/라이다 40m급 미확정 · 기존 `vision/` 틀은 폐기 아님(§12) · `geo_project.pixel_to_gps` 폐기 예정 · 세부 정정 이력·논의는 `docs/session_log.md` 참조
- **참조:** `docs/vision_plan.md` §2(타겟 스펙)/§5.2(버티포트 coarse 캐스케이드)/§5.5(색 항상성)/§7.9(관측성) · `vision/CLAUDE.md`(테스트 규칙표)
