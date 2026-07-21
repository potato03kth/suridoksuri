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
- **개발/테스트 환경:** 메인 개발주체 = **노트북(WSL)**, `.venv` 준비 완료(`source .venv/bin/activate`, 저장소 루트) — `pytest vision/tests/` 47 passed 확인됨. 개발컴 `.venv`도 기존대로 유효. rpi 미설치(headless). RPi에 **Camera Module 3 물리 장착 완료** — 캘리브레이션 착수 가능. 상세 → 메모리 `project_vision_dev_env.md`
- **커밋 규율:** vision 커밋은 메시지에 **`[vision]`** 태그
- **설계 정본:** `docs/vision_plan.md` — 확정 결정·물리 제약·검출 전략·변경내성/관측성·빌드 순서·블라인드스팟. **이 문서는 라이브 진척만** 담는다.
- **대회 규정(2026-07-21 대부분 확정):** ArUco=`DICT_4X4_50` ID23 **50cm×50cm**(원래 계획 가정과 일치 확인) · 버티포트 하기안전구역(빨간 원)=직경2m·선굵기5cm 고리 신규확정 · ③빨간십자·초록색/치수는 **비공개로 확정**(버티포트 유사크기+안전마진 가정) · 성공판정="매끄럽게 보이면"(정성) 확정 · ④단순착륙=GPS+라이다만 확정 · CC 인터페이스·나무조각상 판단기준만 여전히 대기 → 상세 `vision_plan.md` §10.

---

## 트랙 보드

### 👁 vision-정밀착륙 — ▶ 활성 (구현 착수: 단위테스트 → coarse 캐스케이드)

- **내용:** 착륙지점 인식·정밀착륙 시스템(RPi5 온보드). 고전 CV, 타겟별 coarse→fine 2단, 비전 폐루프 <30cm. 설계 정본 `docs/vision_plan.md`.
- **마지막(2026-07-21):** 버티포트 coarse 3단 캐스케이드 구현 완료(§5.2) — 신규 모듈 `WhiteFieldDetector`(mask→원형 blob) / `BlackVMatcher`(matchShapes로 검은 V 형상검증, 1차 bbox 밖 배경 오탐 배제) / `RedRingDetector`(빨강 Hue 양끝 게이팅+최소외접원 피팅, red_ring_detector가 ColorFilter 단일range 한계를 자체 해결). `presets/vertiport_coarse.yaml`로 전체 파이프라인 조립, registry/`__init__` 등록. **설계 교훈:** `ColorFilter`가 `current`를 자기 mask로 지워버려 뒤 단계가 다른 색을 못 봄 → V/ring 검출기는 `current` 대신 `original`을 읽도록 설계(`vision/CLAUDE.md` 필드규칙에 반영). 단위 3개 + 캐스케이드 통합 1개, 총 `pytest vision/tests/` 47 passed. 커밋 예정.
- **다음 (이 순서로 진행, 사용자 확정):**
  1. **관측성 골격 §7.9 다음 항목** — 이중싱크 로거 + JSONL + provenance 헤더
  2. **카메라 인트린식/왜곡 캘리브레이션** — CM3 장착 완료로 착수 가능하나 **체커보드 촬영은 사용자 물리 작업 필요**
- **주의:** rpi 미설치(headless) · Pi4 인코더/라이다 40m급 미확정 · 기존 `vision/` 틀은 폐기 아님(§12) · `geo_project.pixel_to_gps` 폐기 예정 · **버티포트 V 형상매칭은 실물 규격 미확인 상태의 합성테스트로만 검증됨** — 실기체 데이터 확보 후 `BlackVMatcher` 참조 V 템플릿(두께/종횡비)·`max_match_distance` 재검증 필요 · 세부 정정 이력·논의는 `docs/session_log.md` 참조
- **참조:** `docs/vision_plan.md` §2(타겟 스펙)/§5.2(버티포트 coarse 캐스케이드)/§5.5(색 항상성)/§7.9(관측성) · `vision/CLAUDE.md`(테스트 규칙표)
