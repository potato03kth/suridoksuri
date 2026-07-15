---
doc_type: session_status
project: suridoksuri-1
scope: vision 세션 유일 진입점 — 트랙 보드 + 설계 포인터
last_updated: 2026-07-15
---

# vision 세션 진입 상태 문서

> **새 세션 진입:** 아래 vision 트랙 블록을 읽고, 설계 상세는 `docs/vision_plan.md`의 **필요 섹션만** 연다.
> FC 트랙은 `docs/session_status.md`에 별도로 있으며 **vision 세션은 그걸 읽지 않는다**(도메인 간 컨텍스트 격리).
> `/session-log`는 vision을 건드린 세션에서 **이 문서**의 트랙 블록을 갱신한다 (서술 로그는 공용 `docs/session_log.md`).

---

## 공통 상태 (2026-07-15)

- **브랜치:** `dev--vision-computing-module` (현재 FC 트랙과 공용. vision 전용 브랜치 분리는 미결정)
- **개발/테스트 환경:** 개발컴 `.venv` 준비·`pytest vision/tests/` **16 passed**. 개발노트북·그 wsl·rpi 미설치(rpi=headless). 4환경 매트릭스·설치법 → 메모리 `project_vision_dev_env.md` / `vision/requirements.txt` / `vision/CLAUDE.md` 테스트 섹션
- **커밋 규율:** vision 커밋은 메시지에 **`[vision]`** 태그
- **설계 정본:** `docs/vision_plan.md` — 확정 결정·물리 제약·검출 전략·변경내성/관측성·빌드 순서·블라인드스팟. **이 문서는 라이브 진척만** 담는다.
- **대회 규정 대기(열린 항목):** ArUco 딕셔너리/ID · ③빨간십자 규정 · 초록 색·치수 스펙 · 성공 판정·재시도 · CC 인터페이스 → 상세 `vision_plan.md` §10.

---

## 트랙 보드

### 👁 vision-정밀착륙 — ▶ 활성 (관측성 골격 착수)

- **내용:** 착륙지점 인식·정밀착륙 시스템(RPi5 온보드). 고전 CV, 타겟별 coarse→fine 2단, 비전 폐루프 <30cm. 설계 정본 `docs/vision_plan.md`.
- **마지막:** 2026-07-15 — 계획서 갭 8건 반영(§2/§5.1/§5.4/§5.6/§7.1/§7.2/**§7.9 개발디버깅 워크플로 신설**/§10/§11) + `main.py` **headless-safe**(`--display {none|window|file|stream}`, 기본 none=GUI 미호출) + 테스트 규칙표(`vision/CLAUDE.md`)·`requirements.txt`·개발컴 `.venv` 정비. 실테스트 **16 passed**. 커밋 `af32ccf`.
- **다음:** ① **미커버 단위 테스트** — `color` HSV 초록/빨강 우선(정밀착륙 직결) + edge/morphology/fusion 등(대상·규칙 `vision/CLAUDE.md` 표) ② **관측성 골격 §7.9 다음** — 이중싱크 로거 + provenance 헤더(config+git해시+캘리브id) ③ (선행 대기) 카메라 인트린식+왜곡 캘리브레이션 + 실기체 3타겟 데이터
- **주의:** **개발컴만 `.venv` 준비** — 노트북·그 wsl·rpi 미설치(필요 단계에서 `requirements.txt`, rpi=headless) · 대회 상세규정 대기(`vision_plan.md` §10) · 카메라/Pi4 인코더/라이다 40m급 **미확정**(변경 흡수 어댑터로 설계) · 기존 `vision/` 틀은 폐기 아님 — ②조난자 구역 색 파이프라인 + ArUco 모듈 컨테이너로 재편(§12) · `geo_project.pixel_to_gps`는 폐기 예정
- **참조:** `docs/vision_plan.md` · `vision/CLAUDE.md`
