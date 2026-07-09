---
doc_type: session_status
project: suridoksuri-1
scope: vision 세션 유일 진입점 — 트랙 보드 + 설계 포인터
last_updated: 2026-07-09
---

# vision 세션 진입 상태 문서

> **새 세션 진입:** 아래 vision 트랙 블록을 읽고, 설계 상세는 `docs/vision_plan.md`의 **필요 섹션만** 연다.
> FC 트랙은 `docs/session_status.md`에 별도로 있으며 **vision 세션은 그걸 읽지 않는다**(도메인 간 컨텍스트 격리).
> `/session-log`는 vision을 건드린 세션에서 **이 문서**의 트랙 블록을 갱신한다 (서술 로그는 공용 `docs/session_log.md`).

---

## 공통 상태 (2026-07-09)

- **브랜치:** `dev--vision-computing-module` (현재 FC 트랙과 공용. vision 전용 브랜치 분리는 미결정)
- **커밋 규율:** vision 커밋은 메시지에 **`[vision]`** 태그
- **설계 정본:** `docs/vision_plan.md` — 확정 결정·물리 제약·검출 전략·변경내성/관측성·빌드 순서·블라인드스팟. **이 문서는 라이브 진척만** 담는다.
- **대회 규정 대기(열린 항목):** ArUco 딕셔너리/ID · ③빨간십자 규정 · 초록 색·치수 스펙 · 성공 판정·재시도 · CC 인터페이스 → 상세 `vision_plan.md` §10.

---

## 트랙 보드

### 👁 vision-정밀착륙 — ▶ 활성 (계획 완료, 코드 착수 전)

- **내용:** 착륙지점 인식·정밀착륙 시스템(RPi5 온보드). 고전 CV, 타겟별 coarse→fine 2단, 비전 폐루프 <30cm. 설계 정본 `docs/vision_plan.md`.
- **마지막:** 2026-07-09 — **착수 전 컨설팅 완료.** 큰 결정 전부 확정(카메라 RPi Cam Module 3 Wide 표준 IR-cut 컬러 롤링셔터·나디르 하드마운트+고무댐핑·GPS접근+비전폐루프30cm·독립 ROS2 노드+offboard 정밀착륙 서브상태·출력=상대 pose `TargetEstimate`·변경내성/관측성 1급). 계획서(`docs/vision_plan.md`) 작성 + 루트 CLAUDE.md 의존(`vision→fc_ros`) 기록 + 별도 트랙 분리. **미커밋.**
- **다음:** ① 카메라 인트린식+왜곡 캘리브레이션(102° 광각, 없으면 pose 거짓) ② 실기체 데이터 수집(고도별 3타겟) ③ **관측성 골격 먼저** — `vision/main.py` headless-safe 수정 + 구조적 로깅/JSONL 스캐폴딩(이후 모든 개발이 여기 얹힘)
- **주의:** 대회 상세규정 대기(`vision_plan.md` §10) · 카메라/Pi4 인코더/라이다 40m급 **미확정**(변경 흡수 어댑터로 설계) · 기존 `vision/` 틀은 폐기 아님 — ②조난자 구역 색 파이프라인 + ArUco 모듈 컨테이너로 재편(§12) · `geo_project.pixel_to_gps`는 폐기 예정
- **참조:** `docs/vision_plan.md` · `vision/CLAUDE.md`
