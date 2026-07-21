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
- **개발/테스트 환경:** **메인 개발·디버깅 주체 = 노트북으로 전환** (2026-07-21) — 비행 현장에서 실시간 피드백 가능. 노트북 `.venv` 설치 상태는 다음 세션에서 확인. 개발컴 `.venv`는 기존 준비 이력 유지(`pytest vision/tests/` **16 passed**). 그 wsl·rpi 미설치(rpi=headless). **RPi에 Camera Module 3 물리 장착 완료**(2026-07-21) — 카메라 인트린식/왜곡 캘리브레이션 착수 가능. 4환경 매트릭스·설치법 → 메모리 `project_vision_dev_env.md` / `vision/requirements.txt` / `vision/CLAUDE.md` 테스트 섹션
- **커밋 규율:** vision 커밋은 메시지에 **`[vision]`** 태그
- **설계 정본:** `docs/vision_plan.md` — 확정 결정·물리 제약·검출 전략·변경내성/관측성·빌드 순서·블라인드스팟. **이 문서는 라이브 진척만** 담는다.
- **대회 규정(2026-07-21 대부분 확정):** ArUco=`DICT_4X4_50` ID23 100mm(확정) · ③빨간십자·초록색/치수는 **비공개로 확정**(버티포트 유사크기+안전마진 가정) · 성공판정="매끄럽게 보이면"(정성) 확정 · CC 인터페이스·나무조각상 판단기준만 여전히 대기 → 상세 `vision_plan.md` §10.

---

## 트랙 보드

### 👁 vision-정밀착륙 — ▶ 활성 (관측성 골격 착수 / 규정 확정 반영)

- **내용:** 착륙지점 인식·정밀착륙 시스템(RPi5 온보드). 고전 CV, 타겟별 coarse→fine 2단, 비전 폐루프 <30cm. 설계 정본 `docs/vision_plan.md`.
- **마지막:** 2026-07-21 — **종합점검 + 대회규정 다수 확정 반영.**
  - 확정: ArUco `DICT_4X4_50`/ID23/100mm, ③빨간십자·초록색 스펙은 비공개 확정(버티포트 유사크기+안전마진 가정), 성공판정 정성기준("매끄럽게"), 카메라 CM3 정식채용 방향(OIS 실증 조건부), TargetEstimate 프레임 잠정 광학기준, Pi4 인코더는 "일단 Pi5로, 필요시 전환" 방침.
  - **⚠️ 중요 발견:** ArUco 실제 크기(100mm)가 기존 가정(50cm)의 1/5 → GSD 재계산 결과 **fine-lock 가능 고도가 ~20m→~5m로 대폭 축소**(`vision_plan.md` §4.1a). 실기체 데이터로 최우선 검증 필요.
  - 미해결: ④단순착륙 전략(§5.6, 사용자에게 항목 설명 제공 후 결정 대기), CC 인터페이스, 나무조각상 판정기준.
  - (이전 진척: 2026-07-15 계획서 갭 반영 + `main.py` headless-safe. 커밋 `af32ccf`. 2026-07-21 개발주체 노트북 전환 + RPi CM3 물리장착. 커밋 `84850b9`.)
- **다음:** ① **[최우선, 신규] §4.1a fine-lock 고도(~5m) 실측 검증** — 실기체 데이터 수집 시 최우선 확인 ② **미커버 단위 테스트** — `color` HSV 초록/빨강 우선 + edge/morphology/fusion 등(`vision/CLAUDE.md` 표) ③ **관측성 골격 §7.9 다음** — 이중싱크 로거 + provenance 헤더 ④ 카메라 인트린식+왜곡 캘리브레이션 + 실기체 3타겟 데이터(CM3 장착 완료로 착수 가능) ⑤ ④단순착륙 전략 — 사용자 결정 대기
- **주의:** 노트북(WSL) 개발환경 세팅 진행 중(`python3.12-venv` 설치가 sudo 필요 — 사용자 실행 대기) · 개발컴 `.venv`는 기존 이력 유지, rpi 미설치(headless) · **버티포트 표준파일(로컬 카톡캡처) 측정상 빨간 원이 채워진 disc가 아니라 얇은 고리로 보임 — 로고/도면 여부 불확실, 미확정 관찰**(`vision_plan.md` §5.2) · Pi4 인코더/라이다 40m급 여전히 미확정 · 기존 `vision/` 틀은 폐기 아님 — ②조난자 구역 색 파이프라인 + ArUco 모듈 컨테이너로 재편(§12) · `geo_project.pixel_to_gps`는 폐기 예정
- **참조:** `docs/vision_plan.md` §2/§4.1a/§5.2/§7.1/§7.7/§10 · `vision/CLAUDE.md`
