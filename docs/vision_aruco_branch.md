---
doc_type: orchestrator_brief
scope: nominal intrinsics 산출 + ArUco 브랜치 착수 — TargetEstimate 출력 계약 확정까지 (vision_plan.md §9 1·4번)
status: ✅ 완료 (2026-07-24, Phase 1~4 전부 완료 — 브랜치 종료. `vision/CLAUDE.md` "ArUco Phase 4 파이프라인 배선" 절·`docs/vision_status.md` 트랙 보드 참조)
created: 2026-07-24
last_updated: 2026-07-24
---

# ArUco 브랜치 오케스트레이터 브리프

> **다음 세션 진입:** "너는 오케스트레이터이다"로 시작하고 이 문서 하나만 읽으면 된다.
> `docs/vision_status.md`(트랙 보드)·`docs/vision_plan.md`는 필요 섹션만 열되, 이 작업의 지시는
> 여기에 자기완결적으로 있다.
> 프로토콜은 메모리 `feedback_orchestrator_protocol` 준수 — **각 Phase는 fg가 아니면 bg, 세션
> 자기보고는 직접 재현 검증 필수, 진행상황 확인 없이 한 프롬프트로 몰아던지지 말 것.**
> ~~**서브에이전트는 소네트로 생성한다**~~ **[2026-07-25 폐기]** 사용자가 Claude Max를 구독해
> 사용한도 제약이 사라졌다. **서브에이전트 model/effort는 지정하지 않는 것이 기본값**이고,
> 사용자가 특정 모델을 요청하면 그때 그대로 따른다.

---

## 0. 왜 이 작업인가 (한 문단)

2026-07-24, 2차예선 마감임박으로 체커보드 실측 카메라 캘리브레이션을 보류하기로 결정했다(메모리
`project_vision_calibration_deferred`, `vision_plan.md` §9 변경 이력 참조). 그 결과 §9 빌드순서
1~7번은 **nominal(데이터시트/실측 HFOV 근사) intrinsics**로 진행 가능해졌고, 지금까지 미착수였던
**4번 — ArUco 브랜치 → `TargetEstimate` 출력 계약 확정**이 다음 착수 대상이다. 코드베이스 확인
결과 **그린필드**: `vision/modules/`·`vision/core/state.py`에 ArUco/solvePnP/`TargetEstimate`/
calibration 관련 코드가 전혀 없다(`vision/calibration/` 디렉터리 자체가 없음). **RPi 실기체 접근
불필요** — 합성 이미지로 전부 검증 가능, 노트북/WSL `.venv`에서 착수한다.

---

## 1. 확정 전제 (재논의 불필요)

- **타겟:** `cv2.aruco.DICT_4X4_50`, **ID=23**, 물리 크기 **50cm×50cm**(`vision_plan.md` §2/§4.1a 확정).
- **좌표계:** 카메라 광학 프레임 기준으로 출력(잠정 확정, §7.1). body-center 정렬은 카메라 마운트
  확정 후 재검토 대상 — **이번 작업 범위 아님.**
- **intrinsics:** 실측 아님, **nominal**(아래 Phase 1에서 산출). HFOV=75°를 쓰되, 대각/수평 불일치
  미해결(`docs/vision_camera_bringup.md` "미해결로 남긴 판단" 절) — **수평으로 가정하고 진행**,
  yaml 아티팩트에 이 가정을 명시적으로 적어둔다(실측 캘리브레이션 재개 시 검정 대상).
- **센서 해상도:** main 4608×2592 (`calib_capture.py` 확정 스펙, 모드 전환 없음 원칙 유지).
- **카메라:** CAM109-IMX708AF-75 (서드파티 IMX708 클론, 정품 CM3 아님 — 메모리
  `project_vision_dev_env`).
- **`TargetEstimate`은 코어가 뱉는 유일한 중립 dataclass**(§7.1): 상대 pose + 신뢰도 + 타입 +
  frame_id + timestamp. **단위·부호 규약, 불확실성 필드 포함 여부는 여전히 미정** — 이번 작업이
  그 결정을 실제로 내려야 한다(Phase 3, 아래 "리스크/판정" 참조).
- **이번 작업은 §9 8번(폐루프 30cm 검증)이 아니다.** nominal intrinsics로 나온 pose는
  "근사치/미검증" 플래그를 달고 흐른다 — 그 플래그를 지우거나 실제 폐루프 튜닝에 쓰려는 시도는
  scope 밖(실측 캘리브레이션 재개 후에나 유효).

---

## 2. Phase별 실행 (각 Phase 끝에 보고 게이트)

### Phase 1 — nominal intrinsics 산출
- 신규 스크립트(예: `vision/tools/compute_nominal_intrinsics.py`, 하드웨어 비의존 — `calib_analyze.py`
  와 같은 예외로 `.venv` 대상) — 인자: sensor W/H px, HFOV_deg, camera_id.
- 공식: `fx=(W/2)/tan(HFOV_h/2)`, `fy=fx`(정사각 픽셀 가정), `cx=W/2`, `cy=H/2`,
  `distCoeffs=[0,0,0,0,0]`.
- 출력: `vision/calibration/<camera_id>/nominal.yaml` — `camera_matrix`/`dist_coeffs` +
  `source: "nominal_datasheet"` + `accuracy: "unverified"` + `not_for_closed_loop_30cm: true` +
  `hfov_assumption`(대각/수평 중 어느 걸로 가정했는지 값과 함께 명시).
- 단위테스트: 알려진 HFOV로 계산 후 역산(atan2)해 원래 HFOV가 복원되는지 · yaml 스키마 round-trip
  (`calib_analyze.py`의 yaml 왕복 테스트 패턴 재사용).
- **→ 여기까지만 하고 보고.**

### Phase 2 — ArUco 디코드 모듈
- `vision/modules/aruco.py`(가칭) — `cv2.aruco.ArucoDetector(DICT_4X4_50)` 래핑, 기존 파이프라인
  step 계약(`VisionState` 입출력)에 맞춘다. 기존 `Detection` dataclass에 이미 `corners` 필드가
  있음(`vision/core/state.py`) — 재사용할지 신규 필드가 필요한지 검토 후 결정.
- **ID 화이트리스트(23만 통과)** — 다른 ID 검출 시 후보에서 제거하고 §7.4 "거절 이유"로 로깅
  (기존 빨강/초록 검출기의 거절 로깅 패턴 재사용, 새 패턴 발명 금지).
- 단위테스트: `cv2.aruco`로 합성 생성한 마커 이미지로 디코드 성공/실패, 다른 ID 오검출 필터링.
- **→ 보고.**

### Phase 3 — solvePnP + `TargetEstimate` 출력 계약
- `TargetEstimate` dataclass 신설(`vision/core/state.py` 확장 또는 신규 `vision/core/target.py` —
  기존 파일 구조 감안해 결정) — 필드: position(x,y,z) + orientation(rvec 또는 quat) + confidence +
  target_type + frame_id + timestamp + **불확실성 필드 포함 여부**(§7.1 "미정" 항목, 이번에 확정).
- `cv2.solvePnP(objectPoints=50cm 정사각 4코너, imagePoints=디코드된 코너, nominal camera_matrix,
  dist_coeffs)` → rvec/tvec → `TargetEstimate`.
- **단위·부호 규약 확정**(§7.1 "여전히 미정" 항목 — 이번 세션이 실제로 결정해야 함). **오케스트레이터가
  임의로 정하지 말고, 트레이드오프가 있는 선택이면 사용자에게 먼저 확인** — 프로토콜 "정지조건"의
  "실질적 전략 결정"에 해당.
- 단위테스트: **합성 왕복** — 알려진 실제 pose로 투영한 마커 이미지 → solvePnP 복원 pose가 원래
  pose와 허용오차 내 일치(`calib_analyze.py`의 "진짜 K/dist를 알고 합성 투영 후 복원 검증" 패턴을
  그대로 재사용 — 새 검증 방법론 발명 금지).
- provenance echo(§7.3)에 사용한 calib_id(`nominal.yaml` 경로/해시)가 실제로 찍히는지 확인.
- **→ 보고.**

### Phase 4 — 파이프라인 통합 ✅ 완료(2026-07-24)
- `vision/main.py`/`replay.py`에 ArUco 모듈을 fine 단계로 연결 — coarse(§5.2 버티포트 3단 캐스케이드,
  이미 있음)에 이어붙이는 신규 preset(예: `presets/vertiport_fine.yaml`) 또는 기존
  `vertiport_coarse.yaml` 확장 중 선택.
- JSONL 블랙박스에 `TargetEstimate` 필드가 실제로 실리는지 확인(§7.4).
- **`pytest vision/tests/` 전체 통과 확인.**
- **→ 보고, 여기서 완료.**

**완료 기록:** 신규 `presets/vertiport_fine.yaml`(`vertiport_coarse.yaml`과 완전 독립 실행 — 근거는
그 yaml 헤더 주석·`vision/CLAUDE.md` "ArUco Phase 4 파이프라인 배선" 절). solvePnP 배선은
`main.py`/`replay.py` 레벨(신규 `modules/` 모듈 아님 — import 규칙 판단 근거도 같은 절에 기록).
`TargetEstimate`는 JSONL `chosen.target_estimate`에 실린다(`blackbox.log_frame()` 시그니처
불변). `python -m vision.replay <합성 ArUco 프레임 폴더> --preset vision/presets/vertiport_fine.yaml`
실행으로 실제 JSONL에 position/orientation/calib_accuracy/not_for_closed_loop_30cm/calib_id가
찍히는 것을 확인. `pytest vision/tests/` 319 passed(Phase 3 완료 시점 310 + 9). 이로써 ArUco
브랜치(Phase 1~4) 전체가 종료됐다.

---

## 3. 리스크 / 판정

- nominal intrinsics 정확도가 나빠도 ArUco solvePnP(코너 4점 PnP)는 상대적으로 강건해 발산 가능성은
  낮다. 다만 결과 pose의 **절대 정밀도는 "미검증"이 사실** — `TargetEstimate`의 accuracy 플래그가
  하위 소비자(향후 offboard 통합)에게 이 사실을 정확히 전달해야 한다. 플래그를 지우거나 폐루프에
  바로 쓰려는 판단은 이번 스코프 밖.
- **좌표계/단위/불확실성 필드 확정(§7.1)은 실질적 트레이드오프가 있는 결정이다** — 오케스트레이터가
  여기서 막히면 진행 전에 사용자에게 확인할 것(임의로 정하고 진행하지 말 것).
- HFOV 대각/수평 가정 오류가 있으면 nominal fx가 최대 수십 % 틀릴 수 있음 — 그래도 구조적 작업
  (ID 디코드·좌표계약·출력 스키마·상태머신 배선)은 fx 정확도와 무관하게 유효하므로 이번 브랜치를
  막는 요인은 아니다.

---

## 4. 참조

- `docs/vision_plan.md` §2(타겟 스펙)/§4.1a(ArUco 실측)/§7.1(좌표계 계약, 미정 항목)/§9(빌드순서,
  2026-07-24 재배치)
- `docs/vision_status.md` 트랙 보드 — "다음" 목록 1·2번이 이 브리프의 Phase 1·2에 대응
- 메모리 `project_vision_calibration_deferred`(왜 nominal인지 결정 배경) · `project_rpi5_ubuntu_camera_stack`(카메라 스펙)
- 참고 패턴: `vision/tools/calib_analyze.py`의 "합성 왕복 테스트" 방법론(진짜 K/dist를 알고 합성
  투영 후 복원 검증) — Phase 1·3 단위테스트에 그대로 재사용할 것, 새 검증 방법론 발명 금지.
