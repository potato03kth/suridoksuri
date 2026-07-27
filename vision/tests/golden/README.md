# 골든셋 — 재생 회귀 테스트 픽스처

`docs/vision_plan.md` §7.5("골든셋 회귀 테스트") / §7.9 "지금 당장 할 일" 7번.

## 지금 상태: 스캐폴드다 (최종본 아님)

**이 아래 프레임은 전부 합성(synthetic)이다 — 실촬영 데이터가 아니다.**
실기체/실카메라 작업이 이번(2026-07-21) 세션 기준 사용자 승인 대기 중이라(`docs/vision_status.md`
"RPi 작업 허가 대기 중"), 실제 라벨링 데이터셋은 카메라 브링업 이후에나 만들 수 있다. 그 전까지
이 폴더는 **폴더 구조/스키마가 실제로 동작함을 증명**하는 용도다. 타겟 종류 수(2종 + 오탐방지
1종)·고도 티어 수(3단)는 "합리적 최소값"이지 규정된 개수가 아니다 — 늘리거나 줄여도 된다.

## 폴더 구조

```
vision/tests/golden/
  <타겟종류>/
    <고도라벨>/           # 고도가 의미 없는 타겟(no_target)은 이 계층 생략
      frame_000.png        # (여러 프레임을 넣고 싶으면 frame_001.png ... 추가하고
      labels.json           #  labels.json의 "frames" 리스트에 같은 순서로 추가)
```

- `no_target/`처럼 고도 계층이 없는 타겟은, 같은 프레임을 다른 프리셋으로도 회귀검증하고
  싶을 때 `<타겟>/<프리셋변형>/` 형태로 하위 리프를 추가할 수 있다(예: `no_target/distress_coarse/`
  — vertiport_coarse.yaml은 루트 리프, distress_coarse.yaml은 이 하위 리프). 스키마 변경 없음 —
  기존 "리프 디렉터리 하나당 labels.json 하나" 규칙 그대로, 계층 이름만 고도 대신 프리셋 변형을
  가리킬 뿐이다.
- `frame_NNN.png` — `DirFrameSource`가 그대로 읽는 프레임 파일(파일명 정렬 순서 = frame_id,
  `vision/utils/frame_source.py` 참조). 이 폴더에 `telemetry.jsonl`을 추가하면 DirFrameSource가
  자동으로 붙인다(현재 골든셋엔 없음 — 고도/자세 텔레메트리 검증은 `test_replay.py`가 이미
  별도로 커버하고 있어 중복 없음).
- `labels.json` — 이 리프 디렉터리 전체에 적용되는 기대값. 스키마:
  ```json
  {
    "target": "vertiport",
    "altitude_label": "사람이 읽는 설명(스키마 자리표시자 여부 명시)",
    "preset": "vertiport_coarse.yaml",
    "note": "이 케이스가 왜 이런 기대값을 갖는지 설명",
    "frames": [
      {
        "file": "frame_000.png",
        "expect_num_detections": 1,
        "expect_stage_meta": {"white_field": {"candidates": 1}, "...": "..."},
        "known_limitation": false
      }
    ]
  }
  ```
  `expect_stage_meta`는 `null`이면 스테이지별 검증을 생략(전체 detection 개수만 검사) —
  `no_target`처럼 캐스케이드 내부 단계를 특정하지 않는 케이스용. `known_limitation: true`는
  "현재 파이프라인의 실제 동작"이지 "바람직한 동작"이 아님을 표시 — 회귀 테스트는 이 값을
  그대로 assert하되, 검출기를 몰래 고쳐서 통과시키면 안 된다(vision/CLAUDE.md 공통 규칙).

## 합성 카메라 캘리브레이션 픽스처 (`distress/synthetic_calib/`, 2026-07-28 신설)

`labels.json`을 갖지 않으므로 회귀 리프로 잡히지 않는다(`test_golden_regression.py`는
`rglob("labels.json")`으로 리프를 찾는다) — **초록구역 상대 pose 테스트 전용 보조 픽스처**다.

| 파일 | 대상 리프 | 용도 |
|---|---|---|
| `canvas460.yaml` | `distress/10m`, `distress/fine` | 460x460 캔버스 |
| `canvas320.yaml` | `distress/20m` | 320x320 캔버스 |
| `canvas200.yaml` | `distress/40m` | 200x200 캔버스 |

🔴 **실장착 `vision/calibration/cam109-imx708af75/nominal.yaml`을 골든 프레임에 그대로 쓰면
안 된다.** 그건 4608x2592 센서 기준이고 `solvePnP`의 focal은 픽셀 단위라, 460px 캔버스에 쓰면
초점거리도 주점도 안 맞아 거리가 정확히 3.00배(=4608/1536) 나온다(실측 확인).

**값의 출처(지어낸 값 아님):** `generate_synthetic.py`는 카메라 모델로 렌더링하지 않고
`presets/distress_coarse.yaml` 헤더 GSD 표의 픽셀 크기를 그대로 그린다. 그 표의 전제
(HFOV 75°, 폭 **1536px** 다운스케일 프레임)를 만족하는 초점거리는 하나뿐이다 —
`fx = fy = 1536/(2·tan(75°/2)) = 1000.877086 px`. 검산: 3.0m 매트가 10m에서 300.3px(골든
10m 리프가 실제로 300px). 캔버스는 그 프레임을 매트 중심 기준으로 **크롭**한 것으로 해석하며
(크롭은 focal을 바꾸지 않는다) 주점 = 캔버스 중심이다. 각 yaml 헤더에 같은 도출이 적혀 있다.

⚠️ `accuracy: unverified` / `not_for_closed_loop_30cm: true`는 **일부러 nominal.yaml과 같은
보수적 값**을 유지한다 — 합성 카메라라고 provenance 플래그를 해금하면 이 픽스처로 돌린 테스트가
"폐루프 30cm 가능"이라는 거짓 신호를 흘리게 된다.

## 현재 들어있는 것 (2026-07-25 갱신, 최초 2026-07-21c)

| 타겟 | 고도 티어 | preset | 근거 |
|---|---|---|---|
| `vertiport`(①) | 10m/20m/40m | `vertiport_coarse.yaml` | 3단 캐스케이드(white_field→black_v→red_ring) 전체 exercise. 10m/20m은 3단 전부 확인, 40m은 **알려진 한계**(white_field만 후보 내고 black_v 형상매칭 탈락 → 최종 0건, 저해상 스케일에서 고정 `kernel_size=5` morphology가 원인으로 보임) — `generate_synthetic.py` docstring 참조 |
| `distress`(②) | 10m/20m/40m | `distress_coarse.yaml`(전용 검출 모듈 없이 기존 `ColorFilter`+`RectDetector` 조합, 신규 검출 로직 아님) | 초록 매트(3.0m×3.0m×0.105m 라이즈드 플랫폼, 실측)+흰 박스. **[2026-07-22]** 매트 한 변 px은 실측 스펙 + 실측 화각 75°로 역산한 계산값(`generate_synthetic.py`/`distress_coarse.yaml` 헤더 주석/`vision/CLAUDE.md` 참조, 더 이상 임의 placeholder 아님). 10m(~90,000px²)/20m(~22,500px²)는 검출, 40m(~5,625px²)는 `min_area`(8000, 안전마진 반영) 미만인 **물리적으로 타당한** 원거리 미검출 케이스. 이 세 리프의 흰 박스는 `box_ratio` 기본값(0.22, 임의 시각용)으로 그려져 있으나 `distress_coarse.yaml`은 박스를 검증하지 않으므로(전용 fine 모듈이 없던 시절 스캐폴드) 배경 요소일 뿐이다 |
| `distress/fine`(②, **[2026-07-25 신설]**) | ~10m(fine 대역 ≤~15m 내 대표값) | `distress_fine.yaml`(`distress_coarse.yaml` 뒤에 `white_box_detector` 캐스케이드 — §9 "끊어진 체인을 잇는 작업") | 매트는 위 10m 리프와 동일 물리 크기(~90,000px², 같은 고도라 당연히 같음)지만 흰 박스를 **실측 비율 그대로**(20cm/3.0m≈0.0667 선형, 면적비≈0.00444, `vision_plan.md` §2) 그려 `white_box_detector`가 실제로 확정하고 `detection.meta["white_box_detector"]["landing_point_px"]`(박스 옆 착륙점, §5.3 "박스 옆 빈 초록면")를 싣는지까지 실제 재생 경로로 검증한다 |
| `no_target` | — (④ 단순착륙과 동일 조건, §5.6) | `vertiport_coarse.yaml` (루트) + `distress_coarse.yaml` (`no_target/distress_coarse/`) + `distress_fine.yaml` (`no_target/distress_fine/`, **[2026-07-25 신설]**) | 피듀셜 없는 평지에서 가장 정교한 캐스케이드(vertiport)도, 색 기준이 전혀 다른 조난자 coarse(초록 HSV) 필터도, 그 위에 얹은 `white_box_detector` 캐스케이드도 오탐하지 않는지. 세 프리셋은 필터/캐스케이드 기준이 서로 달라 한쪽만 회귀검증하면 다른 쪽 오탐을 놓칠 수 있어 리프를 분리했다 |

**빠진 것 (의도적):** ③ 하기구역(빨간 십자)은 전용 형상판별 검출기가 아직 없어(규정도 비공개
확정 상태, §5.4) 골든셋에서 제외 — 제네릭 rect_detector로 "십자"라고 우기면 허위 검증이 된다.
④ 단순착륙은 애초에 비전 검출 대상이 아니다(§5.6) — 대신 `no_target`이 관련 리스크(오탐 방지)를
간접적으로 커버한다.

**참고:** `no_target/distress_coarse/`는 이 스크립트가 재생성하지 않는다(2026-07-21 감사 세션에서
별도로 추가된 리프로 추정 — 이번 세션에서 발견된 기존 갭이며 이번 세션 범위 밖이라 그대로 둠).
반면 이번 세션에서 새로 추가한 `distress/fine/`과 `no_target/distress_fine/`은 둘 다 아래
재생성 명령으로 완전히 재현된다.

## 재생성

`vision/tests/golden/generate_synthetic.py`가 위 표의 모든 프레임을 만드는 소스다.
타겟 스펙이 바뀌거나 티어를 늘리고 싶으면 이 스크립트를 고치고 다시 실행:

```bash
source .venv/bin/activate
python vision/tests/golden/generate_synthetic.py
pytest vision/tests/test_golden_regression.py -v   # 새 labels.json 기대값이 실제와 맞는지 확인
```

## 실기체 데이터가 들어오면

1. 이 스크립트가 만든 합성 `frame_000.png`를 실촬영 프레임으로 교체(같은 디렉터리, 같은
   파일명 규칙 — 여러 장이면 `frame_001.png`... 추가).
2. `labels.json`의 `expect_num_detections`/`expect_stage_meta`를 실촬영 프레임에서 실제
   파이프라인을 돌려본 값으로 갱신(사람이 눈으로 확인한 "정답"이어야 함 — 파이프라인 출력을
   그대로 베끼면 회귀 테스트가 항상 통과하는 무의미한 테스트가 된다).
3. `altitude_label`에서 "스키마 자리표시자" 문구 제거하고 실제 촬영 고도로 교체.
4. ③ 하기구역용 폴더 + 전용 형상판별 검출기가 생기면 이 README의 "빠진 것" 표를 갱신.
5. 필요하면 `telemetry.jsonl`도 같은 디렉터리에 추가(`vision/utils/frame_source.py`
   `_load_telemetry_jsonl` 포맷 — `{"frame_id":..,"ts":..,"alt":..,"attitude":{...}}` 라인별).

## 왜 replay.py를 통해서 검증하나

`vision/tests/test_golden_regression.py`는 `vision.replay.run_replay()`를 실제로 호출해
`DirFrameSource`로 이 폴더를 읽고 실제 `Pipeline`을 돌려 JSONL 블랙박스에 찍힌 진짜 검출
결과를 `labels.json`과 비교한다(파이프라인 몽키패치 없음). 이렇게 하는 이유: 골든셋의
존재 이유가 "재생 경로 전체(FrameSource→Pipeline→블랙박스)가 앞으로도 계속 같은 입력에
같은 출력을 내는가"를 지키는 것이므로, 재생 CLI를 건너뛰고 `Pipeline`만 직접 부르면
`replay.py`/`frame_source.py` 자체가 깨지는 회귀는 못 잡는다. 캐스케이드 단계별 세부
meta(§5.2)까지 보고 싶은 두 번째 테스트만 보조적으로 `Pipeline.run()`을 직접 호출한다
(이것도 몽키패치가 아니라 replay.py가 내부적으로 쓰는 것과 동일한 실제 호출).
