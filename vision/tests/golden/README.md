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

## 현재 들어있는 것 (2026-07-21c)

| 타겟 | 고도 티어 | preset | 근거 |
|---|---|---|---|
| `vertiport`(①) | 10m/20m/40m | `vertiport_coarse.yaml` | 3단 캐스케이드(white_field→black_v→red_ring) 전체 exercise. 10m/20m은 3단 전부 확인, 40m은 **알려진 한계**(white_field만 후보 내고 black_v 형상매칭 탈락 → 최종 0건, 저해상 스케일에서 고정 `kernel_size=5` morphology가 원인으로 보임) — `generate_synthetic.py` docstring 참조 |
| `distress`(②) | 10m/20m/40m | `distress_coarse.yaml`(이번 세션 신규 — 전용 검출 모듈 없이 기존 `ColorFilter`+`RectDetector` 조합, 신규 검출 로직 아님) | 초록 매트+흰 박스. 40m은 매트 픽셀 면적이 `min_area`(300) 미만이 되는 **물리적으로 타당한** 원거리 미검출 케이스 |
| `no_target` | — (④ 단순착륙과 동일 조건, §5.6) | `vertiport_coarse.yaml` | 피듀셜 없는 평지에서 가장 정교한 캐스케이드조차 오탐하지 않는지 |

**빠진 것 (의도적):** ③ 하기구역(빨간 십자)은 전용 형상판별 검출기가 아직 없어(규정도 비공개
확정 상태, §5.4) 골든셋에서 제외 — 제네릭 rect_detector로 "십자"라고 우기면 허위 검증이 된다.
④ 단순착륙은 애초에 비전 검출 대상이 아니다(§5.6) — 대신 `no_target`이 관련 리스크(오탐 방지)를
간접적으로 커버한다.

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
