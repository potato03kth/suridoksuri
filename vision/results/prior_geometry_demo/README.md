# 사전정보 기반 후보 랭킹 — 종단간 데모 (2026-07-28)

`modules/prior_score.py`(`core/prior_geometry.py` 소비)가 **실제로 무엇을 바꾸는지**를
`python -m vision.replay` 정규 경로로 찍어 둔 재현 산출물이다.

> ⚠️ **합성 데이터다.** 실촬영 프레임은 저장소에 한 장도 없다(`docs/vision_status.md`).
> 다만 이 장면은 `core/prior_geometry.py::project_ground_square()`로 **진짜 원근 투영**해
> 그렸다 — 기존 골든셋(`tests/golden/`, 원근·왜곡 없는 축평행 사각형)과 달리 tilt가
> 물리적으로 일관된다.

---

## 장면

20m AGL, roll 4° / pitch 9°, `synthetic_calib/canvas460.yaml`(fx=fy=1000.877, 460x460).

| | 무엇 | 실측 크기 | 화면 면적 |
|---|---|---|---|
| 진짜 타겟 | ② 조난자 초록 매트 | **3.0m** | ~20,000px² |
| 오탐 | 같은 색·같은 정사각형·같은 원근, **크기만 다름** | **4.5m** | ~52,000px² |

둘 다 `distress_coarse.yaml`의 하드 필터(`min_area=8000` / `max_area=200000`)를 **통과**한다.
색·형상·원근이 전부 같아 **사전정보(고도+자세) 말고는 가를 방법이 없는** 오탐이다.
게다가 오탐이 더 커서 `RectDetector`의 confidence가 더 높다(0.995 vs 0.99) → **기본 순서에서
오탐이 1등**이다.

## 재현

```bash
# (1) 자세 있음 — 스코어러 활성
python -m vision.replay vision/results/prior_geometry_demo/recording \
    --preset vision/presets/distress_coarse.yaml \
    --calib vision/tests/golden/distress/synthetic_calib/canvas460.yaml \
    --log-dir vision/results/prior_geometry_demo/logs --log-name prior_demo

# (2) 자세 없음 — 스코어러 자동 비활성(degrade 실증)
python -m vision.replay vision/results/prior_geometry_demo/recording_no_attitude \
    --preset vision/presets/distress_coarse.yaml \
    --calib vision/tests/golden/distress/synthetic_calib/canvas460.yaml \
    --log-dir vision/results/prior_geometry_demo/logs --log-name prior_demo_no_attitude
```

녹화 폴더 자체는 `vision/tests/test_prior_geometry.py`의 `_scene_frames()`/`_write_recording()`
헬퍼로 만들었다(장면 정의가 데모와 회귀에서 갈리지 않도록 **같은 헬퍼를 재사용**).

---

## 결과 — `logs/prior_demo.jsonl` (자세 있음)

```json
"detections": [
  {"bbox": [47, 49, 146, 144], "confidence": 0.99,
   "prior": {"score": 0.999998, "area_score": 0.999997, "shape_score": 1.0,
             "observed_area_px2": 20015.0, "predicted_area_px2": 19983.518,
             "area_ratio": 1.001575, "observed_anisotropy": 1.014179,
             "max_anisotropy": 1.042186, "incidence_deg": 18.3088}},
  {"bbox": [217, 217, 233, 230], "confidence": 0.995,
   "prior": {"score": 0.55985, "area_score": 0.413134, "shape_score": 1.0,
             "observed_area_px2": 51872.0, "predicted_area_px2": 23106.419,
             "area_ratio": 2.244917, "observed_anisotropy": 1.001915,
             "max_anisotropy": 1.012334, "incidence_deg": 3.711}}
]
```

- **진짜 타겟이 1등**(0.99999 vs 0.55985). confidence 순서(오탐이 앞)가 뒤집혔다.
- 🔴 **오탐이 사라지지 않았다** — 0.56점을 달고 그대로 2등에 남아 있다. 이게 계약이다
  ("좋은 비행 시퀀스는 실패하지 않는 시퀀스" — 거절이 아니라 랭킹).

## 그리고 이것이 **FC로 나가는 pose**를 바꾼다

`main.py`/`replay.py`는 `meta["distress_mat"]`이 붙은 **첫** 검출로 `solvePnP`를 돌린다.
따라서 순위가 곧 `chosen.target_estimate`(= `--target-sink`로 기체에 나가는 좌표)다:

| | 착륙점 픽셀 | `position` z (슬랜트 깊이) | 판정 |
|---|---|---|---|
| 자세 있음 (`prior_demo.jsonl`) | `[120.25, 120.75]` = 진짜 타겟 | **20.83m** | 실제 20m AGL과 정합 ✅ |
| 자세 없음 (`prior_demo_no_attitude.jsonl`) | `[331.0, 331.75]` = 오탐 | **13.23m** | 4.5m 물체를 3.0m로 오인해 거리 **과소평가** ❌ |

거리를 13m로 착각한 채 폐루프에 들어가면 **조기 하강**이다. 랭킹이 막는 것이 바로 이것이다.

## degrade — `logs/prior_demo_no_attitude.jsonl` (자세 없음)

```json
"attitude": null,
"detections": [{"bbox": [217, 217, 233, 230], "confidence": 0.995},
               {"bbox": [47, 49, 146, 144], "confidence": 0.99}]
```

`prior` 키 자체가 없다 — 스코어러가 **자동 비활성**되어 정렬도 점수도 손대지 않았고, JSONL
형태가 이 기능 배선 **이전과 한 글자도 다르지 않다**. `main.py` 라이브 경로에는 아직
AGL/자세 역방향 채널이 없으므로 **현재 실기체 동작이 바로 이 상태**다.

---

## 이 데모가 증명하지 **못하는** 것

- **실촬영 검증은 0이다.** 색·조명·그림자·모션블러·윤곽 열화가 전혀 없는 합성 장면이다.
- 라이다 AGL 오차·고무마운트 자세 잔차(§4.2, **실측된 적 없음**)의 실제 크기를 모른다.
  허용폭(`DEFAULT_AREA_TOLERANCE_LOW/HIGH`, `residual_tolerance_widening()`)은 그 미지를
  넉넉히 잡은 값이지 실측 튜닝값이 아니다.
- 마운트 요각 ψ_m 미측정(`core/frames.py`). 화면 중앙 예측은 ψ_m에 둔감하지만
  (`ground_normal_cam()` docstring), 화면 가장자리 후보의 예측은 영향을 받는다.
