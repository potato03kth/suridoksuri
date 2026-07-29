# 착륙 판단 오버레이 — 화면 버퍼 증거 (2026-07-29)

2차예선 **보고 제출용 영상**을 준비하다 발견한 결함(아래)에 대한 대응
`utils/visualize.py::draw_landing_overlay` + `--report-overlay`가 실제로 화면에 그려짐을
저장된 프레임으로 남긴다. 헤드리스라 창(`--display window`)을 띄울 수 없어
`sink_overlay_demo/`와 같은 방식이다.

## 왜 만들었나 — 기존 오버레이로는 "인식 중"이 안 보였다

`draw_detections()`가 그리는 것은 **초록(0,200,0) 얇은 bbox + 신뢰도 숫자**뿐이다.
우선 타겟인 ② 조난자 구역은 **초록 매트**라 초록 선이 배경에 묻힌다. 게다가 파이프라인이
실제로 판단하는 값들 — **착륙점**(박스 옆 빈 초록면), 흰 박스, 상태머신 상태/명령,
추정 거리 — 은 전부 JSONL에만 있고 **화면에는 한 픽셀도 그려지지 않고 있었다.**

| 파일 | 내용 | 재현 명령 |
|---|---|---|
| `golden_fine_overlay.png` | **커밋된 골든셋만으로 완전 재현 가능**. 노랑=매트 bbox / 하늘색=흰 박스 / **마젠타 십자+원=착륙점**("박스 옆 빈 초록면", 매트 중심이 아니다) / 좌하단 패널=`FRAME`·`DET`·`STATE`·`CMD`·`TARGET`·`DIST`. `DIST 10.01 m`가 골든 라벨 10m와 일치 | 아래 ① |
| `fhd_clip_approach.png` | FHD(1920×1080) 접근 시퀀스 초반(고도 ~21m) — 실제 녹화 영상에서 오버레이가 어떻게 보이는지 | 아래 ② |
| `fhd_clip_near.png` | 같은 시퀀스 후반(고도 ~7.3m). 매트가 커지고 `STATE PRECISION_SERVO`로 진행 | 아래 ② |

### ① 골든셋 재현(완전 재현 가능)

```bash
python -m vision.main vision/tests/golden/distress/fine/frame_000.png \
  --preset vision/presets/distress_fine.yaml \
  --calib vision/tests/golden/distress/synthetic_calib/canvas460.yaml \
  --report-overlay --output vision/results/report_overlay_demo/golden_fine_overlay.png \
  --log-dir /tmp/goldlog
```

🔴 `--calib`을 기본값(`nominal.yaml`, 4608px 기준)으로 두면 **`DIST`가 10배 가까이 틀리게**
나온다 — 캘리브레이션 해상도와 프레임 해상도가 다르면 focal(픽셀 단위)이 어긋나기 때문이고,
이 저장소는 그걸 **조용히 재스케일하지 않는다**(`vision/CLAUDE.md` "캘리브레이션 해상도
불일치" 절). 오버레이가 그 사실을 화면에 드러내 준다.

### ② FHD 클립 재현

입력 합성 클립(초록 매트 3.0m + 흰 박스, 고도 25→4m 접근, 1920×1080/20fps)은
**커밋하지 않는다** — `state_machine_demo/`·`distress_fine_demo/`와 같은 전례(결과물만 남기고
합성 소스 프레임은 저장소 밖). 클립을 만든 뒤 실행한 명령은:

```bash
python -m vision.main <클립>.mp4 --preset vision/presets/distress_fine.yaml \
  --calib vision/calibration/cam109-imx708af75-1920x1080/nominal.yaml \
  --report-overlay --output overlay.mp4 --log-dir /tmp/demolog
```

거리 정확도 실측(합성 참값 대비): 25.00→**24.86m** / 14.41→**14.40m** / 7.35→**7.34m** /
4.00→**3.99m**.

## 이 오버레이가 하지 않는 것

- **착륙점을 계산하지 않는다.** `modules/distress_box.py`가 확정해 둔
  `meta["white_box_detector"]["landing_point_px"]`를 그대로 읽어 그린다(규칙을 두 곳에
  복사하면 한쪽만 고쳐졌을 때 조용히 어긋난다 — `modules/distress_mat.py`와 같은 원칙).
- **기본 경로를 건드리지 않는다.** `--report-overlay`는 opt-in이고, 안 주면 호출 자체가
  없어 산출물이 이 기능 도입 이전과 한 픽셀도 다르지 않다(회귀테스트로 고정).

촬영 절차 전체(실기체 라이브 녹화 / 외부 영상 경로 / 함정)는 `docs/vision_report_video.md`.
