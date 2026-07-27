"""
현장 색 캘리브레이터 (`docs/vision_plan.md` §5.5, §9 빌드순서 5번).

**지오메트리 캘리브레이션(카메라 인트린식, `calib_capture.py`/`calib_analyze.py`)과는
완전히 다른 개념이다** — §9 5번이 "여기서 '캘리브레이터'는 현장 조명 대응용 HSV 임계값
튜너로, 지오메트리 캘리브레이션과는 별개 개념"이라고 명시한다. 이 도구는 렌즈/센서를
전혀 다루지 않는다 — 오직 한 프레임의 ROI에서 뽑은 HSV 통계로 `ColorFilter`/`RedRingDetector`
생성자 인자를 제안할 뿐이다.

## 왜 필요한가

`docs/vision_plan.md` §5.5:
    "레이어드 HSV + 현장 색 캘리브레이터(라이브 ROI 샘플→임계값 자동 설정) — 규정·색이
    대회 직전 공개돼도 코드 수정 없이 대응."
§5.3이 ② 조난자 구역의 "핵심 난제 = 초록 색조 비동일"이라고 못박았다. coarse 색 파이프라인
(`ColorFilter` 등)은 이미 있지만, 지금까지는 사람이 preset yaml의 HSV 숫자를 손으로 추측해야
했다 — 대회 당일 조명/색이 예상과 다르면 이 도구로 현장에서 몇 분 만에 새 임계값을 뽑아
검토·반영할 수 있어야 한다는 게 이 도구의 존재 이유다.

## 흐름

1. **입력**: 프레임 이미지 파일 하나, 또는 녹화 폴더/영상 파일(`vision.utils.frame_source`의
   `open_dir_or_bag()`을 그대로 재사용 — 새 프레임 소스를 만들지 않는다). 정지 이미지 파일은
   `DirFrameSource`(디렉터리 전용)/`BagFrameSource`(영상 컨테이너 전용) 둘 다 원래 대상이
   아니라서 `vision.utils.image_loader.load_image()`로 직접 읽는다(가장 단순한 경로).
2. **ROI 지정**: `--roi x,y,w,h` 정수 좌표만 받는다. **GUI/마우스 인터랙션 없음** — 이 환경
   (WSL·RPi headless)엔 디스플레이가 없고, `cv2.selectROI`는 쓰지 않는다(세션 지시 금지 항목).
3. **통계 산출**: ROI를 HSV로 변환해 채널별 분포를 계산한다. **평균±N×표준편차 대신
   백분위수(percentile) 기반을 기본으로 쓴다** — 이유: 그림자/글레어/과노출 클리핑 같은
   국소 이상치 픽셀이 소수 섞여도(§5.5 "그림자·광택 글레어·과노출 클리핑 대비") 평균과
   표준편차는 그 이상치 쪽으로 쉽게 끌려가 임계값이 부풀거나(표준편차 팽창) 좁아진다
   (평균이 이상치 쪽으로 이동). 백분위수(기본 p5~p95)는 이상치가 전체의 일정 비율
   (기본 5% 미만)이면 그 값 자체를 사실상 무시하고 "주 색상 군집"의 실제 분포 경계를
   따른다 — `tests/test_color_calibrate.py::test_calibrate_roi_percentile_range_ignores_minority_outlier_pixels`가
   이 근거를 합성 이상치 픽셀로 직접 검증한다.
4. **⚠️ 빨강 Hue 랩어라운드 처리**: OpenCV HSV의 Hue는 0~179이고 빨강은 0 부근과 179 부근으로
   갈라진다. 이 코드베이스에 이미 관련 전례가 둘 있다 —
     - `vision/modules/color.py::ColorFilter` — 랩어라운드 **미지원**(§5.4 blind spot,
       `vision/CLAUDE.md` 테스트 규칙표에 회귀테스트로 기록됨). 단일 `hue_range=(low, high)`
       하나만 받으므로 애초에 두 구간을 표현할 수 없다.
     - `vision/modules/vertiport_ring.py::RedRingDetector` — **빨강 Hue 양끝 게이팅**으로
       랩어라운드에 대응하는 실제 구현(`low_hue_max`/`high_hue_min`/`sat_min`/`val_min`
       생성자 인자, `[0,low_hue_max]`∪`[high_hue_min,179]`를 OR로 게이팅). **이 도구가 따르는
       전례는 이쪽이다.**
   이 도구는 ROI Hue 표본을 `--wrap-split-hue`(기본 90) 기준 저/고 두 구간으로 나눠 각 구간의
   표본 비율이 둘 다 `--wrap-min-fraction`(기본 0.05) 이상이면 "랩어라운드"로 판정하고, 두
   구간(low/high)으로 나눠 `RedRingDetector` 호환 파라미터(`low_hue_max`/`high_hue_min`/
   `sat_min`/`val_min`)를 출력한다. 랩어라운드가 아니면 단일 `hue_range`로 `ColorFilter`
   호환 파라미터(`hue_range`/`sat_min`/`val_min`/`val_max`)를 출력한다. 어느 쪽이든 산출
   결과에 "호환 소비자"를 문자열로 명시한다.
5. **출력**: 산출된 임계값을 사람이 그대로 복붙할 수 있는 yaml 조각으로 stdout에 출력하고,
   `--output <path>`를 주면 같은 내용을 파일로도 저장한다. **기존 preset yaml을 자동으로
   덮어쓰지 않는다** — "제안"까지가 이 도구의 범위이고 사람이 검토 후 반영한다. 산출 근거
   (ROI 좌표·샘플 픽셀 수·백분위수·마진·랩어라운드 판정 근거·입력 파일)를 전부 yaml 주석으로
   함께 싣는다.
6. **진단 산출물**: `--diagnostic-dir <dir>`을 주면 (a) ROI 위치를 표시한 오버레이 이미지와
   (b) 채널별 HSV 히스토그램(제안된 임계값을 세로선으로 표시) PNG를 저장한다. 히스토그램은
   `vision/tools/jsonl_view.py`와 동일하게 matplotlib **Agg 백엔드**(headless-safe)를 쓴다.

## 임계값·마진 전부 CLI 파라미터 (매직넘버 금지, §7.3)

백분위수(`--low-percentile`/`--high-percentile`), 랩어라운드 판정 기준(`--wrap-split-hue`/
`--wrap-min-fraction`), 마진(`--hue-margin`/`--sat-margin`/`--val-margin`) 전부 CLI 인자로
노출된다 — 코드에 하드코딩된 숫자가 없다.

## 마진 정책 — 백분위수와 마진은 서로 다른 것을 덮는다 (2026-07-28 확정)

이 절이 `DEFAULT_HUE_MARGIN`/`DEFAULT_SAT_MARGIN`/`DEFAULT_VAL_MARGIN` 기본값의 정식 근거다.
(그 전까지는 셋 다 0이었고 "조명 변동 쿠션 없음"이 알려진 갭으로 기록만 돼 있었다.)

**먼저 개념을 분리해야 한다. 안 그러면 같은 변동을 두 번 센다.**

- **백분위수 밴드(p5~p95)가 이미 흡수하는 것** — 캘리브레이션을 수행한 *그 한 프레임 ROI
  안의 공간적 변동*: 그림자, 광택 글레어, 재질 얼룩, 과노출 클리핑 소수 픽셀. 전부 그
  프레임에 **실제로 찍혀 있는** 정보다.
- **마진이 덮어야 하는 것** — 캘리브레이션한 조건과 *실제 비행 시점 조건이 달라졌을 때의
  이동*: 시각/태양각/구름/화이트밸런스 재수렴/노출 변화로 인한 **분포 중심 자체의 이동**.
  백분위수는 이걸 **원리적으로 볼 수 없다**(그 프레임에 없는 정보다).

따라서 "ROI 표준편차의 N배"를 그대로 마진으로 박으면 프레임 내 변동을 두 번 세게 된다.
아래 3번이 이 위험을 정면으로 다룬다.

### `DEFAULT_HUE_MARGIN = 6` (OpenCV Hue 단위, 양쪽 각각 ±6)

**근거 표본 — 유일한 실측 데이터(2026-07-28 사용자 제공).** 대회 ② 조난자 구역(초록구역,
3.0m×3.0m×0.105m 라이즈드 플랫폼)의 실물 레퍼런스는 공개된 것이 없다. 사용자가 초록구역을
휴대폰으로 찍은 사진에서 랜덤 10점을 뽑고 이상치 2점을 제외한 8점의 Hue:

    136  162  169  146  169  158  146  163      (0~360° 스케일)

1. **스케일 판정 — 0~360°다(검산 완료).** ÷2 하면 OpenCV Hue **68 ~ 84.5**(청록 띤 초록)
   → 초록 매트로 물리적으로 타당하고, 이 저장소가 독립적으로 손튜닝해 둔
   `presets/distress_coarse.yaml`의 `hue_range=[35,85]` **안에 실제로 들어온다**(교차확인).
   다른 해석은 전부 색이 모순된다 — 0~255 스케일 가정은 ×360/255 = 192~238°(**파랑**),
   이미 OpenCV 0~179라는 가정은 ×2 = 272~338°(**마젠타**). 둘 다 초록 매트일 수 없으므로 기각.
2. **절대 Hue 위치는 신뢰하지 않고 쓰지도 않는다.** 카메라 기종·촬영시각·f값·기본 픽쳐
   프로파일이 전부 미상이고 휴대폰 화이트밸런스/색보정 offset이 걸려 있어 중심값(78)은 우리
   카메라로 전이되지 않는다. **애초에 이 도구는 중심을 현장 ROI에서 다시 재므로 중심값이
   필요 없다** — 필요한 건 산포뿐이고, 이 표본에서 신뢰할 수 있는 것도 산포뿐이다
   (사용자가 이 데이터를 준 이유이기도 하다).
3. **산포(직접 재검산):** 0~360° 기준 mean 156.125 / 표본σ 12.1118 / 모집단σ 11.3296 /
   range 33. OpenCV 기준 mean 78.0625 / **표본σ 6.0559 / 모집단σ 5.6648** / range 16.5.
   → **표본σ와 모집단σ가 둘 다 6으로 반올림되므로 어느 추정량을 쓰든 결론이 같다: 6.**
4. **⚠️ 이중계상 위험에 대한 답 — 왜 이 σ를 마진에 써도 되는가, 그리고 왜 딱 1σ인가.**
   이 8점은 **한 장의 사진 안**에서 뽑은 점들이라 엄밀히는 "프레임 내 공간 변동"이고, 그건
   위 정의상 백분위수가 이미 덮는 쪽이다. 그래도 이 σ를 마진 근거로 쓰는 논거는:
     - **(성립하는 부분)** 균질한 무광 매트 한 장 안에서 Hue가 σ≈6이나 흩어지는 원인은
       재질이 아니라 **국소 조명 기하**(음영각·직사광 대 천공광 혼합비·미세 글레어)다.
       이건 태양각이 바뀌거나 구름이 끼었을 때 **면 전체를 이동시키는 것과 같은 물리
       메커니즘**이다. 즉 프레임 내 8점은 서로 다른 국소 조명조건의 *표본*이고, 전역 조건이
       바뀌면 면 전체가 그 분포의 한쪽 끝(그늘 쪽/직사광 쪽)으로 옮겨 앉는다 — **중심 이동
       폭이 프레임 내 산포와 같은 자릿수라고 볼 근거**가 된다.
     - **(성립하지 않는 부분 — 정직하게)** 이건 부등식이 아니다. "프레임 내 σ ≤ 조건 간 σ"를
       보장하는 정리는 없다. 그러므로 **하한 추정이 아니라 자릿수 추정(order-of-magnitude
       proxy)으로만 쓴다.** 이 표본에서 정직하게 끌어낼 수 있는 결론은 "조건 간 이동이
       6 hue 단위쯤 될 수 있다"까지이고, "최소 6이다"가 아니다.
     - **(그래서 1σ이지 2σ·3σ가 아니다)** 현장 ROI가 넓고 얼룩덜룩하면 p5~p95가 이미 이
       변동의 상당 부분을 흡수한 상태라 그 위에 2σ를 더 얹는 건 **진짜로 이중계상**이 된다.
       1σ는 그 겹침을 인정하고 고른 값이다 — 겹치는 경우엔 다소 넉넉해지는 정도로 끝나고,
       ROI가 좁고 균일해서 p5≈p95로 퇴화한 경우(아래 5번)엔 마진이 쿠션 전부를 떠맡는다.
     - 휴대폰의 자동 채도/색조 보정은 saturated 색을 memory color 쪽으로 **압축**하는 경향이
       있어, 원본 센서 σ가 이 값보다 작을 가능성보다는 클 가능성이 높다 — 6이 과대추정일
       위험은 상대적으로 낮다.
5. **왜 0(기존값)을 유지하지 않는가.** 0은 "중립"이 아니라 이 도구의 권장 사용법에서 실제로
   고장난 값이다. 좁고 균일한 ROI를 지정하면 p5==p95가 되어 `hue_range: [60, 60]` 같은 **폭
   0짜리 밴드**가 나오고(이미 관측된 정상 동작), 그 밴드는 조건이 조금만 바뀌어도 타겟을
   통째로 놓친다. 게다가 이 도구의 존재 이유가 "대회 당일 5분 만에 대응"인데, 기억해서 플래그를
   붙여야만 쓸 만해지는 기본값은 그 목적과 어긋난다.
6. **반대편 비용 — 오탐(false positive).** 초록구역 검출은 자갈/잔디/그림자 배경에서 돈다.
   Hue 밴드를 ±6 넓히면 통과대역이 12 단위 넓어진다(Hue 원주 180의 6.7%). 위험 방향은
   **아래쪽(잔디)** — 무성한 잔디는 OpenCV Hue 60~70대까지 올라와 청록 띤 매트(≈78)와
   가깝다. 자갈은 **무채색이라 Hue가 아니라 `sat_min`이**, 그림자는 **`val_min`이** 거르므로
   Hue 마진의 오탐 노출은 사실상 식생 하나다. 이 비용이 감당 가능하다고 판단한 근거:
   (a) ±6을 얹어도 산출 밴드 폭은 대략 12~25 단위로, 이 저장소가 **이미 운용상 받아들이고
   있는** 손튜닝값 `hue_range=[35,85]`(폭 50)보다 여전히 훨씬 좁다 — 즉 현상 대비 오탐이
   늘지 않는다. (b) 잔디/자갈의 최종 배제는 Hue 단독이 아니라 뒤따르는
   `RectDetector`의 면적·형상 필터와 `sat_min`이 함께 담당한다.

### `DEFAULT_SAT_MARGIN = 0` / `DEFAULT_VAL_MARGIN = 0` — 근거가 없어 0을 유지한다

**억지로 채우지 않는다.** 표본은 **Hue 8점뿐**이고 S/V 데이터는 단 하나도 없다.

- **Hue σ에서 S/V 마진을 유도할 수 없다.** V는 조도에 거의 선형으로, S는 색소 반사율 대
  백색광 성분의 비율로 반응한다 — Hue와 물리 경로가 달라 6이라는 숫자를 옮겨 쓸 근거가 없다.
  숫자를 지어내면 그게 바로 §7.3이 금지하는 출처 없는 매직넘버다.
- **더구나 이 두 마진이 밴드를 넓히는 방향이 배경 클래스를 정확히 겨눈다.** `sat_min`을
  내리면 **저채도 = 자갈**이, `val_min`을 내리면 **저휘도 = 그림자**가, `val_max`를 올리면
  **블로운 하이라이트 = 글레어**(백분위수가 일부러 잘라낸 바로 그것)가 들어온다. Hue 마진의
  오탐 노출이 식생 하나뿐인 것과 비교해 비용 프로파일이 명백히 나쁘다.
- **비대칭성:** S/V 마진 부재로 생기는 실패(어두워져서 미검출)는 **즉시 눈에 띄고 몇 초 만에
  복구된다**(`--val-margin`을 주고 재실행). 반면 자갈/그림자 오탐은 알아채기 어렵고 상태머신의
  커밋 게이트를 잘못 통과시킬 수 있다 — `core/state_machine.py`가 모호한 후보를 `HOLD`로
  거절하도록 설계된 것과 같은 철학이다(오검출 < 미검출).
- **→ 현장에서는 `--sat-margin`/`--val-margin`을 명시하라.** `--diagnostic-dir`의 HSV
  히스토그램을 보고 그 자리에서 정하는 것이 지어낸 기본값보다 낫다.
  (`docs/vision_verification_qa_brief.md`의 `--sat-margin 20 --val-margin 20` 예시는 여전히
  유효한 *운영자 선택값*이다 — 다만 조용한 기본값이 되어선 안 된다는 것이 이 절의 결론이다.)

## 과설계 금지 (§5.5가 요구하지 않는 것은 넣지 않았다)

자동 색 추적·머신러닝 분류기·GUI(`cv2.selectROI` 등)·설정 자동 반영(preset yaml 자동 덮어쓰기)
전부 이 도구의 범위 밖이다. §5.5가 요구한 것은 "라이브 ROI 샘플 → 임계값 자동 설정" + "사람
검토" 두 가지뿐이다.

## 하드웨어 의존 없음 (`vision/CLAUDE.md` tools/ 예외 규칙)

`cv2`/`numpy`/`matplotlib`뿐 — `jsonl_view.py`/`calib_analyze.py`와 동일한 예외로 `.venv`
설치 + `tests/test_color_calibrate.py` pytest 대상이다.

## CLI

    python -m vision.tools.color_calibrate <이미지 파일|녹화폴더|영상파일> --roi x,y,w,h
    python -m vision.tools.color_calibrate frame.png --roi 100,100,60,60 \\
        --output vision/presets/_proposed_distress.yaml --diagnostic-dir vision/results/color_calib
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

import matplotlib

matplotlib.use("Agg")  # headless-safe — GUI 강제 호출 금지(WSL/RPi에 디스플레이 없음)
import matplotlib.pyplot as plt

from vision.utils.frame_source import open_dir_or_bag
from vision.utils.image_loader import load_image

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}

# --- 기본값(전부 CLI로 override 가능, §7.3 매직넘버 금지) ---
DEFAULT_LOW_PERCENTILE = 5.0
DEFAULT_HIGH_PERCENTILE = 95.0
DEFAULT_WRAP_SPLIT_HUE = 90
DEFAULT_WRAP_MIN_FRACTION = 0.05
# 마진 기본값 — 정식 근거는 이 파일 상단 docstring "마진 정책" 절 전체. 요약:
#   * 백분위수(p5~p95)는 "그 프레임 안의 공간 변동"을, 마진은 "캘리브 시점↔비행 시점의
#     조건 변화(중심 이동)"를 덮는다. 서로 다른 것을 덮으므로 마진이 필요하다.
#   * HUE=6 — 사용자 제공 실측 8점(초록구역 휴대폰 사진, 0~360° 스케일)의 OpenCV 환산
#     표본σ 6.0559 / 모집단σ 5.6648 → 둘 다 6으로 반올림. 절대 Hue 위치(78)는 카메라/WB
#     미상이라 쓰지 않고 산포만 근거로 삼는다. 1σ인 이유: 표본이 프레임 내 변동이라
#     p5~p95와 부분적으로 겹치므로 2σ 이상은 같은 변동을 두 번 세는 이중계상이 된다
#     (docstring 4번). 오탐 비용은 식생(Hue 60~70대) 방향 하나뿐이며 결과 밴드 폭은
#     기존 손튜닝값 hue_range=[35,85](폭 50)보다 여전히 훨씬 좁다(docstring 6번).
#   * SAT/VAL=0 — 표본이 Hue뿐이라 근거가 없어 숫자를 지어내지 않는다. 게다가 이 두 마진이
#     넓히는 방향이 자갈(저채도)/그림자(저휘도)/글레어(블로운)를 정확히 겨눈다. 현장에서는
#     `--sat-margin`/`--val-margin`을 히스토그램 보고 명시할 것(docstring 마지막 절).
DEFAULT_HUE_MARGIN = 6
DEFAULT_SAT_MARGIN = 0
DEFAULT_VAL_MARGIN = 0

HUE_MAX = 179
CHANNEL_MAX = 255


# ===========================================================================
# 프레임 로드 (frame_source 재사용 — 새 프레임 소스 구현 금지, 세션 지시)
# ===========================================================================


def load_frame(input_path: Path | str, frame_index: int = 0) -> np.ndarray:
    """이미지 파일 또는 녹화 폴더(디렉터리)/영상 파일에서 프레임 한 장을 읽는다.

    디렉터리·영상 파일은 `vision.utils.frame_source.open_dir_or_bag()`(DirFrameSource/
    BagFrameSource)을 그대로 재사용해 frame_index번째 프레임을 취한다. 단일 정지 이미지
    파일은 `DirFrameSource`(디렉터리 전용)도 `BagFrameSource`(영상 컨테이너 전용, cv2.VideoCapture
    기반이라 단일 정지 이미지에 대한 동작이 OpenCV 빌드마다 불안정함)도 원래 대상이 아니므로
    `vision.utils.image_loader.load_image()`로 직접 읽는다(가장 단순하고 확실한 경로).
    """
    input_path = Path(input_path)
    if input_path.is_dir():
        return _nth_frame_from_source(open_dir_or_bag(input_path), frame_index, str(input_path))
    if input_path.suffix.lower() in _IMAGE_SUFFIXES:
        return load_image(str(input_path))
    return _nth_frame_from_source(open_dir_or_bag(input_path), frame_index, str(input_path))


def _nth_frame_from_source(source, frame_index: int, label: str) -> np.ndarray:
    with source:
        for i, record in enumerate(source):
            if i == frame_index:
                return record.image
    raise ValueError(f"load_frame: frame_index={frame_index} — '{label}'에 그만큼의 프레임이 없음")


# ===========================================================================
# ROI 파싱/크롭
# ===========================================================================


def parse_roi(value: str) -> Tuple[int, int, int, int]:
    """`--roi x,y,w,h` 파싱. GUI 없음 — 좌표는 항상 사람이 숫자로 지정한다(세션 지시)."""
    parts = value.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(f"--roi는 x,y,w,h 4개 정수여야 합니다: {value!r}")
    try:
        x, y, w, h = (int(p) for p in parts)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"--roi는 정수만 허용합니다: {value!r}") from e
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError(f"--roi의 w,h는 양수여야 합니다: {value!r}")
    return x, y, w, h


def crop_roi(image: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = roi
    img_h, img_w = image.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(img_w, x + w), min(img_h, y + h)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(
            f"crop_roi: ROI가 프레임 범위를 벗어남 — roi={roi}, frame_size=({img_w}x{img_h})"
        )
    return image[y0:y1, x0:x1]


# ===========================================================================
# HSV 통계 + 임계값 산출 (순수 함수 — 하드웨어/파일 I/O 없음)
# ===========================================================================


def compute_hsv_channels(roi_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ROI(BGR) → (hue, sat, val) 1차원 배열(각 픽셀 1개). int32로 반환(부호 걱정 없이 산술)."""
    if roi_bgr.size == 0:
        raise ValueError("compute_hsv_channels: ROI가 비어 있음(크기 0)")
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[..., 0].reshape(-1).astype(np.int32)
    s = hsv[..., 1].reshape(-1).astype(np.int32)
    v = hsv[..., 2].reshape(-1).astype(np.int32)
    return h, s, v


def detect_hue_wraparound(
    hue: np.ndarray,
    wrap_split_hue: int = DEFAULT_WRAP_SPLIT_HUE,
    wrap_min_fraction: float = DEFAULT_WRAP_MIN_FRACTION,
) -> Tuple[bool, float, float]:
    """Hue 표본을 `wrap_split_hue` 기준 저/고 두 구간으로 나눠 둘 다 `wrap_min_fraction`
    이상의 비율을 차지하면 랩어라운드로 판정한다(`RedRingDetector`의 두 구간 게이팅과 같은
    분할 기준). 한쪽 구간이 노이즈 수준(< wrap_min_fraction)이면 단일 군집으로 본다 —
    그래야 순수 초록/파랑 등 원래 랩어라운드가 없는 색까지 오판하지 않는다."""
    n = int(hue.size)
    if n == 0:
        return False, 0.0, 0.0
    frac_low = float(np.count_nonzero(hue <= wrap_split_hue)) / n
    frac_high = float(np.count_nonzero(hue > wrap_split_hue)) / n
    wraparound = frac_low >= wrap_min_fraction and frac_high >= wrap_min_fraction
    return wraparound, frac_low, frac_high


def _clip_int(value: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, round(value))))


@dataclass
class CalibrationResult:
    """`calibrate_roi()`의 산출물 — 임계값 제안 + 산출 근거(전부 yaml 주석/리포트에 실림)."""

    roi: Tuple[int, int, int, int]
    n_pixels: int
    low_percentile: float
    high_percentile: float
    hue_margin: int
    sat_margin: int
    val_margin: int
    hue_median: float
    sat_median: float
    val_median: float
    hue_wraparound: bool
    wrap_split_hue: int
    wrap_min_fraction: float
    frac_hue_low: float
    frac_hue_high: float
    consumer: str
    params: dict


def calibrate_roi(
    roi_bgr: np.ndarray,
    *,
    roi_box: Tuple[int, int, int, int] = (0, 0, 0, 0),
    low_percentile: float = DEFAULT_LOW_PERCENTILE,
    high_percentile: float = DEFAULT_HIGH_PERCENTILE,
    wrap_split_hue: int = DEFAULT_WRAP_SPLIT_HUE,
    wrap_min_fraction: float = DEFAULT_WRAP_MIN_FRACTION,
    hue_margin: int = DEFAULT_HUE_MARGIN,
    sat_margin: int = DEFAULT_SAT_MARGIN,
    val_margin: int = DEFAULT_VAL_MARGIN,
) -> CalibrationResult:
    """ROI(BGR) → `CalibrationResult`. 순수 함수(파일 I/O 없음) — 합성 패치로 직접 테스트 가능.

    랩어라운드면 `RedRingDetector` 호환(`low_hue_max`/`high_hue_min`/`sat_min`/`val_min`),
    아니면 `ColorFilter(mode=color)` 호환(`hue_range`/`sat_min`/`val_min`/`val_max`) 파라미터를
    `params`에 담는다.
    """
    h, s, v = compute_hsv_channels(roi_bgr)
    n = int(h.size)

    wraparound, frac_low, frac_high = detect_hue_wraparound(h, wrap_split_hue, wrap_min_fraction)

    sat_min = _clip_int(np.percentile(s, low_percentile) - sat_margin, 0, CHANNEL_MAX)
    val_min = _clip_int(np.percentile(v, low_percentile) - val_margin, 0, CHANNEL_MAX)
    val_max = _clip_int(np.percentile(v, high_percentile) + val_margin, 0, CHANNEL_MAX)

    if wraparound:
        low_side = h[h <= wrap_split_hue]
        high_side = h[h > wrap_split_hue]
        low_hue_max = _clip_int(np.percentile(low_side, high_percentile) + hue_margin, 0, HUE_MAX)
        high_hue_min = _clip_int(np.percentile(high_side, low_percentile) - hue_margin, 0, HUE_MAX)
        params = {
            "low_hue_max": low_hue_max,
            "high_hue_min": high_hue_min,
            "sat_min": sat_min,
            "val_min": val_min,
        }
        consumer = "RedRingDetector(vision/modules/vertiport_ring.py) — 빨강 Hue 양끝 게이팅 생성자 인자"
    else:
        hue_lo = _clip_int(np.percentile(h, low_percentile) - hue_margin, 0, HUE_MAX)
        hue_hi = _clip_int(np.percentile(h, high_percentile) + hue_margin, 0, HUE_MAX)
        params = {
            "hue_range": [hue_lo, hue_hi],
            "sat_min": sat_min,
            "val_min": val_min,
            "val_max": val_max,
        }
        consumer = "ColorFilter(vision/modules/color.py, mode=color) 생성자 인자"

    return CalibrationResult(
        roi=roi_box,
        n_pixels=n,
        low_percentile=low_percentile,
        high_percentile=high_percentile,
        hue_margin=hue_margin,
        sat_margin=sat_margin,
        val_margin=val_margin,
        hue_median=float(np.median(h)),
        sat_median=float(np.median(s)),
        val_median=float(np.median(v)),
        hue_wraparound=wraparound,
        wrap_split_hue=wrap_split_hue,
        wrap_min_fraction=wrap_min_fraction,
        frac_hue_low=frac_low,
        frac_hue_high=frac_high,
        consumer=consumer,
        params=params,
    )


# ===========================================================================
# 출력 — 사람이 복붙할 yaml 조각 (자동 반영 아님, "제안"까지가 범위)
# ===========================================================================


def format_yaml_snippet(result: CalibrationResult, *, input_label: str, frame_index: int) -> str:
    lines = [
        "# 현장 색 캘리브레이터(vision/tools/color_calibrate.py) 산출 — 사람이 검토 후 반영할 것.",
        "#   기존 preset yaml을 자동으로 덮어쓰지 않는다 — 이 조각을 복붙해 판단 후 넣는다.",
        f"# 입력: {input_label} (frame_index={frame_index})",
        f"# ROI(x,y,w,h)={list(result.roi)}  샘플 픽셀 수={result.n_pixels}",
        (
            f"# 백분위수 p{result.low_percentile:g}~p{result.high_percentile:g} 기반 "
            f"(평균±표준편차 대신 — 이상치에 강함, 도구 docstring 참조). "
            f"중앙값 H={result.hue_median:.1f} S={result.sat_median:.1f} V={result.val_median:.1f}"
        ),
        f"# 마진: hue±{result.hue_margin}  sat-{result.sat_margin}  val±{result.val_margin}",
        (
            "#   마진 = 캘리브 시점↔비행 시점의 조건 변화(태양각/구름/WB 재수렴) 쿠션. "
            "프레임 내 공간 변동(그림자/글레어)은 위 백분위수가 이미 흡수한다."
        ),
    ]
    if result.hue_wraparound:
        lines.append(
            f"# Hue 랩어라운드 감지됨 (split={result.wrap_split_hue}, "
            f"저구간비율={result.frac_hue_low:.2f}, 고구간비율={result.frac_hue_high:.2f}) "
            "→ low/high 두 구간으로 출력"
        )
    else:
        lines.append(
            f"# Hue 랩어라운드 없음 (split={result.wrap_split_hue} 기준 한쪽 구간 비율이 "
            f"wrap_min_fraction={result.wrap_min_fraction:g} 미만: "
            f"저={result.frac_hue_low:.2f} 고={result.frac_hue_high:.2f}) → 단일 hue_range로 출력"
        )
    lines.append(f"# 호환 소비자: {result.consumer}")
    lines.append("")

    if result.hue_wraparound:
        lines.append("red_ring:  # RedRingDetector 생성자 인자와 동일한 키(vertiport_ring.py)")
        lines.append(f"  low_hue_max: {result.params['low_hue_max']}")
        lines.append(f"  high_hue_min: {result.params['high_hue_min']}")
        lines.append(f"  sat_min: {result.params['sat_min']}")
        lines.append(f"  val_min: {result.params['val_min']}")
    else:
        lines.append("color_filter:  # ColorFilter preset 키(color.py)")
        lines.append("  mode: color")
        lines.append(f"  hue_range: {result.params['hue_range']}")
        lines.append(f"  sat_min: {result.params['sat_min']}")
        lines.append(f"  val_min: {result.params['val_min']}")
        lines.append(f"  val_max: {result.params['val_max']}")

    return "\n".join(lines) + "\n"


# ===========================================================================
# 진단 산출물 (Agg, headless-safe — jsonl_view.py와 동일 패턴)
# ===========================================================================


def save_roi_overlay(frame_bgr: np.ndarray, roi: Tuple[int, int, int, int], out_path: Path) -> Path:
    """ROI 위치를 표시한 오버레이 이미지를 PNG로 저장(cv2, GUI 호출 없음 — imwrite만)."""
    x, y, w, h = roi
    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 2)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)
    return out_path


def save_hsv_histogram(
    h: np.ndarray, s: np.ndarray, v: np.ndarray, result: CalibrationResult, out_path: Path
) -> Path:
    """채널별 HSV 히스토그램(제안된 임계값을 세로 점선으로 표시) — matplotlib Agg."""
    fig, (ax_h, ax_s, ax_v) = plt.subplots(3, 1, figsize=(7, 6.5))

    ax_h.hist(h, bins=60, range=(0, HUE_MAX), color="tab:blue", alpha=0.8)
    ax_h.set_title("Hue (0-179)", fontsize=9)
    if result.hue_wraparound:
        ax_h.axvline(result.params["low_hue_max"], color="tab:red", ls="--", lw=1)
        ax_h.axvline(result.params["high_hue_min"], color="tab:red", ls="--", lw=1)
    else:
        ax_h.axvline(result.params["hue_range"][0], color="tab:red", ls="--", lw=1)
        ax_h.axvline(result.params["hue_range"][1], color="tab:red", ls="--", lw=1)

    ax_s.hist(s, bins=64, range=(0, CHANNEL_MAX), color="tab:green", alpha=0.8)
    ax_s.set_title("Saturation (0-255)", fontsize=9)
    ax_s.axvline(result.params["sat_min"], color="tab:red", ls="--", lw=1)

    ax_v.hist(v, bins=64, range=(0, CHANNEL_MAX), color="tab:orange", alpha=0.8)
    ax_v.set_title("Value (0-255)", fontsize=9)
    ax_v.axvline(result.params["val_min"], color="tab:red", ls="--", lw=1)
    if not result.hue_wraparound:
        ax_v.axvline(result.params["val_max"], color="tab:red", ls="--", lw=1)

    fig.suptitle(f"color_calibrate — ROI={list(result.roi)} (n={result.n_pixels})")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


# ===========================================================================
# CLI
# ===========================================================================


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input", help="프레임 이미지 파일, 또는 녹화 폴더(디렉터리)/영상 파일 경로")
    parser.add_argument(
        "--roi", required=True, type=parse_roi, metavar="x,y,w,h",
        help="샘플링할 ROI 정수 좌표 — GUI 없음(cv2.selectROI 사용 안 함), 반드시 CLI로 지정",
    )
    parser.add_argument(
        "--frame-index", type=int, default=0,
        help="녹화 폴더/영상 입력일 때 사용할 프레임 번호(기본 0)",
    )
    parser.add_argument("--low-percentile", type=float, default=DEFAULT_LOW_PERCENTILE)
    parser.add_argument("--high-percentile", type=float, default=DEFAULT_HIGH_PERCENTILE)
    parser.add_argument(
        "--wrap-split-hue", type=int, default=DEFAULT_WRAP_SPLIT_HUE,
        help="Hue 랩어라운드 판정용 저/고 분할 기준(기본 90, 0~179 중앙)",
    )
    parser.add_argument(
        "--wrap-min-fraction", type=float, default=DEFAULT_WRAP_MIN_FRACTION,
        help="양쪽 구간 모두 이 비율 이상이어야 랩어라운드로 판정(기본 0.05)",
    )
    parser.add_argument(
        "--hue-margin", type=int, default=DEFAULT_HUE_MARGIN,
        help=(
            f"hue 통과대역을 양쪽으로 넓힐 여유(기본 {DEFAULT_HUE_MARGIN} — 실측 8점 산포 1σ, "
            "캘리브 시점↔비행 시점 조명 변화 쿠션. 근거는 모듈 docstring '마진 정책' 절)"
        ),
    )
    parser.add_argument(
        "--sat-margin", type=int, default=DEFAULT_SAT_MARGIN,
        help=(
            f"sat_min에서 뺄 여유(기본 {DEFAULT_SAT_MARGIN} — S 표본이 없어 지어내지 않음. "
            "내리면 저채도 자갈이 들어오므로 현장에서 히스토그램 보고 명시할 것)"
        ),
    )
    parser.add_argument(
        "--val-margin", type=int, default=DEFAULT_VAL_MARGIN,
        help=(
            f"val 경계에 더할 여유(기본 {DEFAULT_VAL_MARGIN} — V 표본이 없어 지어내지 않음. "
            "넓히면 그림자/글레어가 들어오므로 현장에서 히스토그램 보고 명시할 것)"
        ),
    )
    parser.add_argument("--output", default=None, help="산출 yaml 조각을 저장할 파일 경로(선택)")
    parser.add_argument(
        "--diagnostic-dir", default=None,
        help="ROI 오버레이/HSV 히스토그램 PNG를 저장할 폴더(선택)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: 입력 경로 없음 — {input_path}", file=sys.stderr)
        return 1

    try:
        frame = load_frame(input_path, args.frame_index)
    except (FileNotFoundError, ValueError, NotADirectoryError) as e:
        print(f"Error: 프레임 로드 실패 — {e}", file=sys.stderr)
        return 1

    try:
        roi_bgr = crop_roi(frame, args.roi)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    result = calibrate_roi(
        roi_bgr,
        roi_box=args.roi,
        low_percentile=args.low_percentile,
        high_percentile=args.high_percentile,
        wrap_split_hue=args.wrap_split_hue,
        wrap_min_fraction=args.wrap_min_fraction,
        hue_margin=args.hue_margin,
        sat_margin=args.sat_margin,
        val_margin=args.val_margin,
    )

    snippet = format_yaml_snippet(result, input_label=str(input_path), frame_index=args.frame_index)
    print(snippet)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(snippet, encoding="utf-8")
        print(f"저장: {out_path}")

    if args.diagnostic_dir:
        h, s, v = compute_hsv_channels(roi_bgr)
        diag_dir = Path(args.diagnostic_dir)
        overlay_path = save_roi_overlay(frame, args.roi, diag_dir / "roi_overlay.png")
        hist_path = save_hsv_histogram(h, s, v, result, diag_dir / "hsv_histogram.png")
        print(f"진단 산출물: {overlay_path}")
        print(f"진단 산출물: {hist_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
