# vision — Claude 작업 가이드

착륙지점 객체인식 모듈이다.  
사진/영상에서 착륙 가능한 사각형 영역을 탐지하고, 영상의 경우 시간 축으로 확정한다.

---

## 아키텍처 한 줄 요약

**Config-driven callable pipeline.**  
모든 처리 단계는 `VisionState`를 받아 `VisionState`를 반환하는 callable이다.  
파이프라인 자체는 모듈 내부를 모른다 — 순서와 파라미터만 안다.

---

## 파일 역할 표

### core/

| 파일 | 역할 | 수정 빈도 |
|---|---|---|
| `core/state.py` | `VisionState`, `Detection` 데이터 계약 | 낮음 — 필드 추가 시 신중 |
| `core/runner.py` | `Pipeline` 클래스: `from_config()`, `partial()` | 낮음 |

### modules/

모든 모듈은 `__call__(self, state: VisionState) -> VisionState` 인터페이스를 구현한다.

| 파일 | 클래스 | 읽는 필드 | 쓰는 필드 |
|---|---|---|---|
| `color.py` | `ColorFilter` | `current` | `mask`, `current`, `meta` |
| `illumination.py` | `IlluminationModule` | `current` | `current` |
| `denoise.py` | `DenoiseModule` | `current` | `current` |
| `edge.py` | `EdgeDetector` | `current`, `mask` | `mask` |
| `morphology.py` | `MorphologyModule` | `mask` | `mask` |
| `detector.py` | `RectDetector` | `mask` (없으면 `current`) | `detections`, `meta` |
| `background.py` | `BackgroundSubtractor` | `current`, `mask` | `mask` |
| `tracker.py` | `KalmanTracker` | `detections` | `meta` |
| `fusion.py` | `TemporalFusion` | `detections` | `confirmed`, `meta` |

### 그 외

| 파일 | 역할 |
|---|---|
| `registry.py` | 이름 → 클래스 매핑. **새 모듈 등록은 여기에만** |
| `presets/*.yaml` | 시나리오별 모듈 조합. 코드 수정 없이 파이프라인 변경 |
| `config/default.yaml` | 전체 파라미터 기본값 참조용 (실행에 직접 사용하지 않음) |
| `utils/image_loader.py` | 파일 경로 → BGR numpy 배열 |
| `utils/video_reader.py` | 영상 파일 → 프레임 이터레이터 |
| `utils/visualize.py` | bbox 드로잉, 결과 이미지 저장 |
| `utils/geo_project.py` | 픽셀 좌표 → GPS 좌표 (FC 연동 시 사용) |
| `main.py` | CLI 진입점. 이미지/영상 자동 분기 |

---

## VisionState 필드 사용 규칙

```
original    읽기 전용. 모든 모듈이 수정 금지. 시각화/최종 출력 전용.
current     전처리 모듈이 순차 수정하는 작업 이미지 (BGR).
mask        이진 마스크(0/255). ColorFilter → Edge → Morphology 순으로 갱신.
detections  RectDetector가 채운다. Tracker/Fusion이 읽는다.
confirmed   TemporalFusion만 쓴다. 시간 축으로 확정된 단일 결과.
meta        각 모듈의 진단 정보. 키는 모듈 이름으로 네임스페이스를 지킨다.
```

---

## 모듈 권장 실행 순서

**정지 이미지:**
```
ColorFilter → IlluminationModule → DenoiseModule → EdgeDetector → MorphologyModule → RectDetector
```

**영상 (추가 모듈):**
```
ColorFilter → BackgroundSubtractor → IlluminationModule → DenoiseModule
  → EdgeDetector → MorphologyModule → RectDetector → KalmanTracker → TemporalFusion
```

순서를 바꿀 때: 각 모듈이 읽는 필드가 앞 모듈에 의해 채워지는지 위의 필드 표로 확인한다.

---

## 새 모듈 추가하는 법

```python
# 1. modules/새파일.py 작성
from vision.core.state import VisionState

class NewModule:
    def __init__(self, param: float = 1.0):
        self.param = param

    def __call__(self, state: VisionState) -> VisionState:
        # state 필드 읽고 쓰기
        return state
```

```python
# 2. modules/__init__.py 에 추가
from .새파일 import NewModule
```

```python
# 3. registry.py 에 등록
MODULES = {
    ...
    "new_module": NewModule,   # ← 추가
}
```

이후 preset yaml에서 `new_module:` 키로 바로 사용 가능하다.

---

## import 규칙

위반 시 모듈 교체 가능성이 깨진다.

```
core/       ← numpy, opencv만 허용. 다른 vision 서브패키지 import 금지.
modules/    ← vision.core 만 import. 다른 modules 파일 import 금지.
utils/      ← vision.core 만 import. modules import 금지.
main.py     ← presets 경로 + utils + core 만 import.
```

---

## 시나리오 전환

```python
# 코드 수정 없이 yaml 경로만 바꾼다
pipeline = Pipeline.from_config("vision/presets/video.yaml")
pipeline = Pipeline.from_config("vision/presets/low_light.yaml")
```

---

## 디버그: 부분 파이프라인 실행

```python
pipeline = Pipeline.from_config("vision/presets/single_frame.yaml")

# 앞 N단계까지만 실행해서 중간 상태 확인
partial_state = pipeline.partial(3).run(image)
print(partial_state.mask)       # 3단계까지의 마스크
print(partial_state.meta)       # 각 모듈 진단 정보
```

---

## 테스트 실행

```bash
pytest vision/tests/
```
