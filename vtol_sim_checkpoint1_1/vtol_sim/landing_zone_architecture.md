# 착륙지점 인식 파이프라인 — 설계 문서

> 전처리 모듈화 구조 및 추상화 설계  
> 프로젝트 폴더/객체 구조 설계 시 참고용

---

## 1. 설계 철학

이 파이프라인의 핵심 질문은 다음과 같다.

> "알고리즘이 바뀌어도 파이프라인 코드를 수정하지 않으려면 어떻게 설계해야 하는가?"

답은 **"처리 단계"와 "흐르는 데이터"를 분리**하는 것이다.  
각 단계(모듈)는 인터페이스만 맞추면 되고, 파이프라인은 모듈 내부를 전혀 알지 못한다.  
이 원칙 하나가 모든 구조 결정의 근거가 된다.

---

## 2. 추상화 계층 (3-Layer Abstraction)

```
┌─────────────────────────────────────────────┐
│  Layer 3: Pipeline (오케스트레이터)           │
│  모듈 목록을 순서대로 실행. 내부 로직 무지.   │
├─────────────────────────────────────────────┤
│  Layer 2: PipelineModule (인터페이스)         │
│  process(frame) → frame 만 정의.             │
│  모든 알고리즘이 이 계약을 이행한다.          │
├─────────────────────────────────────────────┤
│  Layer 1: PipelineFrame (데이터 컨텍스트)     │
│  프레임 하나의 모든 상태를 담는 컨테이너.      │
│  모듈 간 직접 의존 없이 이 객체로 통신한다.   │
└─────────────────────────────────────────────┘
```

### Layer 1 — `PipelineFrame` : 데이터 컨텍스트

모듈 간 **유일한 통신 수단**이다.  
모듈 A가 모듈 B를 직접 호출하거나 참조하지 않는다.  
A가 `frame.mask`에 쓰면, B가 `frame.mask`를 읽는다.

```python
@dataclass
class PipelineFrame:
    original:   np.ndarray           # 원본 이미지 (불변)
    frame_id:   int                  # 프레임 번호
    mask:       Optional[np.ndarray] # 처리 중인 마스크
    processed:  Optional[np.ndarray] # 전처리된 그레이스케일
    detections: list[Detection]      # 단일 프레임 검출 후보
    confirmed:  Optional[Detection]  # 시간 융합 후 확정 결과
    metadata:   dict                 # 모듈별 진단 정보
```

`original`은 **읽기 전용 계약**이다. 모든 모듈은 이를 수정하지 않는다.  
`metadata`는 모듈이 자유롭게 진단 정보를 남기는 사이드채널이다.  
중간 단계 디버깅 시 `frame.metadata`만 출력하면 어느 단계에서 실패했는지 즉시 파악된다.

---

### Layer 2 — `PipelineModule` : 추상 인터페이스

```python
class PipelineModule(ABC):
    def __init__(self, enabled: bool = True): ...

    @abstractmethod
    def process(self, frame: PipelineFrame) -> PipelineFrame:
        """구현 필수. 입력과 출력이 모두 PipelineFrame."""
        pass

    def __call__(self, frame):
        if not self.enabled:
            return frame          # enabled=False면 프레임 그대로 통과
        return self.process(frame)
```

이 인터페이스가 주는 것:

| 특성 | 설명 |
|---|---|
| **교체 가능성** | 동일한 역할의 모듈을 다른 구현으로 바꿔도 파이프라인 코드 무변경 |
| **비활성화** | `module.enabled = False` 한 줄로 해당 단계 건너뜀 |
| **테스트 격리** | 모듈 하나를 독립적으로 단위 테스트 가능 |
| **확장** | 새 알고리즘 추가 = 이 클래스 상속 + `process` 구현 |

---

### Layer 3 — `LandingZonePipeline` : 오케스트레이터

```python
class LandingZonePipeline:
    def __init__(self, modules: list[PipelineModule]):
        self.modules = modules

    def run(self, image: np.ndarray) -> PipelineFrame:
        frame = PipelineFrame(original=image, frame_id=self.frame_id, ...)
        for module in self.modules:
            frame = module(frame)   # 모듈 내부를 모른다
        return frame
```

파이프라인은 모듈이 무엇을 하는지 모른다.  
`modules` 리스트의 순서와 길이만 안다.  
따라서 모듈 추가/제거/재배치가 **이 클래스를 수정하지 않고** 가능하다.

---

## 3. 구현된 모듈 목록

| 모듈 클래스 | 역할 | 읽는 필드 | 쓰는 필드 |
|---|---|---|---|
| `ColorMaskModule` | HSV 회색 마스크 생성 | `original` | `mask` |
| `IlluminationModule` | CLAHE 조명 보정 | `original`, `mask` | `processed`, `mask` |
| `DenoiseModule` | Gaussian / Bilateral 노이즈 제거 | `processed` | `processed` |
| `EdgeModule` | Canny 엣지 검출 | `processed`, `mask` | `mask` |
| `MorphologyModule` | Closing / Opening 형태 정제 | `mask` | `mask` |
| `RectDetectorModule` | 윤곽선 → 사각형 후보 검출 | `mask` | `detections` |
| `BackgroundSubtractorModule` | MOG2 정적 배경 분리 (영상) | `original`, `mask` | `mask` |
| `KalmanTrackerModule` | 위치 예측 및 보간 (영상) | `detections` | `metadata` |
| `TemporalFusionModule` | 신뢰도 누적 → 착륙지점 확정 (영상) | `detections` | `confirmed` |

---

## 4. 권장 폴더 구조

```
landing_zone_detector/
│
├── core/                          # 추상화 계층 (변경 빈도 낮음)
│   ├── __init__.py
│   ├── frame.py                   # PipelineFrame, Detection 데이터클래스
│   ├── module.py                  # PipelineModule ABC
│   └── pipeline.py                # LandingZonePipeline 오케스트레이터
│
├── modules/                       # 구체 모듈 구현 (자주 교체/추가)
│   ├── __init__.py
│   ├── color.py                   # ColorMaskModule
│   ├── illumination.py            # IlluminationModule
│   ├── denoise.py                 # DenoiseModule
│   ├── edge.py                    # EdgeModule
│   ├── morphology.py              # MorphologyModule
│   ├── detector.py                # RectDetectorModule
│   ├── background.py              # BackgroundSubtractorModule
│   ├── tracker.py                 # KalmanTrackerModule
│   └── fusion.py                  # TemporalFusionModule
│
├── presets/                       # 시나리오별 파이프라인 조합
│   ├── __init__.py
│   ├── single_frame.py            # build_single_frame_pipeline()
│   ├── video.py                   # build_video_pipeline()
│   └── low_light.py               # build_low_light_pipeline()  (예시)
│
├── config/                        # 파라미터 설정 분리
│   ├── default.yaml               # 기본 파라미터
│   └── high_altitude.yaml         # 고고도 특화 파라미터
│
├── tests/                         # 모듈별 단위 테스트
│   ├── test_color.py
│   ├── test_detector.py
│   └── test_fusion.py
│
└── main.py                        # 진입점
```

### 폴더 분리 기준

`core/`는 **추상화 계약**이다. 이 폴더의 파일은 거의 수정되지 않는다.  
알고리즘이 아무리 바뀌어도 `PipelineFrame`의 필드 구조와 `PipelineModule`의 인터페이스는 유지된다.

`modules/`는 **교체 가능한 구현**이다. 새 알고리즘 실험, 기존 모듈 개선이 이 폴더 안에서 완결된다.  
파일 하나 = 모듈 하나 = 독립적으로 테스트 가능한 단위.

`presets/`는 **조합 레시피**다. 어떤 모듈을 어떤 순서로 연결할지 결정하는 팩토리 함수들.  
현장 조건(주야간, 고도, 카메라 종류)마다 다른 프리셋을 정의한다.

`config/`는 **파라미터를 코드에서 분리**한다. 임계값 하나 바꾸려고 코드를 수정하지 않아도 된다.  
yaml 파일만 교체하면 동일한 파이프라인이 다른 환경에서 동작한다.

---

## 5. 확장 패턴: 새 모듈 추가하는 법

예를 들어 원근 보정(Homography) 모듈을 추가하는 경우:

```python
# modules/homography.py

from core.module import PipelineModule
from core.frame import PipelineFrame
import cv2, numpy as np

class HomographyModule(PipelineModule):
    """
    검출된 4각형을 정면 뷰로 펼쳐 재검증.
    왜곡된 카메라 각도에서도 정확한 사각형 판별 가능.
    """

    def process(self, frame: PipelineFrame) -> PipelineFrame:
        for det in frame.detections:
            if len(det.corners) != 4:
                continue
            pts = det.corners.reshape(4, 2).astype(np.float32)
            dst = np.array([[0,0],[100,0],[100,100],[0,100]], dtype=np.float32)
            M = cv2.getPerspectiveTransform(pts, dst)
            warped = cv2.warpPerspective(frame.original, M, (100, 100))
            det.metadata['warped'] = warped   # 필요시 후속 모듈이 활용
        return frame
```

파이프라인에 꽂기:

```python
from modules.homography import HomographyModule

pipeline = LandingZonePipeline([
    *build_single_frame_pipeline().modules,
    HomographyModule(),              # 끝에 추가
])
```

`LandingZonePipeline`, `PipelineFrame`, 기존 모듈 중 **어느 것도 수정하지 않는다.**

---

## 6. 의존 관계 규칙

```
core/          ← 외부 의존 없음. numpy, opencv만 허용.
modules/       ← core/ 만 import. 다른 module/ import 금지.
presets/       ← core/ + modules/ import 가능.
main.py        ← presets/ 만 import.
```

`modules/` 안에서 다른 모듈을 직접 import하면 교체 가능성이 깨진다.  
모듈 간 데이터 공유는 반드시 `PipelineFrame` 필드를 통해서만 한다.

---

## 7. 핵심 요약

```
알고리즘 교체  →  modules/ 안에서 완결
파라미터 조정  →  config/ yaml 수정
시나리오 전환  →  presets/ 함수 선택
새 단계 추가   →  PipelineModule 상속 + process() 구현
파이프라인 자체 →  건드릴 일이 거의 없다
```

이 구조의 목표는 하나다.  
**"가장 자주 바뀌는 것(알고리즘)"이 "가장 덜 바뀌어야 하는 것(파이프라인 구조)"에 영향을 주지 않도록 한다.**
