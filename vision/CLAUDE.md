# vision — Claude 작업 가이드

착륙지점 객체인식 모듈이다.  
사진/영상에서 착륙 가능한 사각형 영역을 탐지하고, 영상의 경우 시간 축으로 확정한다.

> ⚠️ **재설계 진행 중.** 이 문서는 현재 구현된 "무채색 사각형 검출기" 임시 틀을 설명한다.
> 정밀착륙(ArUco/색 2단 검출, 폐루프 유도, 변경내성·관측성) 방향은 **`docs/vision_plan.md`**를
> 먼저 읽을 것. 이 틀은 폐기가 아니라 ② 조난자 구역 색 파이프라인 + ArUco 모듈 컨테이너로 재편된다
> (계획서 §12). `geo_project.pixel_to_gps`는 폐기 예정.

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

## 테스트 방법 & 단위별 테스트 규칙

### 환경 준비 (최초 1회)
vision 은 `cv2`(opencv)·`numpy`·`PyYAML` 이 필요하다. 시스템 파이썬에 없을 수 있으니(특히 Windows 개발 호스트) **venv + requirements 로 격리 설치**한다.

```bash
python -m venv .venv                       # 저장소 루트 (.gitignore 제외됨)
# Windows PowerShell:  .\.venv\Scripts\Activate.ps1
# WSL/RPi(bash):       source .venv/bin/activate
python -m pip install -r vision/requirements.txt
```

⚠️ **OpenCV 변형은 환경별로 다르다**(`vision/requirements.txt` 주석):
- 디스플레이 있는 **개발컴/개발노트북**(`--display window` 사용): `opencv-python`
- 디스플레이 없는 **rpi/headless**: `opencv-python-headless`

### 실행
```bash
pytest vision/tests/            # 전체
pytest vision/tests/ -q -k main # 특정만
```

### 어느 환경에서 무엇을 검증하나
- **개발컴 / 개발노트북 / WSL:** 단위·파이프라인 로직 전체(cv2 CPU로 충분). TDD 주 무대.
- **RPi(실기체):** 위 + 헤드리스 안전(`--display none` 크래시 없음)·실카메라 FrameSource·성능/지연. **`opencv-python-headless` 로.**

### 단위별 필수 테스트 (신규/수정 시 함께 갱신)
모든 모듈은 `__call__(state)->state`. 각 단위 테스트는 최소로 아래를 검증한다.

| 단위 | 필수 검증 | 현재 |
|---|---|---|
| core/runner `Pipeline` | from_config 로드·실행순서·`partial(N)`·unknown module→ValueError | ✅ test_pipeline |
| registry | 등록 이름 전부 실제 클래스 매핑·중복 없음 | ❌ TODO |
| color `ColorFilter` | 모드별 mask 생성·임계값 경계·meta | ✅ test_color (gray+color, 빨강 Hue랩어라운드 미지원은 §5.4 blind spot로 별도 회귀테스트 기록) |
| illumination | current 변형·형상/채널 보존·meta | ❌ TODO |
| denoise | current 변형·형상 보존 | ❌ TODO |
| edge `EdgeDetector` | current/mask→mask 갱신·빈입력 | ❌ TODO |
| morphology | mask 갱신·커널크기 효과·빈 mask | ❌ TODO |
| detector `RectDetector` | mask→detections·min_area 필터·빈 mask 0검출·meta | ✅ test_detector |
| background | 연속프레임 mask 갱신 | ❌ TODO |
| tracker `KalmanTracker` | detections→meta 추적·연속성 | ❌ TODO |
| fusion `TemporalFusion` | detections→confirmed 시간확정·흔들림 억제 | ❌ TODO |
| utils/image_loader | 경로→BGR ndarray·없는 파일 에러 | ❌ TODO |
| utils/video_reader | 프레임 이터레이트·fps·컨텍스트 종료 | ❌ TODO |
| utils/visualize | draw_detections 형상·save_result 파일 생성 | ❌ TODO |
| utils/geo_project | **폐기 예정(plan §12) — 신규 테스트 금지** | 폐기 |
| main.py | `--display` 게이팅: **none=imshow 0회**(헤드리스 안전 불변식)·file→output 강제·stream 미구현 | ✅ test_main |

**공통 규칙 (모든 모듈 테스트):**
1. **선언 필드 계약** — 위 파일표대로 "읽는 필드"만 읽고 "쓰는 필드"만 쓴다.
2. **meta 네임스페이스** — `state.meta["<모듈이름>"]` 기록 확인.
3. **빈/경계 입력** — 검은 이미지·빈 mask에서 크래시 없이 합리적 출력(대개 0검출).
4. **결정론(plan §7.5)** — 같은 입력·같은 config → 같은 출력. 골든셋 회귀는 데이터 수집 후.

**새 모듈 추가 시:** 위 4개 공통 규칙을 담은 `tests/test_<모듈>.py` 를 **같은 커밋에** 추가한다.
