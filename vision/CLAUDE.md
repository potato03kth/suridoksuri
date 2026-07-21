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
| `vertiport_field.py` | `WhiteFieldDetector` | `mask` | `detections`, `meta` |
| `vertiport_v.py` | `BlackVMatcher` | `original`, `detections` | `detections`, `meta` |
| `vertiport_ring.py` | `RedRingDetector` | `original`, `detections` | `detections`, `meta` |

### 그 외

| 파일 | 역할 |
|---|---|
| `registry.py` | 이름 → 클래스 매핑. **새 모듈 등록은 여기에만** |
| `presets/*.yaml` | 시나리오별 모듈 조합. 코드 수정 없이 파이프라인 변경 |
| `config/default.yaml` | 전체 파라미터 기본값 참조용 (실행에 직접 사용하지 않음) |
| `presets/distress_coarse.yaml` | ② 조난자 구역 coarse(§5.3) — 전용 모듈 없이 기존 `ColorFilter(mode=color)`+`RectDetector` 조합. §7.9 항목7 골든셋용으로 신규 추가(신규 검출 로직 아님) |
| `utils/image_loader.py` | 파일 경로 → BGR numpy 배열 |
| `utils/video_reader.py` | 영상 파일 → 프레임 이터레이터 |
| `utils/visualize.py` | bbox 드로잉, 결과 이미지 저장 |
| `utils/geo_project.py` | 픽셀 좌표 → GPS 좌표 (FC 연동 시 사용) |
| `utils/logging.py` | 이중싱크 사람로그(터미널+.log 로테이션) + provenance 헤더(config+git해시+캘리브id) (§7.4/§7.3). `main.py`/`replay.py`에 연결됨 |
| `utils/blackbox.py` | 프레임별 JSONL 블랙박스 + 거절이유 로깅. bounded queue+drop-oldest 비차단 (§7.4). `main.py`/`replay.py`에 연결됨 |
| `utils/stream.py` | `MjpegStreamer` — 라이브 저해상 MJPEG-over-HTTP 스트림(§7.9 항목5, "작동영상 피드백 3경로" (b)). `push_frame()`은 bounded queue+drop-oldest 비차단(`blackbox.py`와 동일 패턴 재사용). 다운스케일·인코딩·HTTP 서빙은 전부 별도 스레드. `main.py --display stream`/`replay.py --display stream`으로 opt-in 연결(항상 켜지지 않음). 기본값 결정 근거는 아래 "라이브 스트림 어댑터 기본값" 참조 |
| `utils/frame_source.py` | `FrameRecord` + `LiveFrameSource`/`DirFrameSource`/`BagFrameSource` 어댑터 + `open_dir_or_bag()` 팩토리 (§7.2/§7.5/§7.9 항목4). Live=실카메라(재시도 후 `ConnectionError`), Dir=녹화폴더(프레임파일+선택적 `telemetry.jsonl`), Bag=단일 비디오파일(+선택적 사이드카 `<basename>.jsonl`) |
| `main.py` | CLI 진입점. 이미지/영상 자동 분기. `--log-dir`/`--log-name`으로 이중싱크 로거+JSONL 블랙박스 실행(항상 on). `--display stream`으로 `MjpegStreamer` opt-in(§7.9 항목5) |
| `replay.py` | 오프라인 재생 CLI(`python -m vision.replay <녹화폴더\|bag> --preset ...`, §7.9 (a)). `open_dir_or_bag`로 Dir/Bag 자동판별 → 동일 `Pipeline`으로 재생 → 로거+블랙박스 기록. **결정론적**(§7.5). `--display stream`으로 `MjpegStreamer` opt-in(§7.9 항목5) |
| `tools/rpi_capture.py` | RPi 헤드리스 캘리브레이션 촬영 — 저해상도 스냅샷 자동갱신(브라우저) + 촬영 트리거(버튼/Enter). GStreamer `libcamerasrc` 서브프로세스 기반. **⚠️ 2026-07-21 확인: 이 RPi에서 현재 작동 불가** — libcamera가 PiSP IPA 없이 빌드돼 있어 `libcamerasrc`가 카메라를 못 봄(picamera2도 동일 원인으로 막힘). 재작업 필요 — 상세·대안 4개는 메모리 `project_rpi5_ubuntu_camera_stack.md` |
| `tools/jsonl_view.py` | JSONL 블랙박스 뷰어/플롯 최소본(§7.9 항목6). `BlackBoxLogger`가 남긴 `.jsonl`을 읽어 시간축 score/latency/state 3단 플롯을 PNG로 저장(`matplotlib` Agg 백엔드, headless-safe). `python vision/tools/jsonl_view.py <jsonl> [--output out.png] [--x-axis ts\|frame_id]`. **하드웨어 의존 없음** — `rpi_capture.py`와 달리 `.venv`에 설치되고(`matplotlib` in `requirements.txt`) `tests/test_jsonl_view.py` 대상이다(tools/의 "CI/pytest 대상 아님" 규칙은 RPi 하드웨어 전용 스크립트에만 적용). |
| `tests/golden/` | 골든셋 회귀 픽스처(§7.9 항목7). `<타겟>/<고도>/frame_NNN.png`+`labels.json` — 구조·스키마·현재 들어있는 것·재생성법은 `tests/golden/README.md`. **⚠️ 전부 합성(synthetic) 데이터** — 실촬영 아님(카메라 브링업 전, `docs/vision_status.md`). `tests/golden/generate_synthetic.py`가 생성 소스(pytest 대상 아님, 수동 재생성 도구) |

---

## 라이브 스트림 어댑터 기본값 (§7.9 항목5, `utils/stream.py`)

vision_plan.md §7.9가 정확한 해상도/포트를 못박지 않아 세션 지시에 따라 합리적 기본값으로 정하고 여기에 기록한다:

- **해상도:** 640x480(VGA) 박스 — 정확히 640x480으로 리사이즈(letterbox)하지 않고 **종횡비를 유지한 채 박스 안에 맞추는 축소**(업스케일 없음). 관찰용 브라우저 `<img>` 태그가 알아서 크기를 맞추므로 letterbox 패딩은 불필요한 복잡도로 판단.
- **포트:** 8080 (`--stream-port`로 변경 가능. `0`을 주면 OS가 임시 포트를 골라준다 — 테스트/포트충돌 회피용).
- **바인딩 주소:** `0.0.0.0`(모든 인터페이스 — RPi가 실제 배포될 네트워크에서 랩탑이 접속 가능해야 하므로). `--stream-host 127.0.0.1`로 로컬만 제한 가능.
- **JPEG quality:** 80, **큐 길이:** 2 — 지연 누적보다 최신 프레임 우선(관찰이 목적, 무손실 기록이 목적 아님. 상시 기록은 `--output`/mp4 덤프 경로가 담당).
- 비차단 큐 패턴은 `utils/blackbox.py`의 `_DropOldestQueueHandler`와 동일 설계(evict-then-insert를 락으로 원자화 — 여러 producer 스레드 동시 push 시 `queue.Full` 경쟁 방지, `blackbox.py`는 단일 로거 호출 경로라 이 레이스가 실질적으로 안 나타나 그대로 둠).

---

## VisionState 필드 사용 규칙

```
original    읽기 전용. 모든 모듈이 수정 금지. 시각화/최종 출력 + 캐스케이드 단계별 원본색상 조회용.
current     전처리 모듈이 순차 수정하는 작업 이미지 (BGR).
mask        이진 마스크(0/255). ColorFilter → Edge → Morphology 순으로 갱신.
detections  RectDetector가 채운다. Tracker/Fusion이 읽는다. 캐스케이드형 검출기는 이전 단계 detections를 읽어 ROI로 쓰고 자기 결과로 덮어쓴다(§버티포트 coarse 캐스케이드).
confirmed   TemporalFusion만 쓴다. 시간 축으로 확정된 단일 결과.
meta        각 모듈의 진단 정보. 키는 모듈 이름으로 네임스페이스를 지킨다.
```

**주의:** `ColorFilter`는 `current`를 자기 mask로 bitwise_and 해버려 mask 밖 픽셀 정보가 사라진다.
버티포트 캐스케이드(`vertiport_v.py`/`vertiport_ring.py`)처럼 뒤 단계가 앞 단계 마스크에 안 걸린 색상을
봐야 하는 경우 `current` 대신 원본이 보존된 `original`을 읽는다.

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
replay.py   ← main.py와 동일 규칙(presets 경로 + utils + core 만). main.py와 헬퍼 상호 import 안 함(각자 얇게 중복 허용).
tools/      ← 이 규칙 밖. RPi 하드웨어 전용 운영스크립트(예: rpi_capture.py의 picamera2/GStreamer) — .venv에 안 깔림, CI/pytest 대상 아님.
              단, 하드웨어 비의존 CLI 도구(예: jsonl_view.py)는 예외 — .venv 설치 + pytest 대상.
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
| vertiport_field `WhiteFieldDetector` | mask→원형 blob 검출·원형도 필터·중심/반지름 meta | ✅ test_vertiport_field |
| vertiport_v `BlackVMatcher` | original 내 어두운 영역 matchShapes 검증·1차 bbox 밖 배경 오탐 배제·불일치 시 detections 제거 | ✅ test_vertiport_v |
| vertiport_ring `RedRingDetector` | 빨강 Hue 양끝 게이팅(랩어라운드 대응)·최소외접원 피팅·중심/반지름 meta | ✅ test_vertiport_ring |
| 버티포트 coarse 캐스케이드 통합(`presets/vertiport_coarse.yaml`) | 3단 전체 파이프라인 end-to-end·단계별 meta 기록·빈 이미지 0검출 | ✅ test_vertiport_cascade |
| utils/image_loader | 경로→BGR ndarray·없는 파일 에러 | ❌ TODO |
| utils/video_reader | 프레임 이터레이트·fps·컨텍스트 종료 | ❌ TODO |
| utils/visualize | draw_detections 형상·save_result 파일 생성 | ❌ TODO |
| utils/geo_project | **폐기 예정(plan §12) — 신규 테스트 금지** | 폐기 |
| utils/logging | 이중싱크 핸들러 구성·콘솔레벨이 파일레벨 억제 안 함·재호출 시 핸들러 중복 안 됨·provenance에 git해시/config/캘리브id | ✅ test_logging |
| utils/blackbox | 프레임/거절이유 JSONL 기록·bounded queue drop-oldest(최신 안 잃음)·close() 큐 가득해도 안전 | ✅ test_blackbox |
| utils/stream `MjpegStreamer`(§7.9 항목5) | 실제 HTTP 서버 기동 → 실제 프레임 push → 실제 클라이언트로 `/stream` 접속해 진짜 MJPEG 바이트 수신·`cv2.imdecode` 디코드 성공·VGA 박스 축소(종횡비 유지, 업스케일 없음)·`push_frame()` 비차단(클라이언트 없음/느린 클라이언트 붙어있어도 논-블로킹, 실측 시간)·`start()` 전 `push_frame` 안전 no-op·idempotent stop/restart | ✅ test_stream |
| utils/frame_source | Dir/Bag: 실제 파일→실제 프레임 디코딩·순서 결정론·telemetry.jsonl(사이드카 포함) frame_id 매칭·빈/누락 입력 에러. Live: 연결 실패 시 재시도 후 `ConnectionError`·읽기 실패 시 `ConnectionError`·`open_dir_or_bag` 디렉터리/파일 자동판별 | ✅ test_frame_source |
| main.py | `--display` 게이팅: **none=imshow 0회**(헤드리스 안전 불변식)·file→output 강제·stream 미구현 · **로거+JSONL 블랙박스 실연결**: 실행 시 실제 `.log`/`.jsonl`이 디스크에 생성되고 detections/latency/provenance가 올바름 | ✅ test_main |
| replay.py | `open_dir_or_bag`로 Dir/Bag 자동판별 재생·실제 프레임 처리로 JSONL(telemetry 포함)/사람로그 실생성·`--output` 지정 시 실제 mp4 기록 | ✅ test_replay |
| tools/jsonl_view.py | 실제 `main.py` 실행으로 만든 진짜 JSONL 로드·행 수=JSONL type=frame 행 수 일치·score/latency 라인 포인트 수=행 수(결측은 nan 구멍, 이어붙이지 않음)·state 미기록 시 안내 텍스트·rejection→세로선·PNG 실파일 생성 | ✅ test_jsonl_view |
| tests/golden 회귀(§7.9 항목7) | `vision.replay.run_replay()`로 골든셋(§ tests/golden/README.md) 실제 재생 → JSONL 검출 개수가 `labels.json` 기대값과 일치·캐스케이드 단계별 meta도 실제 `Pipeline.run()`으로 검증. 몽키패치 없음(실제 파이프라인) | ✅ test_golden_regression |

**공통 규칙 (모든 모듈 테스트):**
1. **선언 필드 계약** — 위 파일표대로 "읽는 필드"만 읽고 "쓰는 필드"만 쓴다.
2. **meta 네임스페이스** — `state.meta["<모듈이름>"]` 기록 확인.
3. **빈/경계 입력** — 검은 이미지·빈 mask에서 크래시 없이 합리적 출력(대개 0검출).
4. **결정론(plan §7.5)** — 같은 입력·같은 config → 같은 출력. **골든셋 회귀 스캐폴드는 2026-07-21c에 합성 데이터로 시작됨**(`tests/golden/README.md`) — 실기체 데이터는 카메라 브링업 이후 교체 예정.

**새 모듈 추가 시:** 위 4개 공통 규칙을 담은 `tests/test_<모듈>.py` 를 **같은 커밋에** 추가한다.
