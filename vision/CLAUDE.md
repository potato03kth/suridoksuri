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
| `presets/distress_coarse.yaml` | ② 조난자 구역 coarse(§5.3) — 전용 모듈 없이 기존 `ColorFilter(mode=color)`+`RectDetector` 조합. §7.9 항목7 골든셋용으로 신규 추가(신규 검출 로직 아님). `min_area`/`max_area`는 실측 스펙 기반 도출값 — 근거는 아래 "distress_coarse.yaml min_area/max_area 도출 근거" 절 |
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
| `tools/rpi_capture.py` | RPi 헤드리스 캘리브레이션 촬영 — 저해상도 스냅샷 자동갱신(브라우저) + 촬영 트리거(버튼/Enter). **2026-07-22b에 GStreamer `libcamerasrc`(작동 불가였음, libcamera PiSP IPA 결여) 대신 V4L2 RAW 직접 캡처+수동 디베이어로 전면 재작성해 브링업 완료** — media-ctl/v4l2-ctl로 rp1-cfe 파이프라인 직접 구성. 대상 media 디바이스(`/dev/mediaN`)는 부팅마다 번호가 바뀔 수 있어 하드코딩 대신 매 호출 동적 탐색(2026-07-22d, 아래 "media 디바이스 동적 탐색" 절). gray-world 화이트밸런스 보정 포함(아래 절). **수동 초점/노출/게인 제어 + 초점 스윕 도구 포함(2026-07-22e, 아래 "수동 초점/노출/게인 제어" 절)** — libcamera 우회 경로라 연속 AF/AE가 없어 방치돼 있던 것에 대한 대응. 상세 경과는 메모리 `project_rpi5_ubuntu_camera_stack.md` |
| `tools/jsonl_view.py` | JSONL 블랙박스 뷰어/플롯 최소본(§7.9 항목6). `BlackBoxLogger`가 남긴 `.jsonl`을 읽어 시간축 score/latency/state 3단 플롯을 PNG로 저장(`matplotlib` Agg 백엔드, headless-safe). `python vision/tools/jsonl_view.py <jsonl> [--output out.png] [--x-axis ts\|frame_id]`. **하드웨어 의존 없음** — `rpi_capture.py`와 달리 `.venv`에 설치되고(`matplotlib` in `requirements.txt`) `tests/test_jsonl_view.py` 대상이다(tools/의 "CI/pytest 대상 아님" 규칙은 RPi 하드웨어 전용 스크립트에만 적용). |
| `tools/calib_analyze.py` | **신설(2026-07-23)** — `calib_capture.py`가 만든 촬영 세트(`<raw_root>/<set>/<distance>m/<stem>.{png,json}`)를 캘리브레이션 아티팩트로. 그룹(set,distance_m)별 `cv2.calibrateCamera` → 사진별 재투영오차로 불량컷 색출·이상치 제외 재캘리브레이션 → **`fx`/`fy` vs `LensPosition` 직선적합으로 L=0(무한대) 외삽**(핵심 목적 — 기체 운용고도 10~40m는 사실상 무한대)·얇은렌즈 물리 일관성 검사(`b/a`→mm, IMX708급 통상 초점거리와 비교)·디옵터 가정(`LensPosition≈1/distance_m`) 직접 검정 → 세트B(2.5m, 최대커버리지)와 외삽값 대조 → fy/fx·주점·HFOV 등 정합성 검사(실패해도 크래시 없이 보고) → 진단 플롯 3종(Agg, `vision/results/`) → `vision/calibration/<camera_id>/<calib_id>.yaml` 아티팩트(그룹별 결과 전부 보존, `recommended`=fx/fy 외삽절편+세트B의 cx/cy/dist_coeffs, 근거는 yaml `note`). 모든 임계값 CLI 파라미터(매직넘버 금지, §7.3). **하드웨어 의존 없음** — `jsonl_view.py`와 동일한 예외로 `.venv` 설치 + `tests/test_calib_analyze.py` 대상. 정확성은 진짜 K/dist를 아는 합성 체스보드 투영 왕복 테스트로 담보(실촬영 사진은 이 스크립트 작성 시점에 아직 없었음 — `docs/vision_camera_bringup.md`). `recommended-source` CLI로 폴백/강제선택 가능, 그룹<2면 적합 생략 + LensPosition 최저 그룹 폴백(사유 yaml 명시, §9 견고성). |
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

## rpi_capture.py gray-world 화이트밸런스 (2026-07-22, `tools/rpi_capture.py`)

`debayer_to_bgr8()`는 libcamera(ISP)를 완전히 우회하는 V4L2 raw 직접 캡처 경로라 화이트밸런스가
전혀 적용되지 않는다 — 실촬영 검증(2026-07-22b, `docs/vision_status.md`)에서 실제로 강한 초록
색편향이 관측됐다. 나중에 실측 데이터로 HSV 색상 탐지(`distress_coarse.yaml` 등)를 검증할 때
이 편향이 왜곡 요인이 되므로 raw 캡처 후처리 단계에서 고쳐둔다.

- **Gray-world 가정 채택 근거:** "전체 이미지의 R/G/B 채널 평균이 같아야 한다"는 가정 기반의
  채널별 게인 스케일링(`apply_gray_world_white_balance()`)만으로 충분하다고 판단 — 체커보드 등
  특정 기준 패치나 조명 스펙트럼 추정이 필요 없어 numpy만으로 구현 가능하고, 새 무거운 의존성
  (colour-science 등)이 불필요하다. 캘리브레이션 촬영(체커보드 코너 검출)은 흑백 대비만 보므로
  이 보정 유무의 영향이 적지만, 그 외 모든 실측 데이터 수집(HSV 색상 탐지 검증 등)에는 영향을
  주므로 기본값은 켜짐(`white_balance=True`)으로 정했다.
- **§5.5 "흰 박스 화이트 앵커"(`vision_plan.md`)와의 관계:** 이 gray-world 보정은 raw 캡처
  단계의 범용 1차 보정이고, §5.5의 흰 박스/기준색 Hue-shift 앵커는 파이프라인 레벨의 프레임별
  정밀 색 항상성 전략이다 — 서로 다른 레이어, 대체 관계 아님. gray-world는 흰 박스 앵커가 아직
  파이프라인에 연결되기 전 단계(raw 촬영/캘리브레이션/골든셋 수집)에서 편향을 줄이는 용도.
- **구현:** `apply_gray_world_white_balance(bgr8)`를 `debayer_to_bgr8()`와 분리된 순수 함수로
  둬 하드웨어 없이 합성 이미지로 단위테스트 가능(`vision/tests/test_rpi_capture.py`). 채널 평균의
  평균을 회색 기준값으로 삼아 각 채널에 `gray/mean_c` 게인을 곱하고 0~255로 클립한다(픽셀별
  보정이 아니라 프레임 전체 통계 1개만 사용하는 가장 단순한 변형). 채널 평균이 0인 완전 검은
  이미지는 0나눗셈을 피해 원본을 그대로 반환.
- **CLI:** `--white-balance`/`--no-white-balance`(`argparse.BooleanOptionalAction`, 기본
  `--white-balance`)로 opt-out 가능. `capture_frame_bgr()`/`_CaptureSession`/`--single-shot` 전부
  이 플래그를 관통시킨다.
- **실측 검증 완료(2026-07-22d, 실카메라):** RPi에서 `--single-shot`을 보정 켜고/끄고 각각
  실행해 B/G/R 채널 평균 비교 — **보정 끄기: B=64.96 G=101.45 R=69.31 (채널간 spread 36.49)**
  / **보정 켜기: B=75.02 G=78.12 R=76.15 (spread 3.10)** — spread가 약 11.8배 줄어 gray-world
  가정이 이 카메라의 실제 초록 편향을 효과적으로 중화함을 확인. (2026-07-22c 세션은 RPi가
  세션 내내 오프라인이라 이 검증을 못 하고 넘겼었음 — 코드/합성테스트만으로 "완료" 처리하지
  않고 다음 세션 최우선 항목으로 명시적으로 남겨뒀던 것이 2026-07-22d에 닫힘.)

---

## rpi_capture.py media 디바이스 동적 탐색 (2026-07-22d, `tools/rpi_capture.py`)

`configure_pipeline()`이 media-ctl/v4l2-ctl로 카메라 파이프라인을 구성할 때 대상 media 디바이스
(`/dev/mediaN`)를 예전엔 `_MEDIA_DEVICE = "/dev/media1"`로 하드코딩했었다 — **이 번호가
재부팅/재연결마다 바뀌는 것이 실기체로 확인돼(2026-07-22d) 하드코딩을 폐기하고 매 호출마다
동적으로 찾도록 바꿨다.**

- **실측으로 확인한 사실:** 카메라 브링업 세션(2026-07-22b)에서는 `/dev/media1`이 카메라
  파이프라인(driver `rp1-cfe`)이었는데, 이번 세션(2026-07-22d) 같은 RPi에서 다시 확인하니
  `/dev/media1`은 `pispbe`(ISP 백엔드, 이 캡처 경로에 안 쓰임)로 바뀌어 있었고 카메라 파이프라인은
  `/dev/media3`으로 옮겨가 있었다. `_MEDIA_DEVICE`가 옛 값을 그대로 가리키고 있어
  `configure_pipeline()`의 첫 `media-ctl -l` 호출이 `CalledProcessError`로 실패하는 것을
  실제로 재현했다(RPi에 존재하지 않는/엉뚱한 media 디바이스를 대상으로 링크 조작을 시도).
- **왜 바뀌는가:** 이 RPi에는 media 디바이스가 5개 있다 — `pispbe` 인스턴스 2개(`/dev/media0`,
  `/dev/media1`, ISP 백엔드), `rp1-cfe` 인스턴스 2개(CSI 포트 cam0/cam1 각각 하나씩, 이 중
  카메라가 실제로 연결된 포트만 센서 엔티티를 가짐), `rpivid` 1개(비디오 디코더, 무관). 리눅스
  미디어 컨트롤러 디바이스 번호는 드라이버가 프로브(등록)되는 순서대로 순차 부여되는데, 이
  프로브 순서는 커널 모듈 로드 순서에 좌우될 수 있어 부팅마다 같은 물리 하드웨어가 다른 번호를
  받을 수 있다 — 이번 세션은 이 현상 자체의 근본 커널 메커니즘(모듈 로드 순서가 왜 바뀌는지)까지는
  규명하지 않았고, 재현된 사실(번호가 실제로 바뀜)만 근거로 하드코딩을 폐기하는 쪽으로 대응했다.
- **판별 기준(실측으로 확정, 찍은 게 아님):** `driver` 이름이 `rp1-cfe`인 것만으로는 부족하다 —
  연결 안 된 CSI 포트도 driver는 똑같이 `rp1-cfe`로 나오지만 센서 엔티티 자체가 토폴로지에 없다
  (`"csi2 (8 pads, 0 link, 0 routes)"`처럼 링크·엔티티가 텅 빔). 그래서 **driver == `rp1-cfe`
  이고, 토폴로지에 `imx708`(이 카메라 센서) 엔티티가 실제로 존재하는 media 디바이스**를 찾는다.
  판별 함수 `_media_ctl_topology_has_camera()`가 이 둘을 문자열 파싱으로 확인한다(정규식 —
  `driver` 줄 + `- entity N: imx708` 줄).
- **구현:** `_find_cfe_media_device()`가 `_iter_media_device_paths()`(=`/dev/media*` 정렬 목록)를
  순회하며 각 디바이스에 `media-ctl -p`를 실행해 위 판별 함수에 넘긴다. **캐시하지 않는다** —
  `configure_pipeline()`이 호출될 때마다(매 프레임 캡처마다) 새로 탐색해, 프로세스 실행 도중
  핫스왑 등으로 바뀌어도 항상 최신 상태를 본다(RPi 하드웨어 특성상 실행 중 안 바뀔 가능성이
  높지만, 캐시로 얻는 이득보다 정확성을 우선했다). 못 찾으면 확인한 모든 디바이스와 각각의
  driver명을 에러 메시지에 담아 `RuntimeError`로 실패 — 다음에 이 문제가 재발해도 원격 세션이
  `for d in /dev/media*; do media-ctl -d $d -p; done`을 다시 돌릴 필요 없이 에러 메시지만으로
  진단 가능하게 함.
- **테스트:** 순수 파싱/탐색 로직(`_media_ctl_topology_has_camera`/`_media_ctl_driver_name`/
  `_find_cfe_media_device`)은 하드웨어 없이 검증 가능 — `vision/tests/test_rpi_capture.py`가
  실기체에서 실제로 받은 `media-ctl -p` 출력(연결된 rp1-cfe/연결 안 된 rp1-cfe/무관한 pispbe
  3종)을 픽스처로 써서 판별 로직을 검증하고, `_run`/`_iter_media_device_paths`를 몽키패치해
  `_find_cfe_media_device()`의 순회·에러 메시지·CalledProcessError 스킵 동작까지 검증한다.
  **`_VIDEO_DEVICE`/`_CSI2_SUBDEV`/`_SENSOR_SUBDEV`(video0/subdev0/subdev2)는 이번 조사에서
  안정적으로 재현돼 그대로 하드코딩 유지** — media 디바이스만큼 자주 흔들리는지는 미확인, 향후
  같은 증상이 재현되면 같은 패턴으로 확장 검토.
- **실측 재검증(2026-07-22d):** 이 동적 탐색이 적용된 코드로 RPi에서 `--single-shot`이 다시
  성공함을 확인(`/dev/media3`을 자동으로 찾아 사용) — 화이트밸런스 채널 평균도 위 절의 수치와
  함께 재확인됨(보정 켜기 spread 3.10, 끄기 spread 36.49 — 같은 실행에서 같이 검증됨).

---

## rpi_capture.py 수동 초점/노출/게인 제어 (2026-07-22e, `tools/rpi_capture.py`)

**문제:** 사용자가 체커보드를 40cm/210cm 거리에서 촬영했더니 **두 거리 모두 프레임 전체가
초점이 안 맞고 과하게 어두웠다.** 원인은 이 V4L2 raw 직접 캡처 경로가 libcamera를 완전히
우회해 연속 AF/AE가 전혀 없다는 것 — 오토포커스 렌즈(dw9807 VCM, `/dev/v4l-subdev3`)의
`focus_absolute`가 기본값(480)에서 **한 번도 움직인 적이 없었고**, 센서(`/dev/v4l-subdev2`)의
`exposure`/`analogue_gain`도 기본값(874/112, `analogue_gain`은 최솟값)에 방치돼 있었다.

### 추가된 것

- **`set_focus_absolute(value, settle_s)`** — `/dev/v4l-subdev3`에 `focus_absolute` 설정 후
  물리 이동 정착시간만큼 `time.sleep`. `--focus`(0~1023)/`--focus-settle-ms`(기본 200) CLI로
  노출.
- **`set_exposure_gain(exposure, gain)`** — `/dev/v4l-subdev2`에 `exposure`/`analogue_gain`
  설정(한쪽만 줘도 됨, 한 번의 v4l2-ctl 호출로 합쳐 보냄). `--exposure`/`--gain` CLI로 노출.
  **범위를 하드코딩하지 않음** — exposure 최댓값은 센서 모드의 vertical_blanking(프레임
  길이)에 따라 달라질 수 있어(다른 모드에 실측 범위를 그대로 적용하면 유효값을 잘못 거부할
  위험), 값을 그대로 v4l2-ctl에 넘기고 범위 위반이면 v4l2-ctl 자체가 거부하게 둔다. focus는
  렌즈 드라이버 고유 범위(센서 모드 무관)라 `_FOCUS_MIN`/`_FOCUS_MAX`(0/1023)로 미리 검증한다.
- **HTTP 미리보기 페이지에 실시간 조정 UI 추가** — 체커보드를 들고 화면을 보면서 위치와 초점을
  동시에 맞춰야 하므로 정적 CLI 인자만으로는 부족했다. focus/exposure/gain 숫자입력폼 1개 +
  focus 전용 -10/+10 빠른버튼 2개(`GET /controls?focus=..&exposure=..&gain=..` → 303 리다이렉트,
  새 JS 없이 기존 `_PAGE`의 순수 HTML 폼 패턴 유지 — 과설계 금지 지시에 따라 슬라이더 등은
  추가하지 않음). `_CaptureSession`은 "원하는 값"과 "마지막 적용값"을 분리해 값이 실제로
  바뀐 경우에만 하드웨어에 적용(불필요한 v4l2-ctl 호출·정착시간 낭비 방지, 스트리밍 직렬화
  락과도 충돌 안 함).
- **`--focus-sweep START:END:STEP`** — focus_absolute를 스윕하며 각각 촬영, 중앙 크롭
  (`--sweep-roi`, 기본 0.6) 기준 라플라시안 분산(`laplacian_sharpness`)이 가장 높은 값을
  보고. 순수 함수(`laplacian_sharpness`/`crop_center_fraction`/`pick_best_focus`/
  `parse_focus_sweep_spec`)로 분리해 하드웨어 없이 합성 이미지로 단위테스트 가능
  (`vision/tests/test_rpi_capture.py`).

### 실측 조사 경과 — "초점 스윕이 평평하다"는 최초 결과는 측정 방법 문제였다

1. **노출 개선 먼저 확인(정성 아님, 실측):** 방치된 기본값(`exposure=874`, `analogue_gain=112`,
   최솟값)으로 어두운 실내에서 `--single-shot` → **mean=25.23**(거의 안 보이는 수준). 수동으로
   `exposure=2400, gain=800`으로 올리자 **mean=126.74**(약 5배) — 노출 방치가 "과하게 어둡다"는
   증상의 실제 원인임을 확인. (세션 동안 저녁으로 넘어가며 실내 주변광이 계속 바뀌어 이후
   같은 설정으로도 mean이 60~207 사이를 오갔다 — 이는 노출 컨트롤 자체의 문제가 아니라 촬영
   시각마다 주변광이 다른 정상적인 변동이므로, 앞으로 촬영 세션마다 밝기를 재확인해야 한다는
   뜻이지 한 번 정한 exposure/gain 값이 항상 통한다는 뜻이 아니다.)
2. **최초 초점 스윕(전체 프레임, step=100, 0~1023) → 평평했다(9~11 범위, 변화 거의 없음).**
   오케스트레이터의 최초 관찰(150 vs 480이 거의 동일)과 일치하는 결과라 "VCM이 물리적으로
   안 움직이는 게 아닌가"라는 의심이 들었으나, **이는 측정 방법의 문제였다** — 아래 3번.
3. **원인 규명 및 해결:** 이 카메라는 바닥에 낮게 고정된 채 촬영 중이라 프레임 하단 대부분이
   렌즈 최단 초점거리보다 가까운(추정 <20cm) 바닥이라 **어떤 focus_absolute 값에서도 항상
   흐리다** — 전체 프레임 기준 라플라시안 분산은 이 "항상 흐린" 넓은 영역에 압도돼 배경의
   실제 초점 변화가 묻혔다. 게다가 최초 step=100은 실제 피크 폭(~100~150 유닛, 아래 4번)보다
   넓어 피크 자체를 건너뛸 수 있었다. **중앙/배경 ROI로 제한 + step을 20~40으로 좁히자** 배경
   클러터(테이블/서랍장, 카메라로부터 약 1.5~2m) 영역에서 뚜렷하고 재현 가능한 피크가
   나타났다 — `--sweep-roi` 기본값(중앙 0.6)도 이 발견에 근거해 정함.
4. **피크 재현성 확인(같은 ROI, 다른 시각 2회 독립 실행):** 1차 `440:640:20` 스윕 → 피크
   `focus=560`(선명도 18.20, 이웃값 대비 baseline ~14.8~15.0에서 뚜렷하게 상승). 2차
   `400:720:20` 스윕(주변광이 밝아진 뒤, roi_mean 140→188) → 피크 `focus=580`(선명도 13.07,
   baseline ~10.3~10.6). **두 번 모두 540~600 좁은 구간에 명확한 피크** — 정확한 피크 위치가
   560→580으로 20유닛 이동한 것은 VCM 기계식 위치결정의 통상적인 반복오차(±수십 유닛) 범위로
   판단, 피크의 존재 자체는 확실히 재현됨. 같은 focus 값(560) 5회 반복 촬영은 편차 ±0.1
   이내로 매우 안정적(재현성 자체는 우수 — VCM이 결정론적으로 반응한다는 뜻).
   **`--focus-sweep 460:660:40`으로 도구를 실행한 결과도 `focus=580`을 골라 위 수동 조사와
   일치**(RPi 실기체, 2026-07-22e).
5. **정착시간(settle) 실측:** 확실히 다른 결과를 내는 focus 값(200→560 등)으로 전환 후
   지연을 0~500ms(및 별도로 극단값 0↔1023 전환에 0~2000ms)로 바꿔가며 촬영 — **지연 값과
   무관하게 각 조건 내에서 결과가 일정**했다. 이는 이 씬에서는 VCM 정착이 우리가 측정한
   가장 짧은 지연(수 ms, subprocess 호출 자체의 오버헤드 포함)보다도 빠르게 끝난다는
   뜻으로 해석된다 — **정착시간이 이보다 길게 걸린다는 증거는 못 찾았지만, "0ms도 항상
   충분하다"고 일반화하기엔 이 씬 하나의 관측이라 근거가 약하다.** 기본값
   `FOCUS_SETTLE_S_DEFAULT=0.2`(200ms)는 세션 지시가 제시한 추정 범위(100~300ms대)의
   중간값으로, 실측이 부정하지 않는 보수적 선택이다 — 필요하면 `--focus-settle-ms`로 언제든
   더 늘릴 수 있다.
6. **⚠️ 미완료 — 40cm급 근접 거리에서의 초점 피크는 이번 세션에서 확인 못 함.** 위 재현된
   피크는 전부 카메라 앞 고정 배경(약 1.5~2m)에 대한 것이다. 40cm급 근접 타겟은 이번 세션
   중 카메라 시야 안에 물리적으로 존재하지 않았고(원격 세션이라 체커보드/타겟을 직접 들거나
   옮길 수 없음), 시야 안의 다른 근접 후보(회전의자 캐스터, 팬 형태 물체)는 과다노출로
   블로운되어 있거나 배경과 같은 거리대인 것으로 확인돼 유효한 근접 타겟이 아니었다.
   **다음 실촬영 세션에서 사람이 체커보드를 40cm와 210cm에 각각 들고
   `--focus-sweep`를 두 거리에서 따로 돌려 실제로 다른 최적값이 나오는지 확인해야 한다**
   (메커니즘 자체 — VCM이 실제로 움직이고 거리별로 다른 지점에서 피크가 남 — 는 이미 배경
   물체로 증명됐으므로, 남은 건 "40cm와 210cm가 실제로 얼마나 다른 값을 요구하는가"라는
   정량적 확인뿐).

### 선명도 지표(`laplacian_sharpness`) 신뢰 전제조건 — 위 조사로 확정

1. **노출을 먼저 확보해야 한다.** 어둡고 대비 낮은 원본에서는 focus_absolute를 전체범위
   스윕해도 지표가 거의 안 움직인다(오케스트레이터의 최초 관찰, 위 1~2번과 일치).
2. **프레임 전체가 아니라 ROI로 제한해야 한다.** 근접 배경(카메라 초점범위 밖이라 항상 흐린
   영역)이 프레임 대부분을 차지하면 그 영역이 전체 분산을 압도해 초점 변화를 못 잡는다.
   `crop_center_fraction`으로 중앙만 보는 것이 기본 완화책(캘리브레이션 타겟은 보통 화면
   중앙에 두고 촬영하므로 일반적으로 합리적인 기본값) — 다만 특정 씬에서는 이마저 부족할 수
   있어 `--sweep-roi`로 조정 가능하게 열어둠.
3. **스윕 step은 좁게(≤40 권장).** 실측 피크 폭이 ~100~150 유닛으로 좁아 step=100은 피크
   자체를 건너뛸 수 있다(실제로 최초 스윕에서 그랬다).

### 다음 캘리브레이션 촬영 가이드

1. 먼저 밝기 확보: `--exposure`/`--gain`을 실내 밝기에 맞게 올린다(참고값 — 이번 세션
   저녁 실내에서 `exposure=2400~2602`, `gain=800~900` 근방이 적당했다. **주변광마다 다시
   맞춰야 한다**, 고정값 아님). HTTP 미리보기 페이지의 숫자입력폼으로 실시간 확인하며
   맞추는 게 가장 빠르다.
2. 그 거리에서 `--focus-sweep <대략시작>:<대략끝>:20 --exposure .. --gain ..`으로 최적
   focus_absolute를 찾는다(예: `--focus-sweep 300:800:20`처럼 넓게 잡아도 되고, 대략적인
   감이 있으면 좁혀도 됨). `--sweep-roi`는 타겟(체커보드)이 화면 중앙을 크게 채우면 기본
   0.6으로 충분, 작게 나온다면 값을 낮춰 더 좁게 잡는다.
3. `[초점 스윕 완료] 최적 focus_absolute=N`으로 나온 값을 그 거리의 최적값으로 기록해 두고,
   HTTP 미리보기 페이지(`--focus N`으로 시작하거나 페이지에서 숫자입력)로 그 값을 확정한 뒤
   촬영 버튼으로 캘리브레이션 사진을 찍는다.
4. **거리가 바뀌면(예: 40cm ↔ 210cm) 최적 focus_absolute도 달라질 가능성이 높다** — 위
   실측(배경 ~1.5~2m에서 540~600 근방)은 하나의 거리에 대한 것일 뿐이므로, 거리를 바꿀
   때마다 2번을 다시 돌려야 한다(자동화돼 있으니 반복 비용은 낮음).

---

## distress_coarse.yaml min_area/max_area 도출 근거 (2026-07-22, `presets/distress_coarse.yaml`)

직전 감사 세션에서 "GSD 미확정 상태의 임의값(물리적 근거 없음)"으로 지적됐던 `rect_detector.min_area`/`max_area`를, ② 조난자 구역 실측 스펙(**3.0m×3.0m×0.105m 라이즈드 플랫폼**, `docs/vision_plan.md` §2/§5.3)이 확정되며 근거 있게 재도출했다.

- **풋프린트 재사용:** 매트는 버티포트 흰 필드(직경 3m)와 정확히 같은 3m 풋프린트 → `vision_plan.md` §4.1 GSD 표의 "3m 피처" 컬럼을 그대로 쓸 수 있다(신규 GSD 컬럼 불필요).
- **화각 보정:** §4.1 표는 계획서 가정 화각 102°(H) 기준인데, 실제 장착 카메라(CAM109-IMX708AF-75)는 실측 75°(`docs/vision_status.md` "카메라 화각이 계획서 가정(102°)과 다름(75°)")다. min_area/max_area 계산은 75°로 다시 했다.
- **공식:** 지상폭 `gw(h) = 2·h·tan(75°/2) ≈ 1.535·h` [m] (coarse 다운스케일 전체프레임, 폭 1536px 기준 — §4.1 "다운스케일(1536px)" 컬럼과 동일 기준). `GSD_down(h) = gw(h)/1536` [m/px]. 매트 한 변(3.0m) 픽셀 길이 = `300cm / (GSD_down(h)×100)`, 면적 = 한 변².
- **계산 결과:** 10m → 한 변 ≈300px, 면적 ≈90,000px² / 20m → ≈150px, ≈22,500px² / 40m → ≈75px, ≈5,625px².
- **마진 반영 임계값:** `min_area=8000`(40m를 확실히 배제, 20m는 실제 검출 면적이 명목값의 ~35%까지 열화돼도 통과할 여유) / `max_area=200000`(10m 명목값의 약 2.2배 — 근접·윤곽 과대추정에 여유를 주되 프레임을 거의 채우는 배경 오탐은 배제).
- **골든셋 정합:** `vision/tests/golden/distress/{10m,20m,40m}/`도 위 계산의 한 변 px(300/150/75)을 그대로 써서 재생성됨 — 10m/20m는 검출, 40m는 마진 반영 임계값 기준으로 미검출(재생성 스크립트: `vision/tests/golden/generate_synthetic.py`).
- 이번 재도출은 **검출 알고리즘 변경이 아니다** — `min_area`/`max_area` 파라미터 값만 교체, `modules/*.py` 로직은 그대로.

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
              단, 하드웨어 비의존 CLI 도구(예: jsonl_view.py, calib_analyze.py)는 예외 — .venv 설치 + pytest 대상.
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
| tools/calib_analyze.py | **★합성 왕복(핵심):** 진짜 K/dist를 알고 합성 투영한 ~20장 사이드카 → 복원 fx/fy/cx/cy 1% 이내·dist 허용오차 내 · fx-vs-LensPosition 직선적합이 알려진 (L,fx) 직선을 복원 · 이상치 검출(코너 오염 이미지가 플래그되고 제외 시 RMS 개선) · 부분 데이터(그룹 1개)에서 크래시 없이 적합 생략+`recommended` 폴백 사유 기록 · yaml 아티팩트 `yaml.safe_load` 왕복 + 필수 키 전부(`checks[].ok`가 python bool인지 — numpy.bool_ 누출 회귀 포함) · `--redetect` PNG 재검출 경로 · CLI(`main()`) end-to-end로 진단 플롯 3종 PNG 실파일 생성 | ✅ test_calib_analyze |
| tests/golden 회귀(§7.9 항목7) | `vision.replay.run_replay()`로 골든셋(§ tests/golden/README.md) 실제 재생 → JSONL 검출 개수가 `labels.json` 기대값과 일치·캐스케이드 단계별 meta도 실제 `Pipeline.run()`으로 검증. 몽키패치 없음(실제 파이프라인) | ✅ test_golden_regression |

**공통 규칙 (모든 모듈 테스트):**
1. **선언 필드 계약** — 위 파일표대로 "읽는 필드"만 읽고 "쓰는 필드"만 쓴다.
2. **meta 네임스페이스** — `state.meta["<모듈이름>"]` 기록 확인.
3. **빈/경계 입력** — 검은 이미지·빈 mask에서 크래시 없이 합리적 출력(대개 0검출).
4. **결정론(plan §7.5)** — 같은 입력·같은 config → 같은 출력. **골든셋 회귀 스캐폴드는 2026-07-21c에 합성 데이터로 시작됨**(`tests/golden/README.md`) — 실기체 데이터는 카메라 브링업 이후 교체 예정.

**새 모듈 추가 시:** 위 4개 공통 규칙을 담은 `tests/test_<모듈>.py` 를 **같은 커밋에** 추가한다.
