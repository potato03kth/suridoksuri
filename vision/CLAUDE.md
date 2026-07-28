# vision — Claude 작업 가이드

착륙지점 객체인식 모듈이다.  
사진/영상에서 착륙 가능한 사각형 영역을 탐지하고, 영상의 경우 시간 축으로 확정한다.

> ⚠️ **재설계 진행 중.** 이 문서는 현재 구현된 "무채색 사각형 검출기" 임시 틀을 설명한다.
> 정밀착륙(ArUco/색 2단 검출, 폐루프 유도, 변경내성·관측성) 방향은 **`docs/vision_plan.md`**를
> 먼저 읽을 것. 이 틀은 폐기가 아니라 ② 조난자 구역 색 파이프라인 + ArUco 모듈 컨테이너로 재편된다
> (계획서 §12). `geo_project.pixel_to_gps`는 **2026-07-28 삭제 완료**(아래 "`utils/geo_project.py` 폐기" 절).

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
| `core/target.py` | **ArUco 브랜치 Phase 3(`docs/vision_aruco_branch.md`)** — `TargetEstimate` dataclass(상대 pose+신뢰도+타입+frame_id+timestamp+uncertainty(항상 None, 자리만)+calib provenance(`calib_accuracy`/`not_for_closed_loop_30cm`/`calib_id`)) + `solve_target_pose()`(`cv2.solvePnP`→rvec/tvec→`cv2.Rodrigues`→quaternion) + `marker_object_points()`(50cm 정사각 4코너, `ARUCO_TARGET_SIZE_M`) + **[2026-07-28] 초록구역(② 조난자) 확장** — `DISTRESS_MAT_SIZE_M`(3.0)/`DISTRESS_MAT_PLATFORM_HEIGHT_M`(0.105)/`MAT_PLANE_REFERENCE` + `order_quad_corners_clockwise()`(approxPolyDP의 보장 없는 코너 순서를 solvePnP 대응 순서로 정규화) + `project_pixel_onto_target_plane()`(착륙점 픽셀 -> 타겟 평면 역투영) + `solve_target_pose(position_at_pixel=...)`(주면 position이 tvec 대신 그 평면상 점 — **None이면 ArUco 경로 무변경**). 아래 "초록구역 상대 pose 산출" 절 참조. `rotation_matrix_to_quaternion()`(scipy 미의존 순수 numpy, 표준 Shepperd's method — `vision/requirements.txt`에 scipy 없어 직접 구현). **순수 기하 계산만(파일 I/O 없음)** — import 규칙("core/ ← numpy, opencv만 허용")대로 yaml 로드는 `utils/calibration_loader.py`에 분리. 좌표계=카메라 광학 프레임, 단위=미터, orientation=quaternion(x,y,z,w) — 전부 `docs/vision_aruco_branch.md` §1 확정 전제, 재논의 대상 아님. **Phase 4(파이프라인 통합) 완료** — `main.py`/`replay.py`가 `state.detections`에서 확정 ArUco 검출을 찾아 `solve_target_pose()`를 호출하고 결과를 JSONL `chosen.target_estimate`에 싣는다(아래 "ArUco Phase 4 파이프라인 배선" 절) | 낮음 |
| `core/state_machine.py` | **신설(§9 빌드순서 6번, `docs/vision_plan.md` §5.1)** — 공통 상태머신 + 안전 폴백. `LandingState`(str Enum: `ACQUIRE`/`CENTER_DESCEND`/`LOCK`/`PRECISION_SERVO`/`TERMINAL`/`HOLD`/`ABORT_ASCEND`) + `Observation`(프레임 단위 관측: `ts`/`frame_id`/`n_candidates`/`center_error_norm`/`fine_locked`/`agl_m`/`target_estimate`/`scale_source`) + `Decision`(`state`/`command`/`reason`/`blind_duration_s`/`scale_source`) + `LandingSMConfig`(매직넘버 금지, §7.3 — `max_blind_duration_s`/`max_drift_estimate_m`/`lock_confirm_frames`/`loss_tolerance_frames`/`center_tolerance_norm`/`max_candidates_for_lock`/`terminal_agl_m`) + `LandingStateMachine.update(obs)->Decision`. **순수 로직(파일 I/O·wall-clock·난수 없음)** — `core/target.py`와 동일 패턴, `Observation.ts`를 그대로 쓰므로 같은 관측열은 항상 같은 상태열(§7.5 결정론). **타겟 종류 무관 공통 골격** — 버티포트/조난자/십자 전용 분기 없음, 타겟별 특수성은 호출자가 `Observation` 필드를 어떻게 채우는지로만 표현된다. **커밋 게이트가 구조적으로 강제됨** — `PRECISION_SERVO`/`TERMINAL`은 오직 `LOCK`의 `_consecutive_fine_locked >= lock_confirm_frames` 통과를 거쳐야만 도달 가능(코드상 다른 진입 경로 없음), 후보가 모호(`n_candidates > max_candidates_for_lock`)한 채 락을 시도하면 `HOLD`로 거절. **안전 폴백 두 갈래** — (a) `CENTER_DESCEND`/`PRECISION_SERVO`에서 검출 상실이 `loss_tolerance_frames`를 넘으면 `HOLD`(재포착 시 `CENTER_DESCEND`로 복귀 가능, 막다른 상태 아님), (b) `TERMINAL`에서 블라인드 지속시간이 `max_blind_duration_s`를 넘거나 근사 이탈추정(마지막 유효 정규화 중심오차×마지막 유효 AGL)이 `max_drift_estimate_m`을 넘으면 `ABORT_ASCEND`(§5.1 "안 보이는데 계속 내려간다" 금지). `TERMINAL` 진입은 `PRECISION_SERVO`에서 `agl_m<=terminal_agl_m`일 때만(AGL이 항상 None이면 구조적으로 절대 진입 못 함 — 크래시 없이 안전하게 폐루프 서보에 계속 머무는 축퇴 동작). `modules/`가 아니다(`__call__(VisionState)->VisionState` 인터페이스 아님, `registry.py` 미등록, preset yaml 미사용) — 프레임 단위 관측을 시간축으로 누적하는 상위 레이어 | 낮음 |
| `core/frames.py` | **신설(2026-07-28, vision↔fc 인터페이스 Phase 1 — `docs/vision_fc_interface.md` §4.3 작업 V3)** — 좌표 프레임 체인 `카메라 광학(cam) → 기체 body FLU → 기체 body FRD`. `R_frd_cam(psi_m)`/`R_flu_cam(psi_m)` + `cam_to_frd`/`cam_to_flu`/`flu_to_frd`와 각각의 역변환 + 쿼터니언 성분순서 어댑터(`quat_xyzw_to_wxyz`/`quat_wxyz_to_xyzw`). **순수 numpy 회전 계산만**(파일 I/O·wall-clock·난수 없음 — `core/target.py`/`core/state_machine.py`와 동일 패턴). `R_flu_cam`은 `R_frd_cam`에서 `R_flu_frd`를 곱해 **유도**한다(두 행렬을 각각 손으로 적어두면 한쪽만 고쳤을 때 조용히 어긋나므로). 🔴 **마운트 요각 `psi_m`은 미측정(§7 U3)** — 저장소 어디에도 값이 없고 물리 측정이 필요해 `MOUNT_YAW_PSI_M_RAD_DEFAULT=0.0` + `MOUNT_YAW_PSI_M_MEASURED=False`로 **값을 지어내지 않고** 파라미터로 뺐다. 그 "미측정" 플래그는 와이어 레코드를 타고 소비자까지 전파된다. **ENU/NED 변환은 의도적으로 구현하지 않았다**(§4.4 — body 프레임 경로가 기체 자세 구독·시간 동기를 통째로 없애므로. 안 쓰는 경로를 만들면 누군가 쓴다). 나디르 하드마운트 가정과 고무마운트 잔차 리스크(§4.5, `지상오차=고도×tan(θ_잔차)`)는 파일 docstring에 정량 수치까지 기록 | 낮음 |
| `core/wire.py` | **신설(2026-07-28, 같은 Phase 1 — 작업 V2 + §5 페일세이프 계약)** — vision→fc 인터페이스 **와이어 포맷(JSON Lines)** + 페일세이프 계약. `SCHEMA_VERSION`/`REQUIRED_TARGET_KEYS`/`REQUIRED_STATE_HINT_KEYS` 상수 + `FailsafeContract`(매직넘버 금지, §7.3) + `gate_confidence()`/`closed_loop_floor_agl_m()` + 레코드 조립 3종(`build_target_record`/`build_invalid_target_record`/`build_state_hint_record`) + `encode_line`/`decode_line` + 왕복 복원(`target_estimate_from_record`/`decision_from_record`) + `validate_record`. **transport-agnostic 순수 로직** — 소켓도 파일도 **클록도** 모른다(타임스탬프를 전부 인자로 받아 같은 입력이면 같은 바이트, §7.5. 실제 클록 샘플링은 어댑터 `utils/target_sink.py::sample_clocks()`). 레코드 타입 2종: `"target"`(TargetEstimate **전량 무손실** + 3프레임 위치 + provenance + 페일세이프 필드)과 `"state_hint"`(상태머신 `Decision`). 🔴 **`command`는 `command_hint` + `command_is_advisory:true` + 타입명 `state_hint` 세 겹으로 "명령 아님"을 형식에 박았다**(§6.3 거부권 권고). 🔴 **위치 필드는 프레임을 이름에 박는다**(`position_cam`/`position_flu`/`position_frd`) — 이 저장소는 `pos_ned`/`vel_ned`가 같은 접미사로 반대 부호 규약을 쓰는 사고 이력이 있다(§4.2). 시계는 **두 클록을 다 싣는다**(`stamp_monotonic_ns` stale판정용 / `stamp_wall_ns` 로그상관용 / `clock_offset_ns` 환산용 / `timestamp` 원본 무손실) — 근거·미검증 2건은 파일 docstring "시계(clock) 계약" 절 | 낮음 |

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
| `distress_box.py` | `WhiteBoxDetector` | `original`, `detections` | `detections`, `meta` |
| `distress_mat.py` | `DistressMatGeometry` | `detections`(코너+`white_box_detector` meta) | `detections`(meta만), `meta` |

### 그 외

| 파일 | 역할 |
|---|---|
| `registry.py` | 이름 → 클래스 매핑. **새 모듈 등록은 여기에만** |
| `presets/*.yaml` | 시나리오별 모듈 조합. 코드 수정 없이 파이프라인 변경 |
| `config/default.yaml` | 전체 파라미터 기본값 참조용 (실행에 직접 사용하지 않음) |
| `presets/distress_coarse.yaml` | ② 조난자 구역 coarse(§5.3) — 전용 모듈 없이 기존 `ColorFilter(mode=color)`+`RectDetector` 조합. §7.9 항목7 골든셋용으로 신규 추가(신규 검출 로직 아님). `min_area`/`max_area`는 실측 스펙 기반 도출값 — 근거는 아래 "distress_coarse.yaml min_area/max_area 도출 근거" 절 |
| `presets/vertiport_fine.yaml` | **신설(ArUco 브랜치 Phase 4)** — `aruco_detector` 단일 스텝. `vertiport_coarse.yaml`의 3단 캐스케이드 **뒤에 이어붙이지 않고 완전히 독립 실행**한다 — `ArucoDetector`(Phase 2)가 ROI 인자 없이 `state.original` 전체 프레임에서 찾고 `state.detections`를 덮어쓰므로, coarse 뒤에 이어붙이면 coarse가 확정한 결과가 조용히 사라진다(설계 판단 근거는 yaml 파일 헤더 주석 참조). ArUco 피덕셜은 스케일/회전에 견고해 ROI 없이도 전체 프레임 탐색이 실용적 |
| `presets/distress_fine.yaml` | **신설(2026-07-25, §9 "끊어진 체인을 잇는 작업")** — `distress_coarse.yaml`(초록 HSV+`RectDetector`) 뒤에 `white_box_detector`(`modules/distress_box.py`)를 잇는 **캐스케이드**(vertiport_fine.yaml과 반대 판단 — 근거는 이 yaml 헤더 주석과 아래 "조난자 fine 파이프라인 배선(체인 잇기)" 절 참조). `rect_detector`의 `min_area`/`max_area`(14000/2,200,000)는 `distress_coarse.yaml` 값을 복사하지 않고 fine 대역(≤~15m, 상한은 `state_machine.py`의 `terminal_agl_m` 기본값 3m 근방)에 맞게 동일 공식으로 재도출(근거는 yaml 헤더 주석) |
| `presets/distress_coarse.yaml` / `presets/distress_fine.yaml` 공통 신규 스텝 `distress_mat_geometry` | **[2026-07-28]** `modules/distress_mat.py` — 초록구역 상대 pose 산출용 기하 태깅. **이 스텝의 존재 자체가 "초록구역 pose 산출기를 쓰라"는 프리셋 주도 신호**(`main.py`가 preset 경로 문자열을 파싱하지 않는다). `distress_fine.yaml`에서는 반드시 `white_box_detector` **뒤**에 와야 한다(앞에 두면 착륙점이 매트 중심으로 조용히 degrade). 아래 "초록구역 상대 pose 산출" 절 참조 |
| `utils/image_loader.py` | 파일 경로 → BGR numpy 배열 |
| `utils/calibration_loader.py` | **신설(ArUco 브랜치 Phase 3)** — `vision/calibration/<camera_id>/nominal.yaml` 로드 어댑터. `CameraCalibration`(camera_matrix/dist_coeffs/image_size/accuracy/not_for_closed_loop_30cm/calib_id 등) 반환, `core/target.py::solve_target_pose()` 입력으로 바로 연결. `compute_nominal_intrinsics.py`가 만드는 `nominal.yaml` 스키마 전용(`calib_analyze.py`의 `<calib_id>.yaml`은 스키마가 달라 별개 — 과설계 금지, 필요해지면 그때 확장). `calib_id`는 로드에 쓴 파일 경로 문자열(§7.3 provenance echo) |
| `utils/video_reader.py` | 영상 파일 → 프레임 이터레이터 |
| `utils/visualize.py` | bbox 드로잉, 결과 이미지 저장. **[2026-07-28] `draw_sink_status()` 추가** — 유도(정밀착륙) 발행 상태 오버레이(소비자 수/마지막 seq/드롭 수/엔드포인트). `--display`와 `--target-sink`가 완전히 독립이라 "화면은 뜨는데 유도 좌표는 아무 데도 안 나가는" 상태가 가능하다는 사각지대 대응(사용자 결정). 소비자 0명=빨강+큰 글씨, sink 꺼짐=주황, 정상=초록. `draw_detections()`가 만든 사본을 **제자리에서** 고치고(사본 재생성 없음) 좌상단 **반투명** 패널에만 그려 검출 결과를 가리지 않는다. ⚠️ 문자열은 **ASCII만**(Hershey 폰트가 한글을 못 그린다). 호출 게이팅(`--display none`이면 비용 0)은 호출자 책임 — 아래 "bind 하드 페일 + 유도 발행 상태 오버레이" 절 |
| `utils/logging.py` | 이중싱크 사람로그(터미널+.log 로테이션) + provenance 헤더(config+git해시+캘리브id) (§7.4/§7.3). `main.py`/`replay.py`에 연결됨 |
| `utils/blackbox.py` | 프레임별 JSONL 블랙박스 + 거절이유 로깅. bounded queue+drop-oldest 비차단 (§7.4). `main.py`/`replay.py`에 연결됨 |
| `utils/stream.py` | `MjpegStreamer` — 라이브 저해상 MJPEG-over-HTTP 스트림(§7.9 항목5, "작동영상 피드백 3경로" (b)). `push_frame()`은 bounded queue+drop-oldest 비차단(`blackbox.py`와 동일 패턴 재사용). 다운스케일·인코딩·HTTP 서빙은 전부 별도 스레드. `main.py --display stream`/`replay.py --display stream`으로 opt-in 연결(항상 켜지지 않음). 기본값 결정 근거는 아래 "라이브 스트림 어댑터 기본값" 참조 |
| `utils/target_sink.py` | **신설(2026-07-28, vision↔fc 인터페이스 Phase 1 — 작업 V1)** — `TargetSink` 포트(§7.2가 이름으로만 지정해 뒀던 것) + `NullSink` + `SocketTargetSink`(**localhost TCP 서버**, 호스트 쪽 발행자) + `sample_clocks()` + `_DropOldestQueue`. 컨테이너 안의 Phase 2 shim 노드가 **클라이언트로 접속**해 JSON Lines를 읽어 ROS2 토픽으로 재발행한다. **TCP를 고른 결정적 근거는 `NetworkMode=host`** — 컨테이너가 호스트 네트워크 네임스페이스를 그대로 공유해 `127.0.0.1`이 추가 설정 0으로 통한다. 반면 UDS는 소켓 파일이 컨테이너 마운트 네임스페이스에 보여야 하는데 마운트가 `/home/suri/drone_ws` 하나뿐이라 **컨테이너 실행 설정 변경(=FC 도메인+배포 절차)** 을 요구해 기각(상세는 파일 docstring). 바인드는 **`127.0.0.1` 고정**(`utils/stream.py`의 `0.0.0.0`과 반대 — `NetworkMode=host`라 0.0.0.0이면 비행 중 유도 스트림이 주변 네트워크에 노출된다). **비차단이 최우선 계약** — `publish()`는 bounded queue에 넣기만 하고 소켓 I/O를 한 줄도 안 한다(인코딩·전송·accept 전부 별도 스레드), 가득 차면 drop-oldest(`utils/blackbox.py`/`utils/stream.py` 패턴 재사용), 느린 소비자는 `send_timeout_s`로 끊는다, `start()` 전/`stop()` 후 `publish()`는 조용한 no-op. **SIGTERM 핸들러**(`install_signal_handlers()`)는 `tools/h264_stream.py::_install_sigterm_handler`와 동일 패턴(핸들러는 이벤트만 세팅) — 이 저장소의 원격 프로세스는 SIGINT가 SIG_IGN이라 SIGTERM이 유일한 신호다. ⚠️ **단 `main.py`/`replay.py`는 이 메서드를 부르지 않는다**(신호당 핸들러가 하나뿐이라 라이브 graceful shutdown을 덮어쓴다 — 2026-07-25 실기체 버그). **rclpy를 import하지 않는다**(Phase 2 shim이 유일한 ROS 접점). **[2026-07-28] 관측 접근자** — `client_count`(기존)/`last_seq`(신설) 둘 다 락으로 보호된다: 화면 오버레이(`utils/visualize.py::draw_sink_status`)가 파이프라인 스레드에서 매 프레임 읽는데 클라이언트 목록은 accept/send 스레드가 만지므로 스레드 교차 접근이 상시 일어난다. 기본값 근거는 아래 "target_sink 소켓 기본값" 절 |
| `ros/shim_core.py` | **신설(2026-07-28, vision↔fc 인터페이스 Phase 2 — 컨테이너 shim)** — JSON Lines 레코드 → **"발행 계획"(ROS 메시지를 서술하는 dataclass)** 순수 변환. `ShimConfig`/`ShimRouter` + `PosePlan`/`StatusPlan`/`LandingTargetPlan`/`ShimOutput` + `validate_record`/`parse_line`/`kv_value`. 🔴 **stdlib만 쓴다 — 취향이 아니라 강제다**: `vision.core.wire`를 import하면 `wire.py → core/target.py → import cv2` 체인 때문에 **fc 컨테이너에서 즉시 죽는다**(cv2 없음, 실측). 그래서 계약 상수(`SCHEMA_VERSION`/`REQUIRED_*_KEYS`)를 **의도적으로 복제**하고, 그 복제가 어긋나는 것은 랩탑 테스트가 `vision.core.wire`와 **직접 대조**해 잡는다(런타임은 격리, 검증은 대조). rclpy도 numpy도 import하지 않아 랩탑에서 그냥 단위테스트된다. **[2026-07-28] `/vision/landing_setpoint` 추가** — `VehiclePose`/`SetpointPlan` + 쿼터니언 순수 수학 4종(`quat_normalize`/`quat_rotate`/`quat_yaw`/`quat_from_yaw`) + `enu_yaw_to_ned_yaw`/`enu_to_pos_ned_n_e_hup` + `build_landing_setpoint()`. 🔴 **절대 좌표를 기억하지 않는다** — 라우터가 들고 있는 상태는 최신 `VehiclePose` 하나뿐이고 목표점은 매 레코드마다 `최신 pose + 그 순간의 상대오차`로 다시 계산된다(EKF 드리프트 상쇄). numpy 금지 제약 때문에 쿼터니언 회전을 수식으로 직접 폈고, 그래서 **로드리게스 공식 독립 재구현과 대조**하는 테스트를 따로 뒀다. 아래 "절대 setpoint" 하위 절 참조 |
| `ros/shim_node.py` | **신설(2026-07-28, 같은 Phase 2)** — `shim_core`의 얇은 rclpy 어댑터. **저장소에서 rclpy를 import하는 유일한 파일**이고 컨테이너 안에서만 실행된다(`python3 -m vision.ros.shim_node`). 소켓 **클라이언트**(vision이 서버)라 재접속 루프를 이쪽이 갖고, EOF를 받으면 pose를 끊고 status만 ERROR로 낸다. 판단은 한 줄도 없다 — "계획 → msg 필드 대입"과 소켓 수명주기뿐이고, 그 **얇음 자체가 회귀테스트 대상**이다(`test_shim_core.py`가 이 파일을 **AST로 파싱해** msg 프레임 상수 참조·와이어 키 직접 읽기를 금지한다 — 소스 문자열 검색은 docstring 산문까지 잡아 쓸모없어진다). SIGTERM 핸들러는 `tools/h264_stream.py`와 동일 패턴. **[2026-07-28] `/mavros/local_position/pose` 구독 + `/vision/landing_setpoint` 발행 추가** — 구독이 생기면서 **executor 데몬 스레드가 하나 늘었다**("발행만 하니 spin 불필요"였던 전제가 깨짐). 소켓 폴링(`_POLL_S=0.2s`) 안에서 `spin_once`를 부르면 자세가 최대 0.2초 묵어 `attitude_stale_s`(0.25s) 예산의 80%를 까먹기 때문. `--no-landing-setpoint`를 주면 구독도 스레드도 안 만들어 이전 단일스레드 동작으로 정확히 되돌아간다 |
| `utils/frame_source.py` | `FrameRecord` + `LiveFrameSource`/`DirFrameSource`/`BagFrameSource` 어댑터 + `open_dir_or_bag()` 팩토리 (§7.2/§7.5/§7.9 항목4). Live=실카메라(재시도 후 `ConnectionError`), Dir=녹화폴더(프레임파일+선택적 `telemetry.jsonl`), Bag=단일 비디오파일(+선택적 사이드카 `<basename>.jsonl`). **`LiveFrameSource`는 2026-07-24 카메라 브링업(`docs/vision_camera_bringup.md`) Phase 4에서 picamera2 백엔드로 재구현됨** — 이전 `cv2.VideoCapture` 구현은 V4L2 raw 경로와 비호환임이 실측 확인돼(`docs/vision_status.md` 2026-07-22b) 폐기. 생성자 인자도 `device`(cv2 정수/경로) → `camera_num`(picamera2 카메라 인덱스) + `resolution`(기본 `nominal.yaml`의 `image_size` 4608x2592와 일치, solvePnP 캘리브레이션과 어긋나지 않게)으로 교체. `create_still_configuration(main={"format": "RGB888", ...})` 요청이 실제로는 BGR 바이트순서를 준다는 picamera2 명명 역전(`tools/calib_capture.py`에서 실기체로 이미 확인된 사실)을 재사용해 별도 색공간 변환 없이 BGR 관례를 만족시킨다. picamera2는 이 `.venv`에 없는 RPi 전용 라이브러리라 `open()` 내부 지연 import로 격리(모듈 최상단 import 금지 — 그러면 이 `.venv`의 `DirFrameSource`/`BagFrameSource` 사용처까지 깨짐). 단위테스트는 `sys.modules`에 가짜 `picamera2` 모듈을 주입해 실기 없이 검증하고, 지연 import 격리 자체도 회귀테스트(소스 텍스트에 최상단 import 없음 + `sys.modules["picamera2"]=None`으로 강제 차단해도 모듈 import는 성공) 대상. **[2026-07-28] AF(오토포커스) 제어 추가 + 저장소 AF 단일 출처가 됨** — `af_mode`(기본 `continuous`)/`lens_position` 생성자 인자와 `AF_MODES`/`LENS_POSITION_MIN,MAX`/`validate_af_args`/`validate_lens_position`/`make_af_controls`가 여기 산다(`tools/h264_stream.py`가 여기서 import). 🔴 **실기체 미검증.** 아래 "LiveFrameSource AF 제어" 절 참조 |
| `main.py` | CLI 진입점. 이미지/영상 자동 분기. `--log-dir`/`--log-name`으로 이중싱크 로거+JSONL 블랙박스 실행(항상 on). `--display stream`으로 `MjpegStreamer` opt-in(§7.9 항목5). **ArUco 브랜치 Phase 4** — `--calib`(기본 `calibration/cam109-imx708af75/nominal.yaml`)로 캘리브레이션을 1회 로드해 재사용, 확정 ArUco 검출이 있으면 `solve_target_pose()` 호출 결과를 JSONL `chosen.target_estimate`에 싣는다(아래 "ArUco Phase 4 파이프라인 배선" 절). **§9 6번 상태머신 배선** — 실행 전체에 걸쳐 `LandingStateMachine` 인스턴스 하나를 재사용(`_run_image`/`_run_video`/`_run_live` 전부, 단일 이미지 경로도 관측 1개짜리로 통과)해 매 프레임 `_build_observation()`으로 `Observation`을 만들고 `update()` 결과를 JSONL `state`/`command`에 싣는다(아래 "공통 상태머신 파이프라인 배선" 절). **[2026-07-25] `_build_observation()`이 ② 조난자 fine(흰 박스)까지 확장됨** — 아래 "조난자 fine 파이프라인 배선(체인 잇기)" 절. **[2026-07-28] `--target-sink` 배선(인터페이스 Phase 1 마무리, §9 작업 V5)** — `--target-sink`/`--target-sink-host`/`--target-sink-port`로 `SocketTargetSink` opt-in(기본 꺼짐=`NullSink`), 세 실행경로(`_run_image`/`_run_video`/`_run_live`) 전부에서 매 프레임 `target`+`state_hint` 레코드 발행. 아래 "`--target-sink` 파이프라인 배선" 절. **[2026-07-28] bind 하드 페일 + 발행상태 오버레이(사용자 결정)** — 기동 실패 시 강등 없이 **종료코드 3**으로 즉사(stderr에 포트 포함), `--display`가 켜져 있으면 매 프레임 `draw_sink_status()`로 소비자 수/seq/드롭을 화면에 표시(`none`이면 비용 0). 아래 "bind 하드 페일 + 유도 발행 상태 오버레이" 절 |
| `replay.py` | 오프라인 재생 CLI(`python -m vision.replay <녹화폴더\|bag> --preset ...`, §7.9 (a)). `open_dir_or_bag`로 Dir/Bag 자동판별 → 동일 `Pipeline`으로 재생 → 로거+블랙박스 기록. **결정론적**(§7.5). `--display stream`으로 `MjpegStreamer` opt-in(§7.9 항목5). **ArUco 브랜치 Phase 4** — `main.py`와 동일한 `--calib`/`TargetEstimate`→`chosen.target_estimate` 배선(헬퍼는 상호 import 안 함 원칙에 따라 얇게 중복). **§9 6번 상태머신 배선** — `main.py`와 동일 원칙(얇게 중복)으로 재생 루프 전체에 걸쳐 `LandingStateMachine` 인스턴스 하나 재사용, `record.telemetry.get("alt")`가 있으면 `Observation.agl_m`으로 흘려보내고 없으면 None으로 우아하게 degrade(아래 "공통 상태머신 파이프라인 배선" 절). **[2026-07-25] `_build_observation()`이 ② 조난자 fine(흰 박스)까지 확장됨** — 아래 "조난자 fine 파이프라인 배선(체인 잇기)" 절. **[2026-07-28] `--target-sink` 배선 + 발행상태 오버레이** — `main.py`와 문자 그대로 같은 CLI 인자/기본 꺼짐/하드페일(exit 3). 🔴 **재생 경로에만 있는 가치: `telemetry.jsonl`의 AGL이 실려 `state_hint`가 `TERMINAL`까지 진행하는 유일한 경로**(main.py는 AGL 경로 자체가 없음). `_solve_target_chosen()` → `_solve_target_estimate()`로 교체(JSONL `chosen` 무변경). 아래 "bind 하드 페일 + 유도 발행 상태 오버레이" 절 |
| `tools/rpi_capture.py` | RPi 헤드리스 캘리브레이션 촬영 — 저해상도 스냅샷 자동갱신(브라우저) + 촬영 트리거(버튼/Enter). **2026-07-22b에 GStreamer `libcamerasrc`(작동 불가였음, libcamera PiSP IPA 결여) 대신 V4L2 RAW 직접 캡처+수동 디베이어로 전면 재작성해 브링업 완료** — media-ctl/v4l2-ctl로 rp1-cfe 파이프라인 직접 구성. 대상 media 디바이스(`/dev/mediaN`)는 부팅마다 번호가 바뀔 수 있어 하드코딩 대신 매 호출 동적 탐색(2026-07-22d, 아래 "media 디바이스 동적 탐색" 절). gray-world 화이트밸런스 보정 포함(아래 절). **수동 초점/노출/게인 제어 + 초점 스윕 도구 포함(2026-07-22e, 아래 "수동 초점/노출/게인 제어" 절)** — libcamera 우회 경로라 연속 AF/AE가 없어 방치돼 있던 것에 대한 대응. 상세 경과는 메모리 `project_rpi5_ubuntu_camera_stack.md` |
| `tools/jsonl_view.py` | JSONL 블랙박스 뷰어/플롯 최소본(§7.9 항목6). `BlackBoxLogger`가 남긴 `.jsonl`을 읽어 시간축 score/latency/state 3단 플롯을 PNG로 저장(`matplotlib` Agg 백엔드, headless-safe). `python vision/tools/jsonl_view.py <jsonl> [--output out.png] [--x-axis ts\|frame_id]`. **하드웨어 의존 없음** — `rpi_capture.py`와 달리 `.venv`에 설치되고(`matplotlib` in `requirements.txt`) `tests/test_jsonl_view.py` 대상이다(tools/의 "CI/pytest 대상 아님" 규칙은 RPi 하드웨어 전용 스크립트에만 적용). |
| `tools/calib_analyze.py` | **신설(2026-07-23)** — `calib_capture.py`가 만든 촬영 세트(`<raw_root>/<set>/<distance>m/<stem>.{png,json}`)를 캘리브레이션 아티팩트로. 그룹(set,distance_m)별 `cv2.calibrateCamera` → 사진별 재투영오차로 불량컷 색출·이상치 제외 재캘리브레이션 → **`fx`/`fy` vs `LensPosition` 직선적합으로 L=0(무한대) 외삽**(핵심 목적 — 기체 운용고도 10~40m는 사실상 무한대)·얇은렌즈 물리 일관성 검사(`b/a`→mm, IMX708급 통상 초점거리와 비교)·디옵터 가정(`LensPosition≈1/distance_m`) 직접 검정 → 세트B(2.5m, 최대커버리지)와 외삽값 대조 → fy/fx·주점·HFOV 등 정합성 검사(실패해도 크래시 없이 보고) → 진단 플롯 3종(Agg, `vision/results/`) → `vision/calibration/<camera_id>/<calib_id>.yaml` 아티팩트(그룹별 결과 전부 보존, `recommended`=fx/fy 외삽절편+세트B의 cx/cy/dist_coeffs, 근거는 yaml `note`). 모든 임계값 CLI 파라미터(매직넘버 금지, §7.3). **하드웨어 의존 없음** — `jsonl_view.py`와 동일한 예외로 `.venv` 설치 + `tests/test_calib_analyze.py` 대상. 정확성은 진짜 K/dist를 아는 합성 체스보드 투영 왕복 테스트로 담보(실촬영 사진은 이 스크립트 작성 시점에 아직 없었음 — `docs/vision_camera_bringup.md`). `recommended-source` CLI로 폴백/강제선택 가능, 그룹<2면 적합 생략 + LensPosition 최저 그룹 폴백(사유 yaml 명시, §9 견고성). |
| `tools/h264_stream.py` | **신설(2026-07-25, ffmpeg Phase 3 "1B 도구화" — `docs/vision_camera_bringup.md` §Phase 3 완료)** — RPi H.264 라이브 디버그 스트림 정식 도구. `Picamera2.create_video_configuration(RGB888, FrameRate)` + `H264Encoder(bitrate=...)` + `FfmpegOutput("-f mpegts -listen 1 tcp://HOST:PORT")`(1A 실증 조합). `SIGTERM` 핸들러로 graceful shutdown 보장(SIGINT는 비대화형 SSH 자식에서 SIG_IGN이 될 수 있어 못 믿음, 1A 실측) + 클라이언트 연결 종료(ffmpeg 프로세스 종료) 감지 시 카메라/인코더 재시작 없이 `FfmpegOutput`만 교체하는 재기동 루프가 기본(`--once`로 단발 모드). `--af-mode continuous\|auto\|manual`(+ `--lens-position`, VCM 실가동범위 0~15.0 디옵터 검증 — 32.0 아님) 지원 — **[2026-07-28] 이 AF 순수 로직의 정의는 `utils/frame_source.py`로 옮겨졌고 여기서는 import해 쓴다**(라이브 파이프라인 경로에도 AF가 필요해졌는데 import 규칙상 `utils/ → tools/`가 불가라 방향을 뒤집었다. 동작·CLI 무변경, `_make_af_controls` 등 기존 이름도 재노출 유지). **`vision/utils/stream.py`(MjpegStreamer)는 병행 운용, 폐기 아님**(용도가 다름 — 이쪽은 카메라 원본, 그쪽은 검출 오버레이). **하드웨어 의존 로직(run_server, picamera2/libcamera 지연 import)은 tools/의 "CI/pytest 대상 아님" 예외** — 단, 해상도/ffmpeg 스펙 파싱·렌즈위치 검증·AF 모드 매핑·fps/라이브니스 통계·output 사망 판정·bounded stop 안전망·SIGTERM 핸들러는 순수 로직으로 분리해 `tests/test_h264_stream.py` 대상(picamera2 지연 import 격리는 `LiveFrameSource`와 동일 패턴). **실측(RPi5)으로 확인된 핵심 함정** — 이 플랫폼(비-VC4, `Platform.PISP`)의 `H264Encoder`는 실제로는 `picamera2.encoders.LibavH264Encoder`(소프트웨어 x264)로 치환되는데, 그 `force_key_frame()`이 설치된 PyAV 버전과 안 맞아 `frame.pict_type = "I"`가 `TypeError`로 인코더 백그라운드 스레드를 크래시시킨다 — 그래서 이 도구는 `force_key_frame()`을 호출하지 않고 `iperiod` 기본값(약 1초 GOP)의 자연 키프레임 재삽입에 기대며, 재접속 후 새 클라이언트가 유효 프레임을 받기까지 실측 약 1~2초(가끔 더 걸림) 지연이 있다. 또한 `FfmpegOutput.stop()`이 `audio=False`면 무기한 대기(`timeout=None`)할 수 있어(재접속 실패로 멈춘 ffmpeg에서 실제로 재현됨) `_stop_output_with_timeout()`이 별도 스레드+bounded timeout+강제 kill로 안전망을 건다. 상세 실측 수치·경과는 `docs/vision_camera_bringup.md` §Phase 3 |
| `tools/color_calibrate.py` | **신설(2026-07-25, §9 빌드순서 5번 — 현장 색 캘리브레이터, `docs/vision_plan.md` §5.5)**. 지오메트리 캘리브레이션(`calib_capture.py`/`calib_analyze.py`, 카메라 인트린식)과는 **완전히 다른 개념** — 렌즈/센서를 다루지 않고, 프레임 ROI(`--roi x,y,w,h`, GUI 없음·`cv2.selectROI` 미사용)의 HSV 분포에서 `ColorFilter`/`RedRingDetector` 생성자 인자를 제안한다. **백분위수(기본 p5~p95) 기반** — 평균±표준편차 대신 쓴 이유는 그림자/글레어/과노출 클리핑 같은 국소 이상치에 강해서(§5.5 "그림자·글레어·과노출 클리핑 대비", `tests/test_color_calibrate.py::test_calibrate_roi_percentile_range_ignores_minority_outlier_pixels`가 합성 글레어 이상치로 직접 증명). **빨강 Hue 랩어라운드 대응** — `--wrap-split-hue`(기본 90) 기준 저/고 두 구간의 표본 비율이 둘 다 `--wrap-min-fraction`(기본 0.05) 이상이면 랩어라운드로 판정해 `RedRingDetector` 호환 파라미터(`low_hue_max`/`high_hue_min`/`sat_min`/`val_min`)를, 아니면 `ColorFilter(mode=color)` 호환 파라미터(`hue_range`/`sat_min`/`val_min`/`val_max`)를 출력 — 어느 소비자와 호환되는지 산출물에 명시한다(`ColorFilter`는 랩어라운드 미지원 blind spot, `RedRingDetector`는 양끝 게이팅 실전례 — 이 도구가 따르는 전례). **출력은 "제안"까지** — 사람이 그대로 복붙할 yaml 조각을 stdout + `--output`으로 내고, 기존 preset yaml을 자동으로 덮어쓰지 않는다. `--diagnostic-dir`로 ROI 오버레이 이미지 + HSV 히스토그램 PNG(matplotlib Agg, `jsonl_view.py`와 동일 headless-safe 패턴) 저장. 입력은 이미지 파일 또는 녹화 폴더/영상(`utils/frame_source.py`의 `open_dir_or_bag()` 재사용 — 새 프레임 소스 미구현). 임계값·백분위수·마진 전부 CLI 파라미터(매직넘버 금지, §7.3). **마진 기본값은 2026-07-28에 `hue=6`/`sat=0`/`val=0`으로 정식 확정** — 백분위수(프레임 내 공간 변동)와 마진(캘리브↔비행 조건 변화)이 덮는 것이 다르다는 구분, 실측 8점 산포 1σ 도출, 이중계상 위험, S/V를 0으로 남긴 이유는 아래 "color_calibrate.py 마진 기본값 확정" 절. **하드웨어 의존 없음** — `jsonl_view.py`/`calib_analyze.py`와 동일한 예외로 `.venv` 설치 + `tests/test_color_calibrate.py` 대상. 정확성은 합성 패치 왕복 테스트(산출된 임계값을 실제 `ColorFilter`/`RedRingDetector`에 먹여 그 패치가 실제로 검출되는지까지 확인, 빨강 랩어라운드 케이스 별도 검증)로 담보 + 골든셋(`tests/golden/distress/10m/`) 실제 프레임으로 `distress_coarse.yaml` 손튜닝값(hue_range=[35,85])과 sanity 교차확인(합성 골든 프레임이 노이즈 없는 단색이라 산출 range가 손튜닝값보다 훨씬 좁게 나오는 게 정상 — 완전 일치 요구 아님) |
| `tests/golden/` | 골든셋 회귀 픽스처(§7.9 항목7). `<타겟>/<고도>/frame_NNN.png`+`labels.json` — 구조·스키마·현재 들어있는 것·재생성법은 `tests/golden/README.md`. **⚠️ 전부 합성(synthetic) 데이터** — 실촬영 아님(카메라 브링업 전, `docs/vision_status.md`). `tests/golden/generate_synthetic.py`가 생성 소스(pytest 대상 아님, 수동 재생성 도구). **[2026-07-25]** `distress/fine/`(`white_box_detector` 캐스케이드, 실측 박스 비율 0.0667) + `no_target/distress_fine/`(오탐 회귀) 추가 — 여전히 합성 |

---

## 라이브 스트림 어댑터 기본값 (§7.9 항목5, `utils/stream.py`)

vision_plan.md §7.9가 정확한 해상도/포트를 못박지 않아 세션 지시에 따라 합리적 기본값으로 정하고 여기에 기록한다:

- **해상도:** 640x480(VGA) 박스 — 정확히 640x480으로 리사이즈(letterbox)하지 않고 **종횡비를 유지한 채 박스 안에 맞추는 축소**(업스케일 없음). 관찰용 브라우저 `<img>` 태그가 알아서 크기를 맞추므로 letterbox 패딩은 불필요한 복잡도로 판단.
- **포트:** 8080 (`--stream-port`로 변경 가능. `0`을 주면 OS가 임시 포트를 골라준다 — 테스트/포트충돌 회피용).
- **바인딩 주소:** `0.0.0.0`(모든 인터페이스 — RPi가 실제 배포될 네트워크에서 랩탑이 접속 가능해야 하므로). `--stream-host 127.0.0.1`로 로컬만 제한 가능.
- **JPEG quality:** 80, **큐 길이:** 2 — 지연 누적보다 최신 프레임 우선(관찰이 목적, 무손실 기록이 목적 아님. 상시 기록은 `--output`/mp4 덤프 경로가 담당).
- 비차단 큐 패턴은 `utils/blackbox.py`의 `_DropOldestQueueHandler`와 동일 설계(evict-then-insert를 락으로 원자화 — 여러 producer 스레드 동시 push 시 `queue.Full` 경쟁 방지, `blackbox.py`는 단일 로거 호출 경로라 이 레이스가 실질적으로 안 나타나 그대로 둠).

---

## target_sink 소켓 기본값 (2026-07-28, `utils/target_sink.py`)

`docs/vision_fc_interface.md`가 포트/큐 길이를 못박지 않아 세션 판단으로 정하고 근거를 여기 남긴다
(`utils/stream.py` 기본값 절과 같은 관례).

- **주소:** `127.0.0.1` **고정**. `stream.py`(MJPEG)는 랩탑 브라우저에서 봐야 해서 `0.0.0.0`이
  기본이지만 이쪽은 반대다 — 컨테이너가 `NetworkMode=host`라 소비자가 이미 같은 네트워크
  네임스페이스 안에 있어 루프백으로 충분하고, `0.0.0.0`으로 열면 **비행 중 유도 스트림이 주변
  네트워크에 그대로 노출**된다. 회귀테스트로 박아 뒀다.
- **포트:** `8091`. `stream.py`의 8080과 인접한 "vision 디버그/IO" 대역으로 묶어 기억하기 쉽게
  하되 겹치지 않게 했고, MAVLink 대역(14550~14580)을 피했다. `port=0`을 주면 OS가 임시 포트를
  고른다(테스트/포트충돌 회피 — `stream.py`와 같은 관례).
- **큐 길이:** `8`(≈10Hz에서 0.8초). `blackbox.py`의 1000과 다른 이유는 **용도가 다르기 때문**이다
  — 블랙박스는 무손실 기록이 목적이라 길게 잡지만, 이건 제어용 스트림이라 **최신성 > 완전성**이다
  (오래된 추정치는 어차피 소비자가 stale로 버린다). 8은 스케줄러 히컵 한 번을 넘길 정도만 주고
  멈춘 소비자가 몇 초치 낡은 데이터를 쌓아두지 못하게 하는 절충값. `stream.py`의 2보다 큰 이유는
  프레임과 달리 레코드가 작고, 유실이 `seq` 구멍으로 소비자에게 보이기 때문.
- **`send_timeout_s`:** `0.5`. 이 시간을 넘겨 못 받는 소비자는 끊는다. `sendall`이 중간에 잘리면
  그 연결의 바이트 스트림은 이미 깨져 살릴 수 없으므로 재접속시키는 게 맞다.
- **`TCP_NODELAY` 켬** — 10Hz 제어 스트림에서 Nagle 지연은 그대로 유도 지연이 된다.
- **`SO_REUSEADDR` 켬** — vision=서버 구조의 비용. 프로세스 재시작 시 `TIME_WAIT`로 같은 포트
  재바인드가 거부되는 것을 막는다.

⚠️ **`stale_timeout_s`/`hold_before_reascend_s`는 일부러 여기 없다** — 소비자(FC)가 자기 제어틱
에서 재는 값이고 vision이 정해 보낼 값이 아니다(§5.4 "침묵을 유일한 권위로 삼는다").

🔀 **문서 권고와 갈리는 지점(사용자 확인 필요):** `docs/vision_fc_interface.md` §8 권고 4번은
**shim=서버/vision=클라이언트**였는데 세션 지시에 따라 **반대로**(vision=서버) 구현했다. §5.4의
핵심 성질(프로세스 사망 → 소비자가 **EOF 즉시** 수신)은 **방향과 무관하게 유지**되고(회귀테스트로
확인), 실제로 바뀌는 것은 **재접속 책임의 소재**뿐이다 — vision=서버면 Phase 2 shim이 재접속
루프를 갖는다.

---

## 컨테이너 ROS2 shim 노드 (2026-07-28, `vision/ros/` — 인터페이스 Phase 2)

Phase 1이 호스트 쪽 절반(`core/wire.py` + `utils/target_sink.py` + `main.py --target-sink`)을
끝내 놓은 뒤, 그 JSON Lines를 **컨테이너 안에서 ROS2 토픽으로 재발행**하는 나머지 절반이다.
`docs/vision_fc_interface.md` §8 권고 R4의 완성.

### 🔴 배치가 `fc_ros/`가 아니라 `vision/`인 이유 (문서 §9 F1과 갈리는 지점)

`docs/vision_fc_interface.md` §9는 이 노드를 **F1(FC 도메인)** 으로 배정하고
`fc_ros/fc_ros/nodes/vision_bridge_node.py`에 두라고 했다. 이번 세션은 **세션 지시에 따라
`vision/ros/`에 뒀고 `fc_ros/`·`fc_bridge/`를 한 줄도 건드리지 않았다.** 실질적 차이:

- ✅ `colcon build` 대상이 되지 않는다 → `docs/rpi_deploy.md`의 배포 절차가 **무변경**이고
  FC의 stale-build 함정(실비행 8건의 근본원인, `4dc30f9`)과 무관해진다. 컨테이너에서는
  `PYTHONPATH=/drone_ws/src/suridoksuri`로 소스를 직접 실행한다.
- ✅ 도메인 경계가 유지된다 — vision이 자기 인터페이스의 양쪽 끝을 다 갖고, 빌드 의존도
  교차 import도 여전히 0이다.
- 🔀 대신 **ROS 패키지가 아니라서 `ros2 run`/launch로 못 띄운다.** FC가 `phase2.launch.py`에
  이 노드를 넣고 싶어지면 그때 `fc_ros/`로 옮기거나 얇은 래퍼를 만들어야 한다 —
  **사용자/FC 결정 사항**이고 그때 옮겨도 `shim_core.py`는 그대로 재사용된다.

### 토픽·메시지 타입 결정

| 토픽 | 타입 | 왜 |
|---|---|---|
| `/vision/target_pose` | `geometry_msgs/PoseWithCovarianceStamped` | §3.4 (b) — 새 빌드 의존 0, `uncertainty`가 갈 6x6 공분산 자리가 **이미 있다**(실측 캘리브 후 채우면 이 파일을 다시 안 열어도 된다) |
| `/vision/target_status` | **`diagnostic_msgs/DiagnosticArray`** | §3.2가 요구한 "두 토픽 `header.stamp` 동기"를 지키려면 header가 필요한데 **`DiagnosticStatus`에는 header가 없다**(실기체 msg 원문 확인 — `level/name/message/hardware_id/values`뿐). header를 가진 것은 `DiagnosticArray`다. 한 배열에 `vision/target` + `vision/state` 두 status를 **같은 stamp로** 싣는다 |
| `/mavros/landing_target/raw` | `mavros_msgs/LandingTarget` | §3.4 (a) — PX4 네이티브 precision-land 피벗의 **유일한 경로**. 🔴 **기본 꺼짐**(`--publish-landing-target`) |
| (c) 커스텀 msg | — | 만들지 않았다. (b)로 표현이 되고 빌드 의존 비용이 실익보다 크다(§3.3) |

### 🔴 `LandingTarget` 함정 2개 — 둘 다 실기체에서 원문 대조로 확정

**① `frame` 상수를 쓰면 안 된다.** `LandingTarget.msg`와 `common.hpp`의 `MAV_FRAME`이
1씩 어긋나 있다: msg `LOCAL_NED=2`인데 실제는 `1`, msg에 `BODY_FRD` 상수는 **아예 없다**(실제 12).
`landtarget_cb`는 `static_cast<MAV_FRAME>(req->frame)` 기준이라 msg 상수를 믿으면
`MISSION`으로 읽혀 `position_valid=false`가 된다. → **정수 리터럴 12**를
`shim_core.MAV_FRAME_BODY_FRD` 한 곳에만 두고, `shim_node.py`가 msg 상수 이름을 참조하면
**AST 회귀테스트가 red**가 된다(파괴검증 D1/D1b).
⚠️ 하필 `LandingTarget.msg`에 `GLOBAL_TERRAIN_ALT_INT = 12`가 있어 "상수를 썼는데 값은 맞는"
경로가 존재한다 — 값이 맞아도 이름이 거짓말이라 금지 목록에 넣었다.

**② `frame=12`에 넣는 좌표는 FRD가 아니라 FLU다 — 브리핑과 반대다.** mavros 2.14.0 원문:
`case MAV_FRAME::BODY_FRD: position = ftf::transform_frame_baselink_aircraft(tr.translation())`.
`baselink`(FLU)→`aircraft`(FRD) 변환을 **플러그인이 한다.** 즉 ROS 쪽 입력은 항상 FLU다.
여기에 와이어의 `position_frd`를 넣으면 변환이 **두 번** 걸려 y·z 부호가 뒤집힌다.
→ pose 토픽이든 LandingTarget이든 **전부 `position_flu`만** 쓴다.
(`position_frd`는 mavros를 거치지 않고 MAVLink를 직접 만들 때를 위해 와이어에 남아 있는 값이다.)

### 침묵/무효/사망 3분법이 토픽 배치를 결정한다 (§5.4)

| 상황 | `/vision/target_pose` | `/vision/target_status` |
|---|---|---|
| 정상 검출 | 발행 | `OK` |
| 안 보임(`valid=false`) | **침묵**(실을 좌표가 없다) | **`WARN` + 사유** ← 여기서 구분된다 |
| 생산자 사망(EOF) | 침묵 | **`ERROR` + `producer_eof`** |
| shim 자신도 사망 | 침묵 | **침묵** |

**→ pose 토픽의 침묵만으로는 "안 보임"과 "죽음"을 못 가른다. 소비자는 반드시 status를 같이
봐야 한다.** pose에 0/NaN을 채워 "계속 발행"하는 안은 기각했다(0은 `core/wire.py`가 "가장
위험한 거짓말"이라 부른 그것). 끊긴 동안 `heartbeat_period_s`마다 나가는 `ERROR` status가
"생산자 사망"과 "shim 사망"을 가르는 유일한 신호다.

### 두 클록의 역할 분담 — Phase 1의 최대 미검증 항목이 닫혔다

**실측(2026-07-28, 실기체):** 호스트와 `fc` 컨테이너의 time namespace inode가
`time:[4026531834]`로 **동일**하고 `/proc/self/timens_offsets`가 양쪽 다 `monotonic 0 0` /
`boottime 0 0`이며 컨테이너 `time.monotonic_ns()`가 호스트 `/proc/uptime`과 일치한다
(145453.053 vs 145453.06). → **`CLOCK_MONOTONIC`은 완전히 같은 기준이다.** `clock_offset_ns`
환산 없이 그냥 뺄셈이 성립한다.

- `header.stamp` ← 레코드의 **`stamp_wall_ns`**. ROS 기본 클록이 `CLOCK_REALTIME`이라
  monotonic(≈145453초)을 넣으면 `node.get_clock().now()`(≈1.7e9초)와 비교하는 mavros/fc_ros
  쪽에서 전부 무의미해진다(파괴검증 D8).
- **stale 판정** ← **`stamp_monotonic_ns`**. shim이 자기 `time.monotonic_ns()`로 나이를 잰다.
  ⚠️ **shim은 stale을 이유로 발행을 막지 않는다** — `age_s`를 KeyValue로 싣고
  `stale_warn_s`를 넘으면 `level`만 올린다. 최종 거부는 FC 몫이다(§5.4).

### 🔴 초록구역 0.105m — z를 고치지 않고 숫자로 수출한다

비전이 재는 것은 **매트 윗면까지의 거리**이고 기체는 그 윗면에 내려앉으므로 착륙 자체엔 보정이
필요 없다. 보정하지 않는 적극적 이유: 라이다 빔이 **매트 위에 떨어지면 라이다도 윗면을 보므로
둘이 일치**하고 **매트 밖 지면에 떨어질 때만** 0.105m 어긋나는데, 어느 쪽인지 코드가 알 수 없다
— 모르는 기하를 가정해 조용히 더하는 것이 이 저장소가 금지하는 그 실수다(캘리브 해상도 불일치를
자동 재스케일하지 않은 것과 같은 판단). 대신 `plane_reference` / `platform_height_m` /
**`ground_agl_minus_vision_z_m`**(= 지면기준 AGL − 비전 z, 빔이 매트 밖일 때)를 KeyValue로
내보내 FC가 스스로 정하게 한다(파괴검증 D9).

### 🔴 `orientation`을 Pose에 싣지 않는다

와이어의 `orientation`은 **카메라 광학 프레임**이다. `PoseWithCovarianceStamped`는 position과
orientation이 **하나의 `frame_id`를 공유**하므로 body FLU 좌표 옆에 cam_optical 쿼터니언을
놓는 것 자체가 프레임 거짓말이고, 이 저장소가 이미 당한 사고 유형(`pos_ned` vs `vel_ned`)과
같은 종류다. → Pose orientation은 **단위 쿼터니언**, 회전 공분산 대각은 `unknown_covariance`,
원본은 KeyValue `orientation_cam_optical_xyzw`로 보존(파괴검증 D5).

### `level`에 `not_for_closed_loop_30cm`을 태우지 않는다

이 플래그는 **지금 100% True**다(실측 캘리브 보류). level에 태우면 level이 **항상** WARN이라
진짜 고장이 묻힌다. 고장이 아니라 **바닥 고도 제약**이므로 KeyValue(`not_for_closed_loop_30cm`
+ 이미 번역된 `closed_loop_floor_agl_m`=3.0)로만 나간다. 해석은 "폐루프 금지"가 아니라
**"최종 커밋 금지"** 다.

### shim 노드 기본값 (매직넘버 금지, §7.3)

- **토픽:** `/vision/target_pose` · `/vision/target_status` · `/mavros/landing_target/raw`.
- **`pose_frame_id`: `base_link`** — mavros가 baselink=FLU로 쓰는 그 이름이다.
- **QoS: BEST_EFFORT / KEEP_LAST / depth=1.** `fc_ros`의 `_MAVROS_QOS`와 같은 계열(§9 F1)이되
  depth만 1인 이유는 제어용 스트림이라 **최신성 > 완전성**이기 때문(`utils/target_sink.py`의
  drop-oldest 큐와 같은 철학). 호환성을 정하는 것은 reliability/durability이지 depth가 아니다.
- **`stale_warn_s`: 0.75 — 🔴 실측 근거 있음(U5).** 이건 **거부 임계값이 아니라 로그 레벨
  임계값**이고 실제 거부는 FC가 자기 제어틱에서 한다. 값의 근거는 아래 U5 실측 절.
- **`unknown_covariance`: 1e6** — `uncertainty`가 None(=지금 항상)일 때 대각에 넣는 "모름".
  0을 넣으면 "분산 0 = 완벽히 정확"이라 `robot_localization` 같은 기성 필터가 절대 신뢰로
  받아들인다(파괴검증 D6).
- **`publish_landing_target`: False** — `px4_config.yaml:214`가 `listen_lt: false`라 구독자가
  아예 없다. 켜는 것은 FC 결정(D2).
- **`reconnect_delay_s`/`heartbeat_period_s`: 1.0** — 재접속은 shim 책임이다(vision이 서버).

### 🔴 U5 실측 — 실제 발행 주파수는 10Hz가 아니라 **약 4.4Hz**다 (2026-07-28 실기체)

`stale_timeout_s`의 근거가 될 숫자라 실카메라로 직접 쟀다. 조건: `main.py live`
+ `distress_fine.yaml` + `nominal.yaml`, `LiveFrameSource` 기본 해상도 **4608x2592**
(=`nominal.yaml`의 `image_size`, solvePnP 캘리브와 어긋나지 않게 하는 그 값), 44.2초 연속.

| 지표 | 값 |
|---|---|
| 프레임 간격 median | **0.2207 s** (= 4.53 Hz) |
| 프레임 간격 p95 | **0.310 s** |
| 컨테이너 `ros2 topic hz /vision/target_status` | **4.35 Hz** (min 0.204s / max 0.918s / std 0.080s) |
| 기동 포함 벽시계 평균 | 3.66 Hz (44.2초, 첫 프레임 6.9초 포함) |

**→ 이 저장소가 여러 곳에서 가정해 온 "10Hz 제어 스트림"은 실제와 2배 이상 다르다**
(`utils/target_sink.py`의 큐 길이 8 = "≈10Hz에서 0.8초" 주석도 실제로는 **≈1.8초**치다 —
안전한 방향이라 이번에 값을 바꾸지는 않았다). 병목은 **4608x2592 전해상도 처리**이고,
해상도를 낮추면 올라가겠지만 그러면 `nominal.yaml` 캘리브와 어긋나 거리가 통째로 틀린다
(`vision/CLAUDE.md` "캘리브레이션 해상도 불일치" 절) — **해상도/주파수 트레이드오프는
사용자 결정 사항**이다.

**FC에 주는 함의:** `stale_timeout_s`를 0.5s 근처로 잡으면 정상 지터(p95 0.310s)와의 여유가
1.6배뿐이라 헛경보가 난다. shim의 `stale_warn_s` 기본값을 **0.75s**(p95의 약 2.4배 =
연속 3프레임 이상 누락)로 정한 근거가 이것이고, 회귀테스트가 이 관계를 고정한다.

### 🔴 `/vision/landing_setpoint` — 상대 오차 → 절대 setpoint (2026-07-28, 사용자 확정 흐름)

사용자 원안은 *"픽스호크 attitude 수신 → 연산 → 목표 setpoint → offboard_node 이동"* 이었고,
**흐름은 그대로 두되 연산 지점만 소켓 뒤(이 ROS 그래프 안)로** 옮긴 절충안이 채택됐다.

**소켓 앞(호스트)에서 안 하는 이유 3가지:**
1. attitude가 소켓 왕복 + vision 발행주기(**실측 4.4Hz**)만큼 늦는다. 접근 중 10°/s 흔들림에
   attitude 250ms 지연 = 2.5° = **10m AGL에서 44cm** — 30cm 요구를 지연 하나로 날린다.
   같은 그래프 안이면 30Hz 자세를 지연 없이 받는다.
2. 와이어(JSONL)가 절대좌표로 바뀌면 `mavros_msgs/LandingTarget` 네이티브 precision-land
   **피벗 경로가 통째로 막힌다**(§8이 명시적 안전장치로 설계해 둔 카드). `position_flu`를
   와이어에 유지하면 그 카드가 산다.
3. attitude 역방향 채널을 만들면 vision이 FC에 의존하게 된다 — 지금은 FC가 죽어도 vision이
   상대 pose를 계속 뱉는다.

#### 🔴 절대 좌표를 **기억하지 않는다** (이 기능의 핵심 계약)

정밀착륙에서 중요한 것은 절대 위치가 아니라 "타겟 대비 오차를 0으로 만드는 것"이다. 목표점을
한 번 계산해 고정하면 **EKF 드리프트가 그대로 착륙 오차로 남는다.** 그래서 매 레코드마다
`목표 = (그 순간의 최신 local_position/pose) + (그 순간의 상대 오차)`로 **다시 계산**한다 —
드리프트가 현재위치와 목표점에 똑같이 실려 상쇄된다. `ShimRouter`가 보관하는 상태는 최신
`VehiclePose` **하나뿐**이고 계산된 절대 좌표는 어디에도 저장되지 않는다(회귀테스트 대상).

#### 변환 유도

```
① p_flu      : 와이어 `position_flu` — body FLU(x=전방, y=좌, z=상) 상대벡터
② q_enu      : /mavros/local_position/pose 의 orientation = base_link(FLU) → map(ENU) 회전
               (mavros 전역 관례로 ROS 쪽은 항상 FLU/ENU. FRD/NED 변환은 mavros가 자기 안에서)
③ Δ_enu      = R(q_enu) · p_flu
④ target_enu = pose.position_enu + Δ_enu        ← 🔴 매번 최신 pose로 다시 더한다
⑤ 발행       : geometry_msgs/PoseStamped, frame_id="map"(ENU) — ②·④와 같은 프레임
```

| 토픽 | 타입 | 왜 |
|---|---|---|
| `/vision/landing_setpoint` | `geometry_msgs/PoseStamped` | **새 빌드 의존 0**(커스텀 msg를 만들면 이 shim이 `vision/`에 사는 이유 자체가 사라진다). `/vision/target_pose`가 `PoseWithCovarianceStamped`를 고른 것과 같은 논리 |

**🔴 왜 `[N, E, h_up]`이 아니라 ENU인가.** `offboard_node._publish_pos_setpoint(pos_ned, ...)`의
`pos_ned`는 `[N, E, h_up]`(3번째가 **위** 양수)인데 이건 NED도 ENU도 아닌 저장소 내부
하이브리드다. 그 숫자를 `geometry_msgs/PoseStamped`(ROS 관례상 `frame_id`가 뜻하는 프레임)에
담으면 **선언한 프레임과 값이 어긋나는 거짓말**이고, 이 저장소가 이미 당한 사고 유형(`pos_ned`
vs `vel_ned`가 같은 접미사로 반대 부호 / cam_optical 쿼터니언을 body Pose에 안 싣기로 한 판단)과
같은 종류다. ENU/`map`으로 내면 소비자는 **이미 저장소에 있는 관용구 한 줄**을 그대로 쓴다:

```python
# fc_ros/adapters/vehicle_state_bridge.py::update_from_pose 와 문자 그대로 같은 줄
pos_ned = np.array([msg.pose.position.y, msg.pose.position.x, msg.pose.position.z])
```

그 `[N, E, h_up]` 삼중항도 `enu_to_pos_ned_n_e_hup()`(단일 출처)로 계산해 KeyValue로
**진단용으로만** 수출한다 — 권위 있는 값은 PoseStamped 쪽이다.

#### 🔴 setpoint의 orientation은 단위 쿼터니언이 아니라 **현재 기수방위**다

`_publish_pos_setpoint` docstring이 못박은 실사고: *"orientation 미설정 시 ROS2 기본값(단위
쿼터니언, ENU yaw=0 = NED yaw=90°)이 실제 헤딩과 무관하게 그대로 발행돼 OFFBOARD 진입 첫 틱에
yaw 점프"*(2026-07-21 flight04 yaw 스핀 사고). 즉 **여기서 단위 쿼터니언은 "모름"이 아니라
"동쪽을 보라"는 명령**이다. 그래서 그 순간 기체의 실제 yaw만 뽑은(roll/pitch=0) 쿼터니언을 실어
"현재 헤딩 유지"를 뜻하게 한다 — 정사각 타겟은 90° 자기대칭이라 타겟에서 방위를 유도하는 것
자체가 물리적으로 무의미하기도 하다.

#### degrade — 3분법을 그대로 따른다

| attitude 상태 | `/vision/landing_setpoint` | `/vision/target_pose` | `/vision/target_status` |
|---|---|---|---|
| 정상 | **발행** | 발행 | `vision/setpoint` = `OK` |
| 아직 못 받음 | **침묵** | **그대로 발행** | `WARN` + `attitude_missing` |
| stale | **침묵** | **그대로 발행** | `WARN` + `attitude_stale` |
| 쿼터니언 노름 0 | **침묵** | **그대로 발행** | `WARN` + `attitude_invalid_quaternion` |
| `valid=false` | **침묵**(좌표가 있어도) | 침묵 | `WARN` + `no_target` |

🔴 **0이나 추측값을 채우지 않는다** — setpoint 자리의 0은 "기체 바로 아래로 가라"가 된다
(`core/wire.py`가 "가장 위험한 거짓말"이라 부른 그것). 🔴 **상대 pose 경로는 attitude 유무와
무관하게 계속 나간다** — 새 기능이 기존 경로를 죽이면 안 된다.
사유를 `vision/target`이 아니라 **별도 `vision/setpoint` status**로 낸 이유: attitude 부재는
"검출이 나쁘다"가 아니라 "절대화 경로가 막혔다"이고, target level에 태우면 mavros 없는 지상시험
에서 level이 **항상** WARN이라 진짜 고장이 묻힌다(`not_for_closed_loop_30cm`과 같은 판단).

#### `attitude_stale_s` = 0.25 — 🔴 위아래 양쪽에서 조인 값 (`stale_warn_s` 복사 아님)

- **위(오차예산)에서:** 폐루프 바닥고도 AGL 3.0m(`closed_loop_floor_agl_m`=`terminal_agl_m`)에서
  10°/s 흔들림 × 0.25s = 2.5° → 지상오차 `3.0·tan(2.5°) = 0.131m`, 30cm 요구의 절반 아래
  (`core/frames.py` §4.5의 `지상오차=고도×tan(θ)` 식). 같은 지연이 10m AGL에서는 0.44m로
  예산을 넘지만, 고고도 추정치로 **최종 커밋하지 않는 것**은 상태머신 커밋 게이트와
  `closed_loop_floor_agl_m`이 이미 담당한다.
- **아래(지터)에서:** 🔴 **실비행 rosbag 48건 실측** — `/mavros/local_position/pose` median
  **29.93Hz**, 메시지 간격 median **33.1ms** / p95 **36.0ms** / p99 **37.2ms**(4개 bag 직접
  측정). 0.25s는 p99의 **6.7배**라 정상 지터로는 안 뜬다. 실제로 걸리는 것은 드물게 관측된
  수백 ms 드롭아웃(같은 bag들에서 최대 211/339/723/**2564**ms)뿐이고 그때 멈추는 것이 의도다.

⚠️ `stale_warn_s`(0.75, vision 4.4Hz 근거)와 **근거가 다른 별개 값**이다 — 누가 "통일"하면 한쪽
근거가 조용히 사라진다(회귀테스트로 고정).

#### executor 스레드가 하나 늘었다

예전엔 구독이 없어 spin을 안 돌렸는데, 구독은 spin 없이는 콜백이 안 뛴다. 소켓 폴링
(`_POLL_S=0.2s`) 안에서 `spin_once`를 부르면 자세가 최대 0.2초 묵어 위 0.25s 예산의 80%를
까먹으므로 **`SingleThreadedExecutor`를 데몬 스레드**에서 돌린다. 두 스레드가 만나는 곳은
`ShimRouter._vehicle_pose` 하나뿐이고 락으로 보호된다(`VehiclePose`가 `frozen=True`인 것도
같은 이유 — 완성된 객체를 통째로 갈아끼워 부분갱신 상태가 안 보이게).
🔴 **`--no-landing-setpoint`면 구독도 스레드도 안 만든다** — 이전 동작으로 되돌리는 escape hatch.

#### ✅ 실기체 종단간 실증 (2026-07-28, RPi + `fc` 컨테이너)

**mavros는 안 돌고 있어** `/mavros/local_position/pose`를 `ros2 topic pub -r 30`으로 직접 주입해
배선을 증명했다. 생산자는 손으로 쓴 JSON이 아니라 **진짜 생산자 코드**(`core/wire.py` +
`utils/target_sink.py`, 카메라만 뺀 `main.py --target-sink` 경로)를 호스트 picam-venv에서 돌렸다.
고정 입력 `p_flu=(10, 0, −8)`(전방 10m·아래 8m), 기수 정북.

| 기체 ENU 위치 | 기수 | 실제 `/vision/landing_setpoint` position | 기대 |
|---|---|---|---|
| (100, 200, 30) | 북 | **(100.0, 210.0, 22.0)** | (100, 210, 22) ✅ |
| (105, 195, 31) | 북 | **(105.0, 205.0, 23.0)** | (105, 205, 23) ✅ |
| (−40.5, 17.25, 12.75) | 북 | **(−40.5, 27.25, 4.75)** | (−40.5, 27.25, 4.75) ✅ |
| (100, 200, 30) | **동** | **(110.0, 200.0, 22.0)** | (110, 200, 22) ✅ |
| (0, 0, 50) | 북 | **(≈0, 10.0, 42.0)** | (0, 10, 42) ✅ |

→ **드리프트 상쇄가 실기체에서 증명됐다**: 상대 델타 `(0, +10, −8)`가 네 위치에서 동일하고
절대 목표점은 전부 다르다. 자세 회전도 실시간 반영된다(북→동에서 목표가 실제로 90° 돌았다).
`frame_id: map`, orientation `(0, 0, 0.7071, 0.7071)` = **현재 기수방위**(단위 쿼터니언 아님).

**degrade 전이 4단계 실제 재현:**

| 단계 | `/vision/landing_setpoint` | `vision/setpoint` status | `/vision/target_pose` |
|---|---|---|---|
| A. attitude 없음 | **침묵**(echo 타임아웃) | `attitude_missing`, `published:false`, **좌표 키 자체가 없음** | (10, 0, −8) **계속 발행** |
| B. attitude 주입 | **발행 시작**(3.98Hz = 생산자 4Hz) | `ok`, `attitude_age_s:0.0296` | 계속 |
| C. attitude 끊김 | **침묵**(0.25s 후) | `attitude_stale`, `attitude_age_s:12.71` | (10, 0, −8) **계속 발행** |
| D. attitude 재개 | **발행 재개** | `ok` | 계속 |

종료 카운터: `pose_pub=53 status_pub=54 setpoint_pub=29 vehicle_pose_rx=222` —
`vehicle_pose_rx=222`가 executor 데몬 스레드가 실제로 30Hz로 돌았다는 증거이고,
`setpoint_pub(29) < pose_pub(53)`이 "자세가 없던 초반에는 침묵했다"는 계약을 카운터로 보여준다.
**SIGTERM graceful 종료도 executor 스레드 추가 후에도 정상**(`RCLError` 없이 위 요약 로그 출력 —
커밋 `b14f42a` 회귀 유지).

⚠️ `attitude_stamp_ns`가 `0`으로 찍히는 것은 `ros2 topic pub`이 header stamp를 안 채우기
때문이며 실제 mavros에서는 채워진다. **실제 비행 telemetry로는 검증하지 않았다.**

#### ⚠️ stamp 동기는 안 했다 (미검증/미구현으로 남긴 것)

목표점은 **그 순간의 최신 자세**로 계산한다 — 레코드의 촬영시각(`stamp_wall_ns`)에 가장 가까운
자세를 골라 쓰는 `message_filters`/pose 링버퍼 방식은 **구현하지 않았다.** 엄밀히는 정지 타겟의
월드 좌표가 시불변이므로 촬영시각 자세를 쓰는 것이 맞지만, 세션 지시가 "매 레코드마다 최신
pose"를 명시했고 드리프트 상쇄가 그쪽을 요구한다. **대신 어긋남을 숨기지 않고 수출한다** —
`attitude_age_s` / `attitude_stamp_ns` KeyValue로 소비자가 직접 잰다. 실비행 데이터로 이 스큐가
문제가 되는지 확인한 적은 **없다.**

### 실행

```bash
docker exec fc bash -lc '
  source /opt/ros/humble/setup.bash
  export PYTHONPATH=/drone_ws/src/suridoksuri:$PYTHONPATH
  python3 -m vision.ros.shim_node
'
ros2 topic echo /vision/target_pose
ros2 topic echo /vision/target_status
ros2 topic echo /vision/landing_setpoint
```

🔴 **`PYTHONPATH`는 반드시 이어붙인다(`:$PYTHONPATH`) — 이 문서의 예전 명령이 틀렸다.**
덮어쓰면 방금 `setup.bash`가 넣은 `/opt/ros/humble/lib/python3.10/site-packages`가 날아가
`import rclpy`가 즉시 죽는다. 2026-07-28 실기체에서 `ModuleNotFoundError: No module named
'rclpy'`로 직접 재현했고(덮어쓰기 실패 / 이어붙이기 성공을 같은 세션에서 대조), 종단간 검증은
고친 형태로 수행했다.

⚠️ `docker exec fc bash -lc "python3 -c 'import rclpy'"`는 **ROS setup을 source하지 않아
실패한다** — 반드시 `source /opt/ros/humble/setup.bash &&`를 앞에 붙일 것.

⚠️ 컨테이너 안 `DiagnosticStatus.level`은 msg상 `byte`라 rclpy가 **1바이트 `bytes`** 를
요구한다(int를 넣으면 `AssertionError`). 실기체에서 실측 확인한 사항이다.

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

## ArUco Phase 4 파이프라인 배선 (2026-07-24, `docs/vision_aruco_branch.md` Phase 4 — 브랜치 종료)

Phase 1(`nominal.yaml`)·Phase 2(`ArucoDetector`)·Phase 3(`solve_target_pose`/`TargetEstimate`/
`load_camera_calibration`)를 실제 파이프라인(`main.py`/`replay.py`)에 배선해 JSONL 블랙박스까지
`TargetEstimate`가 실리도록 완성한 단계.

- **배선 위치 판단: `modules/`의 신규 모듈이 아니라 `main.py`/`replay.py` 레벨.** import 규칙표
  (아래 "import 규칙" 절)가 `modules/ ← vision.core 만 import`인데, `solve_target_pose()`를
  호출하려면 `utils/calibration_loader.py`(utils, core 아님)를 반드시 import해야 한다 —
  `modules/`의 현재 12개 파일 어디에도 `utils` import 전례가 없어(grep 확인) 새 모듈로 만들면
  이 규칙을 위반하는 첫 사례가 된다. 반면 `main.py`/`replay.py`는 애초에 "presets 경로 + utils +
  core 만 import" 규칙이라 `core.target`/`utils.calibration_loader` import가 위반 없이 자연스럽다.
  `core/target.py` docstring이 애초에 "calib = load_camera_calibration(...); estimate =
  solve_target_pose(...)" 호출 순서를 호출자 책임으로 문서화해둔 것도 이 판단과 일치한다.
- **calib은 파이프라인 실행 중 매 프레임 로드하지 않는다** — `main()`/`run_replay()` 시작부에서
  1회만 `load_camera_calibration()`하고 각 프레임 처리 함수(`_run_image`/`_run_video`/프레임
  루프)에 결과 객체만 넘긴다(§7.1 config 레이어드, 파일 I/O 낭비 방지 — 세션 지시).
- **calib 로드 실패는 전체 파이프라인을 막지 않는다** — `FileNotFoundError`를 잡아 경고 로그만
  남기고 `calib=None`으로 계속 진행한다(ArUco와 무관한 `vertiport_coarse.yaml`/
  `distress_coarse.yaml` 등 실행까지 nominal.yaml 부재로 죽으면 안 되므로).
- **`solve_target_pose()`는 확정 ArUco 검출(코너 있음)이 있을 때만 호출한다** — `state.detections`
  에서 `corners is not None and meta.get("aruco_id") is not None`인 첫 항목을 찾는 헬퍼
  (`_find_aruco_detection`, `main.py`/`replay.py` 각자 얇게 중복)로 게이팅한다. 없으면 매 프레임
  무조건 시도하지 않고 조용히 스킵(코너 없인 solvePnP 자체가 실패하므로).
- **`blackbox.log_frame()` 시그니처는 바꾸지 않았다** — 세션 지시대로 기존 `chosen` 파라미터
  ("확정된 결과"라는 의미가 이미 있음) 안에 `{"target_estimate": {...}}` 형태로 얹었다. 기존
  버티포트/조난자 경로가 쓰는 `{"bbox":..., "confidence":...}` 형태의 `chosen`과 키가 겹치지
  않아(둘 다 존재하면 병합) 회귀 없이 공존한다. `target_estimate` dict 필드:
  `position`/`orientation`/`confidence`/`target_type`/`calib_accuracy`/
  `not_for_closed_loop_30cm`/`calib_id`(전부 `TargetEstimate`를 그대로 dict화 — `uncertainty`는
  이번 Phase 항상 None이라 dict에서 제외).
- **preset 판단: 신규 `vertiport_fine.yaml`, `vertiport_coarse.yaml`과 완전히 독립 실행.** 근거는
  위 "presets/vertiport_fine.yaml" 표 행 및 그 yaml 파일 헤더 주석 참조 — 핵심은 `ArucoDetector`
  가 ROI를 받지 않고(Phase 2 완료 코드, 이번 Phase 수정 범위 밖) `state.detections`를 통째로
  덮어쓰므로, coarse 캐스케이드 뒤에 이어붙이면 "ROI로 좁힌다"가 아니라 "coarse 결과를 파괴한다"가
  된다.
- **실측 검증(2026-07-24):** `python -m vision.replay <합성 ArUco 프레임 폴더> --preset
  vision/presets/vertiport_fine.yaml`을 실제로 실행 — JSONL에 다음이 실제로 찍힘(발췌):
  `"chosen": {"target_estimate": {"position": [...], "orientation": [...], "confidence": 1.0,
  "target_type": "aruco_23", "calib_accuracy": "unverified", "not_for_closed_loop_30cm": true,
  "calib_id": ".../calibration/cam109-imx708af75/nominal.yaml"}}`.

---

## 공통 상태머신 파이프라인 배선 (2026-07-25, `docs/vision_plan.md` §9 빌드순서 6번)

`core/state_machine.py`(`LandingStateMachine`)를 `main.py`/`replay.py`에 배선해 JSONL 블랙박스
`state`/`command` 필드가 실제로 채워지도록 완성한 단계 — `utils/blackbox.py::log_frame()`은
애초에 `state` 파라미터가 있었지만(docstring "§5.1 상태머신 프레임별 기록") 아무도 채우지
않고 있었다(ArUco Phase 4가 기존 `chosen` 파라미터를 재사용한 것과 같은 원칙 — **시그니처는
바꾸지 않는다**).

- **배선 위치 판단: ArUco Phase 4와 동일하게 `main.py`/`replay.py` 레벨, `modules/` 아님.**
  상태머신은 애초에 `__call__(VisionState)->VisionState` 파이프라인 모듈이 아니라 프레임
  단위 관측을 시간축으로 누적하는 상위 레이어라 `registry.py`/preset yaml 자체가 부적합한
  개념(세션 지시 확정 판단).
- **`Observation` 구성은 각 파일이 얇게 중복**(`_build_observation()`, import 규칙 — main.py/
  replay.py는 헬퍼를 상호 import하지 않는다): `n_candidates=len(state.detections)`,
  `center_error_norm`=확정/첫 검출 중심의 화면중심 대비 정규화 반경오차, `fine_locked`=
  `_find_aruco_detection(state.detections) is not None`.
- **`fine_locked`은 지금 유일하게 구현된 fine 검증(ArUco ID 확정)만 반영한다.** 버티포트/
  조난자 coarse 전용 프리셋(§9 5번은 coarse까지만 완료, fine 검증 모듈 없음)은 항상
  `fine_locked=False`로 degrade — 상태머신은 이 사실을 모른 채 그대로 안전하게
  ACQUIRE/CENTER_DESCEND에 머문다(타겟 종류 무관 공통 골격 + 커밋 게이트 불변식 둘 다와 자연스럽게
  일치, 별도 분기 불필요).
- **AGL(라이다)은 `main.py`엔 아예 없고(FC 연동 전), `replay.py`만 `record.telemetry.get("alt")`
  로 얻는다** — 텔레메트리에 `alt`가 없으면(사이드카 telemetry.jsonl 부재 등) `None`으로 우아하게
  degrade한다(실기체 telemetry 아직 없음 — AGL 없이도 상태머신이 돌아가는 게 필수 요구사항, 실제로
  `agl_m=None`이면 `TERMINAL` 진입 조건(`near_ground`)이 구조적으로 항상 거짓이 돼 폐루프 서보에
  안전하게 계속 머무는 축퇴 동작이 된다).
- **상태머신 인스턴스는 실행 전체에 하나만** — calib과 동일 원칙(§7.1 config 레이어드)으로
  `main()`/`run_replay()` 시작부에서 1회 생성해 프레임 루프 내내 재사용한다. 프레임마다 새로
  만들면 상태 누적(연속 프레임 카운터 등) 자체가 사라진다.
- **`command`도 같은 김에 채웠다(세션 지시가 명시한 건 `state`뿐이지만, `log_frame`에 이미
  있던 미사용 파라미터라 시그니처 변경 없이 관측성을 넓히는 것으로 판단)** — `Decision.reason`은
  JSONL 스키마에 없어(로그 스키마 변경 금지) 실리지 않는다, in-process 값으로만 남는다.
- **실측 검증(2026-07-25):** 합성 ArUco 프레임 14장(고도가 점점 낮아지다 마지막 3장은 마커가
  화각을 이탈한 것으로 가정한 빈 프레임) + `telemetry.jsonl`(alt 포함)을 실제로
  `python -m vision.replay`로 재생 → JSONL `state`가 실제로
  `CENTER_DESCEND→LOCK→PRECISION_SERVO→TERMINAL`로 진행하고, 블라인드 3프레임 동안은
  `max_blind_duration_s`(기본 2.0s) 미만이라 `ABORT_ASCEND`로 안 빠지고 `TERMINAL`에서
  데드레코닝 하강을 유지함을 확인(`vision/results/state_machine_demo/demo.jsonl`).
  같은 JSONL을 `tools/jsonl_view.py`로 그린 state 서브플롯도 "no state data" 안내 대신 실제
  4단 계단형 라인으로 렌더링됨(`vision/results/state_machine_demo/demo_state.png`) — 코드
  변경 없이 자동으로 그려졌다(`jsonl_view.py` 자체는 이번 세션에서 손대지 않음).

---

## 필드명 정정 — `center_error_px`/`center_tolerance_px` → `_norm` (2026-07-25, 오케스트레이터 발견)

`core/state_machine.py`의 `Observation.center_error_px`와 `LandingSMConfig.center_tolerance_px`는
**이름에 `_px`가 붙어있었지만 실제로는 픽셀이 아니었다** — `main.py`/`replay.py`의
`_build_observation()`이 `dx=(cx-w/2)/(w/2)`, `dy=(cy-h/2)/(h/2)`로 **정규화**한 뒤 그 노름
(0~약1.41)을 넣는다. `center_tolerance_norm` 기본값이 `0.05`인 것도 정규화 값임을 방증한다
(픽셀 단위였다면 이렇게 작은 값이 말이 안 됨). 이름이 거짓말하면 나중에 누가 "픽셀이구나" 하고
픽셀값을 그대로 넣는 순간 즉시 오작동한다 — `center_error_norm`/`center_tolerance_norm`으로
정정했다(`core/state_machine.py` + `main.py` + `replay.py` + `tests/test_state_machine.py`/
`test_main.py` 전부 기계적으로 갱신, `pytest` 전체 회귀 확인 완료). **동작 변경 없음** — 이름만
바뀌었다.

**`drift_estimate` 근사가 tan 항을 생략해 약 1.3배 보수적이다(임계값 재튜닝은 이번 범위 밖):**
`LandingStateMachine`의 `drift_estimate = |center_error_norm| × 마지막 유효 AGL`은
`max_drift_estimate_m`(미터)와 비교되는데, **"정규화 오차 × 미터"는 차원상 미터가 아니다.**
실제 지상 이탈거리 근사는 `정규화오차 × agl × tan(HFOV/2)`이고, 실측 HFOV 75°면
`tan(37.5°)≈0.767`이라 이 tan 항을 생략한 현재 근사값은 실제보다 **약 1/0.767 ≈ 1.3배 크게
(=보수적으로)** 나온다 — 더 쉽게 `ABORT_ASCEND`로 빠지는 안전한 방향이라 급하지 않다.
**계산식 자체는 바꾸지 않았다**(세션 지시 — "정밀도·물리 튜닝에 시간을 쓰지 말 것", 임계값
재튜닝은 실기체 데이터 확보 후에 할 일). 근거는 `core/state_machine.py`의 `drift_estimate`
계산부 주석 참조.

---

## 조난자 fine 파이프라인 배선(체인 잇기) (2026-07-25, `docs/vision_plan.md` §5.3/§9)

바로 위 절이 배선한 `fine_locked`은 그 시점엔 ArUco ID 확정 검출 하나만 반영했다 — ②
조난자 구역(초록구역, 대회 2차예선 우선 타겟으로 확정됨) fine 검출기가 아직 없어서
**`distress_coarse.yaml`만으로는 `fine_locked`가 영원히 False → 상태머신이 `CENTER_DESCEND`에서
절대 못 벗어나는 끊어진 체인**이었다(커밋 게이트가 `Observation.fine_locked=True` 없이는
`LOCK` 이상을 구조적으로 막으므로). 이번 세션은 `modules/distress_box.py::WhiteBoxDetector`
(§5.3 fine 단계 — 매트 안 흰 박스 확인 + **"박스 옆 빈 초록면" 착륙점 산출**)를 만들고
`_build_observation()`을 확장해 이 체인을 이었다.

- **모듈 배치 판단: `modules/`, 캐스케이드 패턴.** `vertiport_v.py`/`vertiport_ring.py`와 동일하게
  이전 단계(coarse, 매트) `detections`를 ROI로 읽어 그 안에서만 찾고 자기 결과로 덮어쓴다.
  `vertiport_fine.yaml`(ArUco)이 **독립 실행**을 택한 것과 반대 판단인데, 이유는 흰 박스가
  ArUco 마커와 달리 ROI 인자 없이 전체 프레임을 덮어쓰는 검출기가 아니라 "매트 안에서만
  찾는" 캐스케이드 검출기이기 때문(근거는 `presets/distress_fine.yaml` 헤더 주석에도 병기).
- **착륙점(landing point) 산출 — "가장 먼 모서리를 안쪽으로 당기기":** 매트 bbox의 네 모서리 중
  흰 박스 중심에서 가장 먼 것을 고르고, 매트 중심 쪽으로 `interior_margin_ratio`(기본 0.3)만큼
  당긴다. 이 방법을 고른 이유는 `modules/distress_box.py::WhiteBoxDetector` docstring에 상세히
  적혀 있다 — 요약하면 (a) 박스가 매트 정중앙에 있는 현재 실측 배치(§2)에서도 고정된 모서리
  순회 순서 덕에 결정론이 깨지지 않고, (b) 박스가 중심을 벗어난 가상 배치에도 별도 분기 없이
  자연스럽게 "반대편"을 가리키며, (c) 매트가 페인트 선이 아니라 실제 0.105m 구조물이라
  가장자리 이탈이 진짜 낙하 위험이라는 점을 안전마진으로 반영한다. **"박스 옆"의 정확한
  방향·거리는 대회측 미회신 상태**(`vision_plan.md` §9 "중요" 각주)이므로 잠정 합리적 기본값이다.
- **`distress_fine.yaml` rect_detector 임계값 재도출** — `distress_coarse.yaml`의 8000/200000을
  그대로 복사하지 않고 fine 대역(≤~15m)에 맞게 같은 공식(gw(h)=1.535h, 1536px 다운스케일
  가정)으로 재계산(14000/2,200,000). 근거는 yaml 헤더 주석.
- **`white_box_detector` 임계값의 altitude-불변 성질** — 박스/매트 면적비(≈0.00444, 실측
  20cm/3.0m)는 매트 픽셀 bbox 대비 **상대값**이라 고도(GSD)에 무관하게 불변이다(매트·박스
  둘 다 같은 GSD로 함께 스케일되므로) — rect_detector의 절대 px² 임계값과 달리 altitude tier별
  재계산이 필요 없다는 게 이 설계의 실질적 이점.
- **`_build_observation()` 확장(main.py/replay.py, 각자 얇게 중복)**:
  - `fine_locked` = ArUco 확정 검출(`_find_aruco_detection`) **또는** 흰 박스 확정 검출
    (`_find_white_box_detection`, `meta["white_box_detector"]["landing_point_px"]` 존재로 판별)
    중 하나라도 있으면 True.
  - `center_error_norm`은 **착륙점 기준**(흰 박스 lock이 있으면 `landing_point_px`, 없으면 기존처럼
    `confirmed`/첫 detection의 bbox 중심으로 폴백)으로 계산 — §5.3 설계 포인트(착륙 목표는
    박스가 아니라 그 옆 빈 초록면)를 상태머신에 들어가는 관측값 레벨에서부터 반영한다.
  - `scale_source`(§5.1 blob 스케일 융합 규칙)는 흰 박스 blob 확정 시에만 채운다(AGL 유효 시
    `"agl"`, 없으면 `"known_size"`) — ArUco는 solvePnP 자체 스케일이라 이 규칙 대상이 아니다.
    이번 세션은 `scale_source`를 실제로 채우는 **첫** 배선이다(그 전까지는 `Observation`/
    `Decision`에 필드만 있고 아무도 채우지 않고 있었음).
  - `core/state_machine.py`는 손대지 않았다(§9 6번 요구 — 타겟 종류 무관 공통 골격, 타겟별
    특수성은 전부 호출자의 `Observation` 구성 쪽에 흡수).
- **실측 검증(2026-07-25):** 초록 매트+흰 박스 합성 프레임 11장(고도가 점점 낮아짐, `alt`
  텔레메트리 포함) + 마지막 3장은 매트가 화각을 이탈한 것으로 가정한 빈 프레임을 실제로
  `python -m vision.replay <녹화폴더> --preset vision/presets/distress_fine.yaml`로 재생 →
  JSONL `state`가 실제로 `CENTER_DESCEND(f0)→LOCK(f1)→PRECISION_SERVO(f2~f4)→TERMINAL(f5~f10)`
  로 진행함을 확인(끊어진 체인이 실제로 이어짐, 이번 세션의 성공 기준). 마지막 3장(블라인드)은
  `ABORT_ASCEND`로 빠졌는데, 이는 `max_blind_duration_s` 초과가 아니라 **근사 이탈추정
  (`drift_estimate = |center_error_norm| × 마지막 유효 AGL`) 초과** 때문이다 — 착륙점을 매트
  중심에서 의도적으로 밀어낸 설계 탓에 `center_error_norm`이 (마트 자체는 화면 중앙에 있어도)
  0에 가깝지 않아 드리프트 추정치가 쉽게 임계값을 넘는다(§ 아래 "미해결/관찰된 상호작용" 참고).
  `vision/results/distress_fine_demo/demo.jsonl` + `tools/jsonl_view.py`로 그린
  `demo_state.png`(5단 계단형 state 라인 — CENTER_DESCEND/LOCK/PRECISION_SERVO/TERMINAL/
  ABORT_ASCEND 전부 실제로 렌더링됨)로 남겨뒀다. **이 데모 입력 프레임 자체는 커밋 저장소
  밖(`/tmp`)에서 임시 생성했다** — `state_machine_demo/`의 ArUco 데모 전례와 동일하게 결과물
  (jsonl/log/png)만 남기고 재현 가능한 합성 소스 프레임은 커밋하지 않는다.
- **미해결/관찰된 상호작용 (버그 아님, 설계 특성으로 문서화):** `state_machine.py`의
  드리프트 안전망(§5.1)은 `center_error_norm`을 "타겟을 얼마나 잘 추적 중인가"의 근사치로
  가정하는데, 조난자 fine의 `center_error_norm`은 "선택한 착륙점이 화면 중심에서 얼마나
  떨어져 있는가"다 — 이 둘은 ArUco(마커 중심 ≈ 실제 타겟 위치)에서는 거의 같지만, 조난자
  fine에서는 착륙점을 의도적으로 매트 중심에서 밀어내므로 구조적으로 다르다. 결과적으로
  조난자 fine 경로는 ArUco보다 블라인드 진입 시 `ABORT_ASCEND`(드리프트 초과 경로)로 더 쉽게
  빠지는 경향이 있다 — 안전 쪽으로 치우친 결과라 이번 세션에서 "고쳐야 할 버그"로 보진
  않았으나(state_machine.py 수정은 이번 세션 범위 밖이기도 함), 향후 실제 착륙점 오프셋
  크기와 안전 임계값(`max_drift_estimate_m`)의 관계를 재검토할 만하다.

---

## `--target-sink` 파이프라인 배선 (2026-07-28, `docs/vision_fc_interface.md` §9 작업 V5)

Phase 1(`core/wire.py`/`core/frames.py`/`utils/target_sink.py`)이 만들어 놓고 **아무도 호출하지
않던** sink를 `main.py`에 실제로 배선한 단계. 그 전까지 `utils/target_sink.py`는 테스트에서만
살아있는 죽은 코드였다.

- **CLI 관례: `--display stream`을 그대로 베꼈다** — 켜는 플래그 1개(`--target-sink`,
  `store_true`) + `--target-sink-host`/`--target-sink-port`. 🔀 **문서와 갈리는 지점**:
  `docs/vision_fc_interface.md` §9 V5는 `--target-sink socket://127.0.0.1:PORT`라는 **URL 형식**을
  적어 뒀는데, 이 저장소에는 URL을 파싱하는 CLI 인자가 하나도 없고 `--stream-host`/`--stream-port`
  라는 확립된 전례가 있어 그쪽을 따랐다(새 파싱 관례를 발명하지 않는다). 기능 차이는 없다.
- **배선 지점: `_run_image`/`_run_video`/`_run_live` 세 경로 전부.** 근거 —
  ① 라이브가 정밀착륙의 본령이고, ② **§7.5(기록·재생)가 "같은 파이프라인으로 오프라인 재생"을
  회귀검증의 최대 레버로 못박고 있다** — 영상 경로에 안 붙이면 sink 계약을 책상에서 회귀로
  잡을 수 없고 조용히 썩는다(실제로 이번 종단간 검증 자체가 영상 경로로 이뤄졌다), ③ opt-in
  이라 꺼져 있으면 세 경로 모두 비용 0이다, ④ `streamer`/`calib`/`state_machine`이 이미 세
  경로에 똑같이 관통되고 있어 `main.py`의 균질성이 유지된다.
- **기본값이 꺼짐이라는 것의 의미 — 레코드 조립 비용조차 0.** `_publish_to_sink()`가
  `isinstance(sink, SocketTargetSink)`가 아니면 **즉시 반환**한다(`if streamer is not None`
  전례와 같은 opt-in 게이트). `NullSink`는 "호출자가 None 분기를 흩뿌리지 않게 하는" 타입
  기본값 역할만 한다. 실측: 같은 입력으로 sink ON/OFF 두 번 돌려 JSONL 80건이 `ts`/`latency`
  제외 **완전 동일**함을 확인.
- **`_solve_aruco_chosen` → `_solve_aruco_estimate`로 바꿨다.** 예전엔 `TargetEstimate`를 얻자마자
  dict로 눌러 담아 객체가 사라졌는데, `build_target_record()`는 dataclass를 요구한다. 이제
  `(estimate, reason)`을 돌려주고 dict 변환은 호출부(`_merge_target_estimate_into_chosen`)가 한다
  — **JSONL `chosen` 형태·내용은 무변경**(위 실측으로 확인). `reason`을 같이 돌려주는 이유는
  §5.4 때문이다: 무효 레코드에 사유(`no_calibration`/`no_target_detection`/`pose_solve_failed`)를
  실으려면 여기서 이미 알고 있는 구분을 버리면 안 된다.
- **§5.4 침묵 금지가 배선의 핵심.** 검출이 없어도 발행을 멈추지 않고 `valid=false`+사유를 보낸다
  — 침묵은 "노드 사망"으로 예약돼 있어(소비자가 EOF/타임아웃으로 잰다) 검출 상실 때 침묵하면
  두 경우가 구분되지 않는다. 종단간 실측에서 마커 이탈 20프레임이 전부
  `valid=false, reason="no_target_detection", position_frd=null`로 나갔다.
- **🔴 발행(publish) 실패는 절대 검출을 죽이지 않는다.** 발행부는 `except Exception`으로 넓게
  삼키고 경고만 남긴다(`BaseException`은 일부러 안 잡는다 — Ctrl+C/SIGTERM은 정상 종료 경로다).
- **🔴 기동(bind) 실패는 반대로 하드 페일이다 — 종료코드 3으로 즉사한다(2026-07-28 사용자
  결정으로 계약이 뒤집혔다).** 아래 "bind 하드 페일 + 유도 발행 상태 오버레이" 절 참조.
- **🔴 `sink.install_signal_handlers()`를 부르지 않는다.** `signal.signal`은 신호당 핸들러가
  **하나**뿐이라, `_run_live`의 `_install_sigterm_handler` 뒤에 sink 핸들러를 걸면 그걸
  **덮어써서** `stop_event`가 영영 세팅되지 않는다 — 실기체에서만 드러났던 SIGTERM graceful
  shutdown 버그가 그대로 재발한다. sink 정리는 `main()`의 `finally`에서 `sink.close()`로 한다
  (루프를 빠져나오면 반드시 거기로 간다). 이 재발 경로를 회귀테스트로 박아 뒀다(파괴검증 D-A5).
- **종단간 실측(2026-07-28, 로컬):** 합성 ArUco 영상 80프레임(60프레임 마커 + 20프레임 이탈)을
  `python -m vision.main ... --target-sink --target-sink-port 18091`로 돌리고, **vision 코드를
  한 줄도 import하지 않는 순수 stdlib 소비자**(`socket.create_connection` → `makefile("r")` →
  `readline()`, 시스템 python3 — 이 인터프리터엔 cv2조차 없다)를 별도 프로세스로 붙여 수신:
  `target=80 (valid=60 / invalid=20), state_hint=80`, `seq` 1~160 단조증가, main() 종료 시 EOF
  수신. FLU/FRD 부호도 규약대로(`position_flu` z=−10.076 / `position_frd` z=+10.076, y 부호 반전).
  소비자 0명으로도, 소비자가 끊겨도 파이프라인은 80프레임을 그대로 처리했다.

---

## bind 하드 페일 + 유도 발행 상태 오버레이 + replay 배선 (2026-07-28, 사용자 결정)

바로 위 배선 절이 남겨 뒀던 **"사용자 확인이 필요한 갈림길"이 닫힌 작업**이다.

> 질문: *"소켓 bind 실패 시 `NullSink`로 강등해 검출을 계속할 것인가, 죽일 것인가."*
> 🔵 **사용자 답(2026-07-28):** *"'유도 좌표가 아무 데도 안 나가는 상황'에 화면은 뜰 수 있는가?
> 만약 그렇다면, 지금은 디버깅이 활발한 상태이니까, 걍 안되면 죽여버릴 수 있도록 하여라.
> 화면에도 보이도록."*

**답은 "뜰 수 있다"이다** — `--display`(창/MJPEG 스트림)와 `--target-sink`는 코드상 **완전히
독립**이라 화면·검출·로그가 전부 멀쩡한 채 유도 좌표만 허공으로 가는 상태가 실제로 가능하다.
그래서 두 갈래로 대응했다: **막을 수 있는 것(bind 실패)은 죽이고, 못 막는 것(소비자 0명)은
화면에 상시 노출한다.**

### 1. bind 실패 = 하드 페일 (계약이 **뒤집혔다**)

- `main.py`/`replay.py` 둘 다 `sys.exit(EXIT_SINK_BIND_FAILED)` = **종료코드 3**. 2(argparse
  사용법)/1(입력 파일 없음)과 겹치지 않게 새로 뒀다 — 스크립트가 "포트 충돌로 못 떴다"를
  다른 실패와 구분할 수 있어야 한다.
- stderr에 **엔드포인트 + 포트 번호 + `ss -ltnp | grep <port>` + 대안 플래그**를 낸다(십중팔구
  포트 충돌이므로 원격 세션이 재조사 없이 바로 진단 가능하게).
- 이제 `utils/target_sink.py::start()`의 문서화된 판단("포트 충돌을 조용히 삼키면 안 된다")과
  **일치**하고, `--display stream`(MjpegStreamer)이 예외를 그대로 올리는 관례와도 맞다.
- ⚠️ **발행(publish) 실패와 정책이 정반대이며, 그래야 한다.** bind는 **기동 시점 1회**라 죽어도
  기체는 지상이고 실패의 뜻은 "명시적으로 켠 발행이 처음부터 안 켜졌다"는 설정 오류다. 발행은
  **비행 중 매 프레임**이라 거기서 죽으면 소비자 사정으로 검출이 통째로 멈춘다. 이 구분은
  `main.py::_publish_to_sink` docstring에 코드 주석으로도 박아 뒀다.
- ⚠️ **`--target-sink` 미지정 기본 실행은 조금도 달라지지 않는다**(소켓 미바인드, 하드 페일
  경로 자체를 안 지남 — 회귀테스트로 고정).
- `_make_target_sink()` 호출은 **`try/finally` 안**에 있다 — `SystemExit`로 죽어도 `finally`가
  blackbox 큐 스레드/파일 핸들을 정리해야 하기 때문(리소스 leak 회귀테스트로 고정).
- 폐기: 옛 계약을 고정하던 `test_sink_bind_failure_degrades_to_nullsink_and_keeps_detecting`.

### 2. 유도 발행 상태 화면 오버레이 (`utils/visualize.py::draw_sink_status`)

**bind는 됐는데 소비자가 0명인 경우가 진짜 사각지대다** — 죽일 수 없다(시작 직후엔 소비자가
아직 안 붙는 게 정상). 그래서 `--display`가 켜져 있으면(`window`/`stream`/`file`) 매 프레임
좌상단에 **소비자 수 / 마지막 발행 seq / 드롭 수 / 엔드포인트**를 그린다.

- **3색 상태:** 소비자 있음=초록 / **소비자 0명=빨강 + 1.4배 큰 헤드라인**("눈에 확 띄게") /
  sink 자체 꺼짐=주황(같은 "유도가 안 나간다"지만 운영자의 명시적 선택이라 구분).
- **`--display none`이면 비용이 정확히 0** — 호출 자체가 게이팅된다(드론 기본 경로, §7.9).
  회귀테스트가 `draw_sink_status` 호출 횟수 0을 단언한다.
- **검출 결과를 안 가린다:** `draw_detections()` **뒤에** 좌상단 패널에만 그리고, 패널은
  `addWeighted`로 **반투명**이라 밑 영상이 비친다. 회귀테스트가 패널 밖 픽셀 비트 동일을 단언.
- **사본을 또 만들지 않는다** — `draw_detections()`가 이미 만든 배열을 제자리에서 고치고 같은
  객체를 돌려준다(매 프레임 전체 복사가 한 번 더 늘어나면 안 된다).
- ⚠️ **오버레이 문자열은 ASCII만.** `cv2.putText`의 Hershey 폰트는 한글을 못 그려 조용히 깨진다
  — 소스 레벨 회귀테스트로 고정.
- 접근자: `SocketTargetSink.client_count`(기존) + **`last_seq`(신설)**. 둘 다 락으로 보호 —
  클라이언트 목록은 accept/send 스레드가 만지고 오버레이는 파이프라인 스레드가 매 프레임 읽는다.
- `--display file`은 창이 없으므로 **저장 파일이 곧 화면**이다 → `_run_image`가 `save_result()`
  대신 오버레이가 얹힌 `annotated`를 쓴다(`--display none`은 기존 `save_result()` 경로 유지).

### 3. `replay.py` sink 배선 — Phase 1 후속 세션이 남긴 갭

🔴 **재생 경로에만 있는 가치:** `replay.py`는 Dir/Bag + `telemetry.jsonl`(AGL) 재생이라
**`agl_m`이 실린 `state_hint`를 회귀로 잡을 수 있는 유일한 경로**다 — `main.py`는 AGL을 받는
경로가 아예 없어(`_build_observation(agl_m=None)`) 상태머신이 구조적으로 `TERMINAL`에 못 간다.
즉 **착륙 최종단계의 유도 힌트가 소비자에게 실제로 흘러가는가**는 여기서만 검증된다.

- `main.py`와 **문자 그대로 같은 CLI 인자 이름**(`--target-sink`/`--target-sink-host`/
  `--target-sink-port`), 같은 기본 꺼짐, 같은 하드페일 종료코드. 정책이 갈리면 회귀검증
  경로로서의 가치가 사라진다.
- `_solve_target_chosen()` → `_solve_target_estimate()`로 교체(main.py가 이미 한 것과 동일한
  이유 — `build_target_record()`는 dict가 아니라 dataclass를 요구). **JSONL `chosen` 형태·내용
  무변경**(기존 회귀테스트 3건이 그대로 통과).
- 실패 사유 6종(`no_calibration`/`no_target_detection`/`mat_geometry_unavailable`/
  `landing_point_unprojectable`/`pose_solve_failed`)도 main.py와 같은 문자열로 내보낸다.

### 종단간 실측 (2026-07-28, 로컬)

- **하드 페일:** 포트 18091을 실제로 점유한 상태에서 `python -m vision.main ... --target-sink
  --target-sink-port 18091` → `[Errno 98] Address already in use` + 진단 메시지, **종료코드 3**.
  `python -m vision.replay ...` 같은 조건에서도 **종료코드 3**(동일 메시지).
- **replay 종단간:** ArUco 80프레임 + `telemetry.jsonl`(frame_id≥3에서 `alt=2.0`)을
  `--target-sink --target-sink-port 18097 --display stream --output ...`로 재생하고 **순수
  stdlib 소비자**(시스템 python3, vision 미import)를 별도 프로세스로 붙임 → `target=80,
  state_hint=80`, seq 1~160 **드롭 0**, 상태열 `CENTER_DESCEND → LOCK → PRECISION_SERVO →
  TERMINAL(×77)`, 첫 TERMINAL 힌트 `reason="near_ground_enter_terminal"`, 재생 종료 시 EOF 수신.
- **오버레이(헤드리스라 창을 못 띄워 화면 버퍼로 증명):** `--display file` 저장 PNG에서
  소비자 0명 = 빨강 `CONSUMERS 0 - GUIDANCE GOES NOWHERE`, sink 꺼짐 = 주황 `SINK OFF - NO
  GUIDANCE OUT`. 소비자가 실제로 붙은 재생 mp4의 60번 프레임에서는 초록 `CONSUMERS 1` +
  `sink 127.0.0.1:18097 seq 122 dropped 0`이 찍혔고 **ArUco 검출 박스와 신뢰도 라벨은 그대로**.

---

## color_calibrate.py 마진 기본값 확정 (2026-07-28, `tools/color_calibrate.py`)

`DEFAULT_HUE_MARGIN`/`DEFAULT_SAT_MARGIN`/`DEFAULT_VAL_MARGIN`이 전부 0이라 산출 임계값에
조명 변동 쿠션이 전혀 없다는 것이 **알려진 갭으로 기록만 돼 있던 상태**(`docs/vision_status.md`
"⚠️ 마진 기본값이 0이다")를 정식으로 닫은 작업. **결론: HUE=6, SAT=0, VAL=0.**
(근거 전문은 `tools/color_calibrate.py` 상단 docstring "마진 정책" 절 — 여기는 요약이다.
"라이브 스트림 어댑터 기본값"·"distress_coarse.yaml min_area/max_area 도출 근거"와 같은
근거기록 패턴.)

- **먼저 개념 분리 — 백분위수와 마진은 서로 다른 것을 덮는다(이 작업의 핵심).**
  - **백분위수 밴드(`DEFAULT_LOW_PERCENTILE=5` ~ `DEFAULT_HIGH_PERCENTILE=95`)가 이미
    흡수하는 것:** 캘리브레이션을 수행한 *그 한 프레임 ROI 안의 공간적 변동*(그림자·글레어·
    재질 얼룩·과노출 클리핑 소수 픽셀). 전부 그 프레임에 실제로 찍혀 있는 정보다.
  - **마진이 덮어야 하는 것:** 캘리브레이션 조건 ↔ *실제 비행 시점 조건의 차이*(시각·태양각·
    구름·화이트밸런스 재수렴·노출 변화)로 인한 **분포 중심 자체의 이동**. 백분위수는 이걸
    원리적으로 볼 수 없다(그 프레임에 없는 정보다).
  - 이 구분 없이 "ROI 표준편차 N배"를 기본값으로 박으면 **같은 변동을 두 번 세게 된다.**
- **`DEFAULT_HUE_MARGIN = 6`(OpenCV Hue 단위, 양쪽 각각 ±6).** 유일한 근거 데이터는 사용자가
  초록구역을 휴대폰으로 찍은 사진에서 뽑은 랜덤 10점 중 이상치 2점 제외 8점의 Hue
  (`136 162 169 146 169 158 146 163`, 0~360° 스케일).
  - **스케일 판정 근거:** ÷2 → OpenCV **68~84.5**(청록 띤 초록), 초록 매트로 타당하고
    `distress_coarse.yaml`의 독립 손튜닝값 `hue_range=[35,85]` **안에 실제로 들어온다**.
    0~255 가정은 192~238°(파랑), 이미-OpenCV 가정은 272~338°(마젠타) — 둘 다 초록과 모순, 기각.
  - **절대 Hue(mean 78)는 쓰지 않는다** — 카메라 기종·시각·f값·픽쳐 프로파일·WB offset 전부
    미상이라 전이되지 않는다. 애초에 이 도구는 중심을 현장 ROI에서 다시 재므로 필요도 없다.
    **산포만 근거로 쓴다**(사용자가 이 데이터를 준 이유). OpenCV 환산 **표본σ 6.0559 /
    모집단σ 5.6648 → 둘 다 6으로 반올림**되어 추정량 선택과 무관하게 결론이 같다.
  - **⚠️ 이중계상 위험에 대한 답:** 이 8점은 한 장의 사진 안에서 뽑았으므로 엄밀히는 위
    정의상 "백분위수가 이미 덮는" 프레임 내 공간 변동이다. 그래도 마진 근거로 쓰는 이유는
    **균질한 무광 매트 안의 Hue 산포를 만드는 원인이 국소 조명 기하**(음영각·직사광 대
    천공광 혼합비)이고, 그건 태양각·구름 변화가 **면 전체를 이동시키는 것과 같은 물리
    메커니즘**이기 때문 — 즉 프레임 내 8점은 서로 다른 국소 조명조건의 표본이고, 전역 조건이
    바뀌면 면 전체가 그 분포의 한쪽 끝으로 옮겨 앉는다. **단 이건 부등식이 아니다** —
    "프레임 내 σ ≤ 조건 간 σ"를 보장하는 정리는 없으므로 **하한이 아니라 자릿수 추정
    (order-of-magnitude proxy)으로만** 쓴다. 그래서 **2σ·3σ가 아니라 1σ다**: ROI가 넓고
    얼룩덜룩하면 p5~p95가 이미 이 변동의 상당 부분을 흡수한 상태라 2σ를 더 얹는 건 진짜로
    이중계상이 된다.
  - **왜 0을 유지하지 않는가:** 0은 중립이 아니라 이 도구의 권장 사용법에서 실제로 고장난
    값이다 — 좁고 균일한 ROI를 주면 p5==p95라 `hue_range: [60, 60]` 같은 **폭 0짜리 밴드**가
    나오고(이미 관측된 정상 동작) 조건이 조금만 바뀌면 타겟을 통째로 놓친다.
  - **반대편 비용(오탐):** 밴드가 12 단위 넓어진다(Hue 원주 180의 6.7%). 위험 방향은 **아래쪽
    (식생)** 하나뿐 — 무성한 잔디는 Hue 60~70대라 청록 띤 매트(≈78)와 가깝다. 자갈은 무채색이라
    `sat_min`이, 그림자는 `val_min`이 거른다. 감당 가능하다고 본 근거: (a) ±6을 얹어도 산출
    밴드 폭(≈12~25)이 **이미 운용상 받아들여지고 있는** 손튜닝값 `[35,85]`(폭 50)보다 여전히
    훨씬 좁아 현상 대비 오탐이 늘지 않고, (b) 최종 배제는 Hue 단독이 아니라 `RectDetector`의
    면적·형상 필터가 함께 담당한다.
- **`DEFAULT_SAT_MARGIN = 0` / `DEFAULT_VAL_MARGIN = 0` — 근거가 없어 0을 유지한다(억지로 안 채움).**
  표본이 **Hue 8점뿐**이라 S/V 데이터가 하나도 없고, V(조도 선형)·S(색소 반사율 대 백색광 비율)는
  Hue와 물리 경로가 달라 6을 옮겨 쓸 근거가 없다 — 지어내면 §7.3이 금지하는 출처 없는
  매직넘버다. 게다가 **이 두 마진이 넓히는 방향이 배경 클래스를 정확히 겨눈다**(`sat_min`↓ =
  저채도 자갈, `val_min`↓ = 그림자, `val_max`↑ = 백분위수가 일부러 잘라낸 글레어). 비대칭성도
  0을 지지한다 — S/V 마진 부재로 인한 미검출은 즉시 눈에 띄고 재실행으로 몇 초 만에 복구되지만,
  자갈/그림자 오탐은 알아채기 어렵고 상태머신 커밋 게이트를 잘못 통과시킬 수 있다
  (`core/state_machine.py`가 모호한 후보를 `HOLD`로 거절하는 것과 같은 철학). **현장에서는
  `--diagnostic-dir` 히스토그램을 보고 `--sat-margin`/`--val-margin`을 명시할 것**
  (`docs/vision_verification_qa_brief.md`의 `--sat-margin 20 --val-margin 20` 예시는 여전히
  유효한 *운영자 선택값*이며, 다만 조용한 기본값이 되어선 안 된다는 것이 이 결정이다).
- **랩어라운드 분기의 마진 방향은 반대다(부호 실수 시 실제 위험한 버그).** `low_hue_max`는
  **키우고** `high_hue_min`은 **줄여야** `[0,low_hue_max]`∪`[high_hue_min,179]` 두 통과대역이
  각각 넓어진다. 부호가 반대면 대역이 조용히 **좁아져** 빨강을 놓친다 — 이 방향 자체가
  회귀테스트 대상이다(아래).
- **파괴검증 완료:** `calibrate_roi()`의 hue 마진 부호를 뒤집으면(4곳 전부 → 7 failed,
  랩어라운드 2곳만 → 정확히 새 랩어라운드 테스트 2건만 failed) 실제로 red가 되고 원복 시
  green으로 돌아옴을 종료코드로 확인. **랩어라운드 부호만 뒤집는 버그를 잡는 테스트는 이번에
  추가된 2건이 유일**했다(그 전에는 조용히 통과했다).
- 이번 작업은 **검출 알고리즘 변경이 아니다** — 기본값 상수 1개 + 근거 문서/주석 + 회귀테스트
  9건. `modules/*.py`, preset yaml 전부 무변경.

---

## 초록구역(② 조난자) 상대 pose 산출 (2026-07-28, `docs/vision_plan.md` §5.3 / `docs/vision_fc_interface.md`)

정밀착륙 인터페이스(`--target-sink`)가 배선된 직후(`b45fdc4`) 종단간 실행에서 드러난 갭을 메운
작업이다: **초록구역은 상태머신이 `CENTER_DESCEND`/`descend`를 내는데 정작 기체에 보낼 상대
pose는 항상 `null`이었다.** pose 산출 경로(`main.py::_solve_aruco_estimate`)가 ArUco 전용이라,
피듀셜이 없는 초록 매트에는 `solvePnP`도 `TargetEstimate` 산출도 한 줄도 없었기 때문이다.
초록구역이 현재 우선 타겟이므로(2차예선) 이 빈칸이 "전체 프로세스 유기적 동작"의 마지막 구멍이었다.

### 고른 접근: 매트 4코너 + 알려진 실측 크기(3.0m) -> `solvePnP` (버린 안 포함)

- ✅ **채택.** `core/target.py::marker_object_points(size_m)`가 애초에 크기를 인자로 받으므로
  (ArUco는 0.50) ArUco와 **똑같은 기계장치**를 그대로 재사용한다 — 새 pose 파이프라인을
  발명하지 않았고 산출물도 완전히 같은 `TargetEstimate` 형식이다. 결정적 이점: **AGL(라이다
  고도)이 필요 없다.** 알려진 크기가 스케일을 준다.
- ❌ **"픽셀 -> AGL로 역투영"**(고도를 알면 GSD가 나오니 픽셀 오프셋을 미터로 환산): `main.py`는
  지금 AGL을 **받는 경로 자체가 없다**(`_build_observation(agl_m=None)`; `replay.py`만
  telemetry에서 읽는다). 라이브에서 항상 None이라 산출이 아예 불가능해 기각.
- ❌ **호모그래피로 평면만 풀고 자세 포기**: `solvePnP`가 같은 입력에서 회전까지 주는데 정보를
  일부러 버릴 이유가 없고, 산출물 형식이 quaternion 자리를 이미 요구한다.

### 배선 구조 — 왜 모듈 하나 + main/replay 두 레벨로 쪼갰나

`solve_target_pose()`를 부르려면 `utils/calibration_loader.py`가 필요한데 import 규칙이
**`modules/ <- vision.core 만`** 이라 위반이 된다(ArUco Phase 4가 같은 이유로 배선을 `main.py`
레벨에 둔 전례). 그래서 역할을 쪼갰다:

    modules/distress_mat.py  : 프레임 안에서 끝나는 순수 기하(코너 정규화 + 착륙점 픽셀 확정)
    main.py / replay.py      : calib 로드 + solvePnP 호출 + TargetEstimate 조립 (각자 얇게 중복)

**pose 산출기 선택은 preset 경로 문자열 파싱이 아니라 파이프라인이 남긴 meta로 한다.** ArUco는
`meta["aruco_id"]`, 초록구역은 `meta["distress_mat"]`(= `distress_mat_geometry` 스텝이 있을
때만 생김)이다. 경로 문자열 관례는 깨지기 쉽고, 같은 `rect_detector`를 쓰는 범용 프리셋
(`video.yaml`)에 3m 매트 크기가 실수로 적용되는 것도 막아야 한다(파괴검증 D7로 회귀 고정).
🔴 **ArUco 경로는 조금도 달라지지 않는다** — 초록 매트는 ArUco 검출이 없을 때만 시도되고,
`position_at_pixel=None`이면 `position`은 예전 그대로 `solvePnP`의 tvec **비트 단위로 동일**하다
(파괴검증 D5/D5b/R3으로 회귀 고정).

### 코너 순서 — 조용한 오작동 1순위 (실측 근거)

`cv2.aruco.detectMarkers()`와 달리 `cv2.approxPolyDP()`는 **코너 순서를 보장하지 않는다.**
골든셋 `distress/10m`에서 `RectDetector`가 실제로 낸 순서는 `[(80,80),(80,380),(380,380),(380,80)]`
= **TL->BL->BR->TR(반시계)** 였고, `marker_object_points()`는 시계방향이라 그대로 넣으면
**감김이 반대인 거울상 대응**이 된다. `order_quad_corners_clockwise()`가 무게중심 기준
`atan2` 정렬(이미지 좌표는 y가 아래로 증가하므로 각도 증가 = 화면상 시계방향) 후 `x+y` 최소
코너를 맨 앞으로 회전시켜 감김과 시작점을 결정론적으로 고정한다.

⚠️ **평면·정면·완전 대칭인 합성 프레임에서는 감김 오류가 거리(z)에 안 나타난다** — 정사각형이
그 대각선에 대해 자기대칭이라 position이 보존되기 때문. 실측(파괴검증 D1)에서 `main.py` 레벨
거리 테스트는 통과하고 `core` 단위테스트(감김/정규화)만 red가 됐다. 원근이 있는 실촬영에서는
position까지 깨지므로, **이 정규화가 "합성에서 안 보이지만 실기체에서 터지는" 종류의 문제**임을
기록해 둔다.

### 회전 4중 모호성 — 구조적으로 해소됨 (흰 박스 단서 불필요)

정사각 매트는 90° 회전 4겹 자기대칭이라 코너 라벨링이 4가지, `solvePnP` 해도 4개다. 그런데
**착륙점 좌표는 4개 해 전부에서 동일하다**: 착륙점은 `project_pixel_onto_target_plane()`로
광선-평면 교점을 구하는데, 자기 법선축 회전은 평면을 자기 자신으로 보내므로 4개 해가 **전부
같은 평면**을 준다. 즉 모호성이 유도 좌표로 새지 않는다(회귀테스트로 고정). 남는 모호성은
`orientation` 쿼터니언뿐이고 정사각형에 대해 물리적으로 무의미한 자유도다.
**흰 박스로 회전을 깨는 방법은 애초에 불가능하다** — 실측 스펙상 박스가 매트 **정중앙**이라
비대칭 단서가 되지 못한다(`vision_plan.md` §2).

### 착륙점 — 매트 중심이 아니다, 그리고 규칙을 재구현하지 않는다

착륙 목표는 "박스 옆 빈 초록면"이고 그 픽셀은 `modules/distress_box.py`가 이미
`meta["white_box_detector"]["landing_point_px"]`로 확정한다. `DistressMatGeometry`는 **그 값을
그대로 소비**한다 — 규칙("가장 먼 모서리를 안쪽으로 당기기")이나 `interior_margin_ratio`
기본값을 두 곳에 복사하면 한쪽만 고쳐졌을 때 조용히 어긋난다. 부수 효과로 이 픽셀은
`_build_observation()`이 `center_error_norm`을 재는 픽셀과 **같아서**, 상태머신이 "중앙에
맞췄다"고 판단한 점과 기체에 보내는 좌표가 어긋나지 않는다.

- fine(`distress_fine.yaml`) -> `landing_point_px` 사용, `target_type="distress_landing_point"`
- coarse(`distress_coarse.yaml`, 흰 박스 단계 없음) -> **매트 중심**으로 degrade,
  `target_type="distress_mat_center"` + `landing_point_source="mat_center"`.
  coarse 대역(40~15m)에선 박스가 해상되지 않으므로 물리적으로 옳고, 상태머신도
  `fine_locked=False`라 `CENTER_DESCEND` 위로 못 올라간다. **소비자가 "coarse 중심"과 "fine
  착륙점"을 구분할 수 있어야** 하므로 `target_type`을 다르게 붙였다.

### 🔴 z=0 평면은 지면이 아니라 **매트 윗면**이다 (0.105m 라이즈드)

검출되는 4코너는 위에서 본 초록 면의 윤곽 = 라이즈드 플랫폼의 **윗면 가장자리**다(초록색은
윗면에만 있고, 라이저 측면은 근-나디르에서 거의 안 보인다). 따라서 산출 거리는 **윗면까지의
거리**다.

**착륙 고도 해석 결론: 윗면 기준이 맞다.** 기체는 매트 윗면에 내려앉으므로 접지 목표면이 곧 이
평면이고 착륙 자체엔 보정이 필요 없다. ⚠️ **단 소비자가 이 z를 라이다 AGL(주변 지면 기준)과
섞으면 정확히 0.105m 어긋난다**(비전 z가 라이다 AGL보다 0.105m 작게 나온다). 그래서
`plane_reference="mat_top_surface"` + `platform_height_m`을 `TargetEstimate.meta`에 실어
와이어 레코드까지 전파한다.

### 실패 사유를 뭉개지 않는다 (§5.4)

`no_target_detection` 하나로 합치면 현장에서 원인을 못 가린다. 초록구역 경로는
`mat_geometry_unavailable`(매트는 봤는데 4코너가 못 씀) / `landing_point_unprojectable`(착륙점
역투영 실패) / `pose_solve_failed` / `no_calibration`을 각각 다른 문자열로 내보낸다.

### ⚠️ 캘리브레이션 해상도 불일치는 조용히 보정하지 않는다

`solvePnP`의 focal은 **픽셀 단위**라 프레임 크기가 캘리브레이션 해상도와 다르면 거리가 그
비율만큼 통째로 틀린다. 실측: 4608px 기준 `nominal.yaml`을 460px 골든 프레임에 그대로 쓰면
거리가 정확히 **3.00배**(=4608/1536) 나온다. **자동 재스케일은 하지 않았다** — 프레임이
다운스케일인지 크롭인지 코드가 알 수 없고, 잘못된 가정으로 조용히 보정하면 진짜 불일치가
숨는다. 대신 `frame_size_px`/`calib_image_size_px`를 `meta`에 실어 사후 진단이 가능하게 했다.
실기체 라이브 경로는 `LiveFrameSource` 기본 해상도가 `nominal.yaml`의 `image_size`와 같아
불일치가 없다.

### 골든셋 합성 카메라 픽스처 (`vision/tests/golden/distress/synthetic_calib/`)

골든 프레임은 카메라 모델로 렌더링된 것이 아니라 GSD 표에서 가져온 픽셀 크기로 그린 것이다.
그 전제(HFOV 75°, 폭 1536px 다운스케일 프레임)를 만족하는 초점거리는 하나뿐이므로
(`fx = 1536/(2·tan(37.5°)) = 1000.877086`) 그걸 역산해 캔버스별(460/320/200) 가상 카메라 yaml
3개를 만들어 뒀다. `accuracy: unverified` / `not_for_closed_loop_30cm: true`는 **일부러
nominal.yaml과 같은 보수적 값**을 유지한다 — 합성이라고 해금하면 테스트가 "폐루프 30cm 가능"
이라는 거짓 신호를 흘린다.

**거리 정확도 실측(2026-07-28):** 10m 라벨 -> 10.0088m(+0.09%) / 20m -> 20.0175m(+0.09%) /
40m -> 40.576m(+1.44%, `min_area`만 낮춘 임시 프리셋. 75px 매트라 1px 오차가 1.3%다).
40m는 `distress_coarse.yaml`이 `min_area=8000`으로 **의도적으로 배제**하므로 정규 경로에서는
`no_target_detection`이 정상이다(프리셋 헤더 주석 참조).

---

## 착륙점 등거리 축퇴 완화 (2026-07-28, `modules/distress_box.py`)

**문제(오케스트레이터 실측 발견):** `WhiteBoxDetector`는 매트 bbox 네 모서리 중 흰 박스 중심에서
가장 먼 것을 골라 착륙점을 만드는데, **실측 스펙상 흰 박스가 매트 정중앙**이라 네 모서리가
이론상 등거리다. 정확한 등거리에서는 `max()`가 첫 최댓값을 돌려줘 결정론은 안 깨지지만,
**실전 입력은 절대 정확히 등거리가 아니다** — 같은 장면인데 정지 이미지에서는 TL, mp4에서는 BR이
선택됐다. 커밋 `e1f8471` 이후 이 착륙점이 `modules/distress_mat.py`를 거쳐 **실제 유도 좌표로
나가므로**, 프레임마다 매트 반대편을 지시하면 기체가 진동한다.

**재현(합성 1px 흔들림 시퀀스 10프레임, 300px=3.0m 매트):**

| 설정 | 서로 다른 착륙점 | 최대 프레임간 점프 |
|---|---|---|
| 완화 전(=`tie_tolerance_ratio=0.0, corner_hysteresis=False`) | **4개**(네 모서리 전부) | **296.98px = 2.970m** |
| 완화 후(기본값) | 1개 | **0.00px = 0.000m** |
| 대조군: 확실히 편심된 박스(좌상단) | 1개(우하단 = 박스 반대편) | 0.000m — **기존 동작 보존** |

**완화 방법 — 선택의 안정성만 손댄다.** `interior_margin_ratio=0.3`(대회측 미회신 잠정값)과
"박스 옆 빈 초록면" 규약은 **무변경**이다.

1. **동률 허용오차 `tie_tolerance_ratio`(기본 0.02)** — 최대거리에서 `ratio × 매트 bbox 대각선`
   이내인 모서리를 전부 동률 후보로 보고 정규 순서(`CORNER_NAMES = tl,tr,br,bl`)의 첫 번째를
   고른다. 300px 매트에서 tol≈8.5px로 실측 잡음 1px을 크게 웃돈다. **`0.0`을 주면 옛 동작과
   정확히 동일**(회귀테스트로 못 박음)이라 되돌리기가 파라미터 한 줄이다.
2. **프레임 간 히스테리시스 `corner_hysteresis`(기본 True)** — 직전 선택이 아직 동률 후보 안에
   있으면 유지. 1+2의 조합은 **폭 `2×tol`의 슈미트 트리거**라, 허용오차 경계에 딱 걸친 배치에서도
   한 번 전환한 뒤 눌러앉지 왕복 진동하지 않는다. 직전 프레임 대응은 매트 bbox IoU로 찾는다
   (`modules/fusion.py::TemporalFusion`과 같은 패턴 — 새 추적기 발명 금지).

**결정론(§7.5) 유지:** 히스테리시스는 wall-clock/난수가 아니라 *입력 시퀀스*만의 함수다. 상태는
인스턴스 단위(`reset()` 제공)라 새 인스턴스는 항상 같은 초기 선택에서 출발한다 —
`modules/tracker.py`/`fusion.py`와 동일한 stateful 모듈 관례.

**진단 3종이 `meta["white_box_detector"]`로 나간다**(§7.4 블랙박스 포렌식): `landing_corner`
(고른 모서리 이름) / `corner_tie_count`(>1이면 그 프레임이 실제 축퇴였다) / `corner_from_hysteresis`
(정규 순서 대신 직전 선택이 유지됨).

---

## `drift_estimate`의 `tan(HFOV/2)` 항 (2026-07-28, `core/state_machine.py`)

**기존 갭:** `TERMINAL` 블라인드 하강의 이탈 추정이 `|정규화오차| × AGL`이었다. `tan(HFOV/2)`
항이 빠져 **차원상 미터가 아니었고**(정규화값 × 미터), 그래서 "미터" 임계값
`max_drift_estimate_m`과 비교하는 것 자체가 어긋나 있었다. "안전 방향이라 그대로 뒀다"고
기록만 돼 있던 것을 이번에 닫았다.

**바로잡은 식:** `|정규화오차| × AGL × tan(HFOV/2)`. 핀홀에서 화면 절반폭이 `fx·tan(HFOV/2)`
이므로 정규화오차 e의 광선각은 `tan(θ)=e·tan(HFOV/2)`, 나디르 카메라 지상거리는 `AGL·tan(θ)` —
**근사가 아니라 핀홀에서는 정확**하다. 단 `center_error_norm`은 x(폭 정규화)와 y(높이 정규화)를
섞은 노름이라 y 성분엔 원래 `tan(VFOV/2)`(<`tan(HFOV/2)`)가 붙어야 한다. 둘 다 `tan(HFOV/2)`로
환산하는 이 식은 그래서 **참값의 상한**이고, 안전 게이트가 원하는 방향(과소평가 없음)이라
의도적으로 고른 근사다(정확히 하려면 상태머신이 dx/dy를 따로 받아야 하는데 그건 "타겟 무관
최소 관측" 원칙을 깬다).

**계수는 화각(도)이 아니라 인트린식에서 유도한다** — `tan(HFOV/2) = (width/2)/fx`
(`half_hfov_tan_from_calibration()`). 이러면 `nominal.yaml`의 `hfov_assumption`이 **수평인지
대각인지 미해결**이라는 문제를 아예 우회한다(`fx`는 가정이 아니라 solvePnP가 실제로 쓰는 값).
`main.py`/`replay.py`가 로드한 캘리브레이션에서 계산해 `LandingSMConfig(half_hfov_tan=...)`로
주입하고, 로드 실패 시에만 `NOMINAL_HALF_HFOV_TAN` 폴백을 쓴다.

- 실측 검증: `nominal.yaml`에서 `2304/3002.6312590261377 = 0.7673269880` = `tan(37.5°)`
  (HFOV 75° 가정과 **자기무모순**). 기록된 "약 1.3배 보수적"도 `1/0.7673 = 1.3032`로 확인됨.

**🔴 소비 파라미터도 같이 내렸다 — `max_drift_estimate_m` 1.0 → 0.75.** 정확해지는 변경이
안전 게이트를 조용히 푸는 것을 막기 위해서다:

| | 발동 조건 (`e × AGL`이 얼마를 넘으면 ABORT_ASCEND) |
|---|---|
| 옛 식 + 옛 임계 1.0 | `> 1.0000` |
| 새 식 + **옛 임계 1.0 그대로 뒀다면** | `> 1.3032` — **1.303배 헐거워짐** ❌ |
| 새 식 + 새 임계 0.75 (채택) | `> 0.9774` — 옛 동작점보다 2.3% 더 빡빡 ✅ |

이제 임계값이 **진짜 미터**를 뜻한다: `TERMINAL` 진입 상한 AGL 3.0m에서 발동점은 지상 이탈
0.750m(중심오차 32.6%)다. 🔀 **0.75는 여전히 실기체 미검증 잠정값**이고 30cm 요구보다 2.5배
크지만, 이건 정밀도 게이트가 아니라 "마지막으로 본 오차가 데드레코닝을 믿을 수 없을 만큼
컸는가"를 보는 **안전 폴백 게이트**라 요구정밀도까지 조이면 상시 ABORT가 난다. 실기체 데이터
확보 후 재튜닝 대상.

---

## LiveFrameSource AF 제어 (2026-07-28, `utils/frame_source.py`)

**기존 갭:** AF 제어는 `tools/h264_stream.py`에만 들어가 있었고 **라이브 파이프라인 경로
(`LiveFrameSource`)의 초점은 드라이버 기본 동작에 방치**돼 있었다. 초점이 안 맞으면 검출
자체가 무의미해지므로 닫았다.

- **기본값 `af_mode="continuous"`** — 기체가 10~40m를 오르내리므로 연속 AF가 맞다.
  `auto`(단발 스캔, `AfTrigger=Start` 2차 호출 필수) / `manual`(+`lens_position` 디옵터)도 지원.
- **`af_mode=None` = AF 미개입**(이 변경 전 동작 그대로). 현장에서 AF가 말썽일 때 되돌릴
  escape hatch다. `main.py --af-mode none`으로 도달 가능.
- **CLI:** `main.py`에 `--af-mode {continuous,auto,manual,none}` / `--lens-position` 추가.
  이름·의미를 `tools/h264_stream.py`와 **일부러 똑같이** 맞췄다(현장에서 두 도구를 번갈아 쓴다).
- **인자 검증은 생성자에서** — 하드웨어를 만지기 전에 실패한다. `lens_position` 상한은
  **실측 하드클램프 15.0**이다(드라이버가 광고하는 32.0은 무효 — 그 값을 쓰면 안 된다).
- **AF 실패는 캡처를 죽이지 않는다.** libcamera 부재/드라이버 거부 시 카메라는 드라이버 기본
  초점으로 계속 돌고, 사유는 `af_error`에 남아 `_run_live`가 사람로그에 `WARNING`으로 찍는다
  (§7.4 침묵 금지). AF 하나 때문에 라이브 파이프라인 전체를 잃는 게 더 나쁘다는 판단.
- **AF 제어의 단일 출처가 여기다.** `tools/h264_stream.py`가 이 모듈에서 import해 쓴다
  (import 규칙상 `utils/ → tools/`는 불가라 방향을 뒤집었다). 복제하면 `LENS_POSITION_MAX`
  같은 **실측 물리값**이 한쪽만 갱신돼 조용히 갈라진다 — 회귀테스트로 박아 뒀다.

### 🔴 실기체 미검증 — 다음 세션이 확인할 것

이 작업은 RPi 접속이 금지된 세션에서 했다. 단위테스트(가짜 `picamera2`/`libcamera` 주입)까지만
했고 **실카메라에서 실행된 적이 없다.** 덧붙여 `h264_stream.py`의 AF 경로조차 2026-07-25에
"크래시 없이 동작"까지만 확인됐고 **초점이 실제로 이동했는지(선명도 지표)는 그때도 미검증**이다.

검증 절차(카메라는 배타적이라 한 번에 하나만 띄울 것):

1. `main.py live --af-mode continuous --display stream`으로 띄우고 사람로그에
   `AF 설정 완료 af_mode=continuous`가 찍히는지 확인(`AF 설정 실패`면 `af_error` 사유를 볼 것).
2. **초점이 실제로 움직였는지**는 `tools/rpi_capture.py`/`calib_capture.py`의 초점 스윕
   방법론을 재사용한다 — `--af-mode manual --lens-position`을 여러 값(예: 0.5 / 3.0 / 10.0)으로
   바꿔가며 프레임을 저장하고, 각 프레임의 **라플라시안 분산(선명도 지표)** 이 렌즈위치에 따라
   실제로 달라지는지 본다. 값이 안 변하면 컨트롤이 먹지 않은 것이다.
3. `--af-mode none`으로도 한 번 띄워 예전 동작(드라이버 기본)과 비교한다.
4. RPi `picam-venv`에는 `pytest`가 없다 — 실기체 검증은 "실행해서 로그/산출물 확인" 방식이어야 한다.

---

## `utils/geo_project.py` 폐기 (2026-07-28)

`docs/vision_plan.md` §12가 지정한 폐기를 실행했다. **삭제**를 골랐고, deprecation 경고 단계를
거치지 않았다 — 근거:

1. **참조가 0건이다.** 저장소 전체를 AST로 훑어 `geo_project` import / `pixel_to_gps` 호출이
   한 건도 없음을 확인했다(`vision/utils/__init__.py`도 빈 파일이라 재노출 경로 없음,
   `registry.py` 미등록, preset yaml 미참조, 테스트 0건). deprecation 경고는 **호출자가 있을 때**
   시간을 벌어주는 장치인데, 호출자가 없으면 죽은 무게만 남는다.
2. **대체가 실제로 존재한다** — 이게 §12의 전제 조건이었다. `core/frames.py`(cam→body 프레임
   체인) + `core/target.py::solve_target_pose()` + `modules/distress_mat.py`(초록구역 상대 pose)
   + `utils/target_sink.py`(소켓 인터페이스)로 상대 pose 폐루프 경로가 끝까지 이어져 있다.
3. **남겨두는 것 자체가 위험하다.** 이 함수는 30cm 요구에 미달하는 바로 그 접근(GPS 절대좌표,
   정확도 한계 ~1~2m)이고, docstring이 "FC 연동 시 자세를 인자로 받도록 확장 필요"라며 **확장을
   권유**하고 있었다. 되살아날 유인이 문서에 실제로 존재한다(아래).
4. 되돌리기는 git 히스토리로 한 줄이다.

### 🔴 남은 stale 참조 (이번 세션 소관 밖 — 다음 세션이 닫을 것)

**FC 도메인 문서가 아직 이 함수를 확장하라고 적어두고 있다.** vision 세션이 손댈 파일이
아니라 그대로 뒀다:

| 파일 | 내용 |
|---|---|
| `docs/pixhawk6c_rpi4_integration_guide.md` | `pixel_to_gps.py (vision)` 블록도, `pixel_to_gps_with_attitude()` **확장 시그니처(필수)**, "P2/P3 작업" 목록 — 전부 피벗 이전 설계다 |
| `docs/flight_plan.md` (241행) | "vision→FC 연동(`pixel_to_gps`로 임의 GPS WP 주입)" |

코드 쪽 재발은 `tests/test_deprecations.py`가 막는다(파일 부재 + import 불가 + 저장소 전체에
`pixel_to_gps` 문자열 재등장 없음). 문서 쪽은 FC 세션이 정리해야 한다.

---

## 착륙점 기하 — 기체 크기 기반 재설계 (2026-07-28, `modules/distress_box.py`)

**드러난 결함(사용자와의 설계 논의).** 2026-07-28에 **기체 최외곽(다리/프롭 끝) 반경이 0.5m를
초과한다**는 사실이 처음 확정됐다. 이걸 기존 착륙점 규칙에 대입하니 구조적 결함이 나왔다:

```
매트 3.0m × 3.0m (반변 1.5m), 0.105m 라이즈드 플랫폼 / 흰 박스 정중앙 0.20m
옛 착륙점 = 매트 bbox 모서리를 중심 쪽으로 interior_margin_ratio(0.3)만큼 당긴 점
          = 중심에서 (±1.05, ±1.05) m
착륙점 → 매트 가장자리 여유 = 1.5 − 1.05 = 0.45 m  <  R (>0.5 m)
⇒ 유도 오차가 정확히 0이어도 기체가 매트 밖으로 삐져나간다.
```

매트는 페인트 선이 아니라 **0.105m 라이즈드 구조물**이라 가장자리에 반쯤 걸치면 기울어져
넘어진다 — 표면적 실패가 아니라 진짜 사고다. `interior_margin_ratio=0.3`은 "대회측 미회신이라
잠정값"으로 들어간 숫자였고 **기체 크기를 한 번도 고려한 적이 없었다.**

### 대체: 거리 `d`를 비율이 아니라 물리량에서 도출한다

부등식 유도 전문·전제·"정중앙 박스가 하한의 최악 경우"인 이유는
`modules/distress_box.py::compute_landing_window()` docstring에 있다(여기서 중복 서술하지 않는다).
요지만:

| | 재는 방향 | 부등식 |
|---|---|---|
| 하한 (박스를 밟지 않는다) | **대각선** — 착륙점은 `(d,d)`, 박스 모서리는 `b·√2` | `√2·d − b·√2 ≥ R` ⟺ `d ≥ b + R/√2` |
| 상한 (매트를 벗어나지 않는다) | **축 방향** — 최단거리는 `M − d`다 | `M − d ≥ R + δ` ⟺ `d ≤ M − R − δ` |

🔴 **방향을 바꿔 재면 조용히 위험해진다.** 상한을 대각선으로 재면 여유를 √2배 과대평가해
"안전하다"고 착각한다(파괴검증 D3이 실제로 그 실수를 재현한다).

착륙점 픽셀은 `착륙점 = 매트중심 + (d/M)·(고른 모서리 − 매트중심)`으로 낸다 — 모서리 선택
규칙(가장 먼 모서리)·`tie_tolerance_ratio`·`corner_hysteresis`는 **무변경**이고, 바뀐 것은
"고른 모서리에서 얼마나 안쪽으로 당기는가"뿐이다. bbox가 원근으로 정사각이 아니어도 축별
스케일이 각각 반영돼 별도 분기가 필요 없다.

### 🔴 이 매트에는 절벽이 있다 (설계 발견)

`R`에 대해 두 부등식을 풀면 안전창이 존재하는 상한이 나온다:

```
R_max = (M − δ − b) / (1 + 1/√2) = (1.5 − 0.30 − 0.10) / 1.7071 = 0.6444 m
```

**실측 R이 0.6444m를 넘으면 3m 매트 위에 "박스 옆" 착륙점이 아예 존재하지 않는다.** 아는 사실이
"0.5m 초과"뿐이므로 이건 가설이 아니라 **실측으로 닫아야 하는 최우선 항목**이다. 코드는 이 값을
`aircraft_radius_max_feasible_m`으로 매 검출 meta에 실어 보낸다.

수치 실증(δ=0.30, b=0.10, M=1.5 기본값):

| R (m) | d_min | d_max | feasible | 채택 d (중점) | 박스 여유 | 매트 여유 | 옛 규칙(d=1.05) 가장자리 여유 |
|---|---|---|---|---|---|---|---|
| 0.30 | 0.3121 | 0.9000 | ✅ | 0.6061 | +0.4157 | +0.5939 | +0.150 |
| 0.50 | 0.4536 | 0.7000 | ✅ | 0.5768 | +0.1743 | +0.4232 | **−0.050** ❌ |
| **0.60**(기본) | 0.5243 | 0.6000 | ✅ | **0.5621** | +0.0536 | +0.3379 | **−0.150** ❌ |
| 0.70 | 0.5950 | 0.5000 | ❌ **해 없음** | — | — | — | **−0.250** ❌ |
| 1.00 | 0.8071 | 0.2000 | ❌ **해 없음** | — | — | — | **−0.550** ❌ |

### 🔴 안전창이 비면 지어내지 않고 **거절**한다

- 그 검출을 `state.detections`에서 빼고 `reject_reasons`에 **`landing_point_infeasible`**
  (`no_white_pixels`와 다른 문자열 — §5.4 사유 뭉개기 금지)을 넣는다.
- 계산된 하한/상한/R 전량을 `state.meta["white_box_detector"]["landing_point_infeasible"]`에
  남긴다. **상위(`main.py`/`replay.py`)가 이걸 읽어 `valid=false`+사유로 발행하는 배선은 아직
  없다** — 다음 세션 몫이고, 지금은 검출이 0건이 되어 기존 `no_target_detection` 경로로 나간다.
- **거절이 필수인 이유:** 착륙점만 빼고 검출을 남기면 `modules/distress_mat.py`가
  `landing_point_px` 부재를 보고 **매트 중심으로 우아하게 degrade**한다 — 매트 중심은 바로 그
  흰 박스 위다. "해가 없다"가 "박스 위에 내려라"로 조용히 뒤집힌다(파괴검증 D5).

### 🔴 `R` 기본값은 미측정 공칭값이다 (`core/frames.py` ψ_m 패턴)

`AIRCRAFT_RADIUS_M_DEFAULT = 0.60` + **`AIRCRAFT_RADIUS_MEASURED = False`**. 저장소 어디에도
기체 치수가 없고(`CA_ROTOR*_PX/PY`는 ±1.0 정규화 값이라 팔 길이가 아니다) 사용자가 준 정보는
"0.5m 초과"뿐이다. 값을 지어내지 않고 파라미터로 빼두되 "미측정" 플래그가 진단 meta
(`landing_geometry.aircraft_radius_measured`)까지 전파된다. 방향의 비대칭이 보수적 선택을
정한다 — **작게 잡으면 매트를 넘어 사고, 크게 잡으면 착륙점이 안 나와 임무 실패(안전)**.

`δ`(`LANDING_DRIFT_ALLOWANCE_M_DEFAULT = 0.30`)는 지어낸 값이 아니라 `docs/vision_plan.md` §1의
요구사항 **"최종 정확도 <30cm"** 이다. ⚠️ 달성이 확인된 오차가 아니라 **요구 오차**다.

### `interior_margin_ratio` 폐기 — 하위호환 no-op

**물리 도출값이 항상 이긴다.** 둘 다 조용히 적용되면 어느 쪽이 이겼는지 모른 채 같은 결함이
재발하므로 "둘 다 적용"은 선택지가 아니었다. 그렇다고 인자를 없애면 이 키를 주는 기존 preset
yaml이 `Pipeline.from_config`에서 `TypeError`로 죽는데, **비행 직전 프리셋 로드가 통째로 실패하는
것이 더 나쁘다**고 판단해 인자는 계속 받되 값을 무시한다. 무시는 침묵이 아니다 — 생성자에서
`DeprecationWarning`을 내고 `state.meta["white_box_detector"]["interior_margin_ratio_ignored"]`로
blackbox까지 알린다(§7.4). `presets/distress_fine.yaml`에서는 제거했다.

### 관측성

- 확정 검출: `det.meta["white_box_detector"]["landing_geometry"]` — `d_m`/`d_min_m`/`d_max_m`/
  `aircraft_radius_m`/`aircraft_radius_measured`/`drift_allowance_m`/`mat_size_m`/
  `white_box_size_m`/`safety_window_bias`/`box_clearance_m`/`mat_edge_margin_m`/
  `aircraft_radius_max_feasible_m`/`box_center_offset_m`/`px_per_m`.
- ⚠️ **정상 경로의 `state.meta["white_box_detector"]` 형태는 일부러 안 바꿨다** — 골든셋
  `labels.json`이 이 dict를 통째로 동등비교하므로, 여기에 물리 파라미터를 넣으면 R 기본값이
  골든 픽스처에까지 복제돼 "단일 출처" 원칙이 깨진다.
- `box_center_offset_m`은 하한 유도가 전제한 "박스는 매트 정중앙"(§2)을 **사후 검증**하라고
  있는 값이다.

### 종단간 실측 (2026-07-28, 로컬)

골든 `distress/fine` 프레임 3장 재생 + 합성 카메라(`synthetic_calib/canvas460.yaml`):

```
fine   distress_landing_point  position = [-0.5590, -0.5590, 10.0088]
coarse distress_mat_center     position = [ 0.0000,  0.0000, 10.0088]
수평 오프셋 0.7906 m  → 축당 d = 0.5590 m   (이론 0.5621 / 옛 규칙 1.0500)
하한: √2·d = 0.7906 ≥ b√2+R = 0.7414 ✅   상한: d = 0.5590 ≤ M−R−δ = 0.6000 ✅
```

이론값과의 0.3% 차이는 `RectDetector`의 매트 bbox 픽셀 양자화(301px vs 300px) 탓이고
안전창 슬랙(박스 +0.049m / 가장자리 +0.341m)이 흡수한다. 산출물은
`vision/results/landing_geometry_demo/`.

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
tools/      ← 이 규칙 밖(`utils/`를 import해도 된다 — 역방향은 금지). RPi 하드웨어 전용 운영스크립트(예: rpi_capture.py의 picamera2/GStreamer) — .venv에 안 깔림, CI/pytest 대상 아님.
              단, 하드웨어 비의존 CLI 도구(예: jsonl_view.py, calib_analyze.py, color_calibrate.py)는 예외 — .venv 설치 + pytest 대상.
ros/        ← **컨테이너(fc, ros:humble) 전용.** shim_core.py = stdlib만(vision.* 도 numpy 도 import 금지 — 컨테이너에 cv2가 없어
              vision.core.wire 체인이 통째로 죽는다). shim_node.py = rclpy + vision.ros.shim_core 만 — **저장소에서 rclpy를 import하는
              유일한 파일**이고 __init__.py가 이것을 import하지 않는다(랩탑엔 rclpy가 없다). 아래 "컨테이너 ROS2 shim 노드" 절 참조.
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
| core/target `solve_target_pose`/`TargetEstimate`(ArUco Phase 3) | **★합성 왕복(핵심, calib_analyze.py 패턴 재사용):** 알려진 실제 pose로 50cm 정사각 4코너를 합성투영(우선 임의 K, 이어서 실제 nominal.yaml intrinsics로도) → solvePnP 복원 pose가 원래 position/quaternion(부호 이중성 고려)과 허용오차 내 일치 · `rotation_matrix_to_quaternion` 다양한 회전에서 단위quaternion+독립 공식으로 재구성한 R과 일치 · provenance echo(calib_accuracy/not_for_closed_loop_30cm/calib_id)가 호출자 값 그대로 반영 · `uncertainty` 항상 None · 코너 순서 오배열 회귀(순환 오배열=position 보존·orientation 붕괴, 비순환 오배열=position 자체 붕괴 — 정사각형 90도 자기대칭 때문, 둘 다 실측 확인) | ✅ test_target |
| core/target 초록구역 확장(2026-07-28) | 3.0m 상수가 ArUco 0.50과 구분됨 · `marker_object_points(3.0)` 재사용이 z=0 시계방향 4코너를 줌 · **알려진 크기만으로 알려진 거리 복원**(10/20/40m, AGL 없음) · 크기 상수를 0.5로 오용하면 정확히 6배 틀림(수치 고정) · `position_at_pixel=None`이 **solvePnP 자신의 tvec과 비트 동일**(ArUco 회귀) · 타겟 중심 픽셀 역투영이 tvec 재현(자기검증 불변식) · 착륙점 픽셀이 실제로 오프셋 좌표를 줌 · **90° 회전 4중 라벨링 전부에서 착륙점 동일**(모호성이 유도로 안 샘) · 감김 뒤집으면 자세가 실제로 깨짐 · 평면과 평행한 시선 거절. **파괴검증 D1/D1b/D2/D5b로 red 확인** | ✅ test_target |
| core/state_machine `LandingStateMachine`(§9 6번) | 정상 시퀀스 전이 ACQUIRE→CENTER_DESCEND→LOCK→PRECISION_SERVO→TERMINAL(순서 단조증가까지 확인) · **안전 폴백 핵심**: 검출 상실이 `loss_tolerance_frames` 초과 시 HOLD(허용치 이내는 유지) + HOLD가 재포착 시 CENTER_DESCEND로 복귀(막다른 상태 아님) · **TERMINAL 블라인드 지속시간 초과 → ABORT_ASCEND**(§5.1 "핵심") + 재포착 시 블라인드 타이머 리셋 회귀 + 근사 이탈추정(중심오차×AGL) 초과도 별도로 ABORT_ASCEND 트리거 · **커밋 게이트**: 후보 모호(fine_locked 시도 중 n_candidates>max_candidates_for_lock) → HOLD로 거절, LOCK 확정 카운트 도중 모호해져도 거절 · **불변식**: fine_locked=False가 지속되면 agl_m이 아무리 낮아도 LOCK/PRECISION_SERVO/TERMINAL 절대 도달 불가(구조적 게이팅, 우회 경로 없음) · agl_m 전부 None이어도 크래시 없이 동작 + TERMINAL 미도달(폐루프 서보 유지, 안전한 축퇴) · 결정론(같은 관측열 → 같은 상태/명령/사유열) · config 필드(`LandingSMConfig`)로 임계값 override 가능 · `scale_source`가 Decision에 그대로 에코. **[2026-07-28] `drift_estimate`의 `tan(HFOV/2)` 항**(위 절) — 옛 식이면 발동하고 새 식이면 발동 안 하는 임계로 tan 항 존재 확인·`half_hfov_tan`이 하드코딩 아닌 config에서 읽힘(계수만 바꿔 결과 반전)·`(width/2)/fx` 유도 + fx<=0 거절·모듈 상수가 **실제 배포된 nominal.yaml**과 일치(둘이 조용히 갈라지는 것 방지)·**게이트가 옛 동작점보다 헐거워지지 않았는지**(`max_drift_estimate_m/half_hfov_tan <= 1.0`). **파괴검증 D6/D7/D8/D9로 red 확인** | ✅ test_state_machine |
| ArUco Phase 4 파이프라인 배선(위 "ArUco Phase 4 파이프라인 배선" 절) | main.py/replay.py 각각: 합성 ID=23 마커 이미지/녹화폴더 실행 → JSONL `chosen.target_estimate`에 position/orientation/calib_accuracy/not_for_closed_loop_30cm/calib_id 실제 기록 · 마커 없는 프레임은 크래시 없이 `chosen=None` · calib 파일 없음(`--calib`/`calib_path` 오지정)도 크래시 없이 target_estimate만 생략+경고 로그 | ✅ test_main(아루코 3개)·test_replay(아루코 3개) |
| utils/calibration_loader `load_camera_calibration`(ArUco Phase 3) | 실제 커밋된 `nominal.yaml` 로드(camera_matrix/dist_coeffs/image_size/accuracy/not_for_closed_loop_30cm/calib_id) · 합성 yaml round-trip(값이 실측/미검증 어느 쪽이든 하드코딩 없이 그대로 반영) · 파일 없음→`FileNotFoundError` | ✅ test_calibration_loader |
| core/frames `R_frd_cam`/`cam_to_flu` 등(인터페이스 Phase 1) | **★§4.3의 RT-1~RT-6를 그대로 구현**(RT-7 ENU 경로는 §4.4가 "vision은 NED 변환 안 함"으로 확정해 구현 자체가 없어 테스트도 없음). RT-1 모든 ψ에서 `RᵀR=I`·`det=+1` · RT-2 cam→frd→cam 및 cam→flu→frd→flu→cam 전체 체인 왕복(1e-9) · **RT-3~5 알려진 방향쌍**(정하방/이미지우측→기체우측/이미지위쪽→기체전방) — **부호 실수를 잡는 유일한 그물**(축 이름만 맞고 부호가 뒤집힌 회전행렬도 RT-1·RT-2는 통과한다, 파괴검증 D1이 Rz(+90)→Rz(-90)으로 실증) · 문서 리터럴 2개와 직접 대조(유도한 `R_flu_cam`이 문서값과 같은지) · ψ_m=90°가 전방 타겟을 실제로 우측으로 돌림(하드코딩 아님) · ψ 합성이 `Rz(ψ)·R(0)` · `MOUNT_YAW_PSI_M_MEASURED is False` 계약 회귀(실측 없이 뒤집으면 red) · FLU↔FRD involution · RT-6 쿼터니언 순서 왕복+항등구현 거부 · 비3차원 입력 `ValueError` · 결정론. **파괴검증 5종(D1~D5)으로 red 확인** | ✅ test_frames |
| core/wire 와이어 포맷·페일세이프 계약(인터페이스 Phase 1) | **★왕복(핵심):** `TargetEstimate`/`Decision` → 레코드 → JSON 한 줄 → 파싱 → dataclass 복원이 **무손실**(uncertainty ndarray·meta·전체 provenance 포함) · 필수키 전량 존재(`REQUIRED_*_KEYS` 대조) · `position_flu`/`_frd`가 진짜 변환 결과(cam 복사면 red) + 계약의 ψ_m이 실제로 먹힘 · **JSONL 프레이밍**: 값에 실제 개행/유니코드가 섞여도 정확히 한 줄이고 `readline()`으로 분리·복원됨 · 인코딩 결정론 · **§5.3 confidence 게이트**: 기본 0.0(비활성)이고 ArUco의 confidence=1.0은 어떤 하한에서도 통과(=현재 no-op임을 고정), 켜면 실제로 거르고 레코드가 valid=false · **§5.5 `not_for_closed_loop_30cm`**: True여도 추정치를 무효화하지 않고 **바닥 고도**(기본 3.0=`terminal_agl_m`과 정렬)를 주며, False면 0.0으로 자동 해금 · **§5.4 valid=false 레코드**: 검출 상실을 침묵 대신 사유와 함께 알리고 **포즈는 0이 아니라 null**(0이면 "기체 바로 아래에 타겟" 이라는 최악의 거짓말) · **§6.3 command**: `command_hint`+`command_is_advisory`로 실리고 맨 이름 `command` 키는 **없어야 함** · 시계 2종+오프셋·원본 timestamp 보존 · `schema_version` 불일치 거절 · 레코드 타입 교차 거절 · numpy 누출 없이 `json.dumps` 통과. **파괴검증 8종(D6~D13)으로 red 확인** | ✅ test_wire |
| utils/target_sink `SocketTargetSink`(인터페이스 Phase 1) | **실제 소켓만 사용(몽키패치 없음, `utils/stream.py` 관례):** 진짜 서버 기동 → 진짜 클라이언트 접속 → 진짜 JSON 바이트 수신·파싱 · 다중 클라이언트 브로드캐스트 · `seq` 단조증가(유실 감지 가능) · **비차단 계약(최우선)**: 소비자 없음/느린 소비자(한 바이트도 안 읽음) 둘 다에서 **단일 `publish()` 최대 지연**을 실측 — ⚠️ **총 시간이 아니라 최대 지연을 봐야 이빨이 생긴다**(직접 소켓 I/O로 망가뜨려도 버퍼가 찬 뒤 `send_timeout_s` 한 번만 멈추고 그 클라이언트를 끊어 총합에 스톨이 묻힌다, 파괴검증 D15에서 실측). 느린 소비자는 수신버퍼를 최소로 줄이고 ~12MB를 밀어 버퍼를 실제로 채운다 · 소비자 끊김 후에도 발행 측 무예외 + 죽은 클라이언트 정리 + **재접속** 가능 · 버스트에서 **최신 레코드 불유실** · **§5.4 EOF**: `stop()`/SIGTERM 종료 시 소비자가 즉시 EOF 수신 · **진짜 SIGTERM 송신**(`os.kill`)으로 graceful shutdown 검증(핸들러 복원 포함) + 워커 스레드에서 `install_signal_handlers()` 무예외 · 루프백 전용 바인드 회귀 · 포트 충돌은 침묵이 아니라 `OSError` · start/stop idempotent · `port=0` 임시포트 · `start()` 전/`stop()` 후 publish는 no-op. **bounded queue 드롭 의미론은 `_DropOldestQueue`를 소비 스레드 없이 결정론적으로 검증**(동시 소비 중 "몇 건 남았나"를 단언하면 근본적으로 플래키 — `test_blackbox.py`의 기존 플래키가 정확히 그 함정, 진단은 test 파일 하단 주석). **파괴검증 7종(D14~D20)으로 red 확인** | ✅ test_target_sink |
| ros/shim_core `ShimRouter`(인터페이스 Phase 2) | **입력을 지어내지 않는다** — 레코드를 손으로 쓴 dict가 아니라 **진짜 생산자 코드**(`vision.core.wire`)로 만들어 넣는다 · **계약 대조**: 복제한 `SCHEMA_VERSION`/`REQUIRED_*_KEYS`/타입명이 `wire.py`와 일치(조용한 드리프트 차단) + `shim_core`가 vision·numpy·rclpy를 import하지 않음을 소스로 고정 · 🔴 `schema_version` 불일치 거절(pose 미생성) · 🔴 유도 입력이 `position_flu`(FRD면 y·z 부호 반전 — 두 값이 실제로 다름을 먼저 단언해 이빨 확보) · 🔴 EOF는 ERROR status만 내고 **pose를 절대 안 냄** + 보류 target 폐기 + 재접속 하트비트가 '생산자 사망'과 'shim 사망'을 가름 · 🔴 `LandingTarget.frame`이 정수 12(msg 상수 2/3이 아님) + `frame=12`에 **FLU**가 들어감 + 기본 꺼짐 + type 손실압축(fiducial/other) + angle/distance가 같은 벡터의 결정론적 투영 · §5.4 `valid=false`는 WARN status만(**좌표가 있어도** pose 미발행 — 게이트가 valid와 좌표존재 **둘 다**임을 고정) · **기본** `unknown_covariance`가 0이 아님(명시 config만 쓰면 기본값을 안 밟아 D6이 조용히 통과했었다) · 회전 공분산은 항상 '모름' · orientation은 단위 쿼터니언이고 원본은 KeyValue로 보존 · `not_for_closed_loop_30cm`이 level을 항상 WARN으로 만들지 않음 + floor 3.0/0.0 해금 · §6.3 `command_hint`+advisory, 맨 이름 `command` 키 부재 · target/state_hint가 **한 stamp로 함께** 나감 + 짝 없을 때 flush + frame_id 불일치 오배치 방지 · `header.stamp`는 wall, `age_s`는 monotonic · 초록구역 z 무보정 + `ground_agl_minus_vision_z_m` 수출 · 깨진 줄에 안 죽고 다음 줄 정상처리 · KeyValue 전부 문자열 + `json.loads` 왕복 · 결정론 · **shim_node 얇음 회귀(AST)**: msg 프레임 상수 참조 금지 · `lt.frame` 대입이 `plan.frame` 경유 1곳 · 와이어 키 직접 읽기 금지 · rclpy import는 shim_node 하나뿐 · `__init__.py` import 0. **파괴검증 10종(S-D1~D10)으로 red 확인**. **[2026-07-28] `/vision/landing_setpoint`** — 🔴 **알려진 방향쌍이 유일한 그물**(축 이름만 맞고 부호가 뒤집힌 변환도 노름·왕복 검사는 통과한다): 기수 정북+전방 10m→정북 10m · 기수 정동+전방 10m→정동 10m(회전이 실제로 먹는지) · FLU +y(좌)→서쪽 · **아래 타겟이 `h_up`을 낮춘다**(`h_up`↔`D` 혼동) · **FRD를 먹이면 다른 값**(두 값이 실제로 다름을 먼저 단언해 이빨 확보) · **우롤 30° 3D 케이스**(yaw만으론 전치/켤레를 못 잡는다 — Rz는 부호가 뒤집혀도 축이 같다) · `quat_rotate`를 **로드리게스 numpy 재구현과 대조**(같은 파일 내 함수로 검증하면 같은 오타를 공유) · 🔴 **드리프트 상쇄**: 기체 위치를 바꿔가며 상대 델타는 동일하고 절대 목표점은 달라짐 + 라우터가 계산된 목표점을 상태로 갖지 않음 + pose 갱신만으론 발행 0 · **degrade 3전이**(없음/stale/노름0 전부 setpoint 침묵 + **상대 pose는 계속 발행** + 사유는 별도 `vision/setpoint` status로) + 임계 바로 안쪽은 통과 + `valid=false`는 **좌표가 있어도** 거절(DS-18이 green이던 pseudo를 잡아 추가 — pose 경로 D7과 같은 함정) + 발행 안 하면 좌표 키를 **아예 안 만든다**(0 채우기 금지) · setpoint 실패가 `vision/target` level을 안 올림(지상시험에서 상시 WARN 방지) · 🔴 **orientation이 단위 쿼터니언이 아니라 현재 기수방위**(flight04 yaw 스핀) + 롤 상태에서도 yaw만 · **기본값 실제 밟기**(인자 없는 `ShimRouter()`가 실제로 발행 + 기본 토픽/프레임 `map`≠`base_link`) + `attitude_stale_s` 기본값이 **실측 p99의 5배 이상**이고 **3m AGL 오차예산 15cm 미만**이며 `stale_warn_s` 복사가 아님 · 세 토픽 stamp 동기 · `[N,E,h_up]` 진단 삼중항. **파괴검증 19종(DS-1~DS-19)으로 red 확인 — 부호 파괴만 8종**(켤레/전치/FLU↔FRD/`h_up`↔`D`/N↔E/델타 뺄셈/`quat_yaw` atan2/ENU→NED yaw) | ✅ test_shim_core |
| registry | 등록 이름 전부 실제 클래스 매핑(값이 실제 클래스 + `vision.modules` 소속 + **인자 없이 생성해 `__call__(state)->VisionState` 실호출**까지 — "매핑만 있고 파이프라인엔 못 쓰는 것" 배제)·중복 없음(**`ast`로 소스 원문 키를 센다** — dict 리터럴은 키가 겹치면 조용히 덮어써서 런타임 dict로는 중복이 있었다는 사실 자체가 안 남는다) · 같은 클래스를 두 이름으로 등록 금지 · **등록 누락 회귀**: `vision.modules.__all__` 전부 등록됨 + 커밋된 preset 7종이 참조하는 스텝 이름 전부 등록됨(미등록이면 `Pipeline.from_config`가 ValueError로 죽어 preset이 통째로 로드 불가). **파괴검증 5종(R-D1~D5)으로 red 확인** | ✅ test_registry |
| color `ColorFilter` | 모드별 mask 생성·임계값 경계·meta | ✅ test_color (gray+color, 빨강 Hue랩어라운드 미지원은 §5.4 blind spot로 별도 회귀테스트 기록) |
| illumination | current 변형·형상/채널 보존·meta | ❌ TODO |
| denoise | current 변형·형상 보존 | ❌ TODO |
| edge `EdgeDetector` | current/mask→mask 갱신·빈입력 | ❌ TODO |
| morphology | mask 갱신·커널크기 효과·빈 mask | ❌ TODO |
| detector `RectDetector` | mask→detections·min_area 필터·빈 mask 0검출·meta | ✅ test_detector |
| background | 연속프레임 mask 갱신 | ❌ TODO |
| tracker `KalmanTracker` | **범위 주의 — 다중트랙 추적기가 아니다**(트랙 ID·검출↔트랙 연관 없음, 매 프레임 `max(confidence)` 검출 1개만 측정치로 쓰는 단일 필터). 최초 검출로 즉시 락(과도구간 없음)·정지 타깃 드리프트 없음·타깃 순간이동 시 예측이 관측으로 수렴(후반 오차 ≤2px, 전반 >10px과 대조) · **연속성 핵심**: 검출 공백에서 학습속도(10px/frame)로 dead-reckoning 계속(`source=predict`, 단조증가) + 재포착 시 필터 재초기화 없이 상태 유지(새 트래커가 같은 측정치로 얻는 값보다 앞서 있음으로 증명) · 초기화 전 빈 프레임은 예측 지어내지 않음(meta 없음) · `process_noise`/`measurement_noise` 생성자값이 실제 추종속도를 바꿈(하드코딩 아님) · 신뢰도 최고 검출만 측정치(검출 순서 무관) · zero bbox 안전·선언필드 계약·결정론. **파괴검증 11종(T-N1~N11)으로 red 확인** | ✅ test_tracker |
| fusion `TemporalFusion` | 임계 미달 시 confirmed/meta 없음·정확히 `min_frames`번째 프레임에 승격·`min_frames`/`iou_threshold` 설정값대로 동작(하드코딩 아님)·확정 카운트 단조증가 · **흔들림 억제 핵심**: 한 프레임 걸러 나타나는 검출은 20프레임을 줘도 미승격, 1프레임 스파이크는 잔여 후보 없이 소멸 · 타깃 소실 시 감쇠→재확정에 프레임 재소요 · IoU 정확값·임계가 병합/분리를 가름·떨어진 두 타깃 독립 누적 + 증거 많은 쪽 확정 · 빈 입력 안전·선언필드 계약·결정론. **파괴검증 12종(F-M1~M12) + 구조결함 수정 파괴검증 12종(F-D1~D12)으로 red 확인**. **[2026-07-28] 구조적 결함 수정 완료** — 예전 `_decay()`가 방금 매칭된 후보의 count까지 깎아 **프레임당 검출 1개짜리 타깃이 영원히 승격되지 않던** 결함(docstring 계약과 정면 충돌)을 고쳤다. 확정된 의미: **`count` = 연속 관측 프레임 수**(관측 프레임 +1, 그 프레임에 겹치는 검출이 몇 개든 +1 / 미관측 프레임 -1). 수정 갈래와 기각한 대안 2건은 `modules/fusion.py::_decay()` 설계 노트 참조. 이에 따라 ① 결함 고정 테스트(`test_current_behaviour_single_detection_per_frame_never_confirms`) 폐기, ② `xfail(strict=True)`였던 `test_documented_contract_*`를 정상 통과로 승격, ③ 결함 산술에 의존하던 `test_iou_threshold_*`(임계 높으면 "확정 못 함" → "후보가 분리됨"으로 계약 직시)와 `test_separate_targets_*`(비대칭을 검출 개수가 아니라 **프레임 수**로 생성 — 옛 방식은 4:4 동률이라 `max`를 `min`으로 바꿔도 통과하는 pseudo였음을 실측 확인) 갱신, ④ 프레임당 최대 +1 규칙 회귀 2건 신설 | ✅ test_fusion |
| vertiport_field `WhiteFieldDetector` | mask→원형 blob 검출·원형도 필터·중심/반지름 meta | ✅ test_vertiport_field |
| vertiport_v `BlackVMatcher` | original 내 어두운 영역 matchShapes 검증·1차 bbox 밖 배경 오탐 배제·불일치 시 detections 제거 | ✅ test_vertiport_v |
| vertiport_ring `RedRingDetector` | 빨강 Hue 양끝 게이팅(랩어라운드 대응)·최소외접원 피팅·중심/반지름 meta | ✅ test_vertiport_ring |
| 버티포트 coarse 캐스케이드 통합(`presets/vertiport_coarse.yaml`) | 3단 전체 파이프라인 end-to-end·단계별 meta 기록·빈 이미지 0검출 | ✅ test_vertiport_cascade |
| ArUco fine 프리셋 통합(`presets/vertiport_fine.yaml`, ArUco Phase 4) | `Pipeline.from_config` 실로드·ID 23 검출+코너·다른 ID 거절·빈 이미지 0검출(coarse와 독립 실행) | ✅ test_vertiport_fine |
| distress_box `WhiteBoxDetector`(§5.3 fine, 2026-07-25) | 매트 내 흰 박스 확인·`landing_point_px`가 매트 bbox 내부에 있음·박스가 매트 좌상단에 치우치면 착륙점이 반대편(우하단)으로 밀림·박스 없음/너무 큼/너무 작음/종횡비 초과 각각 거절+reject_reasons 기록·detections 2개(확정1+거절1) 혼합·빈 detections·zero bbox·original/current/mask 비변형(선언 필드 계약)·결정론. **[2026-07-28] 등거리 축퇴 완화**(위 "착륙점 등거리 축퇴 완화" 절) — 완화 끈 옛 동작이 실제로 매트 반대편까지 점프함(고친 대상 못 박기)·기본값에서 1px 흔들림 10프레임에 착륙점 불변·`tie_tolerance_ratio=0` + `corner_hysteresis=False`가 옛 동작과 정확히 일치·편심 박스는 여전히 반대편 선택(`corner_tie_count==1`)·슈미트 트리거 무진동(전환 경계를 **해석식 아닌 실측 스캔**으로 찾아 대조군이 실제로 진동함을 먼저 확인)·히스테리시스 상태 인스턴스 격리+`reset()`+실제 배선 여부·매트 2개 IoU 교차오염 없음·진단 3종·JSON 직렬화. **파괴검증 D1/D2/D3/D4/D5로 red 확인**. **[2026-07-28] 착륙점 기하 — 기체 크기 기반 재설계**(아래 "착륙점 기하 — 기체 크기 기반 재설계" 절) — 산출 착륙점을 **렌더링된 실제 파이프라인**에서 재 두 부등식(하한은 √2 대각선, 상한은 축 방향)을 R=0.3/0.5/0.6 전부에서 동시 만족·옛 규칙(d=1.05m)이 **R의 알려진 하한 0.5m에서조차** 매트를 넘음을 못 박기(고친 대상 재현)·안전창 경계 손계산 4점(R=0.3/0.5/0.7/1.0, R=0.7·1.0은 infeasible) + 절벽 `aircraft_radius_max_feasible_m=0.6444` 고정·**안전창이 비면 착륙점을 지어내지 않고 거절**(사유 `landing_point_infeasible`, `no_white_pixels`와 구분, 하한/상한/R이 meta에 남고 JSON 직렬화 가능)·🔴 **거절이 필수인 이유를 실제 `DistressMatGeometry` 통과로 증명**(착륙점만 빼고 남기면 매트 중심=박스 위로 조용히 degrade)·정상 경로의 `state.meta` 형태 불변(골든 labels.json 동등비교 보호)·🔴 **인자 0개 생성자로 기본값을 직접 밟는 테스트**(기대값은 상수 재참조가 아닌 손계산 리터럴 0.562132)·`AIRCRAFT_RADIUS_MEASURED is False` + 기본값이 알려진 하한 0.5m 초과 + 미측정 플래그의 meta 전파·매트/박스/드리프트/bias 각각이 실제로 착륙점을 움직임(하드코딩이면 red)·박스 중심 오프셋 수출(정중앙 전제 사후검증)·생성자 인자 검증·`interior_margin_ratio` 폐기(주면 `DeprecationWarning` + 무시 + meta 통보, 물리 도출값이 항상 이김)·preset yaml 값과 모듈 기본값 대조. **파괴검증 17종(D1~D17)으로 red 확인** | ✅ test_distress_box |
| distress_mat `DistressMatGeometry`(초록구역 pose, 2026-07-28) | 코너 순서 정규화가 **실측 approxPolyDP 순서**(반시계)를 시계방향·TL시작으로 바꾸는지(항등이면 red) · 입력 회전/감김 8가지 조합에 불변 · 축퇴 사각형/4점 아님 거절 · coarse는 매트 중심 degrade + `landing_point_source` 명시 · fine은 `white_box_detector`의 `landing_point_px`를 **그대로 소비**(재구현 금지) · 생성자 `size_m`/`platform_height_m`이 실제로 meta에 반영(하드코딩이면 red) · `plane_reference="mat_top_surface"` · 선언 필드 계약(검출 개수 불변, original/current/mask 무변형) · meta 네임스페이스 · 빈 입력 · 결정론 · JSON 직렬화(numpy 누출 없음). **파괴검증 D1/D1b/D3b/D8로 red 확인** | ✅ test_distress_mat |
| ② 조난자 fine 프리셋 통합(`presets/distress_fine.yaml`, 2026-07-25) | `distress_coarse.yaml` 뒤에 `white_box_detector` 캐스케이드 실로드·매트+박스 실제 확정·`Detection.meta`에 `landing_point_px` 실제 기록(`Pipeline.from_config` 경유, 클래스 직접 호출 아님) | ✅ test_replay(`test_distress_fine_preset_confirmed_detection_carries_landing_point_meta`) |
| utils/image_loader | 경로→BGR ndarray(shape/dtype + **채널 순서가 실제로 BGR인지를 3분할 B/G/R 띠로 왕복 확인** — 여기서 RGB로 뒤집히면 뒤의 HSV 색 검출이 통째로 어긋난다)·PNG 무손실 왕복(임의 정규화 없음)·`Path` 객체 수용·1채널 원본도 3채널로 확장·결정론 · 없는 파일→`FileNotFoundError`(메시지에 경로 포함)·디렉터리/쓰레기 바이트/0바이트→조용한 None이 아니라 `ValueError`. **파괴검증 5종(I-D1~D5)으로 red 확인** | ✅ test_image_loader |
| utils/video_reader | **실제 mp4를 써서 실제 디코딩**(몽키패치 없음, `test_frame_source.py`와 같은 원칙) — 전 프레임 이터레이트(BGR shape/dtype) · **순서 보존**(프레임마다 다른 밝기를 심어 확인 — 시간축 모듈 `fusion`/`tracker`가 전적으로 여기 기댄다) · `while True` 루프가 EOF에서 실제로 멈춤(무한루프 회귀) + 소진 후 재이터레이트는 빈 목록 · 결정론 · `fps`/`frame_count`가 인코딩값·실제 이터레이션 수와 일치(하드코딩 아님) · 컨텍스트 종료: `__enter__`가 self 반환 + `__exit__`이 실제로 `VideoCapture`를 release(본문에서 예외가 나도, 닫힌 뒤 읽어도 크래시 없이 빈 목록) · 없는 파일→`FileNotFoundError`·영상 아닌 파일→`IOError`(0프레임으로 삼키지 않음). **파괴검증 7종(V-D1~D7)으로 red 확인** | ✅ test_video_reader |
| utils/visualize `draw_sink_status`(발행상태 오버레이, 2026-07-28) | **소비자 0명이면 경고색이 실제 픽셀로 찍힘**(경고를 정상색으로 바꾸면 red) + 경고 헤드라인이 정상보다 **크다**(잉크량 비교) · 소비자 있으면 정상색이고 경고색은 0픽셀 · sink 꺼짐은 두 색 어느 쪽과도 구분되는 제3색 · seq/dropped 값이 실제로 픽셀에 반영(라벨만 그리고 숫자를 버리면 red) · **검출 그리기 비훼손**: 패널 밖 영역이 비트 단위로 동일 · 제자리 수정(같은 배열 객체 반환)·shape/dtype 보존 · 결정론 · 초소형(64px)~실기체(4608px) 프레임 무크래시 · **오버레이 문자열 ASCII 전용**(Hershey 폰트가 한글을 못 그림 — 소스 레벨 회귀) | ✅ test_visualize |
| utils/visualize `draw_detections`/`save_result` | 형상·파일 생성 | ❌ TODO (위 오버레이 테스트가 이걸 대신하지 않는다) |
| ~~utils/geo_project~~ | **2026-07-28 삭제 완료**(plan §12). 되살아나지 않는지만 묘비 테스트로 지킨다 — 파일 부재·import 불가·저장소 어디에도 `pixel_to_gps` 재등장 없음. **파괴검증 D18로 red 확인** | ✅ test_deprecations |
| utils/logging | 이중싱크 핸들러 구성·콘솔레벨이 파일레벨 억제 안 함·재호출 시 핸들러 중복 안 됨·provenance에 git해시/config/캘리브id | ✅ test_logging |
| utils/blackbox | 프레임/거절이유 JSONL 기록·close() 큐 가득해도 안전 · **[2026-07-28] 플래키 수정** — 예전 `test_bounded_queue_drops_oldest_under_burst`는 "**파일에** 남은 레코드가 ≤ max_queue"를 단언했는데 `__init__`이 `QueueListener`를 이미 start()해 둔 상태라 넣는 동안 리스너가 동시에 파일로 흘려보낸다(파일 건수는 스케줄링 의존 — 생산 루프에 틱당 0.2ms만 양보해도 50건이 남아 실패). drop-oldest가 보장하는 건 "**큐에** 최대 N건"이지 "**싱크에** 최대 N건"이 아니다. 3건으로 쪼갬: ① 소비 스레드 없이 `_DropOldestQueueHandler`만 두고 **매 스텝** `qsize == min(넣은수, 상한)` + 잔여 = 최신 N건(`test_target_sink.py`의 `_DropOldestQueue` 결정론 패턴 재사용. ⚠️ 최종 상태만 보면 "가득 차면 큐를 통째로 비우는" 구현도 50/5에서 우연히 통과 — 파괴검증 D-B3 실측) ② 그 상한이 `BlackBoxLogger`에 실제 배선됐는지 구조 검증(`_queue.maxsize == max_queue` + 핸들러 타입) ③ end-to-end는 **부하 무관 불변식**만(최신 프레임 불유실 + 순서·유일성·부분열). **파괴검증 4종(D-B1~B4)으로 red 확인** | ✅ test_blackbox |
| utils/stream `MjpegStreamer`(§7.9 항목5) | 실제 HTTP 서버 기동 → 실제 프레임 push → 실제 클라이언트로 `/stream` 접속해 진짜 MJPEG 바이트 수신·`cv2.imdecode` 디코드 성공·VGA 박스 축소(종횡비 유지, 업스케일 없음)·`push_frame()` 비차단(클라이언트 없음/느린 클라이언트 붙어있어도 논-블로킹, 실측 시간)·`start()` 전 `push_frame` 안전 no-op·idempotent stop/restart | ✅ test_stream |
| utils/frame_source | Dir/Bag: 실제 파일→실제 프레임 디코딩·순서 결정론·telemetry.jsonl(사이드카 포함) frame_id 매칭·빈/누락 입력 에러. Live: 연결 실패 시 재시도 후 `ConnectionError`·읽기 실패 시 `ConnectionError`·`open_dir_or_bag` 디렉터리/파일 자동판별. **[2026-07-28] AF 제어** — 모드별 컨트롤 조립(auto는 `AfTrigger=Start` 2차 호출 필수)·렌즈위치 범위가 **실측 하드클램프 15.0**(광고값 32.0 거부)·인자 조합 검증이 **생성자에서**(하드웨어 만지기 전에 실패)·기본값이 `continuous`이고 실제로 `set_controls`까지 도달·`af_mode=None` escape hatch는 `set_controls` 0회·**AF 실패가 캡처를 죽이지 않고** `af_error`에만 남음(드라이버 거부/libcamera 부재 둘 다)·`libcamera` 최상단 import 부재 회귀·**`tools/h264_stream.py`가 같은 구현을 재사용하는지**(복제 방지 계약). **파괴검증 D13~D17로 red 확인** | ✅ test_frame_source |
| main.py | `--display` 게이팅: **none=imshow 0회**(헤드리스 안전 불변식)·file→output 강제·stream 미구현 · **로거+JSONL 블랙박스 실연결**: 실행 시 실제 `.log`/`.jsonl`이 디스크에 생성되고 detections/latency/provenance가 올바름 · **§9 6번 상태머신 배선**: 반복 ArUco 프레임 실제 영상 실행 → JSONL `state`가 전부 null 아니고 ACQUIRE에 머물지 않고 실제로 진행(LOCK/PRECISION_SERVO 도달)함을 실제 파이프라인으로 확인, `command`도 함께 실림 · **[2026-07-25] ② 조난자 fine 체인**: 흰 박스 확정 반복 영상(`distress_fine.yaml`)도 ArUco와 별개 경로로 CENTER_DESCEND를 넘어 진행함을 실제 파이프라인으로 확인 · **[2026-07-28] `--target-sink` 배선**: opt-in 불변식(미지정 시 소켓 미기동 + 발행 시도 0) · 매 프레임 `target` 1건 + `state_hint` 1건, `REQUIRED_*_KEYS` 전량 + `seq` 단조증가 + 맨 이름 `command` 키 부재 · **§5.4 침묵 금지**: 검출 없음/calib 없음도 사유가 붙은 `valid=false`로 계속 발행(포즈는 0이 아니라 null) · **🔴 발행 실패 무영향**: 발행마다 예외를 던지는 sink에서도 전 프레임이 JSONL에 남음 · **SIGTERM 비가로채기 회귀**(핸들러 하나 규칙) + 종료 시 sink 정리 · **실소켓 종단간**: 순수 stdlib 소비자가 `readline()`으로 JSONL을 읽고 main() 종료 시 EOF 수신 · **[2026-07-28] bind 하드 페일**(계약 뒤집힘, 옛 강등 테스트 폐기): 종료코드가 `EXIT_SINK_BIND_FAILED`(≠0) + stderr에 포트 번호 + 프레임 0건 처리 + **진짜로 점유된 포트**에서도 동일(몽키패치 없음) + 죽어도 blackbox.close() 보장(leak 회귀) · **[2026-07-28] 발행상태 오버레이**: `--display none`은 `draw_sink_status` 호출 0회(비용 0) / `--display file` 저장본이 `none` 저장본과 다르고 경고색 픽셀이 실제로 있음 / sink 꺼짐은 OFF색으로 구분 / 오버레이 인자가 **실제 sink 카운터**(seq 2,4,6…)여야 함(하드코딩 0이면 red). **파괴검증 8종(D-A1~A8) + 신규 6종(D1~D6)으로 red 확인**  · **[2026-07-28] `tan(HFOV/2)` 배선**: `_landing_sm_config`가 calib 인트린식에서 유도(+calib None/fx<=0 폴백) + **`main()`을 실제로 돌려** nominal과 다른 `--calib`을 준 뒤 그 유도값이 상태머신 생성자까지 도달함을 확인(호출부가 기본 config로 되돌아가면 red). **파괴검증 D10** | ✅ test_main |
| replay.py | `open_dir_or_bag`로 Dir/Bag 자동판별 재생·실제 프레임 처리로 JSONL(telemetry 포함)/사람로그 실생성·`--output` 지정 시 실제 mp4 기록 · **§9 6번 상태머신 배선**: 반복 ArUco 녹화 재생 → JSONL `state` 실제 진행 확인(main.py와 동일) + **telemetry.jsonl의 alt가 실제로 상태머신에 흘러 TERMINAL까지 도달**함을 실제 재생으로 확인(AGL 배선이 진짜로 연결됐다는 증거) · **[2026-07-25] ② 조난자 fine 체인**: 흰 박스 확정 녹화 재생도 CENTER_DESCEND를 넘어 진행 + telemetry alt로 TERMINAL까지 도달 확인(ArUco와 별개 경로) + `landing_point_px`가 실제 `Pipeline.from_config` 경로로 `Detection.meta`에 실림을 확인 · **[2026-07-28] `--target-sink` 배선**: opt-in 불변식(미지정 시 소켓 미기동) · bind 하드 페일이 main.py와 **같은 종료코드**(진짜 점유 포트, 프레임 0건) · **🔴 합격 기준 — AGL이 실린 재생에서 `state_hint`가 `TERMINAL`까지 진행**(main.py는 AGL 경로가 없어 구조적으로 불가, 이 테스트가 유일) · **실소켓 종단간**: 순수 stdlib 소비자가 받은 state_hint에 TERMINAL 실재 + seq 단조증가 + EOF · 오버레이 게이팅(`none`=0회, `stream`=매 프레임 + 실제 seq) | ✅ test_replay |
| tools/jsonl_view.py | 실제 `main.py` 실행으로 만든 진짜 JSONL 로드·행 수=JSONL type=frame 행 수 일치·score/latency 라인 포인트 수=행 수(결측은 nan 구멍, 이어붙이지 않음)·state 미기록 시 안내 텍스트·rejection→세로선·PNG 실파일 생성 | ✅ test_jsonl_view |
| tools/calib_analyze.py | **★합성 왕복(핵심):** 진짜 K/dist를 알고 합성 투영한 ~20장 사이드카 → 복원 fx/fy/cx/cy 1% 이내·dist 허용오차 내 · fx-vs-LensPosition 직선적합이 알려진 (L,fx) 직선을 복원 · 이상치 검출(코너 오염 이미지가 플래그되고 제외 시 RMS 개선) · 부분 데이터(그룹 1개)에서 크래시 없이 적합 생략+`recommended` 폴백 사유 기록 · yaml 아티팩트 `yaml.safe_load` 왕복 + 필수 키 전부(`checks[].ok`가 python bool인지 — numpy.bool_ 누출 회귀 포함) · `--redetect` PNG 재검출 경로 · CLI(`main()`) end-to-end로 진단 플롯 3종 PNG 실파일 생성 | ✅ test_calib_analyze |
| tools/color_calibrate.py(§9 빌드순서 5번) | **★합성 왕복(핵심):** 산출된 임계값을 실제 `ColorFilter`/`RedRingDetector`에 먹여 합성 패치가 실제로 검출되고 배경은 배제되는지 확인(문자열만 맞는 게 아니라 파이프라인이 실제로 동작하는지) · 백분위수가 소수(2~3%) 이상치(글레어/랩어라운드 스필오버)에 강건함을 raw min/max와 대조해 직접 증명 · 빨강 Hue 랩어라운드 감지(0/179 양끝에 걸친 합성 패치)·`wrap_min_fraction` 미만이면 오판 안 함·두 소비자(`ColorFilter`/`RedRingDetector`) 각각 호환 파라미터 키셋 검증 · `parse_roi`/`crop_roi` 경계값·범위이탈 에러 · `load_frame`이 `DirFrameSource`/이미지 직접로드 재사용(frame_index 선택 포함) · 진단 PNG(오버레이+히스토그램) 실파일 생성 · CLI(`main()`) end-to-end(정상/입력없음/ROI범위밖) · 골든셋(`tests/golden/distress/10m/`) 실제 프레임 교차확인(`distress_coarse.yaml` 손튜닝값과 sanity, §9 견고성) | ✅ test_color_calibrate |
| tools/h264_stream.py(ffmpeg Phase 3) | **순수 로직만(하드웨어 없음, `LiveFrameSource`와 동일 예외 패턴):** `parse_resolution`/`build_ffmpeg_listen_spec` 파싱·경계값 · `validate_lens_position`(0~15.0 경계, 32.0 오해 방지 에러 메시지) · `validate_af_args`(manual↔lens-position 필수/배타 조합) · `compute_fps_stats`(워밍업 스킵 기본 내장, 표본부족/비단조 거부) · `count_identical_frame_pairs`(라이브니스) · `_make_af_controls`(continuous/auto/manual→AfMode/AfTrigger 매핑, 가짜 controls 모듈로 검증) · `_is_output_dead`/`_stop_output_with_timeout`(duck-typing, 정상종료 시 kill 안 함·행잉 시 kill 개입 둘 다 검증) · `_install_sigterm_handler`(실제 SIGTERM 전송→stop_event 세팅 확인, 전역 핸들러 복원) · CLI 파싱 기본값/에러종료 · picamera2/libcamera 지연 import 격리 회귀(최상단 import 없음 + `sys.modules`에 None 주입해도 import 성공). **run_server 자체(picamera2/libcamera 실제 하드웨어 연동)는 tools/ 예외로 pytest 대상 아님** — RPi 실기체 기동+랩탑 `cv2.VideoCapture` 실접속으로 검증(§Phase 3 완료 절, `docs/vision_camera_bringup.md`) | ✅ test_h264_stream |
| tests/golden 회귀(§7.9 항목7) | `vision.replay.run_replay()`로 골든셋(§ tests/golden/README.md) 실제 재생 → JSONL 검출 개수가 `labels.json` 기대값과 일치·캐스케이드 단계별 meta도 실제 `Pipeline.run()`으로 검증. 몽키패치 없음(실제 파이프라인). **[2026-07-25]** `distress/fine`(`white_box_detector` 확정+`landing_point_px`)과 `no_target/distress_fine`(오탐 회귀) 리프 추가. **[2026-07-28] 재생성 경로 무결성** — `generate_synthetic.py`를 tmp로 실행해 **커밋된 골든 전체와 바이트 단위 대조**(리프 누락/조용한 드리프트 둘 다 red) + 복구한 `no_target/distress_coarse` 리프 이름 고정. **파괴검증 D11/D12로 red 확인** | ✅ test_golden_regression |

**공통 규칙 (모든 모듈 테스트):**
1. **선언 필드 계약** — 위 파일표대로 "읽는 필드"만 읽고 "쓰는 필드"만 쓴다.
2. **meta 네임스페이스** — `state.meta["<모듈이름>"]` 기록 확인.
3. **빈/경계 입력** — 검은 이미지·빈 mask에서 크래시 없이 합리적 출력(대개 0검출).
4. **결정론(plan §7.5)** — 같은 입력·같은 config → 같은 출력. **골든셋 회귀 스캐폴드는 2026-07-21c에 합성 데이터로 시작됨**(`tests/golden/README.md`) — 실기체 데이터는 카메라 브링업 이후 교체 예정.

**새 모듈 추가 시:** 위 4개 공통 규칙을 담은 `tests/test_<모듈>.py` 를 **같은 커밋에** 추가한다.
