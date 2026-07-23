---
doc_type: orchestrator_brief
scope: RPi5 카메라 정공법 브링업 — libcamera 부활 → picamera2 검증 → ffmpeg 영상
status: 착수 대기 (오케스트레이터 세션이 이 문서로 진입)
created: 2026-07-23
---

# 카메라 브링업 오케스트레이터 브리프

> **다음 세션 진입:** "너는 오케스트레이터이다"로 시작하고 이 문서 하나만 읽으면 된다.
> `docs/vision_status.md`(트랙 보드)·`docs/vision_plan.md`는 필요 섹션만 열되, 이 작업의 지시는 여기에 자기완결적으로 있다.
> 프로토콜은 메모리 `feedback_orchestrator_protocol` 준수 — **각 Phase는 fg가 아니면 bg, 세션 자기보고는 직접 재현 검증 필수, 진행상황 확인 없이 한 프롬프트로 몰아던지지 말 것.**

---

## 0. 왜 이 작업인가 (한 문단)

지난 6세션이 카메라를 **V4L2 raw 직접 캡처로 우회**해 브링업했으나, 그 결과 오토포커스·자동노출·자동화벨이 전무해 수동 스윕(거리당 수 분)·gray-world 수동보정·1.5초 슬라이드쇼로 귀결됐다. 근본원인은 하드웨어도 커널도 아니고 **딱 하나** — Ubuntu 24.04의 `libcamera-ipa` 패키지에 RPi5용 `ipa_rpi_pisp.so`가 빠져 있어 libcamera가 카메라를 0대로 본다. **이건 배포판 패키징 누락일 뿐이라 소스빌드로 살릴 수 있다.** libcamera가 살면 AF/AE/AWB(품질)와 연속 H.264 영상(스트리밍)이 **동시에** 풀리고, 우회책 814줄 + MJPEG + 200줄 포렌치 노트가 통째로 폐기된다.

## 1. 확정 전제 (재논의 불필요)

- **호스트 OS 유지(Ubuntu 24.04). 로컬 prefix 비파괴 빌드.** OS 이주 안 한다 — tailscale/docker/ros/wifi 완화조치/EEPROM 전원설정 전부 무손상, 원격으로 진행 가능. (OS 이주는 헤드리스 최초설정에 물리 키보드·모니터가 필요해 원격 관리와 충돌 → 후퇴선으로만 둔다.)
- **RPi5 카메라·영상·텔레메트리 물리 배치는 확정, 논의 대상 아님:**
  - 카메라(서드파티 IMX708 클론 CAM109) → 호스트 `/dev/video*`·`/dev/media*`
  - 영상 다운링크 → **RTL8812AU USB(호스트)**, 지상국과 통신 (아직 미장착, 목표 하드웨어)
  - MAVLink 텔레메트리 → **픽스호크에 직결된 별도 라디오** (RPi·도커 무관)
  - offboard 제어용 MAVROS → 컨테이너 `fc`(ros:humble), `/dev/ttyACM0`
- **컨테이너 `fc`는 `NET=host`** — 호스트↔컨테이너 네트워크 경계 없음. vision(호스트)→ROS2(컨테이너) 다리는 루프백 IPC 하나면 됨. **영상은 컨테이너를 지나지 않는다.**
- 소스 이미 준비됨: `/home/suri/local-libcamera-src/libcamera` (clone 완료, 빌드의존성 apt 설치 완료, `meson setup` 미실행). 상세는 메모리 `project_rpi5_ubuntu_camera_stack.md`.

## 2. Phase별 실행 (각 Phase 끝에 보고 게이트)

### Phase 1 — libcamera 부활 (핵심, 성패 결정)
- **1a.** `/home/suri/local-libcamera-src/libcamera`에서 meson 옵션에 pisp 파이프라인/IPA가 켜지는지 확인 후 `meson setup build --prefix=/home/suri/local-libcamera` (필요시 `-Dpipelines=rpi/pisp -Dipas=rpi/pisp`). **→ 여기까지만 하고 보고.**
- **1b.** `ninja -C build`. RPi5라 시간 걸릴 수 있으나 로컬 빌드라 네트워크 무관(과거 apt 지연은 네트워크 탓, 빌드엔 무관). **→ 보고.**
- **1c.** 산출물에 **`ipa_rpi_pisp.so` 존재 확인** — 나오면 사실상 승리.
- **1d.** 환경변수(`LIBCAMERA_IPA_MODULE_PATH`/`LD_LIBRARY_PATH`/`PYTHONPATH`)로 로컬 빌드를 가리켜 `CameraManager.singleton().cameras`가 **빈 리스트가 아닌지** 확인. **→ 보고.**
- ⚠️ **1d까지 실패 시 후퇴선:** Bookworm A/B SD카드(빈 카드 굽고 `rpicam-hello` 10분 판정). 이주 아님, "카메라 진짜 되나"만 확인. SD가 부트순서 1순위라(BOOT_ORDER=0xf461, 실측) 카드만 꽂으면 Bookworm 부팅, 빼면 기존 Ubuntu USB 복귀 — 기존 시스템 무손상.

### Phase 2 — picamera2로 6기능 실측 검증
libcamera 살면 picamera2 설치 후 각각 **직접 재현 검증**(pseudo 금지):
① 초점 목표값 1초내 반영 ② 진짜 연속 AF ③ AF↔MF 전환 ④ **AF 윈도우(영역) 지정** ⑤ AE 자동/수동 전환 ⑥ AWB 자동/수동 전환. 되면 지난 세션의 수동 스윕·gray-world·정착시간 조사가 전부 불필요해짐.

### Phase 3 — ffmpeg 영상 (디버깅 + 다운링크 통합)
- `rpicam-vid`/`libcamera-vid` → **연속 H.264** → 랩탑 `ffplay`/`mpv`(브라우저 아님, 저지연 UDP/RTSP). 이게 디버깅 주경로.
- **같은 H.264 인코드를 RTL8812AU 다운링크와 공유** — 계획서 §7.7 EncoderSink 스왑 어댑터가 원래 의도한 구조. 인코드 경로 하나로 디버깅+대회 다운링크 동시 충족.
- MJPEG-over-HTTP(`utils/stream.py`)는 이 시점에 폐기 후보로 전환.

### Phase 4 — 통합·폐기 (별건, 위 검증 후)
- `vision/utils/frame_source.py::LiveFrameSource`를 picamera2 백엔드로 재구현 — 계획서가 정한 정식 이음매(현재 `cv2.VideoCapture` 구현은 V4L2 raw 경로와 비호환, 실측됨).
- **격리 대상(= 우회책 대청소 본체):** `vision/tools/rpi_capture.py`의 V4L2 814줄(`unpack_raw10`/`debayer_to_bgr8`/gray-world/focus-sweep/media 동적탐색) + `vision/CLAUDE.md`의 관련 200줄 포렌식 노트. **libcamera 부활 전엔 아직 유효하므로 Phase 1 성공 전에는 건드리지 말 것.**

## 3. 리스크 / 판정

- Ubuntu 24.04 arm64에서 libcamera 소스빌드 실패 가능성(중간) → **Phase 1d가 판정 지점**. 실패하면 무리한 우회 말고 후퇴선(Bookworm A/B)으로 정확히 전환하고 멈출 것.
- upstream libcamera에 vc4/pisp IPA 소스 자체는 존재함이 이미 확인됨(메모리). 못 빌드될 이유는 옵션/의존성 문제일 가능성이 높으니 로그를 남겨 진단.

## 4. 참조

- 근본원인·진단명령·소스 위치: 메모리 `project_rpi5_ubuntu_camera_stack.md`
- 물리 배치·(C) 다리 설계: `docs/vision_plan.md` §7.6/§7.7/§7.9/§8
- 트랙 보드: `docs/vision_status.md`
