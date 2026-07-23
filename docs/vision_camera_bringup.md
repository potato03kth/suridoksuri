---
doc_type: orchestrator_brief
scope: RPi5 카메라 정공법 브링업 — libcamera 부활 → picamera2 검증 → ffmpeg 영상
status: Phase 1·2 완료 (2026-07-23). Phase 3(영상)·4(통합/폐기) 남음
created: 2026-07-23
last_updated: 2026-07-23
---

# 카메라 브링업 오케스트레이터 브리프

> **다음 세션 진입:** "너는 오케스트레이터이다"로 시작하고 이 문서 하나만 읽으면 된다.
> `docs/vision_status.md`(트랙 보드)·`docs/vision_plan.md`는 필요 섹션만 열되, 이 작업의 지시는 여기에 자기완결적으로 있다.
> 프로토콜은 메모리 `feedback_orchestrator_protocol` 준수 — **각 Phase는 fg가 아니면 bg, 세션 자기보고는 직접 재현 검증 필수, 진행상황 확인 없이 한 프롬프트로 몰아던지지 말 것.**

---

## 🔴 인수인계 — 다음 오케스트레이션 세션은 여기서 시작한다 (2026-07-23 18:00 기준)

### 즉시 할 일 (순서대로)

1. **촬영도구 재시작 + 무중단 검증.** 라파에서 `calib_capture`가 **수정 전 코드로 돌고 있다**(커밋 `49e4520` 이전). **촬영 0장이라 재시작 비용 0.**
   ```bash
   ssh suri@100.67.27.83
   pkill -INT -f vision.tools.calib_capture
   source /home/suri/local-libcamera/env.sh
   cd /home/suri/drone_ws/src/suridoksuri
   $PICAM_PYTHON -m vision.tools.calib_capture --out-dir /home/suri/local-libcamera-src/calib
   ```
   재시작 후 **오케스트레이터가 직접** 검증: 조준 페이지에 `http-equiv="refresh"` 없음 / **단일 curl로 `/stream`을 30초 붙잡아 프레임 경계가 끊김 없이 계속 오는지**(재접속 없음) / 촬영 실패 시 샷 인덱스가 **안 넘어가는지**. `49e4520`의 이 부분은 **아직 실기 미검증**이다(사용자 인스턴스가 카메라를 점유해 2차 인스턴스가 카메라 획득 실패, 6회 재시도 후 정직하게 미검증 보고됨).
2. **사용자가 촬영.** 세트 A(0.5/0.7/1.2m × 15장) → 세트 B(2.5m × 40장). 사람이 직접 보드를 드는 물리 작업이라 오케스트레이터는 대기·모니터링.
3. **캘리브레이션 분석 스크립트 — 아직 없다. 새로 만들어야 한다.** `cv2.calibrateCamera` → 사진별 재투영오차로 불량컷 색출 → 세트별 결과 → **`fx` vs `LensPosition` 직선적합으로 무한대 외삽** → 세트 B(모델 무관 기준값)와 대조. 산출물은 계획서 §6/§7 요구대로 **camera_id에 묶인 버전 아티팩트**(`vision/presets/*.yaml` 스타일, 촬영 시 LensPosition 포함)로 저장하고 **커밋**. 원본 사진은 `vision/data/calibration_raw/`(gitignore) 유지.

### 촬영 스펙 (확정, 재논의 불필요)

- **보드:** A4 인쇄, **10×7칸 = 내부 코너 (9,6)**, 칸 크기 **28.0mm 실측**. 태블릿은 같은 (9,6)에 20.0mm.
  칸 크기는 월드 스케일 인자라 **인트린식에 영향 없고** `tvec`만 스케일한다.
- **거리별 예상 LensPosition:** 0.5m→2.0 / 0.7m→1.43 / 1.2m→0.83 / 2.5m→0.40 (실측 스윕으로 확정할 것. **이 예측이 맞는지 보는 것이 곧 "LensPosition=1/s" 가정의 검정**이다 — 각 거리를 줄자로 재서 기록).
- **세트 A 15장 패턴:** 중앙[정면]·중앙[yaw+35]·중앙[pitch+35]·좌[yaw−30]·우[yaw+30]·상[pitch−30]·하[pitch+30]·좌상/우상/좌하/우하[yaw±25,pitch±25]·중앙[롤45]·좌상[yaw−40,롤30]·우하[pitch+40,롤−30]·중앙[yaw+20,pitch−20,롤15]. 가장자리 존은 u 또는 v=0.2/0.8.
- **세트 B 40장:** u∈{0.15,0.38,0.62,0.85} × v∈{0.2,0.5,0.8} 12위치 × 3기울기 + 롤 4장. 2.5m에서 보드를 ∓117cm 옮기는 것보다 **카메라 팬 ∓25°/틸트 ∓13°가 쉽다**(등가).
- **전부 정면 평행으로 찍으면 축퇴**되어 초점거리와 거리가 분리되지 않는다. 기울기 20~45°, 50° 초과 금지.
- **네 모서리를 반드시 채울 것** — 왜곡계수는 주변부 코너로만 결정된다.

### 알려진 미해결 (촬영 후 처리)

- **도구가 진행상황을 저장하지 않는다.** 재시작하면 샷 인덱스·완료목록·커버리지가 0으로 리셋되고, 같은 번호 재촬영 시 **기존 파일을 조용히 덮어쓴다**. 촬영 중 재시작은 피하고, 부득이하면 `skip`으로 실제 마지막 완료 지점까지 넘길 것.
- 세트 B(2.5m)는 저해상 미리보기상 사각형이 ~11px이라 **자동촬영이 안 될 가능성이 높다** — 수동 버튼 사용.
- **release 빌드에서 연속 AF `Focused` 도달 미재확인**(검증 시점 장면 텍스처 소실). 체커보드가 오면 자연히 확인된다.
- ④ AF 윈도우 PARTIAL — 현장에 사람이 있으면 해소된다.
- `cv2.putText`가 한글을 못 그려 프레임 번인 HUD는 ASCII, HTML 페이지만 한글. 이 분리를 유지할 것.

### 세션 운영 규칙 (사용자 지시)

- **서브에이전트는 소네트로 생성한다.** 이번 세션은 월 사용한도에 걸려 에이전트 하나가 중단된 이력이 있다.
- 나머지는 메모리 `feedback_orchestrator_protocol` 준수.

---

## ✅ 결과 요약 (2026-07-23 세션) — 정공법 성공

**전제가 맞았다.** Ubuntu 패키징 누락이 유일한 병목이었고, 로컬 소스빌드로 해소됐다.
libcamera가 카메라를 인식하고 AF/AE/AWB가 실측으로 동작한다. **우회책(V4L2 raw + 수동
디베이어 + gray-world + 수동 초점스윕)은 이제 폐기 대상이다** — 단 Phase 4는 아직 미착수.

| Phase | 결과 |
|---|---|
| 1a meson setup | ✅ `rpi/pisp` pipeline·IPA 양쪽 enabled, pycamera enabled |
| 1b/1c 빌드 | ✅ 274/274, **`ipa_rpi_pisp.so` 생성** |
| 1d 카메라 열거 | ✅ `cam --list`·Python 양쪽에서 imx708 인식 |
| **1e IPA 서명 수정** | ✅ **아래 "최대 함정" 참조 — 이게 없었으면 전부 무의미했다** |
| 2 6기능 실측 | ✅ 5 WORKS / 1 PARTIAL (④ AF윈도우는 장면 한계) |
| 1f release+설치 | ✅ `-O3` 재빌드 후 로컬 prefix 설치, `env.sh` 확정 |

### ⚠️ 최대 함정 — IPA 서명이 없으면 격리실행 → 수동 컨트롤 전부 FATAL

meson이 crypto를 못 찾으면 `IPA modules signed with : None (modules will run isolated)`이
되고, libcamera는 서명 없는 IPA를 **별도 프로세스로 격리** 실행한다. 그런데 라즈베리파이
IPA는 V4L2 `ControlList`를 IPC 경계 너머로 넘겨야 해서 직렬화기가 죽는다:

```
FATAL Serializer control_serializer.cpp:626 A list of V4L2 controls requires a ControlInfoMap
```

**기본상태 캡처는 IPC를 안 타서 멀쩡하다.** 그래서 1d·2a 검증을 그대로 통과해버렸고,
Phase 2에서 "6기능 전부 FAILS"라는 잘못된 결론까지 나왔다가 뒤집혔다.
**해법:** `sudo apt install libssl-dev` → 재설정 시 `IPA modules signed with : libcrypto`
→ 재빌드. 성공하면 런타임 로그에서 `Public key not valid` 경고와 `Starting worker for IPA
module ... IPC fd` 줄이 **둘 다 사라진다**(= in-process 로드). **`libgnutls28-dev`는 쓰지 말 것**
— nettle/gmp 버전포켓 충돌로 설치 불가.

### 확정된 사용법

```bash
source /home/suri/local-libcamera/env.sh   # 이거 하나면 끝
cam --list                                  # → 1: External camera 'imx708'
$PICAM_PYTHON your_script.py                # picamera2는 별도 venv
```

### 실측으로 확정된 하드웨어 사실

- **VCM 실가동범위는 0~15.0 디옵터.** 드라이버는 32.0까지 광고하지만 15.0에서 하드 클램프
  (40프레임 유지 확인). 서드파티 클론이라 정품 CM3와 다름 — **초점 코드에서 32를 상한으로 쓰지 말 것.**
- 컨트롤 적용에 **4~6프레임** 걸린다. 값은 하드웨어 양자화됨(노출 8000→7993, 게인 1.0→1.123=1024/912).
  **최소 20프레임 창으로 측정하고, 정확히 일치하는지로 판정하지 말 것** — 5프레임 창으로 측정해
  "게인이 안 변한다"고 오판한 사고가 실제로 있었다.

### 남은 확인거리 (정직하게)

- **release 빌드에서 연속 AF의 `Focused` 도달은 재확인 못 했다.** debug 빌드에선 확인됨
  (119프레임/6.04초에 lens 5.563 수렴, lapvar 2.02→4.90). release 검증 시점엔 장면 조도가
  178→91 Lux로 떨어지고 텍스처가 사라져 스윕 전체가 평평(피크비 1.3~1.4배)했고, AF는 피크
  구간(11.50)에 도달했으나 `Failed`를 반환했다 — 신뢰도 임계 미달로 **알고리즘적으로 올바른
  거동**이다. 렌즈 위치제어 자체는 0~15 전 구간 정확. **체커보드 촬영 때 텍스처 있는 장면에서
  재확인할 것.**
- ④ AF 윈도우: 두 윈도우 수렴값이 5.637 vs 5.570으로 0.067 차이 — 원격이라 사람이 물체를 다른
  거리에 못 놓아 깊이 분리 자체가 불가능했다. 메커니즘(윈도우별 독립 재스캔)은 확인됨.
  **현장에 사람이 있는 촬영 세션에서 자연히 해소된다.**

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

### Phase 1 — libcamera 부활 (핵심, 성패 결정) — ✅ **완료 2026-07-23** (아래는 당시 지시 원문)
- **1a.** `/home/suri/local-libcamera-src/libcamera`에서 meson 옵션에 pisp 파이프라인/IPA가 켜지는지 확인 후 `meson setup build --prefix=/home/suri/local-libcamera` (필요시 `-Dpipelines=rpi/pisp -Dipas=rpi/pisp`). **→ 여기까지만 하고 보고.**
- **1b.** `ninja -C build`. RPi5라 시간 걸릴 수 있으나 로컬 빌드라 네트워크 무관(과거 apt 지연은 네트워크 탓, 빌드엔 무관). **→ 보고.**
- **1c.** 산출물에 **`ipa_rpi_pisp.so` 존재 확인** — 나오면 사실상 승리.
- **1d.** 환경변수(`LIBCAMERA_IPA_MODULE_PATH`/`LD_LIBRARY_PATH`/`PYTHONPATH`)로 로컬 빌드를 가리켜 `CameraManager.singleton().cameras`가 **빈 리스트가 아닌지** 확인. **→ 보고.**
- ⚠️ **1d까지 실패 시 후퇴선:** Bookworm A/B SD카드(빈 카드 굽고 `rpicam-hello` 10분 판정). 이주 아님, "카메라 진짜 되나"만 확인. SD가 부트순서 1순위라(BOOT_ORDER=0xf461, 실측) 카드만 꽂으면 Bookworm 부팅, 빼면 기존 Ubuntu USB 복귀 — 기존 시스템 무손상.

### Phase 2 — picamera2로 6기능 실측 검증 — ✅ **완료 2026-07-23** (5 WORKS / 1 PARTIAL, 위 요약 참조)
libcamera 살면 picamera2 설치 후 각각 **직접 재현 검증**(pseudo 금지):
① 초점 목표값 1초내 반영 ② 진짜 연속 AF ③ AF↔MF 전환 ④ **AF 윈도우(영역) 지정** ⑤ AE 자동/수동 전환 ⑥ AWB 자동/수동 전환. 되면 지난 세션의 수동 스윕·gray-world·정착시간 조사가 전부 불필요해짐.

### Phase 3 — ffmpeg 영상 (디버깅 + 다운링크 통합) — ▶ **다음 차례**
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
