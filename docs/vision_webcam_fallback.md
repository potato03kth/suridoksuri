---
doc_type: contingency
project: suridoksuri-1
scope: 예비 경로 — 라파(CSI) 카메라를 못 쓰게 될 경우에만 쓰는 USB 웹캠(UVC) 대안
status: 🟡 대기 — 발동되지 않음 (기본 경로는 라파 카메라)
last_updated: 2026-07-31
---

# 【예비】 USB 웹캠(UVC) 경로

## 0. 🟡 이 문서의 지위 — 읽기 전에 이것부터

**기본 경로는 여전히 라파(CSI) 카메라다.** 이 문서는 *"어쩌면 해야 할지도 모르는"* 작업을
미리 준비해 둔 **조건부 대비책**이고, **지금은 발동되지 않았다.**

| | 기본 (현행) | 예비 (이 문서) |
|---|---|---|
| 카메라 | **라파 CSI 카메라(IMX708)** | USB 웹캠(UVC) |
| 스택 | libcamera 로컬 소스빌드 + picam-venv + picamera2 | `cv2.VideoCapture` + V4L2 |
| 코드 경로 | `main.py live` / `LiveFrameSource` | `main.py uvc` / `UvcFrameSource` |
| 캘리브레이션 | `calibration/cam109-imx708af75*/` (이미 있음) | 웹캠용 신규 산출 필요(§4) |
| 상태 | ▶ **이걸로 진행한다** | 🟡 대기 |

### 발동 조건 (아래 중 하나라도 참일 때만 이 문서를 실행한다)

1. CSI 모듈 교체품 조달이 대회 일정에 못 맞는다.
2. 교체품을 꽂아도 `imx708 ... failed to read chip id 708, error -5`가 재현된다
   (= 모듈이 아니라 다른 데 문제가 있고 그 원인 규명에 시간이 더 든다).
3. 사용자가 명시적으로 "웹캠으로 간다"고 결정한다.

**위 중 아무것도 아니면 → 이 문서로 할 일은 없다.** 웹캠을 사러 갈 필요도, `nominal.yaml`을
만들 필요도 없다. 그냥 `docs/vision_report_video.md` §1의 CSI 복구 절차를 따르면 된다.

### 발동 안 해도 코드가 이미 들어가 있는 이유 (비용 0)

`UvcFrameSource` + `main.py uvc`는 **2026-07-30에 미리 구현해 병합해 뒀다.** 급할 때 코드부터
짜면 늦기 때문이고, 그렇게 해도 **기본 경로에 비용이 0**이기 때문이다:

- `uvc`는 `input` 위치인자의 **opt-in 특수값**이다 — 안 쓰면 코드가 한 줄도 실행되지 않는다.
- `_run_live`의 시그니처·동작은 **무변경**이다(프레임 루프를 `_run_source_loop`로 추출해
  공유할 뿐). 기존 CSI 라이브 경로 테스트가 그대로 통과한다.
- `UvcFrameSource`는 `cv2`만 쓰고 picamera2/libcamera를 건드리지 않는다.

즉 **"쓸 일이 없으면 없는 것과 같다."** 그래서 발동 여부와 무관하게 병합 상태로 둔다.

### 🔵 이미 기각된 대안 — 다시 꺼내지 말 것

**휴대폰 IP Webcam**(MJPEG over HTTP). 2026-07-30에 사용자가 **실제로 시험**했고 **화질 부족으로
기각**됐다. 부수 문제도 있었다 — 무선이라 현장 2.4GHz 링크(메모리
`project_rpi5_tailscale_wifi_drops` 항목19의 RF 손실 건)와 대역을 다툰다. USB 웹캠은 그 두
문제가 다 없어서 예비 경로로 이쪽이 남았다.

---

## 배경 (참고)

2026-07-29 RPi CSI 카메라(IMX708)가 I2C 무응답을 냈다(`docs/vision_report_video.md` §1 —
`failed to read chip id 708, error -5`). 리본 재체결·재부팅으로 복구되지 않아 모듈 자체 문제로
판정됐고, **모듈 교체로 기본 경로를 복구하는 것이 1순위**다. 이 문서는 그게 막혔을 때의 2순위다.

---

## 발동 판단 재료 — 무엇이 쉬워지고 무엇을 잃나

### 🔴 잃는 것 (발동을 망설여야 하는 이유)

| 잃는 것 | 크기 |
|---|---|
| **해상도** | 4608×2592 → 웹캠 1920×1080. GSD 약 1.8배 나빠진다 |
| **ArUco 고고도 검출** | 40m에서 38px → **21px**(디코드 한계 밑). 저고도 전용이 된다 |
| **초점 제어** | libcamera 컨트롤 37종(`AfMode`/`LensPosition`/`AfWindows`…) → 싸구려 웹캠은 대개 **고정초점, 컨트롤 없음** |
| **AE/AWB 세밀 제어** | 위와 같음. 웹캠 드라이버가 주는 만큼만 |
| **기존 캘리브레이션 자산** | `calibration/cam109-imx708af75*/` 3종이 **전부 무효** — 화각부터 다시 재야 한다(§4) |
| **실기체 검증 이력** | 2026-07-25 라이브 1205프레임 등 CSI로 쌓은 검증이 **새 카메라엔 적용 안 된다** |
| **마운트** | CSI 리본 → USB 케이블. 나딜 고정 방식을 다시 잡아야 한다 |

**남는 것:** 우선 타겟인 초록매트 3.0m는 웹캠 VGA로도 40m에서 42px라 색 블롭 검출에 충분하다
(§2). 즉 **2차예선 핵심 타겟은 살아남고, ArUco 고고도와 화질 여유를 내주는 거래**다.

### ✅ 쉬워지는 것

UVC 웹캠은 커널이 `/dev/video*`로 바로 잡고 `cv2.VideoCapture`로 열린다. 이 저장소가 카메라
때문에 겪은 것들이 **전부 해당 없다**:

| CSI 경로의 문제 | UVC에서는 |
|---|---|
| `ipa_rpi_pisp.so` 결여 → 소스빌드 | 불필요 |
| `env.sh` source 안 하면 `libcamera` import 실패 | 불필요 |
| picam-venv 분리(호스트 Py3.12) | 불필요 — 랩탑 `.venv`에서도 그대로 돈다 |
| `/dev/mediaN` 번호가 부팅마다 바뀜 | 해당 없음(단, `/dev/video*` 인덱스는 흔들리니 §3 참조) |
| 카메라 배타성(`Device or resource busy`) | 해당 없음 |

> ⚠️ **오해 방지.** `vision/utils/frame_source.py` 모듈 docstring에 *"cv2.VideoCapture는 V4L2 raw
> 경로와 비호환(isOpened()는 되는데 read()가 실패)"* 이라고 적혀 있는데, 그건 **CSI 베이어 raw**
> 경로에 대한 기록이고 UVC와 무관하다. UVC는 드라이버가 디베이어·포맷변환을 끝낸 프레임을
> 주므로 `cv2.VideoCapture`가 정확히 맞는 API다. 그 전례를 근거로 이 경로를 되돌리지 말 것.

---

> ## ⬇ 아래부터는 **발동한 뒤** 따라가는 실행 절차다
> §0의 발동 조건이 성립하지 않았다면 여기서 읽기를 멈춰도 된다.

---

## 1. 웹캠 고를 때 (구매 시점)

1. **박스에 "1080p 30fps"가 적혀 있는지.** USB 2.0에서 FHD 30fps는 **MJPEG 압축이 있어야만**
   나온다. 해상도만 크고 fps 표기가 없거나 낮으면 무압축 YUYV라 FHD에서 5fps급으로 떨어진다.
2. **렌즈 테두리가 돌아가는지(수동 초점링).** 싸구려 웹캠은 대개 고정초점이고 50cm~1m(화상통화
   거리)에 맞춰져 있다. 돌아가는 모델이면 **무한대로 돌려놓고 테이프로 고정**할 수 있다.
3. **어안/초광각(120°+)은 피한다.** nominal 캘리브레이션이 `dist_coeffs: [0,0,0,0,0]`(왜곡 없음
   가정)이라 배럴 왜곡이 심하면 solvePnP가 크게 틀어진다. 표준 화각(60~80°)이 좋다.
4. **클립이 꺾이는 구조**(나딜 마운트).
5. **가능하면 2개.** 카메라 하나가 이미 죽었다.

전원은 문제되지 않는다 — 웹캠은 200~500mA로 Pi5 USB 예산 안이다(폰 충전과 달리).

---

## 2. 해상도가 실제로 중요한 지점

고도별 타겟 픽셀 크기(화각 60° 가정, `2·h·tan(HFOV/2)/width_px`):

| | 10m | 20m | 40m |
|---|---|---|---|
| **초록매트 3.0m** @ VGA 640 | 166px | 83px | 42px |
| **초록매트 3.0m** @ FHD 1920 | 499px | 249px | 125px |
| **ArUco 0.5m** @ FHD 1920 | 83px | 42px | **21px** |
| (참고) CAM109 4608px @ 75° | 901 / 150px | 450 / 75px | 225 / **38px** |

- **초록구역(3m)이 우선 타겟이고, 그건 VGA로도 넉넉하다** — 40m에서도 42px로 색 블롭 검출에
  충분하다. 싸구려 웹캠으로도 핵심 타겟은 커버된다.
- **ArUco(0.5m)는 고도에서 죽는다.** FHD로도 40m에서 21px이면 디코드 한계 밑이다. 다만 ArUco는
  저고도 정밀착륙 단계에서 쓰므로 10m(83px)면 된다. CAM109도 40m에서 38px로 여유롭진 않았다.
- **되도록 FHD로 스트리밍한다** — VGA와 검출 여유가 3배 차이난다.

---

## 3. 꽂고 나서 — 실제로 뭘 샀는지부터 확인

```bash
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video0 --list-formats-ext     # ★ 이게 진실이다
```

`--list-formats-ext`가 **실제 지원하는 (포맷, 해상도, fps) 조합**을 전부 보여준다. 박스 표기가
아니라 이 출력을 믿는다. `MJPG`(또는 `Motion-JPEG`) 항목에 원하는 해상도가 있는지 확인할 것.

**안정 장치경로를 쓴다.** `/dev/video0`의 인덱스는 USB 재연결·부팅 순서에 따라 흔들린다
(이 저장소는 `/dev/mediaN` 번호가 바뀌어 하드코딩이 깨진 전례가 있다 — `rpi_capture.py`
2026-07-22d). `by-id` 경로는 그 문제가 없다:

```bash
ls -l /dev/v4l/by-id/
# → usb-XXXX_Webcam-video-index0 형태. 이 경로를 uvc: 뒤에 그대로 준다.
```

---

## 4. 화각(HFOV) 실측 → `nominal.yaml` 생성

🔴 **이 단계를 건너뛰면 거리·pose가 통째로 틀린다.** 현재 `vision/calibration/`에는
`cam109-imx708af75` 계열(75°)만 있고, 웹캠 렌즈는 화각이 전혀 다르다.

> 체커보드 캘리브레이션 얘기가 **아니다**(그건 2026-07-24에 보류 결정됨). 아래는 5분짜리
> 화각 실측이고, 산출물은 기존 `nominal.yaml`과 같은 등급(`accuracy: unverified`,
> `not_for_closed_loop_30cm: true`)이다.

**측정:** 카메라에서 거리 `D`(줄자)에 폭 `W`인 물건(A4 짧은 변 = 0.210m 등)을 **화면 중앙에**
두고 한 장 찍는다. 그 물건이 가로로 `p` 픽셀을 차지하면:

```
HFOV = 2 · atan( W · width_px / (2 · p · D) )
```

```bash
# 예: A4 짧은변 0.210m 을 1.00m 거리에서 찍었더니 1920px 프레임에서 268px 이었다면
python3 -c "
import math
W, D, p, width_px = 0.210, 1.00, 268, 1920
print(2*math.degrees(math.atan(W*width_px/(2*p*D))))
"   # → 41.6 (예시 숫자다. 반드시 직접 측정할 것)
```

⚠️ 물건을 **화면 가장자리가 아니라 중앙**에 둔다(가장자리는 왜곡이 섞인다). 거리는 렌즈 앞면
기준으로 넉넉히(≥1m) 잡아야 측정 오차가 덜 실린다.

**생성:** 나온 값으로 `nominal.yaml`을 만든다. `--sensor-width-px`/`--sensor-height-px`는
**실제로 스트리밍할 해상도**여야 한다(센서 물리 해상도가 아니다 — 캘리브레이션은 프레임 해상도에
묶인다).

```bash
python -m vision.tools.compute_nominal_intrinsics \
    --sensor-width-px 1920 --sensor-height-px 1080 \
    --hfov-deg <실측값> --hfov-axis horizontal \
    --camera-id webcam-<모델명>-1920x1080
# → vision/calibration/webcam-<모델명>-1920x1080/nominal.yaml
```

해상도를 여러 개 쓸 거면 **해상도마다 하나씩** 만든다(`cam109-imx708af75-1280x720` 등이 그렇게
되어 있다).

---

## 5. 실행

```bash
python -m vision.main uvc:/dev/v4l/by-id/usb-XXXX-video-index0 \
    --preset vision/presets/distress_fine.yaml \
    --calib vision/calibration/webcam-<모델명>-1920x1080/nominal.yaml \
    --uvc-resolution 1920x1080 \
    --report-overlay --output results/webcam.mp4 --output-fps <실측fps>
```

- `uvc` 하나만 주면 인덱스 0(`/dev/video0`), `uvc:2`면 인덱스 2, `uvc:<경로>`면 그 경로.
- `--display stream`을 주면 브라우저로 검출 오버레이를 볼 수 있다(`http://<host>:8080`).
- `--output-fps`는 종료 시 찍히는 **실측 캡처 fps** 경고를 보고 맞춘다 — 안 맞으면 저장된 mp4가
  배속으로 재생된다(`docs/vision_report_video.md`).
- `--live-retries` / `--live-retry-delay`가 웹캠 모드에도 그대로 적용된다(같은 노브).

### 초점

싸구려 웹캠은 대개 고정초점이라 컨트롤 자체가 없다. 있으면:

```bash
--uvc-autofocus off --uvc-focus <값>     # 단위는 드라이버마다 달라 실측으로 찾는다
```

컨트롤이 없으면 경고 한 줄(`웹캠 초점 컨트롤 미적용`)만 남기고 그대로 진행한다 — 초점 하나
때문에 파이프라인을 죽이지 않는다(`LiveFrameSource`의 AF 실패 처리와 같은 철학).

---

## 6. 🔴 가장 위험한 함정 — 해상도가 조용히 무시된다

싸구려 웹캠은 지원하지 않는 해상도를 요청받으면 **에러 대신 가장 가까운 해상도를 조용히
돌려준다.** `cap.set()`은 그래도 True를 반환한다. 그런데 `nominal.yaml`의 `camera_matrix`는
해상도에 묶여 있어서, **요청 1920 / 실제 640이면 solvePnP 거리·pose가 소리 없이 3배 틀린다.**

`UvcFrameSource`는 여는 즉시 실제 해상도를 되읽어 **다르면 하드 실패**한다:

```
UvcFrameSource: 요청 해상도 1920x1080 가 적용되지 않았다 — 실제 640x480. ...
  1) 지원 조합 확인:  v4l2-ctl -d <device> --list-formats-ext
  2) 그 해상도용 캘리브레이션 생성:  python -m vision.tools.compute_nominal_intrinsics ...
  3) 그 뒤에 --uvc-allow-resolution-mismatch 로 진행
```

`--uvc-allow-resolution-mismatch`는 **실제 해상도용 `nominal.yaml`을 먼저 만든 뒤에만** 쓴다.
안 그러면 이 안전장치를 끄는 것 이상의 의미가 없다.

### 그 외 챙길 것

- **자동노출.** 햇빛 아래 나딜 촬영이면 매트가 날아갈 수 있다. 색 검출은 HSV Hue 기반이라
  노출에 어느 정도 강하지만 하이라이트가 클리핑되면 채도가 죽어 실패한다 → 현장에서
  `python -m vision.tools.color_calibrate`를 한 번 돌릴 것(정확히 이런 상황용 도구다).
- **stale frame.** 파이프라인이 카메라보다 느리면 V4L2 큐에 프레임이 쌓여 낡은 그림을 본다.
  `UvcFrameSource`가 `CAP_PROP_BUFFERSIZE=1`을 best-effort로 걸어 두지만, 백엔드가 무시할 수
  있다. 지연이 의심되면 해상도를 낮춰 파이프라인을 카메라보다 빠르게 만든다.

---

## 7. 이 경로로 되는 것 / 안 되는 것

**된다:**
- 실물 초록매트/ArUco 실시간 검출 — 2차예선 제출 영상
- *"🔴 실카메라 라이브 경로를 한 번도 못 돌렸다"* 미실시 항목 해소
- 상태머신 · `--target-sink` · shim 종단 검증(지상)

**안 된다:**
- **폐루프 30cm 정밀착륙**(`docs/vision_plan.md` §9 8번). nominal 캘리브레이션이라
  `not_for_closed_loop_30cm: true`가 그대로 붙어 나가고, 롤링셔터·고정초점도 예산을 갉아먹는다.
- 다만 §9 8번은 **원래도 실측 캘리브레이션이 전제라 예선 이후 항목**이다 — 이 경로 때문에
  잃는 것은 없다.

---

## 8. 발동했다면 — 문서 상태를 바꿔라

이 문서를 실제로 실행하게 됐다면 **다음 세션이 헷갈리지 않도록** 아래를 같이 갱신한다
(안 하면 "기본은 CSI"라고 적힌 문서를 보면서 웹캠으로 날게 된다):

1. 이 파일 front matter — `status: 🟡 대기` → `🔴 발동 (YYYY-MM-DD, 사유)`, §0 표의 "상태" 행.
2. `docs/vision_status.md` — vision 트랙 보드의 웹캠 항목을 "예비" → "활성"으로.
3. `vision/CLAUDE.md` "UVC(USB 웹캠) 프레임 소스" 절 머리말.
4. 메모리 `project_rpi5_ubuntu_camera_stack.md`(+ `MEMORY.md` 인덱스 줄).
5. `--calib` 기본값(`main.py::_DEFAULT_CALIB_PATH`)이 여전히 `cam109-imx708af75`를 가리킨다 —
   웹캠으로 상시 운용한다면 실행할 때마다 `--calib`을 주든지 기본값을 바꾸든지 정해야 한다.

---

## 9. 관련 문서

- `docs/vision_report_video.md` — 보고 제출용 영상 녹화 절차(정본). §1이 CSI 카메라 증상·복구 절차.
- `docs/vision_status.md` — vision 라이브 트랙 보드.
- `vision/CLAUDE.md` "UVC(USB 웹캠) 프레임 소스" — 구현 근거·설계 판단.
