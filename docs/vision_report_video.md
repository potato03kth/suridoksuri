---
doc_type: procedure
project: suridoksuri-1
scope: vision 작동 영상 녹화 — 2차예선 보고 제출용
last_updated: 2026-07-29
---

# vision 작동 영상 녹화 절차

**질문:** *"지금 vision부 작동하는 상황 영상녹화 가능한가? FHD/HD면 충분하다."*

**답:** 녹화 기능은 **이미 있다**(`--output`이 오버레이 프레임을 mp4로 쓴다). FHD/HD 둘 다 된다.
🔴 **다만 2026-07-29 현재 RPi 카메라가 하드웨어적으로 인식되지 않아 실촬영만 막혀 있다**(§1).
카메라 없이도 만들 수 있는 경로가 두 개 있다(§3, §4).

---

## 1. 🔴 현재 블로커 — 카메라가 libcamera에 0대로 보인다

2026-07-29 16:5x, RPi(tailscale `100.67.27.83`)에서 확인:

```
$ python -m vision.main live --preset ... --live-resolution 1920x1080 --output ...
IndexError: list index out of range        # Picamera2.global_camera_info()[0]
ConnectionError: LiveFrameSource: 카메라 연결 실패 (camera_num=0), 3/3회 재시도 후 포기.

$ sudo dmesg | grep -i imx708
rp1-cfe 1f00128000.csi: found subdevice /axi/pcie@120000/rp1/i2c@80000/imx708@1a
imx708 11-001a: failed to read chip id 708, with error -5      # ← EIO: I2C 응답 없음
imx708: probe of 11-001a failed with error -5
imx708 10-001a: failed to read chip id 708, with error -5
imx708: probe of 10-001a failed with error -5
```

**해석:** 디바이스 트리에는 센서가 선언돼 있는데(오버레이 정상) **센서가 I2C에 응답하지
않는다.** 즉 소프트웨어 문제가 아니다 — 리본 케이블 접촉/역삽입/단선 또는 모듈 자체 문제다.
`/dev/video19~37`은 전부 ISP/코덱 노드이고 센서 노드는 없다. 마지막으로 카메라가 정상
동작한 기록은 **2026-07-25**(`vision/results/logs/vision.jsonl`, 1205프레임)이며, 그 사이
F2 첫 실비행이 있었다.

**조치(사람만 할 수 있음):**

1. 기체 전원을 끄고 **CSI 리본 케이블을 양쪽(카메라 모듈/RPi) 다 재체결**한다 — 접점 방향
   확인(파란 보강판이 커넥터 클립 쪽), 끝까지 밀어넣고 클립 잠금.
2. **재부팅한다.** 드라이버 프로브는 부팅 시 1회뿐이라 케이블만 다시 꽂아도 재부팅 전엔
   안 잡힌다.
3. 확인:
   ```bash
   sudo dmesg | grep -i imx708          # "failed to read chip id"가 없어야 한다
   source /home/suri/local-libcamera/env.sh
   $PICAM_PYTHON -c "from picamera2 import Picamera2; print(Picamera2.global_camera_info())"
   # 빈 리스트 [] 가 아니라 imx708 항목이 나와야 한다
   ```
   ⚠️ 재부팅 후 tailscale 재연결이 늦을 수 있다(메모리 `project_rpi5_tailscale_wifi_drops`).

---

## 2. 경로 A — 실기체 라이브 녹화 (카메라 복구 후, 가장 좋은 그림)

```bash
ssh suri@100.67.27.83
source /home/suri/local-libcamera/env.sh
cd /home/suri/drone_ws/src/suridoksuri
git fetch origin && git checkout origin/dev--vision-computing-module -- vision/   # 코드 갱신

timeout -s TERM 60 $PICAM_PYTHON -m vision.main live \
  --preset vision/presets/distress_fine.yaml \
  --live-resolution 1920x1080 \
  --calib vision/calibration/cam109-imx708af75-1920x1080/nominal.yaml \
  --report-overlay \
  --output /home/suri/tmp/report_fhd.mp4 \
  --log-dir /home/suri/tmp/report_log
```

- `timeout -s TERM <초>`가 **정상 종료 방식**이다 — `main.py`에 SIGTERM 핸들러가 있어
  카메라 release·mp4 close가 보장된다. 비대화형 SSH 자식은 SIGINT가 SIG_IGN이라 Ctrl+C를
  못 믿는다(`vision/CLAUDE.md` h264_stream 절).
- 종료 시 `Saved: ... (N frames)`와 함께 **실측 fps 경고**가 뜰 수 있다(§5-①). 뜨면 그
  값으로 `--output-fps`를 주고 한 번 더 찍는다.
- 랩탑으로 가져오기: `scp suri@100.67.27.83:/home/suri/tmp/report_fhd.mp4 .`
- ArUco(버티포트) 그림이 필요하면 `--preset vision/presets/vertiport_fine.yaml`
  (+ `--calib`은 같은 해상도 것으로).

### 해상도 / fps / 캘리브레이션 조합 — 반드시 세트로 고른다

| 해상도 | 실측 캡처 속도 | `--calib` | 비고 |
|---|---|---|---|
| 4608×2592 (기본) | **~4.4Hz** (실측, U5) | `cam109-imx708af75/nominal.yaml` | 거리 정확하지만 **영상이 뚝뚝 끊긴다**. 제출 영상엔 비권장 |
| **1920×1080 (FHD)** | 미측정(카메라 사망 전 못 잼) | `cam109-imx708af75-1920x1080/nominal.yaml` | **권장.** 16:9 순수 다운스케일이라 화각 동일 |
| 1280×720 (HD) | 미측정 | `cam109-imx708af75-1280x720/nominal.yaml` | 더 빠른 fps가 필요하면 |
| 1536×864 | ~13.6fps (2026-07-25 실측) | (없음 — 필요하면 생성) | 기존 검증에 쓰던 해상도 |

새 해상도가 필요하면:
```bash
python -m vision.tools.compute_nominal_intrinsics \
  --sensor-width-px 1600 --sensor-height-px 900 --camera-id cam109-imx708af75-1600x900
```
🔴 **`--calib`을 해상도에 안 맞추면 화면의 `DIST` 값이 배수로 틀린다**(§5-②).

### 카메라는 배타적이다

`main.py live`가 도는 중에는 `tools/h264_stream.py`를 못 띄운다(`Device or resource busy`).
검출 오버레이를 **보면서** 녹화하려면 `--display stream`을 같이 주고 랩탑 브라우저에서
`http://100.67.27.83:8080`을 연다(프로세스 1개). `h264_stream.py`는 파이프라인을 안 돌릴 때
쓰는 **카메라 원본** 디버그 도구다.

---

## 3. 경로 B — 외부 촬영 영상을 파이프라인에 통과 (카메라 없이 지금 당장 가능)

**휴대폰으로 실제 타겟(초록 매트 + 흰 박스, 또는 ArUco ID23 출력물)을 찍어** 그 영상을
그대로 파이프라인에 넣으면, 같은 검출기·같은 상태머신이 돌고 오버레이가 얹힌 mp4가 나온다.
"실제 물체를 실제로 인식한다"는 그림이라 합성보다 훨씬 낫다.

```bash
python -m vision.main 촬영본.mp4 --preset vision/presets/distress_fine.yaml \
  --report-overlay --output report_out.mp4 --log-dir results/report_log
```

**주의 3가지:**

1. **면적 필터.** `distress_fine.yaml`의 `rect_detector.min_area/max_area`(14000 /
   2,200,000 px²)는 **고도 15~3m에서 보이는 3m 매트** 기준이다. 가까이서 찍어 매트가 화면을
   꽉 채우면 `max_area`를 넘겨 검출 0이 된다. 축소 모형(예: 초록 종이)을 멀찍이 찍거나,
   임시 프리셋을 복사해 두 값을 조정한다(**yaml 원본은 고치지 말 것** — 프리셋을 복사해서 쓴다).
2. **색.** 실제 초록이 `hue_range: [35, 85]` 밖일 수 있다. 그때는 손으로 추측하지 말고
   `python -m vision.tools.color_calibrate <프레임> --roi x,y,w,h --diagnostic-dir ...`로
   제안값을 받아 복사한 프리셋에 넣는다.
3. **거리(`DIST`) 값은 휴대폰 영상에서 의미 없다** — 우리 카메라의 intrinsics가 아니다.
   보고 영상에 거리 숫자를 쓸 거면 경로 A로 찍어야 한다.

---

## 4. 경로 C — 합성 데이터 (최후 수단)

`vision/results/report_overlay_demo/`가 그 예다(골든셋 1프레임은 완전 재현 가능).
전체가 합성이라 **"실제로 인식한다"는 증거로는 약하다** — 파이프라인·상태머신이 유기적으로
도는 것을 보여주는 용도로만 쓴다.

---

## 5. 함정 (실제로 밟은 것들)

### ① 저장 mp4가 배속으로 재생된다

라이브 `--output`의 mp4 fps는 `_LIVE_DEFAULT_OUTPUT_FPS = 20.0` **고정**이었다. 실측 캡처
속도가 4.4Hz(4608px)나 13.6fps(1536×864)면 저장 영상이 각각 **4.5배·1.5배 빨라진다.**
2026-07-29에 `--output-fps`를 추가했고, 종료 시 실측 평균 fps와 10% 이상 어긋나면 경고한다:

```
WARNING: 기록 fps=20.00 / 실측 캡처 fps=4.41 → --output-fps 4.41 권장
```

**기본값은 일부러 그대로 뒀다** — 기존 녹화물/스크립트의 재생속도가 조용히 달라지면 안 되므로.
제출용은 **경고에 나온 값으로 한 번 더 찍는 것**이 정석이다.

### ② 화면의 `DIST`가 틀린다 (캘리브레이션 해상도 불일치)

`solvePnP`의 focal은 **픽셀 단위**라 프레임 해상도 ≠ 캘리브레이션 해상도면 거리가 그 비율만큼
통째로 틀린다. 실측: 1920px 프레임에 4608px 기준 `nominal.yaml`을 쓰면 **2.4배**로 나왔다
(7.35m 참값 → 17.63m). 저장소는 이걸 **자동 보정하지 않는다**(다운스케일인지 크롭인지 알 수
없으므로). → §2 표대로 `--calib`을 해상도에 맞춘다. 맞추면 오차 0.1% 수준으로 떨어진다.

### ③ 초록 매트 위 초록 박스는 안 보인다

`draw_detections()`의 검출색이 (0,200,0) 초록이라 초록 매트 위에서 배경에 묻혔다. →
`--report-overlay`가 노랑(매트)/하늘색(흰 박스)/**마젠타(착륙점)** + 좌하단 상태 패널을
그린다. 착륙점 색이 마젠타인 이유는 빨강이 이미 sink 경고색(`CONSUMERS 0`)이라 한 프레임에서
같은 색이 다른 뜻을 갖게 되기 때문이다.

### ④ 오버레이는 opt-in이다

`--report-overlay` 없이는 호출 자체가 없어 드론 기본 경로의 산출물이 한 픽셀도 달라지지
않는다. 비행 중 켜지 말 것(프레임마다 그리기 비용이 붙는다).

---

## 6. 참고

- 오버레이 구현/색 근거: `vision/utils/visualize.py::draw_landing_overlay`,
  `vision/CLAUDE.md` "착륙 판단 오버레이" 절
- 산출물 예시 + 재현 명령: `vision/results/report_overlay_demo/README.md`
- 카메라 브링업 이력: `docs/vision_camera_bringup.md`, 메모리
  `project_rpi5_ubuntu_camera_stack.md`
- 실측 발행 주파수(U5, 4.4Hz): `vision/CLAUDE.md` "컨테이너 ROS2 shim 노드" 절
