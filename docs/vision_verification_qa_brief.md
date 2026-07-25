---
doc_type: orchestrator_brief
scope: 2026-07-25 야간 세션 산출물에 대한 사용자 질의응답·눈으로 확인 지원 세션
status: ▶ 시작 대기 (2026-07-25 작성)
created: 2026-07-25
last_updated: 2026-07-25
---

# 검증·질의응답 오케스트레이터 브리프

> **진입:** "너는 오케스트레이터이다"로 시작하고 **이 문서 하나만** 읽으면 된다.
> 프로토콜은 메모리 `feedback_orchestrator_protocol` 준수 — **세션 자기보고는 반드시 직접 재현
> 검증, fg가 아니면 bg.** **서브에이전트 model/effort는 지정하지 않는 게 기본값**(2026-07-25
> 사용자 Max 구독으로 "소네트로 생성" 제약 폐기). 사용자가 요청하면 그대로 따른다.

---

## 0. 이 세션의 성격 — 새 기능 개발이 아니다

**2026-07-25 야간 오케스트레이터 세션(사용자 취침 중 자율 진행)의 산출물을 사용자가 직접 눈으로
확인하는 것을 돕는 세션이다.** 사용자가 직접 실행해 보다가 막히는 것, "이게 왜 이러냐"는 질문에
답하고, 필요하면 현장에서 고친다.

**다음 작업(FC 연동 등)을 진행하는 세션은 별도**다 — `docs/vision_next_session_brief.md`.
**둘 다 RPi 카메라를 쓰면 충돌한다**(카메라 배타적) — 실기체 작업 전에 상대 세션이 카메라를
쓰고 있는지 `ps -eo pid,args --no-headers | grep -E "[v]ision.main|[h]264_stream"`로 확인할 것.

**야간 세션이 한 일 요약(상세는 `docs/vision_status.md` 트랙 보드 2026-07-25 항목):**
ffmpeg Phase 3(H.264 스트림) · 공통 상태머신 · ② 초록구역 fine 브랜치 · 현장 색 캘리브레이터 ·
RPi 실기체 통합 검증. `pytest vision/tests/` 330 → **462 passed**. HEAD `d067525`.

---

## 1. 🔴 사용자가 이미 제기한 질문 — "MJPEG 스트림에 ArUco 인식 박스가 안 뜬다"

### 가장 유력한 원인: **오케스트레이터의 브리핑 지시가 틀렸다** (코드 확인 완료)

야간 세션이 사용자에게 아래 명령을 안내했다:
```
$PICAM_PYTHON -m vision.main live --preset vision/presets/distress_fine.yaml --display stream ...
```
그런데 **`distress_fine.yaml`에는 `aruco_detector` 스텝이 아예 없다.** 실제 내용:
```yaml
pipeline:
  color_filter: {mode: color, hue_range: [35, 85], ...}   # ← 초록색만 통과시킴
  morphology: {...}
  rect_detector: {...}
  white_box_detector: {...}
```
즉 **②초록구역 전용 파이프라인**이라, ArUco 마커(흑백)를 들이대면 `color_filter`에서 전부
걸러져 **검출이 0건이고 박스도 당연히 안 뜬다.** 마커 문제가 아니라 **프리셋이 틀린 것.**

**ArUco를 보려면 `vertiport_fine.yaml`을 써야 한다:**
```yaml
pipeline:
  aruco_detector:
    valid_ids: [23]
```
```bash
# RPi에서
cd /home/suri/drone_ws/src/suridoksuri && source /home/suri/local-libcamera/env.sh
$PICAM_PYTHON -m vision.main live --preset vision/presets/vertiport_fine.yaml \
    --display stream --live-resolution 1536x864
# 보는 곳: 브라우저로 http://100.67.27.83:8080/
```

### ✅ 이 진단은 추측이 아니라 실측으로 확인됐다 (2026-07-25, 합성 마커 3케이스)

합성 ArUco 마커(`DICT_4X4_50`)를 만들어 세 경로를 **실제로 실행**해 재현했다:

| 케이스 | 결과 | `state.meta["aruco"]` |
|---|---|---|
| **ID=23 + `vertiport_fine.yaml`**(올바른 경로) | **detections=1**, bbox=(570,230,399,399), `chosen.target_estimate` **있음** | `{confirmed: 1, rejected: 0}` |
| ID=7 + `vertiport_fine.yaml`(화이트리스트 밖) | detections=0 | `{confirmed: 0, **rejected: 1**}` |
| **ID=23 + `distress_fine.yaml`**(사용자가 쓴 경로) | **detections=0** | **키 자체가 없음**(스텝 미존재) |

**→ 사용자가 겪은 증상은 세 번째 줄과 정확히 일치한다.** 그리고 올바른 프리셋에서는 박스가
실제로 그려진다는 것이 첫 줄로 확인됐다.

**`state.meta["aruco"]`가 진단의 결정적 지표다:**
- **키 자체가 없음** → 프리셋에 `aruco_detector` 스텝이 없다(= 프리셋을 잘못 골랐다).
- `rejected > 0` → 마커는 보이는데 **ID가 다르다**.
- `confirmed == 0, rejected == 0` → **마커를 아예 못 봤다**(초점/크기/조도 → 아래 3~5번).

### 그래도 안 뜨면 — 이 순서로 좁혀라 (전부 코드로 확인된 사실 기반)

1. **마커 ID가 23인가?** `ArucoDetector(valid_ids=(23,))`가 기본이고 **다른 ID는 전부 거절**한다
   (`vision/modules/aruco.py:34`). 거절은 조용히 버려지지 않고 **`state.meta["aruco"]["rejected"]`
   카운트로 남는다** — 이게 진단의 핵심이다:
   - `rejected > 0` → **마커는 보이는데 ID가 다르다.** 사용자에게 ID 23 마커를 쓰라고 하거나,
     테스트 목적이면 `--preset`으로 임시 프리셋(다른 valid_ids)을 만들어 확인.
   - `rejected == 0` **이고** `confirmed == 0` → **마커 자체가 검출이 안 되고 있다**(아래 2~4번).
   - 확인 방법: 실행 로그(`--log-dir`)의 JSONL/`.log`를 보거나, 짧게 `--display none`으로
     돌려 `state.meta`를 확인. **추측하지 말고 이 숫자를 먼저 볼 것.**
2. **사전(dictionary)이 `DICT_4X4_50`인가?** `vision/modules/aruco.py:18`이
   `cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)`로 고정돼 있다. 온라인 생성기에서
   기본값(`DICT_6X6_250` 등)으로 뽑았으면 **절대 인식 안 된다.** 대회 규정도 `DICT_4X4_50` ID23이다.
3. **초점.** `LiveFrameSource`(라이브 파이프라인 경로)는 **AF를 전혀 제어하지 않는다**(알려진 갭,
   `docs/vision_next_session_brief.md` §2d). 카메라 드라이버 기본 동작에 맡겨져 있어 근접 마커가
   흐릴 수 있다. **판별법:** `tools/h264_stream.py --af-mode continuous`로 원본 스트림을 띄워
   **눈으로 초점이 맞는지 먼저 확인**하고, 맞는데도 검출이 안 되면 초점 문제가 아니다.
4. **크기/거리.** 규정 마커는 50cm×50cm이고 파이프라인 가정도 그렇다. A4에 뽑은 작은 마커를
   멀리서 들면 픽셀이 부족하다. **가까이(1m 이내) 크게** 들이대고 다시 볼 것.
5. **조도.** 새벽 실측 프레임 밝기 평균이 **0.3/255**였다(거의 암흑). 불을 켜고 볼 것.

### 참고 — 과거에 같은 혼동이 있었다 (재발 주의)
2026-07-24에 사용자가 스트림의 **초록 사각형을 ArUco 인식으로 오해**한 적이 있다. 그때는
`single_frame.yaml`(범용 흑백 사각형 검출기 `RectDetector`)을 쓰고 있었고, 초록 박스는
`draw_detections()`가 그린 **일반 사각형 후보**였지 ArUco가 아니었다.
**`draw_detections()`는 프리셋이 무엇이든 `state.detections`를 똑같이 초록 사각형으로 그린다** —
**박스 색만 보고 "ArUco가 인식됐다"고 판단할 수 없다.** ArUco 여부는 `state.meta["aruco"]`
또는 JSONL의 `chosen.target_estimate` 유무로 판별해야 한다.

---

## 2. 사용자가 눈으로 확인하려는 것들 — 실행법과 기대 동작

### A. MJPEG 검출 오버레이 (설치 0, 브라우저만) — **"인식이 되나"를 보는 용도**
```bash
# RPi
cd /home/suri/drone_ws/src/suridoksuri && source /home/suri/local-libcamera/env.sh
$PICAM_PYTHON -m vision.main live --preset <프리셋> --display stream --live-resolution 1536x864
```
- 보는 곳: **Windows 브라우저** `http://100.67.27.83:8080/`
- 프리셋 선택: **ArUco/버티포트 → `vertiport_fine.yaml`** / **초록구역 → `distress_fine.yaml`**
  / 아무 사각형이나 → `single_frame.yaml`
- 기대: 카메라 영상 위 **초록 사각형**(검출 후보) + **빨간 사각형 "CONFIRMED"**(시간융합 확정).
- 종료: Ctrl+C (또는 다른 셸에서 `kill -TERM <PID>` — SIGTERM도 정상 종료됨, 야간 세션에서 추가)

### B. H.264 원본 저지연 스트림 — **"카메라 원본/초점"을 보는 용도**
```bash
$PICAM_PYTHON -m vision.tools.h264_stream --resolution 1536x864 --af-mode continuous
```
- 보는 곳(디스플레이 있는 PC): **VLC** → 미디어 → 네트워크 스트림 열기 → `tcp://100.67.27.83:8082`
  (또는 `ffplay -fflags nobuffer tcp://100.67.27.83:8082`)
- **랩탑 WSL로는 볼 수 없다** — 디스플레이 없음 + ffplay 미설치(sudo 비밀번호 필요).
- 기대: 30fps 원본 영상, 지연 수십 ms. 검출 박스는 **없는 게 정상**(카메라 원본 전용으로 확정).
- **⚠️ A와 B는 동시에 못 켠다**(카메라 배타적, `Device or resource busy`).

### C. 상태머신·지연 그래프
```bash
python vision/tools/jsonl_view.py <로그>.jsonl --output out.png --x-axis frame_id
```
- `<로그>`는 `main.py ... --log-dir <폴더> --log-name <이름>` → `<폴더>/<이름>.jsonl`
- 이미 만들어진 예시(커밋됨): `vision/results/distress_fine_nominal/demo_state.png`(정상 착륙),
  `vision/results/distress_fine_demo/demo_state.png`(검출 상실→재상승 안전망),
  `vision/results/state_machine_demo/demo_state.png`(ArUco 경로)

### D. 현장 색 캘리브레이터
```bash
python -m vision.tools.color_calibrate <이미지|녹화폴더|영상> --roi x,y,w,h \
    --hue-margin 10 --sat-margin 20 --val-margin 20 \
    --diagnostic-dir vision/results/color_calib
```
- **`python -m` 으로 실행해야 한다**(`vision.*`를 import하므로 직접 경로 실행은 `ModuleNotFoundError`).
- 기대: 복붙 가능한 **yaml 조각을 stdout에 출력**(자동으로 preset을 고치지 않는다) +
  진단 PNG(ROI 오버레이·HSV 히스토그램).
- **⚠️ 마진 기본값이 0이라 조명 변동 쿠션이 없다** — 현장에선 위처럼 마진을 명시할 것.

---

## 3. 사용자에게 아직 확인받지 못한 것 (물어볼 것)

1. **H.264 스트림 화질/체감지연 육안 확인** — 정량(30fps·25~41ms·프레임 100% 디코드)은 검증됐으나
   **사람이 본 적이 없다.** cv2로만 디코드했기 때문에 실제 플레이어와 체감이 다를 수 있다.
   → 2절 B로 확인 요청. 확인되면 `docs/vision_status.md` "미실시 항목" 표 3번을 해소로 갱신할 것.
2. **실제 타겟(초록 매트/ArUco)으로 검출이 되는지** — 지금까지 전부 합성 데이터 + 소등 헤드리스라
   **실물로 검출된 걸 아무도 못 봤다.** 1절이 이것과 직결된다.

---

## 4. 자주 물릴 함정 (야간 세션이 실제로 다 겪은 것 — 재조사 금지)

1. **랩탑(WSL2)은 tailscale 피어가 아니다**(WSL 내부 `172.30.245.10`, tailscale은 Windows 호스트).
   → RPi에서 랩탑으로 **push하는 방식은 NAT에 막힌다.** 항상 **"RPi가 listen, 보는 쪽이 connect"**.
2. **랩탑 sudo는 비밀번호 필요**(RPi는 무암호). 사용자 부재 중 랩탑 apt 설치 불가 →
   스트림 검증은 `.venv`의 cv2(`FFMPEG:YES`)로 `cv2.VideoCapture("tcp://...")`.
3. **비대화형 SSH 백그라운드 자식은 SIGINT가 SIG_IGN으로 막힌다** → 종료는 **SIGTERM**.
4. **SSH로 장기 프로세스를 띄울 때 `&&` 체인이 통째로 `&`로 묶이면 SSH 채널을 붙잡아 타임아웃난다**
   → **런처 스크립트 파일**을 만들어 `setsid nohup /path/launcher.sh > log 2>&1 < /dev/null &`.
   heredoc은 SSH 인용 중첩에서 조용히 실패하니 `printf '%s\n' ... > file`이 안전.
5. **`pgrep -f` / `ps | grep`이 자기 자신(SSH 명령줄)을 매칭한다** — 야간 세션에서 이것 때문에
   ①허위 "잔존 프로세스" 경보 ②**엉뚱한 PID에 SIGTERM을 보내 자기 원격 셸을 죽이는** 사고가 났다.
   → `ps -eo pid,args --no-headers | grep "[v]ision.main"` **대괄호 트릭**을 쓰고, 죽이기 전 args 확인.
6. **파괴검증을 `set -e` + `| tail`로 돌리면 실패를 놓친다**(파이프는 마지막 명령의 종료코드만 봄)
   → 종료코드를 변수로 받아 명시 확인.
7. **RPi `picam-venv`에 `pytest`가 없다** — 실기체 유닛테스트 불가. 억지 설치 말 것.
8. **RPi 저장소는 `git pull`이 막혀 있다**(FC 미추적 비행로그) →
   `cd /home/suri/drone_ws/src/suridoksuri && git fetch origin && git checkout origin/dev--vision-computing-module -- vision/`.
   이후 RPi `git status`에 vision 파일이 M/A로 뜨는 건 **정상**.
9. **카메라는 배타적이다** — `main.py live`와 `h264_stream.py` 동시 실행 불가.

---

## 5. 절대 하지 말 것

- **체커보드 실측 캘리브레이션 재제안 금지**(사용자 2026-07-24 보류 결정, 먼저 꺼내기 전까지).
- **`fc_ros`/`fc_bridge` 접촉 금지**(도메인 밖). FC 연동은 다른 세션·다른 도메인 소관.
- **`vision/utils/stream.py`(MjpegStreamer) 폐기 금지** — H.264와 **병행 운용** 확정.
- **`vision/core/state_machine.py`에 타겟별 분기 추가 금지** — "타겟 종류 무관 공통 골격"이 핵심 요구.
- **사용자 질문에 추측으로 답하지 말 것** — 이 세션의 본질은 "사용자가 본 것"과 "코드의 사실"을
  맞추는 일이다. 모르면 **코드를 열고 실제로 돌려서** 확인한 뒤 답한다.

---

## 6. 참조

- `docs/vision_status.md` — 트랙 보드. **맨 위 "🔴 미실시 항목" 표**(인간 개입 필요 6가지, 지우지 말 것)
- `docs/vision_next_session_brief.md` — **다음 작업 진행 세션용**(이 문서와 역할 분리)
- `vision/CLAUDE.md` — 파일역할표·테스트 규칙표·각 결정의 근거(특히 프리셋별 역할)
- `docs/vision_plan.md` §5.1(상태머신)·§5.2(버티포트)·§5.3(② 조난자)·§5.5(색 항상성)
- 메모리: `feedback_orchestrator_protocol` · `project_rpi5_ubuntu_camera_stack` · `project_vision_dev_env`
