---
doc_type: orchestrator_brief
scope: 2026-07-25 야간 세션 이후 — vision 도메인 §9 빌드순서 1~6번 완료 상태에서의 다음 행동
status: ▶ 시작 대기 (2026-07-25 야간 작성, 아직 미착수)
created: 2026-07-25
last_updated: 2026-07-25
---

# 다음 세션 오케스트레이터 브리프

> **다음 세션 진입:** "너는 오케스트레이터이다"로 시작하고 **이 문서 하나만** 읽으면 된다.
> `docs/vision_status.md`(트랙 보드)·`docs/vision_plan.md`는 필요 섹션만 열어라.
> 프로토콜은 메모리 `feedback_orchestrator_protocol` 준수 — **각 트랙은 fg가 아니면 bg,
> 세션 자기보고는 반드시 직접 재현 검증, 한 프롬프트에 몰아던지지 말 것.**
> **서브에이전트 model/effort는 지정하지 않는 게 기본값이다**(2026-07-25 사용자 Max 구독으로
> "소네트로 생성" 제약 폐기). 사용자가 특정 모델을 요청하면 그때 그대로 따른다.
>
> ---
>
> **⚠️ 이 브리프는 "다음 작업을 진행하는" 세션용이다.** 2026-07-25 야간 세션의 결과물에 대한
> **사용자 질의응답·검증 세션은 별도 문서** `docs/vision_verification_qa_brief.md`가 담당한다.
> 두 세션은 병행될 수 있다 — **둘 다 RPi 카메라를 쓰면 충돌하므로**(카메라 배타적, 3절 9번)
> 실기체 작업 전에 상대 세션이 카메라를 쓰고 있는지 확인할 것.

---

## 0. 지금 상태 한 문단

**2026-07-25 야간 세션(사용자 취침 중 자율 진행)에서 vision 도메인의 `docs/vision_plan.md` §9
빌드순서 1~6번이 전부 완료됐다.** ffmpeg Phase 3(H.264 저지연 디버그 스트림), 공통 상태머신
+안전 폴백, ② 초록구역 fine 브랜치(끊어져 있던 체인을 이음), 현장 색 캘리브레이터, 그리고
RPi 실기체 종단간 통합 검증까지. `pytest vision/tests/` **330 → 462 passed**, 전부
`origin/dev--vision-computing-module`에 푸시됨(HEAD `078ddea` 이후).

**남은 §9 항목은 둘뿐이고 둘 다 vision 세션이 지금 할 수 없다:**
- **7번(offboard 정밀착륙 서브상태 연결)** — `fc_ros`/`fc_bridge` 도메인. 루트 `CLAUDE.md`가
  "vision 세션에서 fc_ros/fc_bridge를 건드리지 않는다"고 명시. **FC 세션이 해야 한다.**
- **8번(폐루프 30cm 검증)** — 실측 체커보드 캘리브레이션 필요. **예선 통과 후**로 보류 확정.

경과 상세는 `docs/vision_status.md` 트랙 보드의 2026-07-25 항목들에 전부 있다(재독 불필요,
필요할 때만 열 것).

---

## 1. 🔴 가장 먼저 확인할 것 — 사용자에게 물어라

이 브리프를 쓴 세션은 **사용자가 자는 동안** 돌았다. 아침에 사용자가 확인해야 판단이 서는 게
두 가지 있다. **추측해서 진행하지 말고 먼저 물어라:**

1. **H.264 라이브 스트림을 눈으로 한 번 봐달라.** 지연·fps·디코드는 전부 정량 검증됐지만
   (30fps, 25~41ms, 프레임 100% 디코드) **화질·색·체감 지연을 사람이 본 적이 없다** —
   랩탑(WSL)에 디스플레이가 없고 `ffplay` 설치에 sudo 비밀번호가 필요해 불가능했다.
   ```
   # RPi에서
   cd /home/suri/drone_ws/src/suridoksuri && source /home/suri/local-libcamera/env.sh
   $PICAM_PYTHON -m vision.tools.h264_stream --resolution 1536x864 --af-mode continuous
   # 보는 쪽(디스플레이 있는 PC)에서
   ffplay tcp://100.67.27.83:8082
   ```
2. **다음 우선순위가 무엇인가** — 아래 2절 후보들 중 어디로 갈지. vision 도메인의 계획된
   빌드순서가 끝났으므로 **다음 방향은 사용자 판단 사항**이다.

---

## 2. 다음 행동 후보 (사용자 확인 후 착수)

### 2a. FC 연동(§9 7번) 착수 조율 — **가장 가치 높음, 단 vision 세션 단독으로는 불가**
vision은 이제 `TargetEstimate`(상대 pose)와 상태머신 `command` 문자열 힌트를 실제로 뱉는다.
소비하는 쪽이 없어서 **비행으로 이어지지 않는 상태**다 — "전체 프로세스가 유기적으로 돌아간다"의
마지막 한 칸. 다만 **도메인 격리 규칙상 vision 세션이 `fc_ros`를 건드리면 안 된다.**
→ 사용자에게 "FC 트랙에서 이걸 받아야 한다"고 제안하고, vision 쪽은 **인터페이스 문서만**
정리해 넘기는 것까지가 범위다(루트 `CLAUDE.md` "도메인 간 의존 관계" 절에 이미 예정으로 기록됨).

### 2b. 실촬영 데이터 확보 — **사용자 물리 개입 필요(정지조건)**
지금 검출기 튜닝·골든셋이 **전부 합성 데이터**다. 실제 자갈/조명/그림자에서의 오탐·미탐을
아무도 모른다. 사용자가 타겟을 놓고 촬영할 수 있는 날이면 이게 최우선이다.
상세는 `docs/vision_status.md` "🔴 미실시 항목" 표 1번.

### 2c. 테스트 커버리지 갭 메우기 — 물리개입 불필요, 언제든 가능
`vision/CLAUDE.md` 테스트 규칙표에 `registry`/`illumination`/`denoise`/`edge`/`morphology`/
`background`/`tracker`/`fusion`/`image_loader`/`video_reader`/`visualize`가 **전부 `❌ TODO`**다.
특히 `fusion`(TemporalFusion)·`tracker`(KalmanTracker)는 영상 프리셋의 실제 경로에 있는데
회귀망이 없다. 새 기능은 아니지만 기존 체인의 신뢰도를 올린다.

### 2d. 알려진 소소한 갭들 (전부 기록만 돼 있고 미처리)
- **색 캘리브레이터 마진 기본값이 0** — 산출 범위에 조명 변동 쿠션이 없다. 현장에서는
  `--hue-margin` 등을 명시해야 한다. 기본값을 바꿀지 판단 필요.
- **`no_target/distress_coarse/` 골든 리프가 `generate_synthetic.py`로 재생성되지 않는다**
  (기존 갭, 2026-07-25에 발견만 함).
- **`drift_estimate`가 `tan(HFOV/2)` 항 누락으로 약 1.3배 보수적** — 안전 방향이라 식은
  그대로 뒀다. 실기체 데이터 확보 후 `max_drift_estimate_m` 재튜닝 대상.
- **AF는 "크래시 없이 동작"만 확인** — 실제로 초점이 이동했는지(선명도 지표)는 미검증.
  `rpi_capture.py`/`calib_capture.py`의 초점 스윕 방법론을 재사용하면 확인 가능.
- **`LiveFrameSource`(`utils/frame_source.py`)는 여전히 AF를 안 건드린다** — AF 제어는
  `tools/h264_stream.py`에만 들어갔다. 라이브 파이프라인 경로의 초점은 드라이버 기본 동작에 맡겨져 있다.

---

## 3. 🔴 이 환경에서 반복해서 물리는 함정 (다음 세션이 반드시 알고 시작할 것)

**전부 2026-07-25 야간에 실제로 겪은 것들이다. 다시 조사하지 마라.**

1. **랩탑(WSL2)은 tailscale 피어가 아니다.** WSL 내부는 `172.30.245.10`이고 tailscale은
   **Windows 호스트**에서 돈다. → **RPi에서 랩탑으로 UDP/TCP를 push하는 방식은 NAT에 막혀
   도달하지 못한다.** 네트워크 경로는 항상 **"RPi가 listen, 클라이언트가 connect"** 방향으로
   설계할 것(MJPEG 8080, H.264 8082 둘 다 이 방향이라 동작한다).
2. **랩탑 sudo에는 비밀번호가 필요하다**(RPi는 무암호 sudo 가능). 사용자 부재 중엔 랩탑에
   apt 설치가 불가능하다. ffmpeg CLI 없이 검증해야 하면 `.venv`의 **cv2 5.0.0이 `FFMPEG:YES`**
   이므로 `cv2.VideoCapture("tcp://...")`로 수신·디코드하면 된다.
3. **비대화형 SSH 백그라운드 자식은 SIGINT가 SIG_IGN으로 막힌다**(`nohup`/`setsid`로도 회피 안 됨).
   원격 프로세스 종료·종료검증은 **반드시 `SIGTERM`**. 새로 만드는 장기실행 도구에는 SIGTERM
   핸들러를 반드시 달 것(`tools/h264_stream.py`·`main.py`에 패턴 있음).
4. **SSH로 장기 프로세스를 띄울 때 `&&` 체인 전체가 `&`로 묶이면 SSH 채널을 붙잡아 명령이
   타임아웃난다.** → **런처 스크립트 파일을 하나 만들어** `setsid nohup /path/launcher.sh > log 2>&1 < /dev/null &`
   로 띄울 것. 그리고 heredoc은 SSH 인용 중첩에서 조용히 실패하니 `printf '%s\n' ... > file`이 안전하다.
5. **`pgrep -f <패턴>` / `ps | grep <패턴>`이 자기 자신(SSH로 보낸 명령줄)을 매칭한다.**
   2026-07-25에 이것 때문에 ① "잔존 프로세스 1개"라는 허위 경보 ② **엉뚱한 PID에 SIGTERM을
   보내 자기 원격 셸을 죽이는** 사고가 각각 일어났다. → `ps -eo pid,args --no-headers | grep "[v]ision.main"`
   처럼 **대괄호 트릭**을 쓰고, 죽이기 전에 PID의 `args`를 눈으로 확인할 것.
6. **파괴검증(테스트 일부러 깨뜨리기)을 `set -e` + `| tail`로 돌리면 실패를 놓친다** —
   파이프라인은 마지막 명령의 종료코드만 본다. **종료코드를 변수로 받아 명시적으로 확인할 것.**
   (실제로 "테스트가 버그를 못 잡는다"고 오판할 뻔했다.)
7. **RPi `picam-venv`에 `pytest`가 없다** — 실기체에서 유닛테스트를 못 돌린다. 실기체 검증은
   "실행해서 로그/산출물 확인" 방식이어야 한다. 억지로 설치하지 말 것.
8. **RPi 저장소는 `git pull`이 막혀 있다**(FC 도메인 미추적 비행로그). vision 파일 갱신은
   `cd /home/suri/drone_ws/src/suridoksuri && git fetch origin && git checkout origin/dev--vision-computing-module -- vision/`.
   이후 RPi `git status`에 vision 파일이 M/A로 뜨는 건 **정상**이다.
9. **카메라는 배타적이다** — `main.py live`와 `tools/h264_stream.py`를 동시에 못 띄운다
   (`Device or resource busy`). 검출 보면서 영상도 보려면 `main.py --display stream`(프로세스 1개).

---

## 4. 절대 하지 말 것

- **체커보드 실측 캘리브레이션 재제안 금지** — 사용자가 2026-07-24에 보류 결정했고, 그 뒤로도
  재개하지 않기로 유지됐다. **사용자가 먼저 꺼내기 전까지 제안하지 마라**(메모리
  `project_vision_calibration_deferred`).
- **`fc_ros`/`fc_bridge` 접촉 금지** — 도메인 밖(루트 `CLAUDE.md`).
- **`vision/utils/stream.py`(MjpegStreamer) 폐기 금지** — H.264와 **병행 운용**이 확정 사항이다
  (MJPEG=검출결과 오버레이 관찰 / H.264=카메라 원본 저지연 디버그). 카메라 배타성 때문에
  둘의 역할 분리가 실측으로도 확정됐다(3절 9번).
- **`vision/utils/geo_project.py`** — 폐기 예정(§12), 신규 테스트 금지.
- **`vision/core/state_machine.py`에 타겟별 분기 추가 금지** — "타겟 종류 무관 공통 골격"이
  §9 6번의 핵심 요구다. 타겟 특수성은 `main.py`/`replay.py`의 `_build_observation()`이 흡수한다.

---

## 5. 참조

- `docs/vision_status.md` — 트랙 보드. **맨 위 "🔴 미실시 항목" 표를 먼저 볼 것**(인간 개입이
  필요해 미룬 6가지 — 지우지 말 것).
- `docs/vision_plan.md` §9(빌드순서)·§5.1(상태머신)·§5.3(② 조난자)·§5.5(색 항상성)·§8(FC 통합)
- `docs/vision_camera_bringup.md` — Phase 1~4 전부 완료 상태(Phase 3가 2026-07-25에 닫힘)
- `vision/CLAUDE.md` — 파일역할표·테스트 규칙표(TODO 현황)·import 규칙·각 결정의 근거
- 메모리: `feedback_orchestrator_protocol` · `project_rpi5_ubuntu_camera_stack` ·
  `project_vision_dev_env` · `project_vision_calibration_deferred` · `feedback_ffmpeg_phase3_not_deferred`(**해소됨** — Phase 3 완료)
