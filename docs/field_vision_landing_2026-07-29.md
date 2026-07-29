---
doc_type: field_procedure
project: suridoksuri-1
scope: 2026-07-29 실기체 비행 — vision_landing:=true 현장 명령줄 시트
last_updated: 2026-07-29
---

# 현장 시트 — `vision_landing:=true` 비행 (2026-07-29)

> **이 비행은 F2 정밀착륙을 실기체에서 처음 돌리는 비행이다.**
> 물리 2건(pusher·자기계)은 `docs/preflight_physical_checklist.md` 가 정본이고,
> 이 문서는 **그 뒤에 오는 소프트웨어 기동 순서와 중단 기준**만 다룬다.

🔴 **명령은 손으로 타이핑한다.** 복사하면 U+00A0 가 섞여 뒤 인자가 앞 인자 값으로 흡수된다
(flight01/02 의 원인). 지금은 `_arg()` 가 기동을 거부하므로 조용히 잘못 날지는 않지만,
현장에서 기동 실패로 시간을 버린다.

---

## 0. 오케스트레이터가 이미 확인한 것 (재확인 불필요)

| | 항목 | 결과 |
|---|---|---|
| ⑤ | `fc_ros` 코드 신선도 | ✅ `install/` ↔ `src` md5 일치 (`9400285273eb…`) |
| ⑤-b | **`fc_bridge` PYTHONPATH** (md5로 못 잡는 사각지대) | ✅ `/drone_ws/src/suridoksuri/fc_bridge/__init__.py` 로 로드 |
| — | **F2-g 수정분 실기체 반영** | ✅ `HANDOFF_ALT_TOL_M=0.5`, `handoff_due(3.1, 3.0, False) = True` |
| — | 저장소 리비전 | ✅ RPi `fc8631d` = 개발컴 배포본 |
| — | 토픽·QoS 계약 | ✅ shim ↔ `VisionTargetBridge` 일치, 양쪽 BEST_EFFORT/KEEP_LAST |
| — | shim 기동 | ✅ 컨테이너에서 정상 기동·1Hz 재접속·SIGTERM 정상 종료 |
| — | 호스트 vision 런타임 | ✅ py3.12.3 / cv2 4.13.0 / numpy 2.5.1, `--target-sink` 존재 |

**남은 미검증은 "카메라+mavros 가 붙은 실제 데이터 흐름" 하나뿐이고, 그게 아래 §3 게이트다.**

---

## 1. 터미널 A — 호스트 vision (target-sink 서버)

```bash
ssh suri@100.67.27.83
cd ~/drone_ws/src/suridoksuri
/home/suri/local-libcamera-src/picam-venv/bin/python3 -m vision.main live \
  --preset presets/vertiport_fine.yaml --target-sink
```

- 포트 **8091** 에 TCP 서버가 뜬다. 이미 점유돼 있으면 **종료코드 3** 으로 즉사한다
  (`ss -ltnp | grep 8091` 로 확인).
- `--target-sink` 를 **안 주면 소켓이 안 열리고**, shim 은 영원히 재접속만 한다
  = **F2 가 조용히 GPS 폴백으로 끝난다.** 이 플래그가 이 비행의 전부다.

## 2. 터미널 B — 컨테이너 shim

```bash
ssh suri@100.67.27.83
docker exec fc bash -lc '
  source /opt/ros/humble/setup.bash
  export PYTHONPATH=/drone_ws/src/suridoksuri:$PYTHONPATH
  python3 -m vision.ros.shim_node
'
```

🔴 **`PYTHONPATH` 는 반드시 이어붙인다(`:$PYTHONPATH`).** 덮어쓰면 `import rclpy` 가 즉사한다.

**정상 신호:** 터미널 A 가 떠 있으면 `Connection refused` WARN 이 **멈춘다.**
계속 나오면 A 가 안 떴거나 `--target-sink` 를 안 준 것이다.

## 3. 🔴 터미널 C — 게이트 (여기서 통과 못 하면 vision_landing 을 켜지 마라)

```bash
ssh suri@100.67.27.83
docker exec fc bash -lc '
  source /opt/ros/humble/setup.bash
  source /drone_ws/install/setup.bash
  ros2 topic hz /vision/target_status --use-wall-time
'
```

그리고 별도 터미널에서:

```bash
docker exec fc bash -lc '
  source /opt/ros/humble/setup.bash
  source /drone_ws/install/setup.bash
  ros2 topic echo /vision/landing_setpoint \
    --qos-reliability best_effort --qos-durability volatile
'
```

| 항목 | 합격 | 불합격이면 |
|---|---|---|
| `/vision/target_status` | **약 4.4 Hz** | 체인이 안 붙었다 — §1·§2 재확인 |
| `/vision/landing_setpoint` | **타겟을 보여줬을 때 흐른다** | 타겟 미검출이거나 mavros pose 미수신 |

⚠️ `landing_setpoint` 는 **mavros `/mavros/local_position/pose` 가 있어야** 나온다
(shim 이 매 레코드마다 그 순간 pose 로 절대좌표를 다시 계산한다). FC 스택이 먼저 떠 있어야 한다.

⚠️ **`ros2 topic echo` 는 QoS 를 맞춰야 보인다.** 위 두 플래그 없이 안 보인다고 체인이
죽었다고 판단하지 마라. 그리고 ros2cli 데몬이 죽으면 토픽은 멀쩡한데 CLI 만 전멸한다
(`ros2 daemon stop && ros2 daemon start`).

## 4. 터미널 D — FC launch

```bash
docker exec fc bash -lc '
  source /opt/ros/humble/setup.bash
  source /drone_ws/install/setup.bash
  export PYTHONPATH=/drone_ws/src/suridoksuri:$PYTHONPATH
  ros2 launch fc_ros phase2.launch.py vision_landing:=true \
    waypoints:="[0.0, 0.0, 25.0, <L>, 0.0, 25.0]"
'
```

`<L>` = 편도 거리(m). **§5 를 먼저 읽고 정하라.**

**이륙 직전 로그 한 줄을 반드시 눈으로 확인한다** (F2-e 가 이걸 위해 있다):

```
거리 상한 감시 활성 300m (… 경로 최원점 <L>m, 탐색 최원점 <L+30>m)
```

**`탐색 최원점` 이 300 에 가까우면 `range_limit_m:=<더 큰 값>` 을 같이 준다.**
⚠️ 단 **오늘 형상(편도 150 m 이하)에서는 실측 최대가 112.6 m 라 발동하지 않는다** — §5-1 참조.

---

## 5. 경로 길이 — SITL 스윕 실측 (2026-07-29, 오늘 코드 `44027a5`)

> ⚠️ **이 절은 스윕 결과로 통째로 교체됐다.** 처음에 적었던 "편도 100 m 이상 권장,
> 60 m 이하 위험"은 **틀렸다** — 상태기계는 25 m 에서도 완주했다. 진짜 제약은 공간이다.

편도만 바꿔가며 `vision_landing=true transition_alt=25.0`, `range_limit_m` 기본 300 으로 측정:

| 편도 | FOLLOWING 체류 | 오버슈트 | HOLD 복귀 | 래치 AGL | **총 수평 이탈** | 결과 |
|---|---|---|---|---|---|---|
| 150 m | 8.2 s | 48.1 m | 12.6 s | 24.8 m | **188.0 m** | 통과 |
| 60 m | 2.4 s | 49.3 m | 12.6 s | 24.9 m | **112.6 m** | ✅ 완주 |
| 40 m | 1.0 s | 68.3 m | 17.0 s | 24.9 m | **111.8 m** (횡 −17.2 m) | ✅ 완주 |
| 25 m | **0.2 s** | 73.2 m | 17.6 s | 25.0 m | **102.5 m** | ✅ 완주 |

### 5-1. 🔴 결론 — 편도를 줄여도 공간이 안 줄어든다

**편도를 150 → 25 m 로 6배 줄여도 총 수평 이탈은 188 → 102 m, 절반밖에 안 준다.**
역천이 오버슈트가 **~70 m 에서 포화**하기 때문이다.

> ## 현장 판정 기준
> ### **비행 방향 최소 140 m × 폭 40 m.** 없으면 편도 축소로는 해결되지 않는다.
> 편도 25 m 든 60 m 든 **기체는 이륙지점에서 100~123 m 까지 나간다.**
> 그만한 공간이 없으면 줄일 것은 경로가 아니라 **"FW 로 날지 여부"** 다.

### 🔴 런 간 편차가 크다 — 평균이 아니라 **최악값**으로 잡아라

편도 60 m 를 **3번** 돌렸더니 총 이탈이 **106.4 / 112.6 / 122.6 m** 로 갈렸다(**편차 16 m**).
역천이 오버슈트 자체도 같은 60 m 에서 **49.3 m 와 59.5 m** 로 나왔다.

| 편도 | 관측 총 이탈 | 표본 |
|---|---|---|
| 25 m | 102.5 m | 1 |
| 40 m | 111.8 m | 1 |
| **60 m** | **106.4 / 112.6 / 122.6 m** | **3** |
| 150 m | 188.0 m | 1 |

**⚠️ 현장이 비행 방향으로 딱 120 m 라면 편도 60 m 는 빠듯하다** — 관측 최대가 이미 122.6 m 다.
그 경우 **편도 40 m 이하**로 잡아라(단 **20 m 이하는 금지**, §5-4). 40 m 도 표본이 1건뿐이라
60 m 와 같은 편차를 가정하면 **여유가 10 m 남짓**이다.

- **횡방향도 짧을수록 벌어진다** — 40 m 런에서 **−17.2 m**(60 m 런은 −1.2 m).
- **`range_limit_m` 300 은 어느 길이에서도 안전하다**(실측 최대 122.6 m).
  **현장에서 만질 이유가 없다** — §4 의 "300 에 가까우면 키워라"는 이 조건에선 발동 안 한다.

### 5-2. F-11 / B7 은 실제로 일어났으나 임무를 깨지 않았다

FOLLOWING 창은 예고대로 소멸했다(**8.2 → 0.2 초**). 그런데도 완주한 이유:

- **종점 포착이 결승선 OR 거리원**(`8bbcb94`, F-10)이라 **0.2 초여도 놓치지 않는다.**
- HOLD 가 오버슈트를 `v_approach` 램프로 되돌려 **`hold_timeout=30 s` 를 못 넘겼다**(최악 17.6 s).

⇒ F-11 은 여전히 유효한 결함이지만 **오늘의 블로커는 아니다.**

### 5-3. `transition_alt=25` 가 F2 를 눈에 띄게 쉽게 만든다

네 런 **모두 `VISION_SEARCH` 진입 0.2~0.6 초 만에 래치**했다(24.8~25.0 m, 상한 33.6 m 아래).
정렬·나선·재탐색이 **한 번도 필요 없었다.**

> 🎯 **그러므로 오늘 F2 가 실패한다면 원인은 상태기계가 아니라 검출이다.**
> 현장에서 F2 가 안 먹으면 파라미터를 만지지 말고 **§3 게이트(토픽이 흐르는가)부터** 다시 보라.

### 5-4. 아직 확인 안 된 것

- **편도 20 m 이하 미실시** — 20 m 면 FOLLOWING 잔여거리가 `d_end_thresh=10 m` 아래라
  **0 틱**이 될 수 있다. 진짜 벼랑일 가능성이 있으니 **20 m 이하로는 가지 마라.**
- 편도 80/100 m 미실시(60 m 완주로부터 단조 추정).

> ✅ **F2-d(유도 상실 — shim 자신이 죽는 경우)는 이제 실증됐다**(`V7b`, 오늘 형상).
> shim 이 통째로 죽으면 `link ERROR` 가 **안 나가므로** F2-c 로는 안 잡히는데,
> `vision_lost_timeout=5.0` 이 **Δ5.835 s** 에 메웠다:
> `vision 유도 상실 5s 지속 (setpoint age=6.0s status age=6.0s, AGL 16.4m) → GPS 착륙 폴백`.
> 폴백 직전 2.7 s 는 **고도를 붙들고**(17.7→16.4 m, 추측 하강 금지) 수평만 유지했고,
> 21.4 s 뒤 `DONE`. **호버로 남지 않는다.**

---

## 6. 중단 기준 (하나라도 걸리면 vision_landing 을 끄고 날린다)

| | 신호 | 뜻 |
|---|---|---|
| 1 | `/vision/target_status` 가 4.4 Hz 로 안 흐른다 | 체인 미연결. **F2 는 조용히 GPS 폴백** |
| 2 | 🔴 **비행 방향 140 m × 폭 40 m 를 확보 못 했다** | 편도를 줄여도 안 준다(§5-1). 120 m 뿐이면 **편도 40 m 이하**, 그마저 여유 10 m. 안 되면 **FW 로 안 나는 쪽을 택하라** |
| 3 | HOLD 가 `hold_timeout` 30 s 로 끝난다 | WP1 반경 3.0 m 진입 실패. **F2 align 도 같은 조건에 막힌다**(SITL 최악 17.6 s, 여유 있음) |
| 4 | 이륙지점과 착륙지점의 **지면 높이차가 크다** | AGL 이 이륙지점 기준 **평탄지 가정**이라 그 차이만큼 그대로 틀린다 |
| 5 | 편도 **20 m 이하** | FOLLOWING 0 틱 가능성. **미검증 구간** |

**`vision_landing:=false` 로 떨어뜨리는 것은 언제나 안전한 후퇴다** — 그 경로는 오늘 이전과
코드가 같고 노드 테스트가 고정하고 있다.

> ✅ **이제 전용 SITL 런으로도 확인됐다**(`V8b`, 오늘 형상 60 m/25 m/상한 기본).
> `vision_landing` 을 **아예 주지 않은** 상태에서
> `WP1 도달·안정 (dist=0.6m speed=0.1m/s) → LANDING → DONE`, `exit=0` 119.5 s.
> `VISION_SEARCH`·`PRECISION_LAND` 는 한 번도 안 나오고, 기동 로그에
> `비전 정밀착륙 활성` 줄도 없다. 이륙 전 거리상한 문구도 `경로 최원점 60m` 만 찍고
> `탐색 최원점 …` 항이 빠진다 — **꺼져 있으면 문구·판정이 종전과 동일**하다는 계약대로다.

---

## 7. 이 비행에서 처음 실기체에 노출되는 값 (전부 SITL 근거뿐)

| 값 | 근거 | 위험 |
|---|---|---|
| 래치 상한 **33.57 m AGL** (자동) | SITL 실측 | 25 m 탐색고도면 항상 통과 — 문제 없을 것 |
| 시한 배수 **1.6** | **SITL 2런뿐.** 핸드오프가 "실비행 근거 아님" 명시 | **바람으로 정렬이 자주 끊기면 짧다** |
| `handoff_agl` **±0.5 m** 허용대 | F2-g, `climbing_reached` 선례와 같은 값 | 3.5 m 에서 AUTO.LAND 인계 |
| `vision_latch_frames` (연속 vision 프레임) | 래치가 제어틱이 아닌 프레임을 센다 | 프레임 드랍 시 래치 지연 |

**이 넷은 오늘 비행이 곧 검증이다.** 로그를 반드시 회수한다.
