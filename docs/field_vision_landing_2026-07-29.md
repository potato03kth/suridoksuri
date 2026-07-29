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

---

## 5. 경로 길이 — 짧게 갈 때의 하한

**현장 공간 제약으로 150 m 보다 짧게 간다면 아래를 먼저 본다.**

### 5-1. 필요한 물리 공간 (경로 길이만으론 안 끝난다)

```
축방향 소요 ≈ L + 47m      (역천이가 종점을 약 47m 지나친다, d_end_thresh=10 기준)
횡방향 소요 ≈ ±39.4m       (WP1 중심 탐색 나선 반경 30m + 풋프린트 반폭)
```

🔴 **경로를 줄여도 탐색 반경 30 m 는 안 줄어든다.** 공간이 빠듯하면 경로가 아니라
**탐색 반경을 같이 줄여야** 한다 — 둘 다 launch 인자로 있다:

```
vision_search_radius:=<값>    vision_retry_radius:=<값>
```

| 편도 `L` | 축방향 총 소요 | 비고 |
|---|---|---|
| 150 m (yaml 기본) | 197 m | 지오펜스 224/300 m, 여유 76 m |
| 100 m | 147 m | |
| 80 m | 127 m | |
| 60 m | 107 m | **아래 5-2 위험대** |

### 5-2. ⚠️ 짧은 경로에서 깨지는 것 (F-11 / B7 — 기존 미해결 항목)

- `_FW_LOOKAHEAD = 70.0` 은 `offboard_node.py:102` **하드코딩 상수**다.
  **launch 로 못 바꾼다** — 현장에서 완화할 수단이 없다.
- 개선계획 문서: *"70 m 고정이 **40 m 경로에서 추종 구간을 없앤다**"*
- B7 실측: FOLLOWING 창 상한 = `(L − d_end_thresh) / v_cruise` = `(L − 10) / 18` 초.
  **L=40 이면 1.7 초**, L=60 이면 2.8 초, L=100 이면 5.0 초.
- 직선 편도 경로에서는 lookahead 클램프 자체는 무해하다(종점 방향 = 경로 방향).
  **위험한 것은 FOLLOWING 창이 0 에 수렴해 TRANSITION_FW 직후 바로 역천이로 넘어가는 것.**

**잠정 권고(SITL 스윕 결과 나오기 전):** **편도 100 m 이상**을 권한다.
**80 m 는 경계**, **60 m 이하는 검증 없이 날리지 마라.**

> SITL 세션이 **60/80/100 m 스윕**을 돌리는 중이다. 결과가 나오면 이 절의
> "명시적 하한 숫자"로 교체한다.

---

## 6. 중단 기준 (하나라도 걸리면 vision_landing 을 끄고 날린다)

| | 신호 | 뜻 |
|---|---|---|
| 1 | `/vision/target_status` 가 4.4 Hz 로 안 흐른다 | 체인 미연결. **F2 는 조용히 GPS 폴백** |
| 2 | 이륙 직전 `탐색 최원점` 이 `range_limit_m` 에 근접 | 비행 중 OVERRIDE 로 끊긴다 |
| 3 | HOLD 가 `hold_timeout` 30 s 로 끝난다 | WP1 반경 3.0 m 진입 실패. **F2 align 도 같은 조건에 막힌다** |
| 4 | 이륙지점과 착륙지점의 **지면 높이차가 크다** | AGL 이 이륙지점 기준 **평탄지 가정**이라 그 차이만큼 그대로 틀린다 |

**`vision_landing:=false` 로 떨어뜨리는 것은 언제나 안전한 후퇴다** — 그 경로는 오늘 이전과
코드가 같고 노드 테스트가 고정하고 있다.

---

## 7. 이 비행에서 처음 실기체에 노출되는 값 (전부 SITL 근거뿐)

| 값 | 근거 | 위험 |
|---|---|---|
| 래치 상한 **33.57 m AGL** (자동) | SITL 실측 | 25 m 탐색고도면 항상 통과 — 문제 없을 것 |
| 시한 배수 **1.6** | **SITL 2런뿐.** 핸드오프가 "실비행 근거 아님" 명시 | **바람으로 정렬이 자주 끊기면 짧다** |
| `handoff_agl` **±0.5 m** 허용대 | F2-g, `climbing_reached` 선례와 같은 값 | 3.5 m 에서 AUTO.LAND 인계 |
| `vision_latch_frames` (연속 vision 프레임) | 래치가 제어틱이 아닌 프레임을 센다 | 프레임 드랍 시 래치 지연 |

**이 넷은 오늘 비행이 곧 검증이다.** 로그를 반드시 회수한다.
