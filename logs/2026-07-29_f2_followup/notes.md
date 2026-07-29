---
doc_type: sitl_run_log
project: suridoksuri-1
track: 🎯 fc-정밀착륙 (F2)
scope: F2 후속 결함 a~g SITL 검증 런 기록
created: 2026-07-29
---

# F2 후속 결함 SITL 검증 — 런 기록

대상 커밋: `db7aa9c`(a~f) · `3d73bfb`(g).
환경: WSL `Ubuntu-22.04` / **격리 클론 `/root/ws_f2b`** / `PX4_DIR=/root/PX4-vehicle`
(`c890d9db0a` + F-17/F-4 + `crsf_rc` 패치, `make px4_sitl_default` = `ninja: no work to do`
로 **stale 아님을 확인**했고 바이너리 md5 는 재빌드 전후 동일 `4d717ab5…`).

> 🔴 **`PYTHONPATH` 를 격리 클론으로 앞에 붙여야 한다.** 전역
> `/usr/local/lib/python3.10/dist-packages/easy-install.pth` 가
> `/root/drone_ws/src/suridoksuri/fc_bridge` 를 가리키고 있어서, 안 그러면
> **`fc_bridge`(= `precision_land.py` 의 F2 판정)가 옛 클론에서 로드된다.**
> `colcon build` 가 옮기는 것은 `fc_ros` 뿐이라 이 함정은 md5 대조로도 안 잡힌다.
> 실행 래퍼: `/mnt/c/sitl7_xfer/f2b_run.sh`.

## 런 요약

| 런 | 설정 | 결과 | 무엇을 보여주나 |
|---|---|---|---|
| `V1_vision_off` | `vision_landing` 기본(false), `range_limit_m` 기본 300 | ARM→CLIMBING→TRANSITION_FW→STREAMING→FOLLOWING→TRANSITION_MC→**OVERRIDE**→DONE | 종전 경로 정상. OVERRIDE 는 F2 와 무관한 **기존** 거리상한 발동(300m 경로 + 역천이 오버슈트) — 아래 F2-e 참조 |
| `V1b_vision_off` | 위 + `range_limit_m=1200` | ❌ `CLIMBING 타임아웃 120s` → exit 2 | **환경 문제**(직전 런의 gz 잔류, 런 간격 8초). 코드와 무관 |
| **`V2_vision_happy`** | `vision_landing=true`, `range_limit_m=1200`, fake_vision 정상 발행 | ✅ **HOLD→VISION_SEARCH→PRECISION_LAND→LANDING→DONE 완주** (195.3s) | **a·b·e 실증 + g 발견** (아래) |
| `V3`/`V4`/`V5` | — | ❌ `px4_dead`·`mavros_not_connected` (exit 3) | **다른 세션(`/root/ws_c1b`)이 같은 배포판에서 SITL 캠페인 중** — 아래 "단일 테넌트" |
| `V6` | — | ⏸ **미실행** | 위와 같은 이유. 대기 큐가 `RACED_WITH_OTHER_SESSION` 으로 물러났다 |

## `V2_vision_happy` — 실측 증거

```
[  0.0s] 비전 정밀착륙 활성 — … / 래치 상한 33.6m AGL (검출 상한 자동)
[  1.6s] 거리 상한 감시 활성 1200m (… 경로 최원점 300m, 탐색 최원점 330m (=WP1 300m + 반경 30m))
[ 81.1s] 래치 보류 — AGL 50.0m 가 래치 상한 33.6m 초과 …
[ 86.2s] 래치 보류 — AGL 42.3m 가 래치 상한 33.6m 초과 …
[ 91.2s] 래치 보류 — AGL 34.7m 가 래치 상한 33.6m 초과 …
[ 92.5s] 타겟 래치 (연속 3프레임, N=299.8 E=-0.2) → PRECISION_LAND
[ 92.5s] 정밀착륙 진입 — 래치 AGL 32.8m, 하강예산 37s → 시한 60s (하한 60s)
[131.6s] 정밀착륙 AGL=3.4m …            ← 여기까지 0.775 m/s 로 정상 하강 (38s)
[133.6s] 정밀착륙 AGL=3.1m …
   ⋮        (3.1m 에서 22초간 정지)
[154.6s] 정밀착륙 타임아웃 60s 초과 (수평오차 0.48m, AGL 3.1m) → GPS 착륙 폴백
```

- **F2-a ✅** 래치가 AGL 50.0/42.3/34.7m 에서 **3회 거부**되고 상한 아래
  **32.8m 에서 성립**했다. 검증 세션 `R5` 는 같은 지점에서 **49.6m 에 래치**했다.
- **F2-b ✅** 시한이 래치 고도에서 재계산된다(`하강예산 37s → 시한 60s`).
  이 런은 32.8m 래치라 하한(60s)이 이겼다. 49.6m 래치면 93.2s 가 된다(실기체에서
  직접 호출해 확인 — `docs/rpi_deploy.md` 검증 절차 참조).
- **F2-e ✅** 경고/정보 문구가 **탐색 최원점 330m** 을 따로 계산해 찍는다.
  `V1_vision_off`(vision 꺼짐)에서는 종전대로 `경로 최원점 300m` 만 찍는다 —
  **꺼져 있을 때 문구·판정이 무변경**이라는 계약이 지켜진다.
  ⚠️ 같은 런의 `range_guard.max_horiz_m` 이 **343.5m** 다 — 역천이 오버슈트만으로
  기본 상한 300m 를 넘는다. **단 이건 A1(직선 300m) 시나리오 한정**이다:
  대회 경로는 폐회로라 마지막 WP 가 출발점 근처이므로 `d(WP1) + 탐색반경` 은
  작다. 위험한 것은 **직선/편도 경로에서 `vision_landing:=true`** 를 켜는
  경우이고, 그때는 `range_limit_m` 을 같이 키워야 한다. 어느 쪽인지는 이제
  이륙 직전 로그 한 줄이 말해 준다(그게 F2-e 의 목적이다).
- **F2-g 🔴 여기서 새로 발견** — 위 3.1m 22초 정지. `_pl_alt` 하한이
  `handoff_agl`(3.0m)과 같은 숫자라 기체는 그 위에 정착하고, 단측 판정
  `agl <= 3.0` 은 **원리적으로 성립할 수 없다.** `R5` 가 못 본 이유는 그 런의
  fake_vision 이 `command_hint="land"` 를 42s 에 냈기 때문이다(advisory 경로).
  ⇒ 커밋 `3d73bfb` 에서 허용대(±0.5m, `climbing_reached` 선례와 같은 값)로 수정.

## 🔴 이 배포판의 SITL 은 단일 테넌트다 (2026-07-29 실측)

`px4`/`mavros` 가 **14540/14580 을 고정**으로 쓴다. 다른 세션(`/root/ws_c1b`,
F-10 재검증 캠페인)이 돌고 있는 동안 내 런은 전부 `px4_dead`(exit 3)로 죽었다
(`V3`/`V4`/`V5`). 두 세션이 겹치면 **둘 다** 못 산다.

재시도 이력: 남의 캠페인이 비는 창을 기다렸다가 `V3` 를 띄웠지만, 실행 도중
상대가 다음 시나리오를 시작해 포트를 가져가면서 `mavros_not_connected` 로 죽었다.
그 다음 `V5` 는 대기 큐가 상대 런을 감지하고 **스스로 물러났다**
(`RACED_WITH_OTHER_SESSION`) — 겹치면 **남의 런도 같이 죽기** 때문이다.
⇒ **동시 세션 환경에서 이 박스의 SITL 은 선착순 배타 자원으로 다뤄야 한다.**

> 🔴 **더 나쁜 것:** `tools/sitl/run_scenario.py` 의 정리 루틴이
> **`pkill -f "gz sim"` 을 시스템 전역으로** 부른다(세션이 하나뿐이던 시절의 코드).
> 즉 내 런이 끝날 때마다 **남의 gz 가 같이 죽는다.** 이번에는 내 클론에만
> 로컬 가드를 넣어(다른 세션 `run_scenario.py` 가 떠 있으면 pkill 을 건너뛴다)
> 피했지만, **공용 파일에는 아직 그대로 남아 있다** — 동시 세션이 상시화된
> 지금은 상류 수정 대상이다.

## 미실시 (정직하게)

| 항목 | 상태 |
|---|---|
| F2-c 생산자 사망 SITL 실증 (`V5`) | ❌ 박스 점유로 미완. **노드 테스트(`_step_precision_land` 를 실제로 실행) + 파괴검증**으로만 확인 |
| F2-f align 타임아웃 SITL 실증 (`V6`) | ❌ 위와 같음. 재현 인자는 준비돼 있다(아래) |
| F2-d 유도 상실 SITL 실증 | ❌ 위와 같음 |
| `vision_landing=false` 완전 완주(HOLD→LANDING) | ⚠️ `V1` 은 거리상한 OVERRIDE 로 끝났고 `V1b` 는 환경 문제로 실패. **HOLD→LANDING 종점까지 간 런은 이번에 없다** (다만 그 경로는 `db7aa9c` 이전과 코드가 같고 노드 테스트가 고정한다) |

## 다음 세션이 그대로 이어받을 재현 명령

박스가 비면 아래 3건만 돌리면 된다(`/mnt/c/sitl7_xfer/f2b_run.sh` 사용,
`PYTHONPATH`·`PX4_DIR`·`DRONE_WS` 는 그 래퍼가 잡는다).

```bash
# ① vision_landing=false 회귀 — HOLD→LANDING 종점까지
bash f2b_run.sh V3_vision_off "" --launch-arg range_limit_m=1200.0

# ② F2-c 생산자 사망 (PRECISION_LAND 한복판에서 죽인다)
#    래치는 AGL 33.6m 아래에서 서고(V2 실측 t≈11.5s) 거기서 3m 까지 38초가 더
#    걸리므로, 25s 면 확실히 PRECISION_LAND 안이다.
bash f2b_run.sh V5_producer_death \
  "--enu-x 0.0 --enu-y 300.0 --enu-z 0.2 --on-match VISION_SEARCH --linkdead-at 25" \
  --launch-arg range_limit_m=1200.0 --launch-arg vision_landing=true

# ③ F2-f align 타임아웃 → 나선 강제 진행
#    🔴 `wp1_land_radius` 는 launch 인자로 **선언돼 있지 않다** — 주면 위생검사가
#       RuntimeError 를 던져 launch 자체가 실패한다. 선언된 인자만 쓴다.
#    v_approach=0.5 로 슬루를 늦춰 역천이 오버슈트(≈47m) 복귀가 align 30s 안에
#    끝날 수 없게 만든다. hold_timeout=20 이 HOLD 도 같은 이유로 끊는다.
bash f2b_run.sh V6_hold_timeout_align "" \
  --launch-arg range_limit_m=1200.0 --launch-arg vision_landing=true \
  --launch-arg v_approach=0.5 --launch-arg hold_timeout=20.0 \
  --launch-arg vision_search_timeout=45.0
```

기대 로그: ② `vision 생산자 사망(link ERROR) — AGL …m → GPS 착륙 폴백` 이
**시한을 다 채우기 전에** 뜬다. ③ `탐색고도 정렬 타임아웃 30s 초과 … → 나선 강제 진행`.

---

# 이어받은 세션 (2026-07-29 15:0x~) — 실기체 F2 첫 비행 직전 검증

## 이 세션이 바꾼 환경 (앞 절의 기록과 다른 점)

| 항목 | 앞 세션 | 이 세션 |
|---|---|---|
| 격리 클론 커밋 | `3d73bfb` | **`44027a5`** (`git fetch origin` + `--ff-only`) |
| `fc_ros` 설치본 | 위에 맞춰짐 | `colcon build --packages-select fc_ros` 재수행 |

**왜 올렸나 — 오늘 실기체가 나는 코드가 `44027a5` 이기 때문이다.** `3d73bfb..44027a5`
의 코드 델타는 문서가 아니다:

```
fc_ros/fc_ros/nodes/mission_node.py     |  2 +-   waypoints 기본 300m→150m 편도
fc_ros/fc_ros/nodes/offboard_node.py    |  8 +-   transition_alt 50→25 (+F2-g 주석)
fc_ros/fc_ros/params/fc_ros_params.yaml | 28 +-   transition_alt 25 / vision_search_timeout 120→180
tools/sitl/run_scenario.py              | 136 +   gz 정리 전역 pkill → 자기 프로세스 그룹 (bc3229e)
```

부수효과 하나가 중요하다: **`bc3229e` 가 앞 절이 "상류 수정 대상"이라 적어둔
전역 `pkill -f "gz sim"` 을 이미 고쳐 놨다.** 그래서 앞 세션이 자기 클론에만
넣었던 로컬 가드(`f2b_guard.py`)는 이제 필요 없다 — 이 세션은 가드를 **넣지도
지우지도 않았다**(클론 워킹트리는 `git status` 클린).

### 진입 위생검사 실측 (`/mnt/c/sitl7_xfer/f2c_verify.sh`)

```
fc_bridge      : /root/ws_f2b/src/suridoksuri/fc_bridge/__init__.py
precision_land : /root/ws_f2b/src/suridoksuri/fc_bridge/execution/precision_land.py
offboard_node  : /root/ws_f2b/install/fc_ros/lib/python3.10/site-packages/fc_ros/nodes/offboard_node.py
설치된 params  : transition_alt 25.0 / vision_search_timeout 180.0 / range_limit_m 300.0
                 vision_landing false / vision_latch_max_agl 0.0(자동) / precision_land_timeout 60.0
PX4            : make px4_sitl_default → "ninja: no work to do", md5 4d717ab5… (앞 세션과 동일)
단일 테넌트    : run_scenario.py / px4 / gz sim / mavros_node 전부 0건 — 박스 비어 있음
```

> 🔴 **`wsl.exe -d Ubuntu-22.04 -- bash -c '…$VAR…'` 에서 `$VAR` 가 바깥 셸에서
> 먼저 전개된다**(작은따옴표 안이어도). 이 세션에서 `export PYTHONPATH=$SRC:$PYTHONPATH`
> 를 그렇게 한 줄로 넣었더니 **ROS 가 넣어준 PYTHONPATH 가 통째로 날아가**
> `fc_ros.nodes` 를 못 찾았다(진단에 시간을 썼다). 명령은 **스크립트 파일로 만들어
> 경로만 넘겨라** — `f2b_run.sh` 는 파일이라서 애초에 안전하다.

## 🔴 `gz sim` 이 안 뜨면 PX4 는 6분 뒤에 죽는다 (V6 1차 실측)

V6 1차는 `exit=4 reason=px4_died` 로 끝났다. **코드와 무관한 환경 고장**인데,
증상 순서를 남겨 둔다 — 다음 세션이 코드 문제로 오인하기 딱 좋은 형태다.

```
[run_scenario] 기동 px4 pid=578          ← 이때 `gz sim` 이 같이 떠야 하는데 안 떴다
[run_scenario] MAVROS connected=true 확인
[mavros.param] PR: Failed to get parameter type: CBRK_SUPPLY_CHK   (×6, 180s)
[mavros.param] PR: Failed to get parameter type: NAV_DLL_ACT       (×14, 180s)
[run_scenario] 경고: 프리플라이트 우회 파라미터 검증 실패 ['CBRK_SUPPLY_CHK','NAV_DLL_ACT']
[offboard_node] ARM 요청
→ 산출물: … (exit=4 reason=px4_died)   elapsed 383.1s
```

커널 쪽 근거(dmesg):

```
[238182.502343] logger[997823]: segfault at 7f0800000002 ip 0000563b39871645
                error 6 in px4[563b395ed000+4dd000]
[238182.502370] potentially unexpected fatal signal 11.
```

- 런 중 `pgrep -af "gz sim"` **0건**이었다. PX4 SITL 은 gz 와 lockstep 이라
  시뮬레이터가 없으면 파라미터 응답조차 못 한다 → MAVROS 의
  `Failed to get parameter type` 은 **원인이 아니라 증상**이다.
- 브링업이 **30s → 383s** 로 늘어난 것도 전부 이 때문이다(우회 파라미터 1건당
  `set_preflight_bypass(timeout_s=180)` 를 통째로 태운다).
- 남긴 ulg 가 1.8MB 뿐이다(정상 런은 40MB 대). 로거 스레드가 초반에 죽었다.
- 재실행하니 `gz sim --verbose=1 -r -s …/worlds/default.sdf` 가 정상 기동했다.
  ⇒ **일과성**. 디스크(923G 여유)·메모리(4.0G 여유)는 원인이 아니다.

> 다음 세션 체크: `기동 px4 pid=…` 직후 `pgrep -af "gz sim"` 이 비어 있으면
> **그 런은 이미 죽은 것이다.** 380초를 기다리지 말고 바로 끊고 다시 띄워라.

## `V6_hold_timeout_align` (2차) — **F2-f ✅ 실증**

설정: A1 + `range_limit_m=1200 vision_landing=true v_approach=0.5 hold_timeout=20
vision_search_timeout=45`. fake_vision **없음**(래치가 서면 align 을 안 거친다).
`exit=0 reason=done`, 252.5s.

```
[  1.20s] ARM_TAKEOFF     ARM 요청
[  2.20s] CLIMBING        CommandTOL 이륙 요청 alt=50.1m AMSL
[ 30.24s] TRANSITION_FW   운용 고도 50.0m 도달 → transition_fw
[ 44.06s] STREAMING/FOLLOWING
[ 63.04s] TRANSITION_MC   경로 추종 완료 -> transition_mc
[ 68.86s] HOLD            MC 전환 완료 -> HOLD (WP1 복귀)
[ 88.73s] VISION_SEARCH   홀드 타임아웃 → VISION_SEARCH (비전 착륙 활성)
[178.70s] LANDING         탐색 2회차 실패 (타임아웃 45s) → GPS 착륙 폴백
[208.65s] DONE            착륙 완료 (disarmed) -> DONE
range_guard: max_horiz_m 352.4  at ENU(-3.9, 352.4, 38.6)   상한 1200m 미접촉
```

핵심 구간 원문(`node.log`):

```
WP1 홀드 dist=45.0m speed=3.8m/s stable=0/10
WP1 홀드 dist=39.7m speed=0.8m/s stable=0/10
WP1 홀드 dist=34.9m speed=0.6m/s stable=0/10
[WARN] WP1 홀드 타임아웃 20s 초과 (dist=34.9m)
       탐색 1회차 — 고도 25m AGL 반경 6.1→30m 링간격 12.2m 속도상한 0.5m/s (풋프린트 단폭 18.8m)
       홀드 타임아웃 → VISION_SEARCH (비전 착륙 활성)
       탐색 1회차 [align] r=6.1m rev=0.00 t= 5/45s seen=False
       탐색 1회차 [align] r=6.1m rev=0.00 t=25/45s seen=False
[WARN] 탐색고도 정렬 타임아웃 30s 초과 (AGL 41.8m/25m, WP1 거리 30.4m) → 나선 강제 진행
       탐색 1회차 [spiral] r=6.1m rev=0.00 t=30/45s seen=False
       탐색 1회차 [spiral] r=6.8m rev=0.06 t=35/45s seen=False
       탐색 1회차 [spiral] r=7.5m rev=0.11 t=40/45s seen=False
[WARN] 탐색 1회차 실패 (타임아웃 45s) → 15m 재탐색
       탐색 2회차 — 고도 15m AGL 반경 3.7→18m 링간격 7.3m 속도상한 0.5m/s
[WARN] 탐색 2회차 실패 (타임아웃 45s) → GPS 착륙 폴백
```

- **F2-f ✅** `wp1_land_radius` 를 못 건드리는 상태에서도 재현에 성공했다.
  `v_approach=0.5` 가 슬루를 늦춰 역천이 오버슈트 복귀가 20s 안에 안 끝나고
  (`dist=34.9m`), **HOLD 를 끊은 바로 그 조건이 align 에서도 그대로 안 선다**
  (`WP1 거리 30.4m`, `AGL 41.8m/25m` — 수평·수직 **둘 다** 미달). 30s 상한이
  `align → spiral` 을 강제했고 나선이 실제로 커졌다(`r=6.1 → 6.8 → 7.5m`).
- **수정 전이었다면** 두 회차를 align 에 갇힌 채 통째로 태웠을 것이다.
  이 런의 회차 상한은 시험용 45s 였지만 **오늘 실기체의 정식값은 180s** 다
  (`fc8631d`) — 즉 수정이 없었으면 **2×180 = 360초(6분)** 를 WP1 근처에서
  제자리 호버로 태운 뒤에야 GPS 폴백으로 갔다. 임무 끝단 배터리에 직격이다.
- 부수 관측: 강제 진행 시점의 AGL 이 **41.8m 로 래치 상한 33.6m 위**다.
  이 상태에서 타겟이 보였어도 F2-a 게이트가 래치를 막는다 — 즉 **F2-f 와 F2-a
  는 같은 런에서 동시에 걸릴 수 있고**, 그때 기체는 "나선은 도는데 래치는 안
  서는" 상태가 된다. 하강은 `_vs_ramp` 가 계속 끌어내리므로 시간이 풀어준다.

## `V9_today_shape` — 오늘 형상(150m 편도 · 25m · vision on · 상한 기본 300)

설정: `vision_landing=true transition_alt=25.0 waypoints=[0,0,25, 150,0,25]`,
**`range_limit_m` 은 주지 않았다**(기본 300). fake_vision 은 WP1 상공 타겟.

```
[  2.82s] ARM_TAKEOFF     ARM 요청
[  3.83s] CLIMBING        CommandTOL 이륙 요청 alt=25.2m AMSL (지면 0.2+25.0)
[ 25.47s] TRANSITION_FW   운용 고도 25.0m 도달 → transition_fw
[ 38.69s] STREAMING/FOLLOWING
[ 46.90s] TRANSITION_MC   경로 추종 완료 -> transition_mc
[ 52.91s] HOLD            MC 전환 완료 -> HOLD (WP1 복귀)
[ 65.53s] VISION_SEARCH   WP1 도달·안정 (dist=0.6m speed=0.2m/s) → VISION_SEARCH
[ 66.13s] PRECISION_LAND  타겟 래치 (연속 3프레임, N=150.2 E=0.2)
[ 93.99s] LANDING         인계고도 도달 (3.5m AGL, 목표 3.0m±0.5) → LANDING (AUTO.LAND)
```

```
거리 상한 감시 활성 300m (이륙지점 기준 수평, 경로 최원점 150m,
                          탐색 최원점 180m (=WP1 150m + 반경 30m))
FOLLOWING 시작 pos=[13.9,1.8] tgt=[83.9,0.0] cte=1.8m mode=OFFBOARD seg=13/150
WP1 홀드 dist=48.1m speed=7.9m/s stable=0/10      ← 역천이 오버슈트
WP1 홀드 dist= 7.1m speed=5.0m/s stable=0/10
WP1 도달·안정 (dist=0.6m speed=0.2m/s) → VISION_SEARCH
타겟 래치 (연속 3프레임, N=150.2 E=0.2) → PRECISION_LAND
정밀착륙 진입 — 래치 AGL 24.8m, 하강예산 27s → 시한 60s (하한 60s)
정밀착륙 AGL=20.7m 수평오차=0.27m 하강 state=LOCK age=0.21s t= 6/60s
정밀착륙 AGL= 4.9m 수평오차=0.21m 하강 state=LOCK age=0.03s t=26/60s
인계고도 도달 (3.5m AGL, 목표 3.0m±0.5) → LANDING (AUTO.LAND)
range_guard: max_horiz_m 188.0  at ENU(0.1, 188.0, 25.2)
```

**오늘 비행에 대한 실측 답 4개.**

1. **`range_limit_m` 기본 300 으로 충분하다.** 이륙 전 로그가
   `탐색 최원점 180m` 을 찍고 경고가 아니라 **정보**로 나온다. 감시가 잰
   최대 수평거리는 **188.0m**(오케스트레이터 추산 224m 보다 작다).
   ⚠️ 단 이 값은 하니스가 ~10s 간격으로 뜨는 표본이라 **첨두를 놓친다** —
   `WP1 홀드 dist=48.1m` 로부터 실제 첨두는 **≈198m**(150+48)로 봐야 한다.
   그래도 300m 대비 **여유 100m** 다.
2. **`transition_alt=25` 는 래치를 첫 프레임에 세운다.** HOLD 고도 24.8m 가
   래치 상한 33.6m 아래라 `VISION_SEARCH` 진입 **0.6초 만에** 래치했다.
   V2(50m 천이)에서 34.7m까지 3회 거부되며 11.5초를 쓰던 구간이 통째로 사라졌다.
   시한도 `하강예산 27s → 60s(하한)` 으로 여유가 크다.
3. **F2-g 수정(`3d73bfb`)이 실제로 먹었다.** V2 에서 3.1m 에 22초 정지 후
   타임아웃으로 끝나던 종료가, 여기서는 `인계고도 도달 (3.5m AGL, 목표
   3.0m±0.5)` 로 **정상 인계**됐다. 하강도 20.7→4.9m 를 20초에 = 0.79 m/s 로
   명목(0.8)대로 걸었고 수평오차는 내내 **0.03~0.40m**.
4. **역천이 오버슈트는 48.1m** (yaml `:248` 의 "≈47m" 와 일치).
   HOLD 는 그 48m 를 `v_approach=5` 램프로 되돌아와 **12.6초**에 `dist=0.6m` 로
   안정 — `hold_timeout=30s` 안이다.

> ⚠️ 이 런은 `exit=4 reason=px4_died` 로 끝났다. **다만 죽은 시점은
> `인계고도 도달 → LANDING(AUTO.LAND)` 이후**라 F2 구간은 전부 관측됐다.
> 오늘 두 번째 PX4 사망이다(V6 1차 = ARM 시점). WSL SITL 쪽 불안정이며
> 코드 경로와 무관하다 — 다만 **다음 세션은 런당 1회 재시도를 기본으로 잡아라.**

---

# 짧은 경로 스윕 — "편도 몇 m 아래부터 깨지나"

동기: 오늘 현장 공간이 좁아 편도가 150m 보다 **짧아진다**(길이 미정). 문서상
짧은 경로 위험 3건이 걸려 있다 — F-11(`_FW_LOOKAHEAD=70.0` 은
`offboard_node.py:102` **하드코딩**이라 현장에서 못 바꾼다), B7(FOLLOWING 창
상한 = `(L − d_end_thresh)/v_cruise`), 그리고 역천이 오버슈트 ≈47m.

공통 설정: `vision_landing=true transition_alt=25.0`, **`range_limit_m` 미지정
(기본 300)**, fake_vision 은 WP1 상공. 시나리오는 A1(`waypoint_frame=local`).

## 실측 대조표

| 편도 | FOLLOWING 체류 | 역천이 오버슈트 | HOLD 복귀(→dist 0.6m) | 래치 AGL | `range_guard.max_horiz_m` | 결과 |
|---|---|---|---|---|---|---|
| **150m** | 8.2s | 48.1m | 12.6s | 24.8m | 188.0 (첨두 ≈198) | F2 전구간 통과 (PX4 는 AUTO.LAND 중 사망) |
| **60m** | **2.4s** | **49.3m** | 12.6s | 24.9m | **112.6** (횡 −1.2m) | ✅ `exit=0 done` **94.75s 완주** |
| **40m** | **1.0s** | **68.3m** | **17.0s** | 24.9m | **111.8** (횡 **−17.2m**) | ✅ `exit=0 done` **98.75s 완주** |
| **25m** | **0.2s** | **73.2m** | **17.6s** | 25.0m | **102.5** (횡 +1.0m) | ✅ `exit=0 done` **99.74s 완주** |

## `S60_short` — 편도 60m, 완주

```
[  1.63s] ARM_TAKEOFF     ARM 요청
[  2.63s] CLIMBING        CommandTOL 이륙 요청 alt=25.1m AMSL (지면 0.1+25.0)
[ 22.65s] TRANSITION_FW   운용 고도 25.0m 도달 → transition_fw
[ 35.87s] STREAMING
[ 36.07s] FOLLOWING       OFFBOARD 확인 → FOLLOWING
[ 38.47s] TRANSITION_MC   경로 추종 완료 -> transition_mc      ← FOLLOWING 체류 2.4s
[ 44.48s] HOLD            MC 전환 완료 -> HOLD (WP1 복귀)
[ 57.10s] VISION_SEARCH   WP1 도달·안정 (dist=0.6m speed=0.1m/s)
[ 57.70s] PRECISION_LAND  타겟 래치 (연속 3프레임, N=60.2 E=0.2)
[ 85.54s] LANDING         인계고도 도달 (3.5m AGL, 목표 3.0m±0.5) → LANDING
[ 94.75s] DONE            착륙 완료 (disarmed) -> DONE
```

```
거리 상한 감시 활성 300m (이륙지점 기준 수평, 경로 최원점 60m,
                          탐색 최원점 90m (=WP1 60m + 반경 30m))
FOLLOWING 시작 pos=[13.7,1.5] tgt=[60.0,0.0] cte=1.5m mode=OFFBOARD seg=13/60
FOLLOWING tick=20 mode=OFFBOARD cte=1.5m pos=[44.7,1.5] tgt=[60.0,0.0] seg=44/60
WP1 홀드 dist=49.3m speed=7.9m/s stable=0/10     ← 오버슈트가 경로장의 82%
WP1 홀드 dist= 8.0m speed=4.8m/s stable=0/10
WP1 도달·안정 (dist=0.6m speed=0.1m/s) → VISION_SEARCH
타겟 래치 (연속 3프레임, N=60.2 E=0.2) → PRECISION_LAND
정밀착륙 진입 — 래치 AGL 24.9m, 하강예산 27s → 시한 60s (하한 60s)
인계고도 도달 (3.5m AGL, 목표 3.0m±0.5) → LANDING (AUTO.LAND)
range_guard: max_horiz_m 112.6  at ENU(-1.2, 112.6, 26.0)
```

읽는 법 세 가지:

1. **`tgt` 가 종점에 고정된다.** 150m 런에서는 `tgt=[83.9,0]`(= 현재위치 +70m
   lookahead)였는데 60m 런에서는 첫 틱부터 `tgt=[60.0,0.0]` — 경로 끝으로
   클램프됐다. 즉 **60m 에서는 `_FW_LOOKAHEAD=70` 이 이미 경로장보다 길다.**
   F-11 이 말하는 상태에 **이미 들어와 있다**. 그런데도 깨지지 않은 이유는
   진입 시점의 잔여거리(60−13.7 = 46.3m)가 SITL 선회반경(~37m)보다 **아직
   컸기** 때문이다. 이 여유가 사라지는 지점이 진짜 하한이다.
2. **오버슈트는 경로장에 비례하지 않는다 — 거의 상수(48~49m)다.** 150m 에서
   48.1m, 60m 에서 49.3m. 그래서 **총 수평 이탈 = 편도 + 49m** 로 잡으면 된다
   (60m → 109m, 실측 112.6m).
3. **HOLD 복귀 시간도 상수(12.6s)다** — 되돌아올 거리가 오버슈트뿐이라서다.
   `hold_timeout=30s` 대비 여유 17s. **짧은 경로가 HOLD 를 깨지는 않는다.**

## `S40_short` — 편도 40m, 완주. **여기서부터 열화가 보인다**

```
[  2.21s] ARM_TAKEOFF
[  3.61s] CLIMBING        alt=25.1m AMSL
[ 22.84s] TRANSITION_FW   운용 고도 25.0m 도달 → transition_fw
[ 36.06s] STREAMING
[ 36.26s] FOLLOWING       OFFBOARD 확인 → FOLLOWING
[ 37.26s] TRANSITION_MC   경로 추종 완료 -> transition_mc     ← FOLLOWING 체류 **1.0s**
[ 44.47s] HOLD            MC 전환 완료 -> HOLD (WP1 복귀)
[ 60.89s] VISION_SEARCH   WP1 도달·안정 (dist=0.8m speed=0.2m/s)
[ 61.49s] PRECISION_LAND  타겟 래치 (연속 3프레임, N=40.2 E=0.2)
[ 89.73s] LANDING         인계고도 도달 (3.5m AGL, 목표 3.0m±0.5)
[ 98.75s] DONE            착륙 완료 (disarmed) -> DONE
```

```
거리 상한 감시 활성 300m (… 경로 최원점 40m, 탐색 최원점 70m (=WP1 40m + 반경 30m))
FOLLOWING 시작 pos=[13.5,1.7] tgt=[40.0,0.0] cte=1.7m mode=OFFBOARD seg=13/40
WP1 홀드 dist=68.3m speed=8.0m/s stable=0/10       ← 오버슈트가 경로장의 171%
WP1 홀드 dist=36.2m speed=5.0m/s stable=0/10
WP1 홀드 dist= 6.5m speed=5.0m/s stable=0/10
WP1 도달·안정 (dist=0.8m speed=0.2m/s) → VISION_SEARCH        ← HOLD 진입 후 17.0s
타겟 래치 (연속 3프레임, N=40.2 E=0.2) → PRECISION_LAND
정밀착륙 진입 — 래치 AGL 24.9m, 하강예산 27s → 시한 60s (하한 60s)
range_guard: max_horiz_m 111.8  at ENU(**−17.2**, 110.5, 29.2)
```

40m 은 **완주했지만 세 가지 지표가 동시에 나빠진다**:

- **FOLLOWING 체류 1.0s.** `docs/sitl_vtol_auto_path_spec.md:385` 의 B7 예측
  (`(40−10)/18 ≈ 1.7s`)과 같은 자릿수다. 실측이 더 짧은 것은 진입 시점에
  기체가 이미 13.5m 를 지나 있어서다(잔여 26.5m). **경로 추종이 사실상 없다.**
- **오버슈트가 48~49m 에서 68.3m 로 튀었다.** 60m·150m 에서 거의 상수였던 값이
  40m 에서만 커진다. 원인은 `tgt` 클램프다 — FOLLOWING 진입 시 잔여거리
  26.5m 가 SITL 선회반경(~37m)보다 **작아서**, FW 가 목표점을 향해 곧장 가는
  대신 선회로 감싸며 지나친다(= F-11 이 말하는 상태).
- **횡방향 이탈 −17.2m.** 60m 런의 최원점은 ENU(−1.2, 112.6) 로 거의 직선인데
  40m 런은 ENU(−17.2, 110.5) 다. 경로 옆으로 17m 를 벗어났다는 뜻이고,
  **좁은 현장에서는 이게 진짜 위험**이다(총 수평 이탈 111.8m = 경로장의 2.8배).

HOLD 는 그래도 버틴다: 68.3m 를 `v_approach=5` 로 되돌아오는 데 **17.0s**,
`hold_timeout=30s` 안이다(여유 13s). 래치도 24.9m 에서 즉시 섰다.

## `S25_short` — 편도 25m, **여기서도 완주한다**

```
[ 22.63s] TRANSITION_FW   운용 고도 25.0m 도달 → transition_fw
[ 36.05s] FOLLOWING       OFFBOARD 확인 → FOLLOWING
[ 36.25s] TRANSITION_MC   경로 추종 완료 -> transition_mc     ← FOLLOWING 체류 **0.2s (2틱)**
[ 43.86s] HOLD            MC 전환 완료 -> HOLD (WP1 복귀)
[ 61.49s] VISION_SEARCH   WP1 도달·안정 (dist=0.7m speed=0.1m/s)   ← HOLD 후 17.6s
[ 61.89s] PRECISION_LAND  타겟 래치 (연속 3프레임, N=25.2 E=0.2)
[ 90.13s] LANDING         인계고도 도달 (3.5m AGL, 목표 3.0m±0.5)
[ 99.74s] DONE            착륙 완료 (disarmed) -> DONE
FOLLOWING 시작 pos=[13.6,1.7] tgt=[25.0,0.0] cte=1.7m seg=13/25
WP1 홀드 dist=73.2m speed=7.5m/s stable=0/10       ← 오버슈트가 경로장의 293%
정밀착륙 진입 — 래치 AGL 25.0m, 하강예산 27s → 시한 60s (하한 60s)
range_guard: max_horiz_m 102.5  at ENU(1.0, 102.4, 24.3)
```

# 🔴 결론 — 오늘 현장에서 쓸 숫자

## 1. 상태기계는 25m 편도까지 안 깨진다

**25 / 40 / 60 / 150m 네 길이 전부 `HOLD → VISION_SEARCH → PRECISION_LAND →
LANDING → DONE` 을 완주했다.** F-11/B7 이 예고한 "FOLLOWING 구간 소멸"은
**실제로 일어났지만**(체류 8.2 → 2.4 → 1.0 → **0.2s**), 그것이 임무를
깨지는 않았다. 이유는 두 가지다.

- 종점 포착이 **거리 원 OR 결승선**(F-10 수정, `8bbcb94`)이라 FOLLOWING 이
  0.2초여도 종점을 놓치지 않는다.
- HOLD 가 오버슈트를 `v_approach=5` 램프로 되돌리는 구조라, 오버슈트가 커져도
  **복귀시간이 12.6 → 17.6s** 로만 늘고 `hold_timeout=30s` 를 못 넘겼다.

## 2. **그런데 경로를 줄여도 필요한 공간은 줄지 않는다** ← 오늘 가장 중요한 숫자

| 편도 | 종점 너머 오버슈트 | 이륙지점 기준 총 수평 이탈 |
|---|---|---|
| 150m | 48.1m | ≈198m |
| 60m | 49.3m | **112.6m** |
| 40m | 68.3m | **111.8m** |
| 25m | 73.2m | **102.5m** |

**오버슈트는 경로장에 비례하지 않고 ~70m 에서 포화한다.** 그래서 편도를
150 → 25m 로 6배 줄여도 총 수평 이탈은 **198 → 102m, 절반밖에 안 준다.**

> ### 현장 판정 기준
> **비행 방향으로 "편도 + 75m" 의 여유가 없으면 이 임무는 성립하지 않는다.**
> 편도 25m 든 60m 든 **기체는 이륙지점에서 100~115m 까지 나간다.**
> 추가로 **횡방향 최대 17.2m**(S40 실측 ENU E=−17.2m)를 잡아라.
> ⇒ 실무적으로 **비행 방향 120m × 폭 40m** 를 확보하지 못하면 편도를
> 줄이는 것으로는 해결되지 않는다. 줄여야 하는 것은 **`transition_alt` 이나
> 경로가 아니라 "FW 로 날지 여부"** 다.

## 3. `range_limit_m` 기본 300 은 어느 길이에서도 안전

실측 최대 112.6m. 이륙 직전 로그도 전부 경고 아닌 정보로 나왔다
(예: `거리 상한 감시 활성 300m (… 경로 최원점 40m, 탐색 최원점 70m)`).
**현장에서 `range_limit_m` 을 만질 이유가 없다.**

## 4. `transition_alt=25` 는 F2 를 눈에 띄게 쉽게 만든다

네 런 모두 **`VISION_SEARCH` 진입 0.2~0.6초 만에 래치**했다(래치 AGL
24.8~25.0m, 상한 33.6m 아래). 정렬·나선·재탐색을 한 번도 안 거쳤다.
시한도 `하강예산 27s → 60s(하한)` 이라 여유가 33s다.
**즉 오늘 비행에서 F2 가 실패한다면 원인은 상태기계가 아니라 "검출이 되느냐"다.**

---

# `V5b_producer_death` — **F2-c ✅ 실증** (앞 세션 미실시분)

설정: A1 + `range_limit_m=1200 vision_landing=true`,
fake_vision `--linkdead-at 25`(= `VISION_SEARCH` 관측 후 25초에 `vision/link`
를 ERROR 로 바꾼다. 프레임 발행도 같이 끊긴다).

```
래치 보류 — AGL 50.0m 가 래치 상한 33.6m 초과 (검출 신뢰구간 밖의 좌표로는 하강을 시작하지 않는다)
래치 보류 — AGL 42.9m 가 래치 상한 33.6m 초과 …
래치 보류 — AGL 36.3m 가 래치 상한 33.6m 초과 …
타겟 래치 (연속 3프레임, N=300.0 E=0.0) → PRECISION_LAND
정밀착륙 진입 — 래치 AGL 32.8m, 하강예산 37s → 시한 60s (하한 60s)
정밀착륙 AGL=32.1m 수평오차=0.02m 하강 state=LOCK age=0.22s t= 2/60s
정밀착륙 AGL=24.2m 수평오차=0.01m 하강 state=LOCK age=0.10s t=12/60s
[ERROR] vision 생산자 사망(link ERROR) — AGL 23.6m → GPS 착륙 폴백
```

- **F2-c ✅** 폴백이 **`t=13/60s`** 에 걸렸다. 수정 전이라면 유도가 영원히 안
  올 것이 확정된 상태에서 **남은 47초를 호버로 태운 뒤에야** 내려갔다.
  임무 끝단 배터리에 직결되는 47초다.
- **F2-a 재현(부수)** 래치가 50.0 / 42.9 / 36.3m 에서 **3회 거부**되고 32.8m
  에서 성립했다. V2 와 같은 결과이며, `transition_alt=50` 일 때만 나타난다
  (오늘 형상 25m 에서는 첫 프레임에 바로 선다 — 위 S-런 4건 참조).

전이 타임라인 / `exit=0 done` 164.2s / `range_guard.max_horiz_m` 334.6m
(상한 1200m 미접촉):

```
[  1.80s] ARM_TAKEOFF   [  3.41s] CLIMBING     [ 31.25s] TRANSITION_FW
[ 43.86s] STREAMING/FOLLOWING                  [ 63.09s] TRANSITION_MC
[ 68.90s] HOLD          [ 80.32s] VISION_SEARCH
[ 93.34s] PRECISION_LAND  타겟 래치 (연속 3프레임, N=300.0 E=0.0)
[106.15s] LANDING         vision 생산자 사망(link ERROR) — AGL 23.6m → GPS 착륙 폴백
[133.99s] DONE            착륙 완료 (disarmed)
```

---

# 이 세션의 미실시 (정직하게)

| 항목 | 상태 |
|---|---|
| **F2-d 유도 상실 SITL 실증** | ❌ **미실시.** 재현 인자는 준비돼 있다(아래 ①). `--linkdead-at`(F2-c)과 달리 **`--exit-at`** 를 써야 한다 — `fake_vision.py` 가 통째로 죽으면 `vision/link` ERROR 조차 안 나가서 `link_dead` 가 **서지 않고**(`vision_target_bridge.py:74-77` 3분법), `_pl_lost_elapsed` 시한(`vision_lost_timeout=5.0`)이 잡아야 정상이다. 그 경로는 이번에 **한 번도 실행되지 않았다** |
| **`vision_landing=false` 종점 완주(HOLD→LANDING)** | ❌ **여전히 미실시.** 앞 세션 `V1`(거리상한 OVERRIDE 종료)·`V1b`(환경 실패)에 이어 이번에도 못 돌렸다 — 짧은 경로 스윕에 밀렸다. **다만 `V6` 가 `vision_landing=true` 로 `_exit_hold` 의 반대 분기를 태우지 않았을 뿐, `HOLD → … → LANDING → DONE` 종점 도달 자체는 이번 세션 6런 중 5런에서 관측됐다** |
| 편도 80 / 100m | ❌ 미실시. 60m 가 완주했으므로 그 사이는 단조로 안전하다고 **추정**했다(실측 아님) |
| 편도 20m 이하 | ❌ 미실시. 20m 면 FOLLOWING 진입 시 잔여거리가 `d_end_thresh=10m` 아래라 **FOLLOWING 이 0틱**이 될 수 있다 — 여기가 진짜 벼랑일 가능성이 있으나 확인 못 했다 |

## 다음 세션이 그대로 이어받을 재현 명령

```bash
# ① F2-d 유도 상실 (shim 자체 사망 = setpoint·status 둘 다 침묵)
bash /mnt/c/sitl7_xfer/f2b_run.sh V7_guidance_lost \
  "--enu-x 0.0 --enu-y 300.0 --enu-z 0.2 --on-match VISION_SEARCH --exit-at 25" \
  --launch-arg range_limit_m=1200.0 --launch-arg vision_landing=true
# 기대: `vision 유도 상실 5s 지속 (setpoint age=…s status age=…s, AGL …m) → GPS 착륙 폴백`
# (스크립트로도 있다: /mnt/c/sitl7_xfer/f2c_v7.sh)

# ② vision_landing=false 종점 완주
bash /mnt/c/sitl7_xfer/f2c_v8.sh

# ③ 짧은 경로 추가 스윕 (편도 20m — FOLLOWING 0틱 여부)
bash /mnt/c/sitl7_xfer/f2c_short.sh 20
```

이 세션이 새로 만든 헬퍼(전부 `/mnt/c/sitl7_xfer/`):
`f2c_verify.sh`(진입 위생검사) · `f2c_inspect.sh <RUN_ID>`(산출물 요약) ·
`f2c_diag.sh`/`f2c_diag2.sh`(px4 사망 원인) · `f2c_short.sh <편도m>`(스윕) ·
`f2c_copy.sh <RUN_ID…>`(ulg 제외 내보내기).
