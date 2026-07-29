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
| `V3`/`V4`/`V5` | — | ❌ `px4_dead` (exit 3) | **다른 세션(`/root/ws_c1b`)이 같은 배포판에서 SITL 캠페인 중** — 아래 "단일 테넌트" |

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

> 🔴 **더 나쁜 것:** `tools/sitl/run_scenario.py` 의 정리 루틴이
> **`pkill -f "gz sim"` 을 시스템 전역으로** 부른다(세션이 하나뿐이던 시절의 코드).
> 즉 내 런이 끝날 때마다 **남의 gz 가 같이 죽는다.** 이번에는 내 클론에만
> 로컬 가드를 넣어(다른 세션 `run_scenario.py` 가 떠 있으면 pkill 을 건너뛴다)
> 피했지만, **공용 파일에는 아직 그대로 남아 있다** — 동시 세션이 상시화된
> 지금은 상류 수정 대상이다.

## 미실시 (정직하게)

| 항목 | 상태 |
|---|---|
| F2-c 생산자 사망 SITL 실증 (`V5`) | ❌ 박스 점유로 미완. 노드 테스트 + 파괴검증으로만 확인 |
| F2-f align 타임아웃 SITL 실증 (`V6`) | ❌ 위와 같음 |
| `vision_landing=false` 완전 완주(HOLD→LANDING) | ⚠️ `V1` 은 거리상한 OVERRIDE 로 끝났고 `V1b` 는 환경 문제로 실패. **HOLD→LANDING 종점까지 간 런은 이번에 없다** (다만 그 경로는 `db7aa9c` 이전과 코드가 같고 노드 테스트가 고정한다) |
