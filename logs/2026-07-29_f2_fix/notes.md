# F2 정밀착륙 — 블로커 2건 수정 후 SITL 재검증 (2026-07-29)

수정 대상은 검증 세션(`logs/2026-07-29_f2_verify/`, 커밋 `6e3ea0c`)이 실증한 블로커 2건과
노출·관측 2건이다. 배경·근거 정본은 `docs/fc_precision_land_handoff.md` **§8-1**.

**환경:** WSL `Ubuntu-22.04` · 격리 클론 `/root/ws_f2`(`/root/drone_ws` 미접촉) ·
`PX4_DIR=/root/PX4-vehicle`(패치본, `make px4_sitl_default` = "ninja: no work to do" 로
바이너리 최신 확인) · 시나리오 `A1`(직선 300m, `waypoint_frame=local`) ·
합성 vision 발행기 `tools/sitl/fake_vision.py`(4.4Hz 실측 주파수).

ulog 는 `.gitignore` 로 제외(런당 38~45MB). 원본은 위 클론에 남아 있고 파일명은
각 `meta.json` 의 `"ulogs"` 에 있다.

---

## 런 3건

| run | launch 인자 | 결과 | 무엇을 보는가 |
|---|---|---|---|
| `G1_base` | (vision 인자 없음 = YAML `false`) | **exit=done 152.9s** | 종전 경로 무변경 회귀 |
| `G2_vision` | `vision_landing:=true` | **exit=done 176.9s** | F2 해피패스 완주 |
| `G3_single_frame` | `vision_landing:=true vision_search_timeout:=25.0` + vision 프레임 **1건만** | **exit=done 181.7s** | 🔴 단발 오탐 1프레임에 탐색이 중단되지 않는가 (핸드오프 §6-1 #5) |

### G1_base — `vision_landing=false` 회귀

```
HOLD 67.89 → LANDING 78.30 → DONE 121.76      (warn 1 / error 0)
```
- 비전 관련 로그 **0건**(`grep -c "vision\|비전\|탐색"` = 0) — 파라미터가 꺼져 있으면
  구독조차 만들지 않는다는 계약 그대로다.
- **`LANDING` 이 timeline 에 다시 잡힌다.** 종전 정규식으로는 이 줄
  (`WP1 도달·안정 (…) → LANDING`)이 안 맞아 `R1_base` 에서 LANDING 이 통째로 누락됐다
  (HOLD 72.50 → DONE 127.98). 관측 계층 수정의 실측 확인.

### G2_vision — `vision_landing=true` 해피패스

```
HOLD 67.49 → VISION_SEARCH 77.90 → PRECISION_LAND 78.50 → LANDING 118.15 → DONE 143.59
```
- **launch 인자가 실제로 먹는다** — `vision_landing:=true` 로 노드가
  "비전 정밀착륙 활성 — 탐색고도 25m 반경 30m / 재탐색 15m 반경 18m" 를 찍었다.
  수정 전에는 위생검사가 미선언 인자를 거부해 **launch 자체가 실패**했다.
- **래치 0.60s**(VISION_SEARCH 진입 1785267321.3585 → 래치 1785267321.9580).
  4.4Hz 에서 3프레임 = 0.68s 예상과 일치한다. 종전(제어틱 계수) 참조값은 0.50s 였고,
  그 0.50s 는 **프레임 1~2개**에 불과했다.
- 로그 문구도 `연속 3프레임` 으로 바뀌어 단위가 드러난다.
- 하강 **0.73 m/s**(49.8m → 23.1m / 36.2s, 설정 `vision_descend_speed=0.8`),
  하강 구간 수평오차 max 0.33m / mean 0.04m.
- `command_hint="land"` 인계 힌트에 `LANDING (AUTO.LAND)` 로 전이 → disarm.

> 참조: 검증 세션이 폐기 클론에서 **BLOCKER-1 한 줄만** 넣은 `R5_patched_handoff`
> 는 exit=done 185.8s / DONE 149.27s / 래치 0.50s 였다. 이번 값(176.9s / 143.59s /
> 0.60s)은 그 재현 기준과 같은 자리이고, 래치만 의도한 방향으로 0.1s 늘었다.

### G3_single_frame — 🔴 블로커 2 의 직접 반증실험

vision 프레임을 **딱 1건**만 쏘고 침묵시켰다(`--emit-count 1`). 하트비트는 살아 있어
`link_dead` 폴백이 아니라 순수하게 "래치가 서는가"만 본다.

```
VISION_SEARCH 78.70
  탐색 1회차 [align→spiral] 25s 타임아웃 → 15m 재탐색     ← 래치 0건
  탐색 2회차 25s 타임아웃 → GPS 착륙 폴백
LANDING 128.57 → DONE 149.20
```
- **래치 로그 0건.** 수정 전(제어틱 계수)이라면 그 1프레임이 `stale_timeout` 1.0s
  동안 valid 로 남아 **0.3s 만에 래치**되고 곧장 `PRECISION_LAND` 로 갔다.
- 덤으로 확인된 것 2가지: ①`vision_search_timeout:=25.0` 이라는 **float launch 인자가
  실제로 전달**된다(노출 수정 확인) ②재탐색은 **1회뿐**이고 그 뒤 GPS 폴백이다
  (무한 재탐색 없음, `search_pass_next`).
- 하니스가 `탐색 2회차 실패 (…) → GPS 착륙 폴백` 을 **LANDING 진입으로 인식**한다 —
  종전 정규식에는 이 경로가 아예 없어 F2 폴백 런의 결말이 timeline 에서 사라졌다.

---

## 재현 명령

```bash
# WSL Ubuntu-22.04, 격리 클론
export PX4_DIR=/root/PX4-vehicle
source /opt/ros/humble/setup.bash && source /root/ws_f2/install/setup.bash
cd /root/ws_f2/src/suridoksuri

# G1
python3 tools/sitl/run_scenario.py A1 --outdir logs/2026-07-29_f2_fix --run-id G1_base \
    --launch-arg range_limit_m=1200.0

# G2 (fake_vision 을 먼저 백그라운드로)
python3 tools/sitl/fake_vision.py --enu-x 6.0 --enu-y 302.0 --enu-z 0.2 --rate 4.4 \
    --on-match VISION_SEARCH --landhint-at 40.0 \
    --watch-log logs/2026-07-29_f2_fix/G2_vision/node.log &
python3 tools/sitl/run_scenario.py A1 --outdir logs/2026-07-29_f2_fix --run-id G2_vision \
    --launch-arg range_limit_m=1200.0 --launch-arg vision_landing=true

# G3 — 프레임 1건만
python3 tools/sitl/fake_vision.py --enu-x 6.0 --enu-y 302.0 --enu-z 0.2 --rate 4.4 \
    --on-match VISION_SEARCH --emit-count 1 \
    --watch-log logs/2026-07-29_f2_fix/G3_single_frame/node.log &
python3 tools/sitl/run_scenario.py A1 --outdir logs/2026-07-29_f2_fix --run-id G3_single_frame \
    --launch-arg range_limit_m=1200.0 --launch-arg vision_landing=true \
    --launch-arg vision_search_timeout=25.0
```

⚠️ `fake_vision.log` 끝의 `ExternalShutdownException` traceback 은 런 종료 시
`pkill` 로 죽인 흔적이다(정상). 세 런 모두 `error_count=0`.

## 아직 안 한 것

핸드오프 §6-1 장애주입 6종 중 이번에 실증한 것은 **#5(단발 오탐)** 하나뿐이다.
#1 생산자 SIGKILL(`--linkdead-at`) · #2 veto 지속(`--veto-at`) · #3 setpoint 침묵 ·
#4 shim 사망(`--exit-at`) · #6 나선 완주는 `fake_vision.py` 에 인자가 이미 있으나
**미실시**다. 그리고 이번 범위 밖으로 남겨둔 F2-a~f(래치 고도 게이트,
`precision_land_timeout` 정합, `_step_precision_land` 의 `link_dead` 미감시, 죽은 변수
`_pl_lost_elapsed`, `_check_path_within_range` 탐색반경 미가산, HOLD 타임아웃 후 align
재차단)도 그대로다.
