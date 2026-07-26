# tools/sitl — SITL-7 VTOL 회귀 캠페인 하니스

계획서: `docs/sitl_vtol_campaign.md` (환경 1장 / 산출물 2장 / 시나리오 3장 / 지표 4장 / 세션분할 5장)

## 한 줄 호출

**WSL 배포판 `Ubuntu-22.04` 안에서 도는 스크립트다.** 호스트(Windows/다른 배포판)에서는
항상 아래 한 줄 형태로만 부른다 — `bash -lc '...'` 에 복잡한 셸 구문을 넣으면 깨진다(실측).

```powershell
# 실행 (시나리오 1건)
wsl.exe -d Ubuntu-22.04 -- bash /root/drone_ws/src/suridoksuri/tools/sitl/run_scenario.sh A1

# 시나리오 사이 정리 — 이게 기본이다 (gz 잔류 프로세스 확실히 제거)
wsl.exe --terminate Ubuntu-22.04

# 분석
wsl.exe -d Ubuntu-22.04 -- bash /root/drone_ws/src/suridoksuri/tools/sitl/analyze_run.sh A1
```

배포판 안에서 직접:

```bash
source /opt/ros/humble/setup.bash && source /root/drone_ws/install/setup.bash
python3 tools/sitl/run_scenario.py A1
python3 tools/sitl/analyze_run.py A1
```

## 파일

| 파일 | 역할 |
|---|---|
| `scenarios.yaml` | 계획서 3장의 시나리오 정의 전량. 실행 인자·타임아웃·장애주입 스펙 |
| `run_scenario.py` | 1건 실행: PX4 SITL → MAVROS → 프리플라이트 우회 → `ros2 launch` → 감시 → 수집 → 정리 |
| `run_scenario.sh` | 위의 소싱 래퍼 (호스트에서 부르는 진입점) |
| `analyze_run.py` | `node.log` + `.ulg` → `metrics.json` + `verdict.md` (계획서 4장 지표 전량) |
| `analyze_run.sh` | 분석 소싱 래퍼 |

## 산출물

```
logs/2026-07-27_sitl_vtol_campaign/<scenario_id>/
  node.log      # ros2 launch (telemetry_node + offboard_node) stdout — 상태 전이의 1차 근거
  mavros.log    # MAVROS stdout (4MB 초과 시 head/tail 만 남김)
  *.ulg         # 이 런에서 새로 생긴 PX4 ulog 전부 ⚠️ git 미포함, 아래 참조
  meta.json     # 시작/종료 시각, launch 인자 전문, 종료사유, 상태 타임라인, 장애주입 결과
  metrics.json  # analyze_run.py 산출 — 지표 원값 + 각 지표의 null 사유
  verdict.md    # PASS/FAIL/WARN/NULL 판정표 + 근거 수치 + 경고 전량 + 미산출 지표 목록
```

**⚠️ `.ulg` 는 git 에 올라가지 않는다** (`.gitignore` 에 이 디렉터리만 예외 등록).
SITL ulog 는 런 1건당 **약 40MB**(A1 실측 39.9MB / 122초 비행, gzip 14.3MB)로,
실비행 ulog(0.6~2MB)와 자릿수가 다르다. `SDLOG_PROFILE` 은 이미 기본값 1(최소 세트)이라
더 줄일 여지가 없고, 시나리오 26건이면 1GB 급이 된다 → CLAUDE.md 공통규칙의
"대용량 raw 바이너리" 예외. 원본은 WSL 안
`/root/drone_ws/src/suridoksuri/logs/2026-07-27_sitl_vtol_campaign/<id>/` 에 남으며
파일명은 `meta.json` 의 `ulogs` 에 기록된다. **ulog 에서 뽑은 수치는 전부
`metrics.json` 에 들어가 커밋되므로 판정 근거 추적은 git 만으로 가능하다.**

## 시나리오 id ↔ 계획서 3장 대응

계획서의 22개 시나리오 중 파라미터 스윕이 있는 것은 실행 단위로 쪼갰다(총 26 런).

| 계획서 | 실행 id |
|---|---|
| A1~A4, B1~B8 | 그대로 |
| C1 (천이고도 저/고) | `C1a`(20m) `C1b`(120m) |
| C5 (`d_end_thresh` 스윕) | `C5a`(10) `C5b`(30) `C5c`(60) |
| C6 (OVERRIDE FW/MC) | `C6a`(FOLLOWING 중) `C6b`(HOLD 중) |
| C2~C4, C7~C10 | 그대로 |

## 장애주입 훅

`scenarios.yaml` 의 `inject:` 목록. 트리거 3종 × 액션 4종.

```yaml
inject:
  - on_state: FOLLOWING     # node.log 상태 진입 (또는 on_vtol_state: 1 / at_s: 60)
    delay_s: 8.0
    action: override        # set_mode(mode:) | override | param_set(param:,value:) | probe(topic:)
```

발화 시각·rc·출력은 `meta.json` 의 `inject_results` 에 남는다.
**현재 `probe` 경로만 실사용 검증됨(A1).** `set_mode`/`override`/`param_set` 은 S4에서 첫 사용이다.

## 종료 코드

| 코드 | 뜻 |
|---|---|
| 0 | DONE 도달 (완주) |
| 2 | `timeout_s` 초과 — 미완주. 산출물은 정상 수집됨 |
| 3 | 브링업 실패 (PX4/MAVROS 미기동, 또는 `boot_timeout_s` 안에 offboard_node 첫 로그 없음) |
| 4 | `ros2 launch` 또는 PX4 가 DONE 없이 죽음 |
| 5 | 시나리오 정의/사용법 오류 |

### 두 개의 시계

`offboard_node` 는 `__init__` 에서 **플래너를 동기 실행**한다(정적 감사 E-11 — 2WP 직선은 즉시,
꺾임 경로는 45~130초). 그래서 시계가 두 개다:

- `--boot-timeout-s` (기본 300) — `ros2 launch` 기동 → offboard_node **첫 로그**까지. 이 구간이
  플래너 계산 시간이다. 초과하면 exit 3 (`node_boot_timeout`).
- `timeout_s` (시나리오별) — offboard_node 첫 로그 → DONE 까지. **플래너 시간이 미션 예산을
  잡아먹지 않는다.** 실제 플래너 소요는 `meta.json` 의 `planner_blocking_s` 에 기록된다.

## 함정 (전부 실측 — 어기면 시간 날린다)

1. **PX4 콘솔(`pxh>`)을 파일로 리다이렉트 금지.** 비-TTY 재출력 루프로 20초에 195MB,
   방치 시 GB급. `run_scenario.py` 는 PX4 의 stdout/stderr 를 `/dev/null` 로, stdin 도
   `/dev/null` 로 끊는다. **이 부분은 절대 "로그 보려고" 바꾸지 마라.**
   PX4 상태는 `ros2 topic`/`ros2 service`/`ss -uln` 으로 본다.
   (`node.log`·`mavros.log` 는 pxh 가 아니라서 파일 저장이 안전하다.)
2. **`wsl.exe -d ... -- bash -lc '...'` 는 복잡한 셸 구문이 깨진다** — `$(seq)`+`for`+`if`
   조합에서 작은따옴표 안인데도 `syntax error near unexpected token`. 그래서 모든 로직이
   스크립트 파일에 있고 호출은 한 줄이다. cwd 경고 `Failed to translate \\wsl.localhost\...`
   는 무해.
3. **프로세스 정리는 `wsl.exe --terminate Ubuntu-22.04` 가 기본.** `pkill -f px4` 만으로는
   `gz sim` 이 남아 다음 런이 이전 gz 서버에 얹힌다(중복 인스턴스). `run_scenario.py` 도
   정리를 시도하지만 그것만 믿지 마라.
4. **프리플라이트 우회는 SITL 벤치 전용** — `CBRK_SUPPLY_CHK=894281`, `NAV_DLL_ACT=0`.
   실기체 파라미터에 절대 넣지 마라. MAVROS 파라미터 동기가 끝나야 설정되므로
   `run_scenario.py` 가 최대 180초까지 재시도한다.
   ⚠️ **`ros2 param set /mavros/param` 은 동기 완료 전에도 `Set parameter successful` 을
   돌려준다** — 실제로는 MAVROS 가 `PR: Unknown parameter to set: ...` 로 버린다(A1 1차
   실행 실측, 하니스는 OK로 보고했는데 값은 안 들어갔다). 그래서 `set` 뒤에 반드시
   `get` 으로 되읽어 검증하고, 결과를 `meta.json` 의 `preflight_bypass.readback` 에 남긴다.
8. **같은 시나리오를 재실행하면 이전 런의 `.ulg` 를 반드시 지워야 한다.** `run_scenario.py` 가
   시작 시 자동으로 지우고, `analyze_run.py` 는 `meta.json` 의 `ulogs` 로 파일을 특정한다 —
   둘 중 하나라도 없으면 **이전 런 ulog + 새 node.log** 조합으로 분석되는 사고가 난다
   (A1 재실행에서 실측).
5. `fc_bridge` 는 `pip install -e .` 금지 — 이미 `.pth` 방식으로 구성돼 있다.
6. `mavros.guided_target: PositionTargetGlobal failed because no origin` 은 알려진 코스메틱.
   **그 외 경고는 전부 `verdict.md` 에 실린다 — 스크립트가 무해성을 판단하지 않는다.**
7. **launch 파일을 고쳤으면 `colcon build --packages-select fc_ros` 를 다시 돌려야 한다.**
   `/root/drone_ws/install` 은 심볼릭이 아니라 복사본이다.

## 알려진 한계

- 시나리오는 **직렬 실행만** 가능하다(gz 서버·UDP 14540/14580 점유). 동시 실행 금지.
- `analyze_run.py` 의 상태창은 `node.log` 벽시계 ↔ ulog boot 시각을 **`vehicle_command`
  앵커**(ARM / NAV_TAKEOFF / DO_VTOL_TRANSITION ×2)의 **1차식 회귀**로 맞춘다.
  **lockstep SITL 의 시뮬 클록은 벽시계와 속도가 다르다** — A1 실측에서 시뮬이 벽시계보다
  8.3% 느려, 상수 오프셋만 쓰면 65초 구간에서 5.4초가 벌어지고 DONE 창이 끝<시작으로
  뒤집혔다. 1차식으로 잔차 0.72초까지 줄였지만 **±1s 경계 판정에는 여전히 여유가 없다**
  (`metrics.json` 의 `time_alignment.max_abs_residual_s` 로 매 런 확인할 것).
  미션이 이륙 전에 끝나 앵커가 2개 미만이면 scale=1 상수 오프셋으로 폴백하며 그 사실이
  `time_alignment.reason` 에 남는다.
- `setpoint_jump` 은 위치 setpoint 가 NaN 이었다가 재개되는 구간(속도 setpoint 제어 중)을
  점프로 세지 않는다(`max_gap_s=0.5`). 그 구간은 `resumption_gaps` 로 따로 보고된다.
- 계획서 4장의 setpoint 임계 `3 × v_approach × dt` (=1.5m) 는 **FW 순항의 정상 lookahead
  전진량**(`v_cruise × dt` ≈ 2m/틱)보다 작다 — 순항 중 상시 초과한다. 그래서 판정은
  **상태 경계 ±1s 내 위반만** 본다. 전체 위반 수는 참고값이다.
- `수직 가속` 판정은 계획서 문구대로 "천이 구간 제외"만 적용하므로 **착륙 접지 충격이
  판정을 지배한다**(A1: 4.1g). 비행 중 부드러움은 같은 항목의 `excl_touchdown`
  (disarm−5s 이후 제외) 값을 봐야 한다.
- `geometric_cte` 는 원본 waypoints 직선 폴리라인 기준이라, 곡선을 그리는 eta3 플래너
  경로에서는 "계획경로 이탈"이 아니다. 1차 근거는 `node.log` 의 L1Guidance cte 다.
- 바람 주입(C4)의 gz 측 주입 방법은 **아직 확정되지 않았다** — `scenarios.yaml` C4 주석 참조.
