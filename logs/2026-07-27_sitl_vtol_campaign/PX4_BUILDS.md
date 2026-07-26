# 이 캠페인의 런 ↔ PX4 빌드 대응표

SITL-7 S4 에서 **SITL 이 실기체와 다른 PX4 를 쓰고 있었다**는 것이 확인됐다.
같은 시나리오라도 어느 PX4 에서 돌았는지 모르면 결과를 해석할 수 없으므로 여기 남긴다.

| PX4 커밋 | 별칭 | 소스 디렉터리 | 오프보드 course 가드 | 비고 |
|---|---|---|---|---|
| `9bb0d365c4` | SITL 기존 | `/root/PX4-Autopilot` (태그 `sitl7-orig-head`) | **없음** (revert `1499238f1c` 포함) | 실기체가 쓰지 않는 코드 |
| `c890d9db0a` | 실기체 | `/root/PX4-vehicle` (worktree) | **있음** (가드 `2e59c98b7c` 포함, revert 없음) | 실기체 탑재본과 동일 |

가드란 `src/modules/fw_mode_manager/FixedWingModeManager.cpp` 의
`control_auto_position()` 에서 course 추종 분기를
`_vehicle_status.nav_state == NAVIGATION_STATE_GUIDED_COURSE` 로 제한하는 조건을 말한다.
가드가 없으면 오프보드가 채우지 않은 `course` 필드의 초기값 `0.0f` 가
"정북(0 rad) 유지" 라는 유효 명령으로 해석된다 (`msg/PositionSetpoint.msg:36`).

## 런 대응

| 런 디렉터리 | PX4 | 결과 | 근거 |
|---|---|---|---|
| `A1` `A2` `A3` `A4` `B1` `B6` | `9bb0d365c4` | A3 만 timeout(6.5km 폭주), 나머지 done | meta.json `px4_dir=/root/PX4-Autopilot`, 당시 HEAD (S4 시작 시점에 `sitl7-orig-head` 태그로 고정) |
| `A1_pxvehicle` | `c890d9db0a` | done — 기존 A1 과 동등 | meta.json `px4_head` (S4 에서 하니스에 추가) |
| `A3_pxvehicle` | `c890d9db0a` | **done — L자 완주** | 〃 |
| `A3_pxvehicle_try1_simstall` | `c890d9db0a` | timeout — **PX4 무관, 시뮬 정지** | 〃. CLIMBING 중 gz 실시간율이 0.79→0.05 로 급락해 미션 시계만 흘렀다. 같은 조건 재실행(`A3_pxvehicle`)이 완주했으므로 일회성 환경 실패로 판단하고 보존만 한다 |
| `B8_pxvehicle` | `c890d9db0a` | **done — 후방 300m 완주** | S5(Phase 2). 비-정북 레그 최초 실증 |
| `B2_pxvehicle` | `c890d9db0a` | done — 완만곡선 4WP 완주 | 〃 |
| `B3_pxvehicle` | `c890d9db0a` | done — 직각코너 완주 | 〃 |
| `B7_pxvehicle` | `c890d9db0a` | done — 단거리 40m 완주 | 〃. FOLLOWING 체류 1.0s |
| `B4_pxvehicle` | `c890d9db0a` | done — U턴 135° 완주 | 〃 |
| `B4_pxvehicle_try1_bringupfail` | `c890d9db0a` | exit 3 — **PX4 무관, 브링업 실패** | 〃. `mavros_not_connected` (MAVROS 가 120s 안에 FCU 접속 실패). 이륙 전이라 ulog 34.5MB 만 남았다. `wsl --terminate` 후 동일 조건 재실행(`B4_pxvehicle`)이 완주했으므로 일회성 환경 실패로 판단하고 보존만 한다 |
| `B5_pxvehicle` | `c890d9db0a` | done — 사각폐곡선 5WP 완주 | 〃. 플래너 블로킹 263.5s (기본 `--boot-timeout-s` 300s 에 근접) |
| `C2_pxvehicle` | `c890d9db0a` | done — 동쪽 300m(헤딩 90°) 완주 | S6(Phase 3 전반부). 정렬 잔류오차 −0.4° |
| `C1a_pxvehicle` | `c890d9db0a` | done — 천이고도 20m 완주 | 〃. 천이 첫 틱 setpoint 고도 계단 **+30.12m**(h_up 19.92→50.03) 실측 |
| `C1b_pxvehicle` | `c890d9db0a` | done — 천이고도 120m 완주 | 〃. 계단 **−69.78m**(h_up 119.77→49.99). 종점 포착 실패 → 종점 주위 2회 선회 후 종료 |
| `C8_pxvehicle` | `c890d9db0a` | done — 통제 HOME(35.9078/126.5310/3.0m) 완주 | 〃. `CommandTOL 이륙 요청 alt=53.0m AMSL (지면 3.0+50.0)` |
| `C5b_pxvehicle` | `c890d9db0a` | done — `d_end_thresh=30` | 〃. 종점 통과 +26.1m |
| `C5c_pxvehicle` | `c890d9db0a` | done — `d_end_thresh=60` | 〃. 종점 통과 없음(−1.5m) |
| `C10_pxvehicle` | `c890d9db0a` | **timeout(exit 2) — 예측된 결과** | 〃. `entry_mode=mid_flight`. ENTRY 무한대기(정적 감사 E-3). 480s 동안 WP0 반대방향으로 **5.85km 이탈**, disarm 없음 |

**C5a(`d_end_thresh=10`)는 별도 런을 만들지 않았다** — `A1_pxvehicle` 이 같은 경로·같은
기본값(`d_end_thresh=10`)이므로 그 수치를 C5 스윕의 10 지점으로 인용한다(S6).

S4 이후의 런은 `meta.json` 의 `px4_head` / `px4_dir` 와 `verdict.md` 헤더의
"PX4 빌드" 줄로 스스로를 식별한다 — 이 표에 의존하지 않아도 된다.
표가 필요한 것은 그 필드가 없던 기존 6건뿐이다.
