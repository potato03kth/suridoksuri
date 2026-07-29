# B2 — 판정

- 목적: 완만 곡선 4WP(30°급 꺾임) — eta3 NR 경로(_trapz 수정) 실행 검증
- 실행: 2026-07-29T04:46:22.879249+00:00 ~ 2026-07-29T05:01:04.717438+00:00 (경과 824.5s)
- 종료: `timeout` (exit=2)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 150.0,0.0,50.0, 300.0,80.0,50.0, 450.0,200.0,50.0] range_limit_m:=800.0`
- 저장소 HEAD: `bc3229e`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 04_46_35.ulg (meta.json 기록: 04_46_35.ulg)
- 요약: FAIL 2, NULL 7, PASS 3, WARN 1

- 시각 정렬: `wall = 1.00000 x ulog + 1785300405.920` (앵커 1개, 최대 잔차 0.000s). 시뮬 클록이 벽시계보다 +0.0% 빠름/느림 — 상수 오프셋만 쓰면 0.00s 벌어진다
  - ⚠️ 앵커가 1개뿐(또는 구간이 5초 미만)이라 scale=1 상수 오프셋으로 폴백 — 시뮬/벽시계 클록 드리프트가 보정되지 않았다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **FAIL** | 관측 상태: ARM_TAKEOFF; 종료사유=timeout | DONE 상태 도달 |
| disarm 확인 | **NULL** | ulog 상 armed(=2) 구간이 없음 — ARM 실패 | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **FAIL** | seq=[3], 정천이 Nones / 역천이 Nones | 3→1→4, 4→2→3 |
| setpoint 점프 | **PASS** | 임계 1.5m, 경계±1s 위반 0건 / 전체 위반 0건 / 샘플 4021개, 최대 0.1129m (0.5643 m/s). 경계 최대: -. 스트림 재개 갭 1건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **PASS** | 피크 \|az\|=0.1183 m/s² (0.0121g) @794.904s state=ARM_TAKEOFF; 접지 제외값 없음(disarm 시각을 몰라 접지 구간을 제외할 수 없음) | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **NULL** | 역천이 구간(vtol_state==2 또는 TRANSITION_MC 상태창)을 특정할 수 없음 — 역천이가 일어나지 않았을 수 있다 | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **NULL** | TRANSITION_FW 상태창이 없음 (상태 미도달이거나 시각정렬 실패) | 오버슈트 없이 단조수렴, 정렬완료 err ≤ wp0_heading_tol |
| CLIMBING 오버슈트 | **NULL** | CLIMBING 상태창 없음 (시각정렬 실패 또는 상태 미도달) | transition_alt 대비 최대 AGL ≤ +10% |
| 정천이 고도손실 | **NULL** | 정천이 구간(vtol_state==1)을 특정할 수 없음 — 천이 미발생 가능 | ≤ 5m |
| 순항 고도편차 | **NULL** | FOLLOWING 상태창 또는 순항고도 기준을 알 수 없음 | ±3m |
| FW cte | **NULL** | node.log 에 'FOLLOWING tick= ... cte=' 샘플이 없음 (FOLLOWING 미진입이거나 20틱 미만 체류) | 직선 ≤ 2m |
| 경고/타임아웃 | **WARN** | node.log 12건/8종, mavros.log 97건/11종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 639.7291 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.368 | 807.024 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 2 | ≈1785301263.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785301263.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785301263.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785301263.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785301263.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 49317, exit code -2, cmd '/root/ws_f5/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __node: |
| node.log | ERROR | 1 | ≈1785301263.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 49319, exit code -2, cmd '/root/ws_f5/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node:=o |
| node.log | WARN | 1 | ≈1785300623.6 | stdout 중계(비-ROS 포맷) | [offboard_node-2] [Eta3ClothoidPlannerV3] WARNING: NR pos residual 9.450m is large. affine correction guarantees WP passage but curve may be deformed. |
| node.log | WARN | 1 | ≈1785301263.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | WARN | 42 | 1785300417.1 |  | PR: Failed to get parameter type: NAV_DLL_ACT |
| mavros.log | ERROR | 16 | 1785300451.5 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | ERROR | 13 | 1785300395.8 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 6 | 1785300394.1 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 6 | 1785300396.9 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-16-18-1-128-0-0-0-0-0-0-0-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 4 | 1785300392.3 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785300400.1 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785300408.9 |  | TM: RTT too high for timesync: 2012.32 ms. |
| mavros.log | WARN | 1 | 1785300395.9 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | WARN | 1 | 1785300402.9 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785301264.9 |  | UAS Executor terminated |

## 미산출 지표 (null)

- **disarm 확인**: ulog 상 armed(=2) 구간이 없음 — ARM 실패
- **역천이 감속률**: 역천이 구간(vtol_state==2 또는 TRANSITION_MC 상태창)을 특정할 수 없음 — 역천이가 일어나지 않았을 수 있다
- **TRANSITION_FW 헤딩**: TRANSITION_FW 상태창이 없음 (상태 미도달이거나 시각정렬 실패)
- **CLIMBING 오버슈트**: CLIMBING 상태창 없음 (시각정렬 실패 또는 상태 미도달)
- **정천이 고도손실**: 정천이 구간(vtol_state==1)을 특정할 수 없음 — 천이 미발생 가능
- **순항 고도편차**: FOLLOWING 상태창 또는 순항고도 기준을 알 수 없음
- **FW cte**: node.log 에 'FOLLOWING tick= ... cte=' 샘플이 없음 (FOLLOWING 미진입이거나 20틱 미만 체류)

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
