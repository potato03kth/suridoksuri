# 2026-07-21_flight04

- **비행 조건:** (기체/모드/launch 인자: vehicle_type:=mc transition_alt:=4.0 waypoints:=[0.0,0.0,4.0, -4.24,-4.24,4.0, 0.0,0.0,4.0])
- **관찰:** straight up
- **결론:** half success

## 사용자 보고 현상 분석 (2026-07-21, 비행 직후)

사용자 증언: "중간에 2.5미터 부근에서 잠시 정지, 그후 쭉 올라감. 4미터로 보이는 지점에서
순간 yaw회전 두바퀴 후 착륙." ulog(`log_34`, pyulog 직접 디코드, `dist_bottom`+`vehicle_attitude`
쿼터니언+`vehicle_control_mode`+`trajectory_setpoint`+`battery_status`+`failsafe_flags` 대조) 결과 **두
현상 모두 재현·원인 특정됨, 서로 다른 두 개의 독립된 원인.**

### ①"2.5m 부근 잠시 정지" — PX4 배터리 페일세이프로 확정

- t=7.86s 고도(dist_bottom) 2.28m→t=8.37s 2.57m까지 상승 후, t=8.87~12.96s 사이 1.6~2.0m로
  주저앉아 정체(약 5초) → t=13.5s부터 재상승.
- 같은 구간 `battery_status`: t=3.16s 11.44V/12.7A(remaining 0.81)에서 t=6.16s 10.09V/31.6A로
  급락, remaining이 t=7.36s 0.13, t=9.76s 0.03, t=10.36s **0.00**까지 붕괴(고전류 28~35A 지속).
- `failsafe_flags`에 t=7.24s경 `battery_warning=1`, `battery_low_remaining_time=1` 동시 세팅 확인,
  같은 시각 모드타임라인상 `nav_state`가 `user_intention`(AUTO_TAKEOFF) 불변인 채
  `failsafe=1`로 **AUTO_LOITER 강제 전환**(PX4 자체 페일세이프, 조종기 아님) — t=12.04s
  `battery_warning` 해소와 함께 AUTO_TAKEOFF 재진입, 상승 재개.
- **결론: 정지가 아니라 PX4 배터리 저잔량 페일세이프로 인한 AUTO_LOITER 강제진입 — 이함 후
  급격한 전류소모(고도 상승 중 최대 32A)로 배터리 잔량추정치가 순간 붕괴한 것.** roll은 이
  구간 -6~-8° 수준으로 안정적이라(H12류 롤폭주와는 무관), 기존 열린가설 중 **H1/H2(배터리
  노후/용량부족)와 정합** — `docs/mc_hw_open_hypotheses.md` 참조.

### ②"4m 지점 순간 yaw 스핀 후 착륙" — fc_ros 코드 버그로 확정(신규)

- t=18.18s `nav_state`→OFFBOARD, t=18.5~22.7s 사이 `vehicle_attitude` yaw가 -94°→(0 경유)→
  +196°로 **연속 방향으로 약 268~290° 회전**, 최대 yaw rate **198.8°/s**(t=19.19s) — 이
  구간 고도(dist_bottom)는 3.85~4.4m(정확히 사용자가 말한 "4m 지점"과 일치).
  `quat_reset_counter` 불변(EKF 리셋 아님, 실제 자세 변화 확인). t=20.38s AUTO.LAND
  요청 이후 yaw가 176~182°에서 안정.
  (참고: 실측 회전량은 약 270~290°로, "두 바퀴"라는 인상은 순간 최대 200°/s의 빠른 회전이
  준 체감일 가능성 — 실제 720°까지는 아님. 단정하지 않고 기록만.)
- **원인 특정:** `trajectory_setpoint`를 직접 디코드한 결과 t=18.364s에 `yaw` 값이
  -97.64°→**+90.00°**로 순간점프(같은 틱에 `yawspeed=nan`), 이후 FOLLOWING/HOLD 구간 내내
  90.00°로 고정 발행됨. 90°(NED)는 ENU 기준 yaw=0(정체성 쿼터니언)을 NED로 환산한 값과
  정확히 일치 — `fc_ros/fc_ros/nodes/offboard_node.py::_publish_pos_setpoint()`(562~574행)가
  `PoseStamped`의 `pose.orientation`을 **한 번도 설정하지 않음**(위치 x/y/z만 채움) →
  ROS2 메시지 기본값(단위 쿼터니언, ENU yaw=0)이 그대로 나가 **현재 기체 헤딩과 무관하게
  yaw=90°(NED) 커맨드가 OFFBOARD 진입 첫 틱부터 발행됨**. FOLLOWING 상태 시작 시점 기체의
  실제 yaw는 -97.64°였으므로 최초 발행 순간 약 188° 순간점프 커맨드가 나갔고, 자세제어기가
  이를 쫓아가며 관측된 급격한 yaw 기동이 발생한 것으로 판단됨.
  - 부가 요인: 이 비행은 WP1(경로 끝점)이 시작지점과 매우 가까워(`cte=0.9m`) FOLLOWING이
    1틱(0.1s) 만에 즉시 완료 → `_step_hold()`(688~697행)로 전환되는데, 이 함수는
    `_step_following()`의 `_mc_pos_ramp` 슬루레이트 제한 없이 **목표점(wp1, cruise_alt)을
    직접 발행** — `trajectory_setpoint.position`이 t=19.365s에 (1.29,-0.02,-3.08)→
    **(0.00,0.00,-4.00)**로 약 1.6m를 한 틱(0.2s) 만에 순간이동(허용 슬루 한도
    `v_approach`=5.0m/s×0.2s=1.0m 초과) — yaw 점프와 별개로 위치도 함께 순간점프해 자세
    보정을 가중시켰을 가능성.
  - **2026-07-20 flight01 제어상실 사고(`docs/session_status.md` 🚁 트랙 "그 전" 항목,
    `8ea5e35`)의 재발 변종으로 판단** — 그때는 위치 순간점프가 원인이었고 `_mc_pos_ramp`로
    위치만 슬루제한했는데, **yaw는 애초에 어떤 상태에서도 설정된 적이 없어 이 경로 자체가
    미수정 상태로 남아있었음**. `_step_hold()`도 위치 슬루제한이 빠져있어 짧은 경로(WP1이
    시작점과 가까운 케이스)에서 위치점프가 재발할 수 있음.
- **미적용:** 이 세션에서는 분석만 수행, `offboard_node.py` 수정은 하지 않음(사용자 확인 후
  진행 여부 결정 — 실비행 코드라 SITL 회귀검증 없이 바로 고치는 것은 이 저장소 관례상
  보류). 다음 세션 후보 수정: ① `_publish_pos_setpoint()`에 실제 원하는 목표 헤딩(또는
  최소한 "현재 yaw 유지")을 명시적으로 설정 ② `_step_hold()`에도 `_step_following()`과
  동일한 슬루레이트 제한 적용.
