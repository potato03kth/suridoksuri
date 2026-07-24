---
doc_type: orchestrator_brief
scope: home_amsl 수렴판정 수정(PR #4, 병합완료) 검증 중 우연히 발견된 STREAMING 진입 직후 고도 오버슈트(3.0m 목표 → 5.5m대) 조사
status: ⏳ 부차 원인 2건 수정 완료, 진짜 근본원인은 다음 세션(사용자 주도 오케스트레이션)으로 이월 — `docs/sitl_verification_log.md` "작업 H-2" 체크리스트 실행이 다음 단계
created: 2026-07-24
last_updated: 2026-07-24
---

> **⚠ 다음 세션 진입점은 이 문서가 아니라 `docs/sitl_verification_log.md` "작업 H-2" 섹션이다.**
> 이 문서(§1)의 1차 가설(`climbing_reached()` 속도무시)은 부차 원인으로 정정·수정 완료(안전,
> 유지). 조사 중 드러난 **진짜 지배적 원인 후보는 이 문서 범위 밖의 훨씬 오래된 이슈** —
> `offboard_node.py`의 `self._home_amsl` 세션 내 재사용(수정 완료, 안전) 자체도 부차적이고,
> 검증 도중 완전히 신선한 SITL(재사용 아님)에서도 `home_amsl`·CommandTOL 계산이 둘 다 정확한데
> 실제 상승고도가 목표(3.0m)가 아니라 **요청한 AMSL 절대고도값(50.4) 자체**에 수렴하는 현상을
> 발견했다. 이건 **2026-07-11 "작업 H-2"가 이미 예견하고 전용 진단 체크리스트까지 만들어뒀던
> "geoid/ellipsoid 혼동 → 과상승" 시나리오와 증상이 정확히 일치**하는데, 그 체크리스트가
> "⏳ SITL 재검증 대기" 상태로 한 번도 실행된 적이 없었다(오늘 처음 안 사실). **다음 세션은 이
> 문제를 새로 파헤치지 말고 `docs/sitl_verification_log.md` "작업 H-2" 섹션(631~693행)의 기존
> 체크리스트를 `PX4_HOME_LAT/LON/ALT`로 통제된 조건에서 그대로 실행할 것.** 상세 경과는
> `docs/session_log.md` 2026-07-24 "STREAMING 오버슈트 조사" 항목(과 그 뒤 "⚠ 정정" 항목) 참조.
> 아래 원본 내용은 조사 과정의 기록으로만 유지한다.

# 다음 세션 브리프 — STREAMING 오버슈트 조사 (1차 가설 정정, 근본원인은 작업 H-2로 이월)

> **다음 세션 진입:** 이 문서 하나만 읽으면 된다. `docs/session_status.md`(트랙 보드)는
> "🚁 mc-실기체" 블록만, `docs/mc_hw_open_hypotheses.md`는 필요하면 참고(단 이 문서의 현상은
> 그 문서 범위인 "이함직후 롤폭주·추력부족"과 다른 카테고리 — 억지로 끼워맞추지 말 것).
> **코드 수정 전 반드시 SITL로 재현·검증할 것**(이 프로젝트 관례) — 이 노트북 SITL 환경은
> 이미 구축돼 있다(아래 §2), 재구축 불필요.

---

## 0. 지금까지 뭐가 끝났나 (한 문단, 상세는 링크만)

2026-07-24 세션에서 2026-07-23 저녁 실비행 사고(오프보드 3m 상승 명령이 30m로 실행 + 오프보드
미이행)를 분석해 근본원인(`offboard_node.py::_cb_home`이 `home_position`을 단발 스냅샷하는
설계 갭)을 특정하고, `home_amsl_confirmed()`(N회 연속 수렴판정) 수정을 구현·SITL로 재현검증까지
완료해 **PR #4로 `dev--vision-computing-module`에 병합 완료**했다(`pytest` 162 전부 통과, SITL
재현 시 26.7m급 오차·AUTO_LOITER 고착 둘 다 재현 안 됨, 미션 정상 완주 확인). 상세는
`docs/session_log.md` 2026-07-24 항목 + `logs/2026-07-23_flight01/notes.md`.

**이 브리프의 범위는 그 검증 과정에서 우연히 관찰된 별개 현상이다** — SITL 재현 비행에서
`CLIMBING`(목표 3.0m)이 정상 완료 판정됐음에도 실제 고도가 계속 올라가 **5.54m까지 오버슈트**된
뒤 서서히 3.0m 근방이 아니라 그냥 하강해버리는 게 관측됐다. home_amsl 수정 자체와는 무관하다고
판단해 그 PR에는 포함하지 않고 분리했다.

---

## 1. 현상 요약 (재분석 불필요, 이미 확정된 사실)

- 데이터: `logs/2026-07-24_sitl_streaming_overshoot/05_07_03.ulg`(PX4 SITL 원본 ulog, 29.2초) +
  `notes.md` + `analysis_auto.md`(이미 `analyze_flight.py`로 1차 분석 완료).
- **AGL 시계열(육안 확인 완료):** t≈9s부터 상승 시작 → t≈11s대에 3.0m 통과(가속 중, 감속
  안 됨) → **t=13.52s에 피크 5.54m** → 이후 5m대에서 수 초 진동하다 서서히 하강, t≈25s 착지.
- **`nav_state` 타임라인(analyze_flight.py, 교차검증 필요 — §4 참조):** `AUTO_TAKEOFF`가
  t=5.11~13.21s(8.1초) 유지되고, **AGL 피크(13.52s)가 이 AUTO_TAKEOFF 구간 거의 끝자락**에서
  발생 — `OFFBOARD` 확정은 t=13.21s로 피크 바로 직전.
- **유력 가설(코드 근거 있음, 미검증):** `fc_bridge/execution/state_logic.py::climbing_reached()`가
  속도를 보지 않고 `pos_ned[2]`가 목표±0.5m에 들어오기만 하면 "도달"로 판정 → 아직 상승 관성이
  남은(감속 전) 시점에 너무 이르게 `STREAMING`으로 넘어갔을 가능성. 이 시점엔 아직 OFFBOARD
  권한이 없어(`offboard_node.py` STREAMING 분기, 약 345~373행) 우리가 스트리밍하는 setpoint를
  PX4가 소비하지 않고, PX4는 계속 자기 AUTO_TAKEOFF를 진행 — 결과적으로 실제 정지는 PX4의
  AUTO_TAKEOFF 자체가 끝나는 시점(t=13.21s)에야 일어나고, 그때 이미 5m대. OFFBOARD 확정 후
  STREAMING의 `_mc_pos_ramp = state.pos_ned`(매 틱 현재위치 재캡처, 2026-07-20 flight01 사고
  대응으로 이렇게 설계됨)가 그 오버슈트된 고도를 그대로 "정상 위치"로 락인해버린 것으로 추정.
- **왜 이번 PR에 안 넣었나:** home_amsl 수정은 "이륙 *목표 고도*가 틀렸다"는 문제이고, 이건
  "이륙이 *목표 고도에서 못 멈춘다*"는 다른 층위의 문제 — 원인 후보(속도 무시 판정, PX4
  AUTO_TAKEOFF 자체 감속 부족, STREAMING의 늦은 OFFBOARD 확정)가 최소 3갈래라 섣불리
  손대면 또 다른 회귀를 만들 위험. 이번 세션은 관측·가설 수립까지만 하고 분리했다.

---

## 2. SITL 재현 (이미 구축된 환경, 재구축 불필요)

이 노트북 E드라이브에 `Ubuntu-22.04` WSL 배포판이 이미 있다(`docs/wsl_dev_env_setup.md` 섹션 F,
`docs/session_status.md` "환경 참조" 절). 진입 전 **반드시 존재 확인부터** (`wsl -d Ubuntu-22.04
-- echo ok`) — 없으면 그 문서대로 재구축, 있으면 아래 그대로 재사용.

```bash
wsl -d Ubuntu-22.04 --cd ~

# T1 — PX4 SITL (콘솔을 파일로 리다이렉트 금지 — pxh> 폭주로 로그 GB급 됨, §F 참조)
cd ~/PX4-Autopilot && HEADLESS=1 make px4_sitl gz_x500

# T2 — MAVROS (포트 14580, PX4 버전별로 다를 수 있음 — 안 붙으면 px4-rc.mavlink에서 실측)
ros2 launch mavros px4.launch fcu_url:=udp://:14540@localhost:14580

# T3 — 벤치 arm에 필요(SITL 전용, 실기체 파라미터에는 미적용)
ros2 param set /mavros/param CBRK_SUPPLY_CHK 894281
ros2 param set /mavros/param NAV_DLL_ACT 0

# T4 — 재현
cd ~/drone_ws/src/suridoksuri && git pull   # PR #4 병합분 받기
colcon build --packages-select fc_ros && source install/setup.bash
ros2 launch fc_ros phase2.launch.py vehicle_type:=mc transition_alt:=3.0 \
    waypoints:="[0.0,0.0,3.0, -3.0,-1.0,3.0, 0.0,0.0,3.0]"
```

재현 후 새 `.ulg`는 `~/PX4-Autopilot/build/px4_sitl_default/rootfs/log/<날짜>/`에 생긴다 —
`offboard_node`가 남긴 ARM 요청 로그의 epoch를 UTC로 변환하면 파일명(`HH_MM_SS.ulg`)과 정확히
일치한다(2026-07-24 확인된 방법). E드라이브(`/mnt/e/...`) 경유로 복사해 `logs/`에 커밋할 것.

---

## 3. 조사 순서 제안 (강제 아님, 판단은 다음 세션이)

1. **`nav_state`/`vehicle_control_mode.flag_control_offboard_enabled` 원본 재디코드**로
   §1의 타임라인부터 교차검증(analyze_flight.py 표를 곧이곧대로 쓰지 말 것 — §4 참조).
   정확히 언제 AUTO_TAKEOFF가 끝나고 OFFBOARD가 확정되는지, 그 순간 AGL이 얼마인지 확정.
2. `vehicle_local_position.vz`(수직속도)를 같이 보고, `climbing_reached()`가 "도달" 판정한
   순간의 속도가 0에 가까운지 아직 큰지 확인 — 크면 위 가설 ① 확정.
3. 확정되면 후보 수정: `climbing_reached()`에 속도조건 추가(예: `abs(vz) < 임계값`도 같이
   요구) — 단 `climbing_reached()`는 `state_logic.py`의 순수함수라 pytest로 먼저 회귀 없는지
   확인 후 SITL 재검증(이 프로젝트 관례, 실비행 코드).
4. 만약 속도조건을 추가해도 여전히 오버슈트되면(PX4 AUTO_TAKEOFF 자체의 감속 부족 문제라면)
   — 이건 우리 코드로 못 고치는 영역일 수 있으니 그 시점에 범위를 다시 사용자와 상의할 것.

---

## 4. 주의 (재확인 필요한 데이터 이상)

- `analysis_auto.md`의 `nav_state` 표가 t=0.0을 이미 `14:OFFBOARD`로 보여주는 등 앞뒤가 안
  맞는 값이 섞여 있다 — 이 SITL 인스턴스에 같은 세션 안에서 수동 arm/disarm 테스트를 먼저
  했던 부작용으로 추정되나 미조사. **표를 그대로 인용하지 말고 원본에서 재확인할 것.**
- 이 `.ulg`는 **home_amsl 회귀검증용으로 급조한 것**(정식 `record_flight.sh` 절차 미사용,
  rosbag·launch.log 없음) — 재현 시엔 가능하면 `record_flight.sh --sitl`로 정식 수집해
  launch.log(offboard_node 자체 로그)까지 같이 남길 것을 권장.

---

## 5. 참조

- `docs/session_log.md` 2026-07-24 항목 — home_amsl 수정 전체 경과(이 브리프의 전제)
- `logs/2026-07-24_sitl_streaming_overshoot/` — 이번에 관측된 원본 ulog+notes.md+analysis_auto.md
- `docs/wsl_dev_env_setup.md` 섹션 F, `docs/session_status.md` "환경 참조" — 이 노트북 SITL 재사용법
- 메모리 `project_fc_sitl_laptop_env.md` — SITL 환경 함정 3건(fc_bridge import, 프리플라이트 우회, 콘솔 로그 폭주)
- `fc_ros/fc_ros/nodes/offboard_node.py` STREAMING 분기(약 345~373행)·`_step_climbing()`(약 460행대)
- `fc_bridge/execution/state_logic.py::climbing_reached()`
