# SITL-3 튜닝 노트 & 추후 확인 사항

> 2026-06-30 기준. SITL-3 PASS 후 남은 튜닝 노브와 후속 검증 항목을 한눈에.
> 배경: `docs/sitl3_fix_plan.md`(해결 기록), 메모리 `feedback_px4_fw_offboard_position`.

---

## 1. 튜닝 노브

| 파라미터 | 현재값 | 위치 | 의미 / 언제 조정 |
|---|---|---|---|
| `_FW_LOOKAHEAD` | 70.0 m | offboard_node.py 상수 | FW 위치목표 전방거리. **목표를 빙빙 돌면(orbit) ↑**, **경로 모서리 질러가면 ↓**. 선회반경(~37 m)보다 커야 함 |
| `_HOLD_STABLE_REQ` | 10 틱 | offboard_node.py 상수 | WP1 착륙 전 연속 안정 요구 틱 (10 Hz → 1 초) |
| `wp1_land_radius` | 3.0 m | yaml | WP1 착륙 도달 반경. 정착이 안 되면 ↑ |
| `wp1_land_speed` | 1.5 m/s | yaml | 착륙 전 안정 수평속도 임계. 정착이 안 되면 ↑ |
| `hold_timeout` | 30.0 s | yaml | WP1 홀드 타임아웃(초과 시 강제 착륙). 경고 뜨면 위 둘 완화 |
| `d_end_thresh` | 10.0 m | yaml | 역천이 시작 거리. **WP1 오버슈트 줄이려면 ↑ (30~50 m)** |
| `l1_dist` | 20.0 m | yaml | L1 `compute()`용 — 현재 `cte` 진단에만 사용(조향은 `_FW_LOOKAHEAD`) |
| `wp0_heading_tol` | 0.05 rad | yaml | 천이 전 heading 정렬 허용오차 (≈2.9°) |

---

## 2. 현재 값 메모 (복구하지 않기로 결정 — 2026-06-30)

| 값 | 현재 | 비고 |
|---|---|---|
| `v_cruise` | 20.0 | SITL 종단감속 가시화용. FW에선 TECS가 속도 관장 → 경로추종 무관, **그대로 유지** |
| `waypoints` | `[0,0,50, 300,0,50]` (직선 300 m) | 테스트 경로. 실제 미션 좌표 주입 시 yaml **두 곳**(offboard_node·mission_node) 동시 수정 |

> 사용자 결정: 임시값을 되돌리지 않는다. 실제 미션 좌표 주입 시점에 `waypoints`만 교체.

---

## 3. 추후 확인 / 검증 (SITL-4·5)

- [ ] **SITL-4 전체 사이클** — 직선 외 L자/사각형 경로에서 FW 추종·코너 처리 확인. (FW는 타이트 곡률 못 따름 → WP 직선 레그 기준)
- [ ] **천이 가속도 측정** — flight_plan SITL-4 항목.
- [ ] **WP1 착륙 정밀도** — GPS 기준 WP1 도달 오차 측정. 오버슈트 크면 `d_end_thresh` ↑.
- [ ] **다구간 lookahead** — `target_point_ned`는 경로 끝에서 클램프. 다 WP 미션에서 세그먼트 전환 거동 확인. (WP 직전 lookahead 동적 축소는 flight_plan L687 향후개선 항목)
- [ ] **SITL-5 RPi4 배포** — apt 바이너리만, 소스빌드 금지.

---

## 4. 알아둘 거동 (버그 아님)

- **역천이 오버슈트** — FW 관성으로 WP1을 수십 m 지나친 뒤 MC로 복귀. HOLD가 WP1으로 되돌려 착륙. 줄이려면 `d_end_thresh` ↑.
- **속도·고도 제어 주체** — FW 구간 속도는 PX4 TECS, 고도는 위치 setpoint z(`_cruise_alt`, WP 고도 고정)가 담당. (역천이 감속은 PX4 back-transition)
- **MC↔FW 비대칭** — MC는 velocity setpoint OK, FW는 위치만. 새 상태 추가 시 어느 모드인지 먼저 확인.

### 죽은 코드 정리 (2026-06-30)

velocity→위치 전환으로 무효가 된 코드 제거:
- 적응형 `a_max`(cross-track 정체 감소) 일체 — 상태변수 6개 + 파라미터 4개(`a_max`/`error_stall_steps`/`accel_reduction`/`accel_min_frac`) 삭제.
- `_gamma`(상승각 vD용) — 고도는 위치 setpoint z가 담당하므로 삭제.

**보류 (제거 안 함):** `v_profile`/`apply_terminal_decel`/`speed_profile` — FW에선 v_cmd가 쓰이지 않아 효과는 없으나, ⓐ `L1Guidance` 생성자 API, ⓑ 레거시 `offboard_follower`(pymavlink), ⓒ `v_terminal ≥ 스톨×1.1` **안전 테스트**(`test_params`)와 얽혀 있어 단독 제거 시 부수효과 큼. 비용도 1회 계산뿐이라 유지. (제거하려면 L1 API·레거시·안전테스트까지 함께 정리 필요 — 별도 작업)

---

## 5. 더 견고한 대안 (미채택, 참고)

- **AUTO.LAND 직접 전환** — 명시적 역천이 대신 FW에서 바로 `AUTO.LAND` → PX4가 역천이+착륙을 네이티브 처리. TRANSITION_MC/HOLD keepalive 문제 자체가 사라짐. 단 착륙지점 정밀도는 PX4에 위임.
- **AUTO.MISSION** — WP 미션 업로드 후 PX4 내장 VTOL 미션 로직 사용(`mission_uploader`/`MissionNode` 존재). 가장 견고하나 동적 목표(vision)엔 덜 유연. → 현재는 **OFFBOARD+위치 채택**(향후 vision 연동 대비).
