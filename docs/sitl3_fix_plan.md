# SITL-3 해결 기록 — FW 천이·경로추종·착륙

> **2026-06-30 SITL-3 PASS.** 이 문서는 당초 "수정 계획"을 대체하는 **해결 기록**이다.
> 핵심 교훈: PX4 **고정익(FW) 오프보드는 velocity setpoint를 무시하고 위치 setpoint만 추종**한다.
> 멀티콥터(MC)는 velocity OK — 이 **MC↔FW 비대칭**이 모든 버그의 뿌리였다.
> 튜닝/추후 확인: `docs/sitl3_tuning_notes.md` · 메모리: `feedback_px4_fw_offboard_position`

---

## 1. 최종 상태머신

```
ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING
  → TRANSITION_MC → HOLD → LANDING → DONE
```

| 구간 | 모드 | setpoint | 비고 |
|---|---|---|---|
| ARM_TAKEOFF / CLIMBING | AUTO.TAKEOFF | — | 천이 고도 도달 |
| TRANSITION_FW Phase 1·2 | MC OFFBOARD | velocity(zeros) + yaw_rate | hover + heading 정렬 |
| TRANSITION_FW Phase 3 / ACTIVE | OFFBOARD | **위치 (WP1)** | 직선 천이 |
| STREAMING / FOLLOWING | FW OFFBOARD | **위치 (lookahead 70 m)** | GPS 경로 추종 |
| TRANSITION_MC | OFFBOARD | **위치 (전방 70 m 먼점)** | 직선 역천이 감속 |
| HOLD | MC OFFBOARD | **위치 (WP1)** | WP1 복귀·홀드 |
| LANDING | AUTO.LAND | — | WP1 수직 착륙 |

---

## 2. 근본 원인 (단일)

**PX4 FW 오프보드는 velocity setpoint를 무시한다.** 위치(`x,y,z`)만 추종하고 velocity/acceleration은 무시한다(PX4 공식 문서). velocity만 주면 FW가 추종할 경로가 없어 기본 **"flower-like pattern"(반경 ~37 m 원) 선회**로 빠진다.

이 하나가 세 증상을 모두 만든다:

- **버그 1 (원호 천이)** — 당초 "PX4 천이 컨트롤러가 방향 권한 약화"로 오진. 실제는 FW 진입 **직후** velocity 무시 → flower-pattern. 천이 자체(`vtol=1` 구간)는 직선이었음(SITL 로그 확인).
- **버그 2 (경로가 초기 heading 종속)** — FW가 velocity 방향을 무시 → 진입 자세대로 직진.
- **FOLLOWING 미진입** — STREAMING의 `vel_aligned_with_path` 15° 게이트가 영원히 불만족(속도가 경로로 정렬될 수 없음).

> 당초 `frame_id="base_link"` 가설은 **기각**: 실행 설정은 `local_origin`이었고 NED↔ENU 변환은 전부 정상이었다.

검증(PX4 문서): velocity-only setpoint는 FW 미지원, type_mask 부적합 시 "flower-like pattern" — 로그의 원 선회와 일치.

---

## 3. 적용된 수정

파일: `fc_ros/fc_ros/nodes/offboard_node.py`, `fc_bridge/guidance/l1_guidance.py`

1. **위치 setpoint 발행** — `_publish_pos_setpoint()`: NED`[N,E,h_up]`→ENU `PoseStamped` → `/mavros/setpoint_position/local`. `local_position/pose`와 동일 프레임 → GPS(EKF) 추종. FW 활성 구간 전부 이걸로.
2. **lookahead 목표** — `L1Guidance.target_point_ned(pos, lookahead=70 m)`: 경로 위 전방점. **lookahead > 선회반경(~37 m)** 필수 — 작으면 목표를 orbit(flower-pattern 재발).
3. **STREAMING 단순화** — 속도 발행·`vel_aligned` 15° 게이트 삭제 → "위치 발행 + OFFBOARD 확인 → FOLLOWING".
4. **TRANSITION_MC keepalive** — 역천이 대기 중 위치 발행 추가. 없으면 무발행 → offboard 상실 failsafe → RTL (이전 **151 m RTL** 버그).
5. **역천이 직진** — 끝점(근접) 대신 `현재위치 + 최종방향·70 m` 먼점 목표 → 근접목표 급선회(**동향 45° 꺾임**) 제거.
6. **WP1 착륙 (HOLD 상태 신규)** — 역천이 오버슈트 후 MC로 WP1 복귀·홀드(`wp1_land_ready`: dist<3 m · speed<1.5 m/s · 10틱) → 그 자리 AUTO.LAND. MC는 근접목표 추종 가능.

**당초 계획에서 폐기된 것**:
- 사전가속(Phase 2.5) — **불필요**. 가속해도 FW의 velocity 무시는 동일.
- 버그 1·2를 별개 문제로 본 프레이밍 — **단일 원인**.

---

## 4. 검증

- 단위 테스트: `python -m pytest fc_ros/test fc_bridge/tests -q` → **108 passed**
  - 신규 순수 함수 테스트: `target_point_ned`(전방성/끝점클램프/경로복귀/선회반경초과), `wp1_land_ready`(근접·정착/경계).
- SITL(직선 300 m): 이륙 → 헤딩정렬 → **직선 천이** → **FW 직선 추종** → **직선 역천이 감속** → **WP1 복귀·홀드** → **WP1 착륙** 정상.

---

## 5. 관련 문서

- `docs/sitl3_tuning_notes.md` — 튜닝 노브 & 추후 확인 사항
- 메모리: `feedback_px4_fw_offboard_position`(FW 위치 setpoint 필수), `feedback_px4_mc_offboard_yaw`(MC yaw rate), `project_sitl_state`
