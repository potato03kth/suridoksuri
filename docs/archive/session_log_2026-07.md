---
doc_type: session_log_archive
project: suridoksuri-1
period: 2026-07-03 ~
---

# 세션 로그 아카이브 — 2026-07

> `docs/session_log.md`에서 이동된 과거 세션 기록 (최신이 위).
> 현행 로그는 최근 8개 세션만 유지하며, 초과분은 `/session-log` 실행 시 이 디렉터리로 이동된다.

---

## 2026-07-03 — 실기체 MC 브링업 (RPi5/24.04 + PX6C)

**브랜치:** `dev--vision-computing-module`
**목적:** RPi5(Ubuntu 24.04) + Pixhawk 6C **순수 MC** 테스트기체에 fc_ros 배포·검증 (SITL-5 변형)

### 완료

- **`vehicle_type` 런타임 파라미터 추가** — `"vtol"`(기본)|`"mc"`. MC는 FW 천이 2단계(TRANSITION_FW/TRANSITION_MC) 생략하고 CLIMBING→STREAMING, FOLLOWING→HOLD 직행. 코드 분기만, **VTOL 동작 불변**. 순수함수 `after_climb_state`/`after_following_state` + 테스트 4개 추가(90 passed)
- **launch 런타임 오버라이드** — `phase2.launch.py vehicle_type:=mc` + yaml 기본값. 코드 교체 없이 파라미터로 MC 전환
- **RPi5 배포 환경 구축** — Docker `ros:humble` 컨테이너(이름 `fc`, 항상 `sudo`), MAVROS·numpy 설치, fc_ros colcon 빌드, fc_bridge+vtol_sim은 PYTHONPATH(`/drone_ws/src/suridoksuri`)로 로드
- **Pixhawk 6C 펌웨어 ArduCopter→PX4 교체** — PC 데스크톱 QGC로 플래시, 에어프레임/캘리브레이션 재설정, **수동비행 검증 성공**

### 결정

- **RPi5(24.04)는 Docker Humble로 운용** — Humble이 22.04 전용이라. **개발컴은 22.04/Humble 유지**(업그레이드 안 함). 네이티브 Jazzy 미채택("오류 나면 안 됨" 우선 → 검증된 Humble 환경 재현)
- **MC 추종은 위치 setpoint 재사용** — 속도+L1 복원 안 함. 속도는 PX4 MPC가 관장, `v_terminal`/`decel_dist`는 MC에서 무의미
- **MC 검증은 코드포크가 아니라 파라미터 스위치로** — SITL은 gz_x500(=MC)로 선검증

### 다음 세션

1. **MAVROS 링크 문제 해결** — RTT 2~5초·heartbeat 플래핑·935 params 정체. 태블릿 QGC 끊고 **USB 직결**로 링크 안정화부터
2. **AUTO.TAKEOFF 미실행 진단** — offboard가 이륙명령 발행 안 함. (a) MAVROS 서비스 미준비인지 (b) PX4 GPS 락 없어 AUTO.TAKEOFF 거부인지 `statustext`로 판별
3. **커밋** — `vehicle_type` 변경 등 이번 세션 전체 미커밋

### 주의

> **근본 교훈: 6C는 ArduCopter였다.** 우리 코드·SITL 검증은 전부 **PX4 전용**(모드명·AUTO.TAKEOFF·OFFBOARD·vtol_state). 실기체는 PX4 확인부터.
> **AUTO.TAKEOFF는 GPS 락 필수** — 수동비행 성공 ≠ GPS 락. 실내/벤치 불가.
> **웨이포인트 비퇴화 필수** — 시작=끝 동일하거나 초단거리 레그면 플래너 divide-by-zero(NaN).
> **이번 세션 전체 미커밋.**
