---
doc_type: session_status
project: suridoksuri-1
scope: FC 작업단위(코드 A~E / SITL-1~5) 진입 전 환경 상태 및 브랜치 전략
last_updated: 2026-06-18
---

# FC 세션 진입 상태 문서

> 새 세션에서 FC 작업을 시작할 때 이 문서를 먼저 읽는다.  
> 작업단위별 내용·테스트는 `docs/flight_plan.md` 참조.

---

## 브랜치 전략

| 단계                | 브랜치                         | 작업                                         |
| ------------------- | ------------------------------ | -------------------------------------------- |
| 현재                | `dev--vision-computing-module` | 통합 launch 기동(SITL-2) 확인 후 `main` 병합 |
| 작업 A~E / SITL-1~5 | `dev--fc-vtol-sitl` (가칭)     | `main`에서 분기, VTOL 비행 사이클 구현·검증  |

> 작업단위 정의는 `docs/flight_plan.md`로 재구성됨: **[코드] 작업 A~E**(Claude 자율, pytest) + **[SITL] SITL-1~5**(사람 수행) + 후속(작업 F·SITL-6). 구 "세션 A~F" 명칭은 폐기.

**통합 launch 기동 (= flight_plan SITL-2):**  
`ros2 launch fc_ros phase2.launch.py` 기동 시 TelemetryNode + OffboardNode가 오류 없이 뜨는지 확인.  
→ 이 기동을 위해 **작업 A**(`fc_ros_params.yaml` flat 변환 + 신규 파라미터)가 **선행** 필요 (아래 "다음 작업 순서" 참조).

**작업단위(A~E / SITL-1~5) 브랜치 생성 절차 (main 병합 후):**

```bash
git checkout main
git checkout -b dev--fc-vtol-sitl
```

---

## WSL drone_ws 상태

| 항목               | 내용                                                  |
| ------------------ | ----------------------------------------------------- |
| 연결 방식          | 직접 복사 — Windows repo와 별도 (`~/drone_ws/src/`)   |
| 마지막 colcon 빌드 | 2026-06-18 (MissionNode 버그 수정 반영됨)             |
| 기체 모델          | `gz_x500` (SITL-1에서 `gz_standard_vtol`로 전환 예정) |

### 코드 동기화 절차

Windows에서 코드를 수정·커밋한 후 SITL 테스트 전 WSL에서 수행:

```bash
# WSL 터미널에서 실행
cd ~/drone_ws
git pull
colcon build --packages-select fc_ros fc_bridge
source install/setup.bash
```

> `source install/setup.bash`는 빌드 후 매번 실행해야 변경사항이 반영된다.

---

## SITL 기동 명령

### 현재 (gz_x500)

```bash
# T1 — PX4 SITL
cd ~/PX4-Autopilot && make px4_sitl gz_x500

# T2 — MAVROS
ros2 launch mavros px4.launch fcu_url:=udp://:14540@localhost:14557

# T3 — fc_ros
cd ~/drone_ws && source install/setup.bash
ros2 launch fc_ros phase2.launch.py
```

### SITL-1 이후 (gz_standard_vtol)

T1만 변경:

```bash
cd ~/PX4-Autopilot && make px4_sitl gz_standard_vtol
```

---

## QGC ↔ WSL 연결 (PX4 재기동마다)

```bash
# Step 1 — IP 확인 (WSL)
WIN_IP=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'); echo "Windows IP: $WIN_IP"

# Step 2 — PX4 콘솔
pxh> mavlink start -x -u 14551 -r 4000000 -t <WIN_IP>

# Step 3 — QGC (Windows)
# Application Settings → Comm Links → Add → UDP 14551 → Connect
```

상세 절차: `docs/sitl_verification_log.md` "Windows QGC ↔ WSL SITL 연결" 섹션 참조.

---

## 다음 작업 순서

```
1. 작업 A — fc_ros_params.yaml flat 변환 + 신규 파라미터 (Claude 자율, pytest)
         ↓
2. 동기화 + colcon build
         ↓
3. SITL-2 — ros2 launch fc_ros phase2.launch.py 기동 확인 (통합 launch 기동)
         ↓
4. dev--vision-computing-module → main 병합
         ↓
5. dev--fc-vtol-sitl 브랜치 생성
         ↓
6. 작업단위 진입 (docs/flight_plan.md 권장 실행 순서: SITL-1 / 작업 A~E 병행)
```

---

## 작업단위별 선행 조건 요약

> 상세·테스트는 `docs/flight_plan.md` 참조. [코드]=Claude 자율(pytest), [SITL]=사람 수행.

| 작업단위                                   | 유형          | 선행             | SITL 기체          |
| ------------------------------------------ | ------------- | ---------------- | ------------------ |
| 작업 A — params/YAML 정비                  | [코드]        | 없음             | — (pytest)         |
| 작업 B — 종단 감속 헬퍼                    | [코드]        | 없음             | — (pytest)         |
| 작업 C — 상태머신 ① 이륙·상승·천이         | [코드]        | A, (SITL-1 상수) | — (pytest)         |
| 작업 D — 상태머신 ② 역천이·착륙            | [코드]        | C                | — (pytest)         |
| 작업 E — 긴급 override                     | [코드]        | C                | — (pytest)         |
| SITL-1 — VTOL 환경 전환 + 상수             | [SITL]        | 환경             | `gz_standard_vtol` |
| SITL-2 — launch 통합 기동                  | [SITL]        | A                | `gz_standard_vtol` |
| SITL-3 — 경로 추종 검증                    | [SITL]        | B·C·D, SITL-2    | `gz_standard_vtol` |
| SITL-4 — 전체 사이클 통합                  | [SITL]        | E, SITL-3        | `gz_standard_vtol` |
| SITL-5 — RPi4 배포                         | [배포]        | SITL-4           | RPi4 실기체        |
| 작업 F / SITL-6 — 임의 WP 생성·추종 (후속) | [코드]/[SITL] | SITL-4           | `gz_standard_vtol` |
