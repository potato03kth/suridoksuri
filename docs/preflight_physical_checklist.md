---
doc_type: procedure
project: suridoksuri-1
scope: 비행 전 물리 문제 해결 체크리스트 — 현장에서 그대로 따라가는 절차
last_updated: 2026-07-29
---

# 비행 전 물리문제 해결 체크리스트

> **쓰는 법:** 세션에 **"비행전 물리문제 해결을 위한 체크리스트"** 라고 말하면 이 문서를 연다.
> 근거·규명 과정은 `docs/fc_ground_diagnostics_2026-07-29.md` 에 있고, 여기는 **현장 실행용**이다.
> ①②는 **미해결 결함**이라 통과하지 못하면 그 비행은 같은 실패를 반복한다.

**전제:** 펌웨어는 `Build Jul 28 2026 20:27:22`(F-17/F-4 패치 + `crsf_rc`)여야 한다.
다르면 `docs/px4_v6c_patch_build.md` §11-6 부터 볼 것.

---

## ① pusher 추력 — **D단계 통전 테스트** (최우선)

flight02 에서 pusher 가 **100 % 지령(PWM 1900)으로 7.8초 고정된 구간의 배터리 전류가
호버보다 낮았다**(21.49 A vs 21.99 A). 프롭 달린 모터가 풀스로틀이면 불가능한 값이다.
FC 지령·PWM 출력·프롭 기계는 전부 정상으로 확인됐으므로 **원인은 ESC·모터·배선·아밍**에 있다.

### 절차

1. ⚠️ **pusher 프롭을 제거한다.** (MC 프롭도 빼두면 더 안전)
2. 배터리 연결 → ESC 기동음(비프) 청취. **pusher ESC 만 비프 패턴이 다르거나 없으면 그 자리에서 원인 확정**
3. 기체를 고정하고 pusher 에 저스로틀 지령을 넣는다
4. 판정

| 결과 | 뜻 | 조치 |
|---|---|---|
| **안 돈다** | ESC·모터·배선·아밍 확정 | ESC 교체 우선(가장 흔함) → 모터 → 커넥터 |
| **돈다** | 전기 경로 정상 | 프롭 장착 후 전류 측정으로 재확인. 공력/장착 방향 재점검 |

> 계측기(오실로스코프)는 **필요 없다.** 회전 여부만 보면 갈린다.

### 통과 기준
- pusher 가 지령에 따라 **회전**하고, 프롭 장착 상태에서 스로틀을 올렸을 때 **배터리 전류가 눈에 띄게 상승**한다.

---

## ② 자기계 — **야외 GPS fix 후 4방위 측정**

지상 실측에서 `mag_strength_gs` 가 기수에 따라 **−10 %** 변했다(북동 0.4213 G → 남서 0.3792 G).
\|B\| 는 캘리가 완전하면 방향 무관 상수여야 하므로 보정 불완전이다.
`EKF2_MAG_CHK_STR = 0.2`(하한 80 %) 기준으로 **남서에서 75.4 % = 탈락**이다.

### 절차

1. **GPS 3D fix 확보 후 2분 정차** — EKF 수렴 대기.
   ⚠️ 재부팅 직후 측정하면 미수렴 값이 나와 오판한다(2026-07-29 에 실제로 오판했다)
2. **위치를 고정한 채 제자리 회전.** 바닥에 표식을 쓰고, **벽 같은 기준물로 각도를 만들지 말 것**
   (기준물을 쓰면 방향과 위치가 함께 바뀌어 원인이 섞인다)
3. 북 → 동 → **남** → 서 4방위, 각 30초 정차
4. 각 방위에서 아래를 원격으로 뜬다 (요청하면 세션이 실행)

```bash
ssh suri@100.67.27.83 'sudo docker exec fc python3 \
  /drone_ws/src/suridoksuri/tools/px4_params/nsh_cmd.py /dev/ttyACM0 \
  "listener estimator_status" "listener estimator_innovation_test_ratios"'
```

### 통과 기준

| 항목 | 기준 |
|---|---|
| **`test_ratio`(남쪽 헤딩)** | **< 1** ← 이게 본 합격선 |
| `mag_strength_gs` 방향 간 편차 | 10 % 이내 |
| `mag_strength_gs` / `mag_strength_ref_gs` | **80 ~ 120 %** (`EKF2_MAG_CHK_STR`) |
| `cs_mag_fault` | OFF 유지 |

> 야외에서는 `mag_strength_ref_gs` 와 `mag_inclination_ref_deg` 가 **실제 값으로 나온다**(실내는 `nan`).
> 실내에서 `pre_flt_fail_mag_field_disturbed: False` 는 합격이 아니라 **채점 안 함**이었다.

### 통과 못 하면
- 재캘리 시 **기수를 남쪽에 둔 자세를 충분히** 포함(QGC 는 샘플이 차면 자동 종료되므로 각 면을 천천히)
- 그래도 안 되면 **자기계 물리 위치**(간섭원과의 거리) 문제 — `CAL_MAG0_ODIAG 0.0959`(정상 ±0.05 초과)가 soft-iron 잔차를 가리킨다
- `CAL_MAG1_PRIO = 0` 이라 **보조 자기계 교차검증이 꺼져 있다** — 활성화 검토 가능(단 현재 MAG1 offset 44.6 % 로 MAG0 보다 나쁨)

---

## ③ RC 링크 (플래시 이후 필수)

CRSF 드라이버가 빠져 조종기가 두절된 사고가 있었다(`docs/px4_v6c_patch_build.md` §11-2).

```bash
ssh suri@100.67.27.83 'sudo docker exec fc python3 \
  /drone_ws/src/suridoksuri/tools/px4_params/nsh_cmd.py /dev/ttyACM0 "crsf_rc status"'
```

| 항목 | 기대 |
|---|---|
| `crsf_rc status` | `command not found` 가 **아니어야** 함 |
| `UART device` | `/dev/ttyS1` (TELEM3) |
| `Invalid CRCs` | **0** |
| 조종기 전원 ON 후 스틱 | QGC 에서 채널 반응 확인 |

**킬스위치·모드 스위치 실물 확인**(`RC_MAP_KILL_SW=11`, `RC_MAP_FLTMODE=10`, `RC_MAP_ARM_SW=12`).

---

## ④ 배터리 — **지표를 믿지 말 것** (조치 아님, 주의사항)

`remaining` 은 이륙 4.5~7.7초에 **항상 0 %** 가 된다(11건 중 기동 7건 전부). 오탐이다.
`/mavros/battery` 의 `0 %` 도 같다. **비행 시간과 무부하 전압으로 직접 관리한다.**

> 🔴 **`COM_LOW_BAT_ACT` 를 켜지 말 것.** 지금은 `0`(Warning)이라 오탐을 무시해서 비행이 가능하다.
> `2`/`3` 으로 바꾸면 **만충 팩으로도 이륙 7초에 자동 착륙**한다. 근거는
> `docs/fc_battery_gate_survey.md` 최상단 배너.

---

## ⑤ 코드 신선도 (물리는 아니지만 같이 걸린다)

stale colcon build 가 실비행 8건의 근본원인이었다(`4dc30f9`).

```bash
ssh suri@100.67.27.83 'cd ~/drone_ws/src/suridoksuri && git log --oneline -1 && \
  md5sum fc_ros/fc_ros/nodes/offboard_node.py \
  ~/drone_ws/install/fc_ros/lib/python3.10/site-packages/fc_ros/nodes/offboard_node.py'
```
두 md5 가 **일치**해야 한다. 다르면 컨테이너에서 `colcon build --packages-select fc_ros`
(`--symlink-install` 금지, 절차는 `docs/rpi_deploy.md`).

---

## ⑥ launch 인자 — 손으로 직접 타이핑

flight01/02 를 날린 원인이다. 문서·채팅에서 복사하면 **U+00A0 가 섞여** 뒤 인자가 앞 인자 값으로
흡수된다. 이제 `_arg()`/`_check_choice()` 가 **기동을 거부**하므로 조용히 잘못 나는 일은 없지만,
**현장에서 기동 실패로 시간을 버리지 않으려면 직접 타이핑**한다.

---

## 판정 요약

| | 항목 | 통과 기준 | 상태 |
|---|---|---|---|
| ① | pusher 통전 | 회전 + 전류 상승 | ⏸ 미수행 |
| ② | 자기계 4방위 | 남쪽 `test_ratio < 1` | ⏸ 미수행 |
| ③ | RC 링크 | `Invalid CRCs 0` + 스틱 반응 | ✅ 2026-07-29 확인 |
| ④ | 배터리 | (지표 무시, 설정 변경 금지) | ✅ 결정 완료 |
| ⑤ | 코드 신선도 | md5 일치 | 비행 당일 재확인 |
| ⑥ | launch 인자 | 직접 타이핑 | 현장 |

**①② 중 하나라도 통과하지 못하면 천이·헤딩 정렬이 같은 방식으로 실패한다.**
