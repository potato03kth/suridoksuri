---
doc_type: procedure
project: suridoksuri-1
scope: 2026-07-28 PX4 재플래시 — 플래시 후 대조·복구 체크리스트
last_updated: 2026-07-28
---

# 플래시 후 대조 체크리스트

**전제:** 사용자가 QGC로 Pixhawk 6C를 F-17/F-4 패치본(`px4_fmu-v6c_default.px4`)으로
직접 플래시한다. 이 문서는 **플래시가 끝난 뒤** 검증자가 그대로 따라가는 절차다.

- 백업 원본: `px4_params_2026-07-28_pre-flash.json` / `.txt` / `.csv` / `.params` (1437개)
- 결정적 항목 하이라이트: `CRITICAL_PARAMS.md`
- 도구: `tools/px4_params/dump_px4_params.py`, `render_params.py`, `compare_px4_params.py`
- 빌드 근거: `docs/px4_v6c_patch_build.md`

---

## 0. 플래시 전 baseline (2026-07-28 02:38 KST 실측, 이미 채취 완료)

```
HW arch: PX4_FMU_V6C
HW type: V6C000000
PX4 git-hash: c890d9db0a300795594fd5ba6c045be9ebd71c09
PX4 version: 1.18.0 40 (17956928)
PX4 git-branch: main
OS: NuttX / Release 11.0.0 (184549631)
Build datetime: Jul  7 2026 13:20:51        <-- 교체 판별의 기준선
Build uri: localhost
Build variant: default
Toolchain: GNU GCC, 14.2.1 20241119         <-- 교체 판별의 기준선
PX4GUID: 00060000000033363532333351190028002b
MCU: STM32H7[4|5]xxx, rev. V
```

파라미터 개수 1437, `SYS_AUTOSTART = 13000`.

> ⚠ **플래시 직전에 `.px4` 파일 sha256을 반드시 재확인**한다
> (`docs/px4_v6c_patch_build.md` §4-2/§7). 기대값
> `f1c16e2b3799a352a73d1e9c4cfe31fb4b6775d89b1e5ccc189fc3e5c5b47dda`,
> 크기 1,837,236 B. 순정은 1,837,292 B / `210ace1b…`.

---

## 1. 플래시 후 — 펌웨어가 실제로 교체됐는지 판별

`ver all` 을 QGC의 MAVLink Console 또는 아래 스크립트로 뜬다.

```bash
# RPi5 에 USB를 되꽂은 뒤, 컨테이너 fc 안에서
sudo docker exec fc python3 /tmp/ver_all.py /dev/ttyACM0
```
(스크립트가 없으면 `tools/px4_params/` 옆에 두거나 QGC MAVLink Console 에서 `ver all` 직접 입력)

### 판정표

| 항목 | 플래시 전 | 플래시 후 기대 | 판정 |
|---|---|---|---|
| **Build datetime** | `Jul  7 2026 13:20:51` | **`Jul 28 2026 ...`** (오늘 빌드) | **날짜가 바뀌어야 교체 성공.** `Jul 7` 그대로면 플래시가 안 먹은 것 |
| **Toolchain** | `GNU GCC, 14.2.1 20241119` | **`GNU GCC, 10.3.1 20210621`** | 패치본은 WSL의 arm-none-eabi-gcc 10.3.1 로 빌드됐다 (`px4_v6c_patch_build.md` §2). **가장 강한 식별자** |
| PX4 git-hash | `c890d9db0a…` | **동일** `c890d9db0a…` | **같은 것이 정상.** 패치가 버전 문자열을 안 바꾼다 — PX4가 `git describe` 를 `--dirty` 없이 부르기 때문(§4-2) |
| PX4 version | `1.18.0 40 (17956928)` | 동일 | 같은 커밋이므로 동일이 정상 |
| HW arch / PX4GUID | `PX4_FMU_V6C` / `0006…0028002b` | 동일 | 같은 보드인지 확인용 |

> 🚨 **git-hash 가 같다고 "플래시 실패"로 오판하지 말 것.**
> 교체 여부는 **Build datetime + Toolchain** 두 줄로만 판정한다.
> 반대로 이 두 줄이 그대로면 **플래시가 안 들어간 것**이다 (stale build 계열 함정, `4dc30f9`).

> 참고: `ver all` 로는 **패치본인지 순정인지**를 구별할 수 없다. 위 판정은
> "Jul 7 빌드에서 오늘 빌드로 바뀌었다"까지만 보증한다. 패치가 실제로 들어갔는지는
> 비행 거동(F-17 천이 헤딩 / F-4 course)으로 확인한다 — `px4_v6c_patch_build.md` §6.

---

## 2. 플래시 후 — 파라미터 대조 (자동)

### 2-1. 재덤프

```bash
# RPi5 호스트에서 (스크립트는 저장소에 있음)
cd ~/drone_ws/src/suridoksuri && git pull
sudo docker cp tools/px4_params/dump_px4_params.py fc:/tmp/dump_px4_params.py
sudo docker exec fc python3 /tmp/dump_px4_params.py /dev/ttyACM0 /tmp/px4_params_post.json
```

노트북으로 회수:

```bash
ssh suri@100.67.27.83 'sudo docker exec fc cat /tmp/px4_params_post.json' \
  > logs/2026-07-28_px4_flash/px4_params_2026-07-28_post-flash.json
```

> ⚠ MAVROS가 떠 있으면 시리얼을 물고 있어 덤프가 실패한다.
> `fuser /dev/ttyACM0` 로 무점유 확인 후 실행. (덤프 스크립트는 pyserial 불요 —
> 컨테이너 `fc` 에 pyserial이 없어서 raw termios + pymavlink 파서로 직접 연다.)

### 2-2. 자동 비교

```bash
python3 tools/px4_params/compare_px4_params.py \
  logs/2026-07-28_px4_flash/px4_params_2026-07-28_pre-flash.json \
  logs/2026-07-28_px4_flash/px4_params_2026-07-28_post-flash.json

# 결정적 항목만 보고 싶으면
python3 tools/px4_params/compare_px4_params.py <before> <after> --critical-only
```

차이 나는 항목만 `CHANGED` / `MISSING` / `ADDED` 로 뽑아준다.
종료 코드 **0 = 완전 일치**, **1 = 차이 있음**.

### 2-3. 판정 기준

| 결과 | 해석 | 조치 |
|---|---|---|
| 차이 0건 | 정상. 같은 버전이라 PX4가 파라미터를 유지했다 | 그대로 진행 |
| `CHANGED`/`MISSING` 에 `CAL_MAG*` 포함 | **자기계 캘리브레이션 소실** | §4 복구 → 안 되면 재캘리브레이션 |
| `CAL_ACC*` / `CAL_GYRO*` 소실 | IMU 캘리브레이션 소실 | §4 복구 |
| `SYS_AUTOSTART` 가 13000 이 아님 | **기체 프레임이 리셋됨.** 나머지 전부 기본값일 가능성 | §4 전량 복구 필수 |
| `ADDED` 만 소수 | 패치로 새 파라미터가 생겼을 수 있음 | 값 확인만, 문제 아님 |
| 부동소수 마지막 자리만 다름 | REAL32 왕복 오차 | 비교기가 `rel_tol=1e-6` 로 이미 흡수 — 뜨면 실제 변경 |

### 2-4. 눈으로도 확인할 최소 6종

```
SYS_AUTOSTART   = 13000
FW_AIRSPD_TRIM  = 15.0
FW_AIRSPD_MIN   = 10.0
FW_AIRSPD_MAX   = 20.0
MPC_THR_HOVER   = 0.5
CAL_MAG0_ID     = 396809     (CAL_MAG1_ID = 396321)
```

---

## 3. 자기계가 특히 중요한 이유 (잊지 말 것)

2026-07-25에 헤딩 65° 오차가 났고, 원인은 자기 환경이 아니라 **저장된 캘리브레이션 값**
이었다(커밋 `655f539`). 그 값들이 `CRITICAL_PARAMS.md` §1에 전량 박제돼 있다.
플래시로 날아가면 **비행 전 반드시 복구하거나 재캘리브레이션**한다 — 그냥 날리면 7/25가 재현된다.

---

## 4. 복구 절차 — 파라미터가 날아갔을 때

### 4-A. QGC로 전량 복원 (권장)

1. QGC를 기체에 연결.
2. **Vehicle Setup → Parameters → Tools → Load from file...**
3. `logs/2026-07-28_px4_flash/px4_params_2026-07-28_pre-flash.params` 선택.
   (탭 구분 QGC 표준 포맷 — `sysid compid name value MAV_PARAM_TYPE`, 1437행)
4. QGC가 변경될 항목 목록을 보여준다 → 확인 후 적용.
5. **기체 재부팅**(전원 재인가). 재부팅 전에는 일부 값이 반영되지 않는다.
6. 재부팅 후 §2-1 재덤프 → §2-2 비교 → 차이 0건이 될 때까지 반복.

> QGC가 "일부 파라미터가 기체에 없다"고 경고할 수 있다. 같은 커밋이므로 원래는
> 안 나는 게 정상이며, 나면 **펌웨어가 예상과 다른 것**이니 §1 판정표부터 다시 본다.

### 4-B. 특정 그룹만 복원

`CAL_MAG*` 만 날아간 경우 등: `.params` 파일에서 해당 줄만 남긴 축약 파일을 만들어
같은 방법으로 로드한다.

```bash
head -8 logs/2026-07-28_px4_flash/px4_params_2026-07-28_pre-flash.params > /tmp/mag.params
grep -P '\tCAL_MAG' logs/2026-07-28_px4_flash/px4_params_2026-07-28_pre-flash.params >> /tmp/mag.params
```

### 4-C. 복구가 실패하면

- 자기계: QGC **Sensors → Compass** 재캘리브레이션.
- 가속도계/자이로: QGC **Sensors → Accelerometer / Gyroscope** 재캘리브레이션.
- 프레임: **Airframe → Standard VTOL** (`SYS_AUTOSTART = 13000`) 재선택 후
  §4-A로 나머지 전량 재로드.
- 어느 경우든 **재캘리브레이션 후 다시 덤프해서 백업을 갱신**한다.

---

## 5. 이 작업 중 금지 사항

- 파라미터 **쓰기 금지** (복구 절차 §4를 실행할 때만 예외, 그때도 사용자 승인 후).
- **ARM·모터 회전 금지.**
- 검증이 끝나면 **MAVROS/덤프 스크립트를 반드시 내려 `/dev/ttyACM0` 을 해제**한다.
  `fuser /dev/ttyACM0` 로 무점유 확인.
