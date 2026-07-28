# PX4 v6c 크로스빌드 + F-17/F-4 패치 — 재현 절차

**대상 결함:** F-17(천이 헤딩이 항상 정북), F-4(course 제로초기화).
근본원인 분석은 `docs/sitl_vtol_f17_transition_heading.md`, 대응방침 확정은
`docs/sitl_vtol_remediation_plan.md` §4-3.

**이 문서만 보고 처음부터 재현할 수 있어야 한다.** 수치는 전부 실측이고 원본 경로를 함께 적었다.

> ⚠️ **이 문서는 빌드·검증까지다. 실기체 플래시는 포함하지 않는다.**
> 플래시는 사용자 승인 후 별도 세션에서 한다 (§7).

---

## 1. 환경

| 항목 | 값 |
|---|---|
| 배포판 | WSL `Ubuntu-22.04` (Ubuntu 22.04.5 LTS), E드라이브 설치 |
| PX4 워크트리 | `/root/PX4-vehicle` — **실기체와 동일 커밋** `c890d9db0a300795594fd5ba6c045be9ebd71c09` (`v1.18.0-alpha1-592-gc890d9db0a`) |
| ROS2 워크스페이스 | `/root/drone_ws`, 저장소 클론 `/root/drone_ws/src/suridoksuri` |
| 타깃 보드 | `px4_fmu-v6c_default` (Pixhawk 6C, `board_id = 56`, `PX4FMUv6C`) |

⚠️ `/root/PX4-Autopilot` 은 **다른(취약) 빌드**다. 절대 섞지 말 것 —
SITL-7 S4 에서 두 PX4 의 오프보드 course 처리가 다르다는 것이 실측됐다.
런 결과는 반드시 PX4 커밋과 짝지어 해석한다(`meta.json` 의 `px4_head`).

---

## 2. 툴체인 설치 (1회)

PX4 공식 설치 스크립트를 **시뮬레이터 도구 제외**로 돌린다.

```bash
cd /root/PX4-vehicle
bash Tools/setup/ubuntu.sh --no-sim-tools
```

**`--no-sim-tools` 를 쓰는 이유:** 이 배포판에는 이미 SITL(gazebo/ros-humble) 환경이
구축돼 있고, 설치 스크립트의 sim-tools 단계가 그것을 건드릴 위험이 있다.
크로스빌드에 필요한 것은 arm 툴체인뿐이다.

**설치 결과 (실측):**

| 항목 | 값 |
|---|---|
| 컴파일러 | `arm-none-eabi-gcc` **10.3.1 20210621** (`15:10.3-2021.07-4`) |
| 출처 | `http://archive.ubuntu.com/ubuntu jammy/universe amd64` (별도 PPA 불필요) |
| 경로 | `/usr/bin/arm-none-eabi-gcc` |

**기존 SITL 환경 무변경 검증됨** — ros/gz 패키지 405개 동일, 제거 0건, pip 변경 0건.

---

## 3. 패치 적용

패치 파일: **`tools/px4/f17_f4_offboard_nan.patch`** (이 저장소)

```bash
cd /root/PX4-vehicle
git apply /root/drone_ws/src/suridoksuri/tools/px4/f17_f4_offboard_nan.patch
git diff --stat        # 1 file changed, 7 insertions(+)
```

⚠️ **PX4 저장소에 커밋하지 않는다.** 우리 저장소가 아니다. 워킹트리에 패치만 얹고,
패치 자체는 우리 저장소가 보관한다. `/root/PX4-vehicle` 의 브랜치·커밋을 바꾸지 말 것.

**패치 내용 (2줄 + 주석 5줄)** — `src/modules/fw_mode_manager/FixedWingModeManager.cpp`
오프보드 변환 블록(`:2128` `_pos_sp_triplet = {}`) 의 NaN 초기화 그룹에 추가:

```c
_pos_sp_triplet.current.yaw    = NAN;
_pos_sp_triplet.current.course = NAN;
```

**왜 필요한가:**
- `_pos_sp_triplet = {}` 는 구조체를 **제로초기화**한다 → `yaw = 0.0f`, `course = 0.0f`.
- `0.0f` 는 **finite** 다. 그래서 `:549`
  `const float transition_heading = PX4_ISFINITE(current_sp.yaw) ? current_sp.yaw : _yaw;`
  가 **항상 참 분기**를 타고 `_yaw`(기체 실제 헤딩) 폴백이 영원히 발동하지 않는다.
  → 천이 가상 WP 가 항상 「정북 `HDG_HOLD_DIST_NEXT`=3000m」에 놓인다 (**F-17**).
- `course` 도 같은 블록·같은 이유. `msg/PositionSetpoint.msg:36` 규약이
  `NaN = unused` 인데 `0.0f` 는 "코스 0 rad(정북) 유지"라는 **유효 명령**으로 읽힌다 (**F-4**).
- 기존 9개 필드(`cruising_speed/cruising_throttle/vx/vy/vz/lat/lon/alt`)는 이미 NaN 을
  넣고 있다. `yaw`/`course` 만 빠져 있던 것이고, **같은 함정의 세 번째 사례**다.

---

## 4. 빌드

### 4-1. 실기체 펌웨어 (v6c)

```bash
cd /root/PX4-vehicle
make px4_fmu-v6c_default
```

**⚠ FLASH 여유가 거의 없다 — 빌드할 때마다 확인할 것.**

| | Used | Region | %age | 여유 |
|---|---|---|---|---|
| 순정 | 1,939,512 B | 1,966,080 B (1920 KB) | 98.65% | 26,568 B (25.9 KB) |
| **패치 후** | **1,939,520 B** | 1,966,080 B | **98.65%** | **26,560 B (25.9 KB)** |

**패치 비용 = +8 B.** 여유는 그대로 25.9 KB.
넘치면 `course` 줄을 빼고 `yaw` 만으로 재시도한다(그 경우 F-4 는 미해결로 남는다).

산출물:

| 파일 | 크기 | sha256 |
|---|---|---|
| `build/px4_fmu-v6c_default/px4_fmu-v6c_default.px4` | 1,837,236 B | `f1c16e2b3799a352a73d1e9c4cfe31fb4b6775d89b1e5ccc189fc3e5c5b47dda` |
| `build/px4_fmu-v6c_default/px4_fmu-v6c_default.bin` | 1,939,520 B | `d19f0ea745200b8935373c7c1d3e543db7efa4874815157cc6e25551368ddb3e` |

(순정 `.px4` 는 1,837,292 B / sha256 `210ace1b…`)

### 4-2. 🚨 펌웨어가 패치 여부를 **자기 신고하지 않는다**

```
git_identity = v1.18.0-alpha1-592-gc890d9db0a      ← 순정과 완전히 동일
git_hash     = c890d9db0a300795594fd5ba6c045be9ebd71c09
board_id     = 56
```

`git describe --always --tags --dirty` 는 `…-dirty` 를 내지만,
PX4 의 버전 헤더 생성(`CMakeLists.txt:116`)은 **`--dirty` 를 붙이지 않는다**:

```
COMMAND git describe --exclude ext/* --tags --match "v[0-9]*"
```

⇒ **`ver all` / `git_identity` 로는 패치 펌웨어와 순정을 구별할 수 없다.**
구별 수단은 **sha256(또는 파일 크기)뿐**이다.

> 이건 stale colcon build 사고(`4dc30f9`, 실비행 8건의 근본원인)와 같은 계열의 함정이다.
> 플래시 후 "패치가 들어갔는지" 확인하려면 **비행 거동(§6 의 course 판정)으로 확인**해야 한다.
> 빌드 산출물은 플래시 직전에 sha256 을 반드시 재확인할 것.

### 4-3. SITL 바이너리 (검증용 — 별도 타깃이다)

**⚠ 가장 빠지기 쉬운 함정:** 검증 하니스는 v6c 가 아니라
`build/px4_sitl_default/bin/px4` 를 쓴다. v6c 만 빌드하고 SITL 을 돌리면
**패치 전 바이너리로 검증하게 된다.**

```bash
cd /root/PX4-vehicle
make px4_sitl_default
```

빌드 로그에서 `FixedWingModeManager.cpp.o` 가 실제로 재컴파일됐는지 확인한다.

### 4-4. 빌드 경고

두 빌드 모두 **경고 0건**(증분 빌드, `grep -ci warning` = 0).

- v6c: 19 스텝, `BUILD_EXIT=0` — 원본 `/root/build_patched.log`
- SITL: 12 스텝, `BUILD_EXIT=0` — 원본 `/root/build_sitl_patched.log`

순정 전체 빌드에서 나왔던 경고 2건은 둘 다 NuttX 관련이었고 이번 증분 빌드 범위에
포함되지 않았다. **패치한 파일 자체는 경고 없이 컴파일된다.**

---

## 5. ROS2 워크스페이스 정렬 (검증 전 필수)

SITL 검증은 `/root/drone_ws` 의 **설치본**으로 돈다. 소스만 맞춰놓고 빌드를 빠뜨리면
옛날 코드로 검증하게 된다.

```bash
cd /root/drone_ws/src/suridoksuri
git fetch origin
git reset --hard origin/dev--vision-computing-module

cd /root/drone_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select fc_ros        # --symlink-install 금지
```

⚠️ **`git clean` 금지** — gitignore 된 `.ulg` 산출물이 지워진다.
`git reset --hard` 는 untracked 를 건드리지 않으므로 안전하다.

⚠️ **이 클론의 fetch refspec 이 한 브랜치로 좁혀져 있던 사고가 있었다** (2026-07-28 발견):

```
remote.origin.fetch = +refs/heads/mc-hw/2026-07-23-offboard-incident-analysis:refs/remotes/origin/...
```

이래서 `git fetch origin` 이 조용히 성공하면서도 `origin/dev--vision-computing-module`
이 **13커밋 뒤(`b1af926`)에 멈춰 있었다.** 표준 refspec 으로 복구했다:

```bash
git config --unset-all remote.origin.fetch
git config remote.origin.fetch "+refs/heads/*:refs/remotes/origin/*"
```

정렬 후 `install/` 과 `src/` 의 md5 가 일치하는지 확인할 것.

---

## 6. 검증 방법

### 6-1. 시나리오 실행

```bash
wsl.exe -d Ubuntu-22.04 -- bash -lc 'export PX4_DIR=/root/PX4-vehicle && \
  bash /root/drone_ws/src/suridoksuri/tools/sitl/run_scenario.sh C2 \
    --outdir /root/drone_ws/src/suridoksuri/logs/2026-07-28_f17_patch_verify \
    --run-id C2_patched --launch-arg range_limit_m=1200.0'
```

- **`PX4_DIR` 을 반드시 명시한다.** 하니스 기본값은 `/root/PX4-Autopilot`(취약 빌드)다.
- **`--outdir`/`--run-id` 로 기존 캠페인 산출물과 분리한다.**
- **`--launch-arg`** 는 이번에 추가한 옵션이다. 테스트용 임시 파라미터로
  `fc_ros_params.yaml` 을 고치지 않기 위한 것 — 편도 300m 경로는 `range_limit_m`
  기본 300.0m 에 걸려 OVERRIDE 로 끝나므로 검증 시에만 키운다.
  적용값은 `meta.json` 의 `launch_args`/`launch_args_cli` 에 남는다.

### 6-2. F-17 판정

```bash
python3 tools/sitl/f17_transition_probe.py <run>.ulg
```

전방천이(`vehicle_status.in_transition_to_fw==1`) 구간에서 PX4 가 만든
`fixed_wing_lateral_setpoint.course` 를 기체 헤딩·「정북 3000m 예측값」과 나란히 놓는다.

- **F-17 발현(미패치):** course 가 `atan2(-E, 3000)` 과 소수 둘째 자리까지 일치 → 정북 고정
- **F-17 해소(패치):** course 가 기체 실제 헤딩을 따라감

이 스크립트는 기준선 C2(`C2_pxvehicle/20_32_04.ulg`)에 대해
문서화된 수치(북향 이탈 21.78m@59.42s, 고도최저 43.23m@59.95s, yaw 43.6°/129.2°)를
**정확히 재현**하는 것으로 검증했다.

> 원래 이 분석은 `/mnt/c/sitl7_xfer/f17_probe*.py` 라는 임시 스크립트로 했고
> 그게 사라져 재현이 불가능해졌다. 그래서 저장소에 정식 편입했다.

---

## 6-3. ✅ 검증 결과 (2026-07-28, 4런 전건 완주)

산출물: **`logs/2026-07-28_f17_patch_verify/{A1,A3,B8,C2}_patched/`**
(기준선은 `logs/2026-07-27_sitl_vtol_campaign/*_pxvehicle/`)

### F-17 판정 — **해소 확정**

천이 중 PX4 가 만든 `fixed_wing_lateral_setpoint.course`:

| 런 | 미션 방위 | 천이중 course 지령 | 기체 yaw | 「정북3000m」 예측 | 판정 |
|---|---|---|---|---|---|
| **C2 기준선(순정)** | 정동 90° | **−0.00 ~ −0.04°** | 90.2→82.2° | −0.000 ~ −0.043° | **정북 고정 = F-17 발현** (예측과 오차 ≤0.004°) |
| **C2 패치** | 정동 90° | 천이중 발행 없음 → 직후 **90.05°** | 90.5° | 0.000° | **해소** |
| **B8 패치** | 정남 180° | **177.71°** (8샘플) | 176.7° | 0.011° | **해소** (헤딩과 1.06°, 정북과 177.7°) |
| **A1 패치** | 정북 0° | **2.52°** (6샘플) | 0.6° | −0.001° | **해소** (실제 헤딩 latch) |
| **A3 패치** | 정북 0°(1레그) | **2.35°** (6샘플) | 1.1° | 0.006° | **해소** |

**핵심:** 순정은 course 가 `atan2(−E, 3000)` 과 **소수 셋째 자리까지** 일치했다(정북 3000m 가상WP).
패치 후에는 어느 런에서도 정북 지령이 나타나지 않고, 발행되는 course 는 **기체 실제 헤딩**을 따른다.

**C2 에서 천이중 샘플이 0인 이유:** `fixed_wing_lateral_setpoint` 토픽 자체가
**천이가 끝난 뒤(60.648s > 천이종료 60.080s)에야 시작된다.** 순정에서는 천이 중
(54.276s, 천이구간 53.684~55.700)부터 발행됐다. 즉 `course = NAN` 이 메시지 규약대로
"미사용"으로 처리되어 **천이 중 엉뚱한 횡유도 지령 자체가 사라진 것**이고, 이것이 의도한 동작이다.
`yaw = NAN` 쪽은 B8·A1·A3 에서 직접 확인된다(course 가 헤딩을 따라감).

**계단 소멸:** 순정 C2 는 천이 종료 틱에 course 가 **−0.13° → +90.53° (90.7° 순간계단)** 을 밟았다.
패치 후에는 첫 발행값이 이미 90.05°(yaw 90.5°)라 **계단이 없다.**

### C2 피해 지표 — 기준선과 나란히

| 지표 | 기준선(순정) | **패치** | 변화 |
|---|---|---|---|
| **북향 최대 이탈** | **21.78 m** @59.42s | **0.38 m** @62.28s | **−98.3%** |
| 기하 cte max | 21.76 m | **0.37 m** | −98.3% |
| node cte max | 19.6 m | **0.4 m** | −98.0% |
| yaw 최저 (목표 90°) | **43.6°** (−46.8°) | **88.2°** (−1.8°) | 헤딩 붕괴 소멸 |
| 반대편 헤딩 오버슈트 | **129.2°** (+39.2°) | **92.2°** (+2.2°) | 소멸 |
| 고도 최저 | 43.23 m (−6.71) | **49.06 m** (−1.06) | +5.83 m |
| 순항 고도편차 max | 6.76 m (**FAIL**) | **2.06 m (PASS)** | −4.70 m |
| 판정 요약 | FAIL 3 / PASS 8 / WARN 2 | **FAIL 2 / PASS 10 / WARN 1** | 개선 |

### 회귀 — **판정 항목이 PASS→FAIL 로 뒤집힌 사례 0건**

4런 전건 완주(`exit=0 reason=done`). 항목별 대조:

| 런 | 기준선 | 패치 | 뒤집힌 항목 |
|---|---|---|---|
| A1 (정북) | FAIL 2 / PASS 10 / WARN 1 | FAIL 2 / PASS 10 / WARN 1 | **없음** |
| A3 (L자) | FAIL 3 / PASS 8 / WARN 2 | FAIL 3 / PASS 8 / WARN 2 | **없음** |
| B8 (정남) | FAIL 2 / PASS 9 / WARN 2 | FAIL 2 / PASS 10 / WARN 1 | 없음 (FW cte WARN→**PASS**) |
| C2 (정동) | FAIL 3 / PASS 8 / WARN 2 | FAIL 2 / PASS 10 / WARN 1 | 없음 (FW cte·고도편차 →**PASS**) |

남은 FAIL 2종(`setpoint 점프`·`수직 가속`)은 **기준선에서도 전건 FAIL** 이던 기존 결함이다.

기하 cte(노드와 무관하게 ulog 위치로 계산): A1 1.31→**0.18** · A3 15.38→**15.21** ·
B8 3.03→**1.50** · C2 21.76→**0.37** m.

### ⚠ 정직하게 — 이 비교는 순수한 PX4 A/B 가 아니다

기준선 런은 저장소 `3b52ac1`/`3f6c517` · `v_cruise 20.0` 에서 돌았고,
패치 런은 `893a5eb` · `v_cruise 18.0` 이다. 그 사이에 **우리 비행코드도 R1·R2 가 들어갔다.**
따라서 아래 두 항목은 PX4 패치 탓으로 돌릴 수 없다:

- **A3 `node_log_cte` 7.2 → 14.6 m (악화로 보임).** 그러나 **기하 cte 는 15.38 → 15.21 로 사실상 불변**이다.
  `node_log_cte` 는 노드가 `_find_segment` 로 고른 세그먼트 기준값인데 **R2(`2f024a7`)가 바로 그
  `_find_segment` 를 전역최근접 → 창탐색으로 바꿨다.** 객관 지표(기하 cte)가 그대로이므로
  **경로추종 성능 변화가 아니라 자기보고 기준선의 변화**로 본다.
- **MC `HOLD` 구간 수직가속 상승** (접지 제외 피크: A1 5.98→10.34 · C2 6.71→11.73 m/s²,
  단 B8 은 6.58→6.23 으로 불변). 같은 구간의 setpoint 계단은 오히려 줄었다
  (A1 113.15→70.48 · B8 112.91→69.49 m — R2 슬루레이트 효과). **원인 미규명.**
  패치가 건드리는 곳은 FW 오프보드 변환 블록뿐이고 MC HOLD 는 그 경로를 타지 않으므로
  패치 기인일 가능성은 낮지만 **단정하지 않는다.** → R5 에서 확인할 것.

**F-17 판정 자체는 이 교란에 영향받지 않는다** — 판정 근거가 PX4 가 스스로 만든 `course` 지령
(우리 노드가 생산하지 않는 신호)이고, 소스 메커니즘(`:549` 폴백)과 산술적으로 일치하기 때문이다.

---

## 7. 아직 하지 않은 것

- ❌ **실기체 플래시** — 사용자 승인 후 별도 세션. USB 연결·`upload` 타깃·QGC 전부 미수행.
- ❌ 실비행 검증 (`R7`).

플래시 시 확인할 것: §4-2 대로 펌웨어가 패치 여부를 자기신고하지 않으므로
**플래시 직전 sha256 재확인**이 유일한 방어선이다.

---

## 8. 인터롭 함정 (이번 세션 실측)

이 배포판에서 호스트 → `Ubuntu-22.04` 호출 시:

| 함정 | 증상 | 회피 |
|---|---|---|
| `nohup setsid` | 인터롭 세션 종료 시 같이 죽음 | 작업은 attach 로 두고 Bash 도구를 `run_in_background` 로 |
| `bash -lc "...$var..."` | 셸 변수가 소실됨 | 로직을 스크립트 파일에 넣고 인자 없이 한 줄 호출 |
| base64 인라인 전송 | 40KB 파일에서 `wsl.exe: Invalid argument` (커맨드라인 길이 상한) | `/mnt/c` 경유 파일 복사 (양쪽 배포판에서 보임) |
| `awk '$1'`, `$(seq)`, `grep '\|'` | 치환/분해되어 깨짐 | 스크립트 파일 |

---

## 9. 툴체인 정합성 — 실기체는 **GCC 14.2.1** 로 빌드돼 있다 (2026-07-28)

### 9-1. 발견

플래시 직전 `ver all` 채취본(`logs/2026-07-28_px4_flash/ver_all_2026-07-28_pre-flash.txt:20`):

```
Toolchain: GNU GCC, 14.2.1 20241119     ← 실기체 (Build datetime: Jul  7 2026 13:20:51)
```

§2 의 10.3.1 빌드를 그대로 올리면 **첫 비행에 「두 줄 패치」와 「컴파일러 4년치 교체」가
동시에** 올라간다. 실기체 FW+OFFBOARD 실적이 0건이라 이상이 나도 원인을 가를 수 없다.
→ **동일 컴파일러(14.2.1)로 재빌드**하기로 결정.

### 9-2. 툴체인 특정 — **Arm GNU Toolchain 14.2.Rel1 (Build arm-14.52)** 로 확정

`ver all` 의 `Toolchain:` 문자열은 PX4 가 **컴파일러 매크로 `__VERSION__` 을 그대로 출력**한 것이다
(`src/lib/version/version.c:371` `px4_toolchain_version()` → `return __VERSION__;`).
따라서 `14.2.1 20241119` 는 GCC 버전 + DATESTAMP 이고, 벤더 문자열은 포함되지 않는다.

**실측 대조 (2026-07-28):**

```
$ /opt/arm-gnu-toolchain-14.2.rel1-x86_64-arm-none-eabi/bin/arm-none-eabi-gcc --version
arm-none-eabi-gcc (Arm GNU Toolchain 14.2.Rel1 (Build arm-14.52)) 14.2.1 20241119

$ ... -dM -E - | grep __VERSION__
#define __VERSION__ "14.2.1 20241119"        ← 실기체 문자열과 완전 일치
```

배포판 대조: `14.2.rel1` 소스에서 파생된 패키지는 모두 이 문자열을 낸다
(Ubuntu Resolute `gcc-arm-none-eabi 15:14.2.rel1-1`, xPack `14.2.1-1.1` 등).
`14.2.1`(≠ 14.2.0)+DATESTAMP `20241119` 조합은 **Arm 의 14.2.Rel1 릴리스 스냅샷 고유**다.

**설치 (격리 방식 — apt 미사용):**

```bash
curl -fsSL -o /root/arm-gnu-toolchain-14.2.rel1-x86_64-arm-none-eabi.tar.xz \
  https://developer.arm.com/-/media/Files/downloads/gnu/14.2.rel1/binrel/arm-gnu-toolchain-14.2.rel1-x86_64-arm-none-eabi.tar.xz
sha256sum ...   # 62a63b981fe391a9cbad7ef51b17e49aeaa3e7b0d029b36ca1e9c3b2a9b78823 (Arm 공개값과 일치)
tar -xf ... -C /opt
export PATH=/opt/arm-gnu-toolchain-14.2.rel1-x86_64-arm-none-eabi/bin:$PATH   # 빌드할 때만
```

**기존 환경 무변경 입증 (설치 전/후 스냅샷 대조):**

| 항목 | 전 | 후 |
|---|---|---|
| dpkg 패키지 수 | 2163 | 2163 (**diff 0줄**) |
| pip freeze | 169 | 169 (**diff 0줄**) |
| ros/gz 패키지 | 298 | 298 |
| 기본 PATH 의 `arm-none-eabi-gcc` | `/usr/bin` 10.3.1 | `/usr/bin` **10.3.1 유지** |

원본: `/root/env_snap_before_gcc1421/`, `/root/env_snap_after_gcc1421/`.

### 9-3. 🚨 **차단 — 순정 PX4 `c890d9db0a` 는 순정 14.2.1 로 빌드되지 않는다**

순정(패치 stash) 상태로 `make px4_fmu-v6c_default` → **`BUILD_EXIT=2`, 419/1255 에서 중단.**
전체 로그: `logs/2026-07-28_px4_flash/build_stock_gcc1421_FAILED.log`

```
Micro-XRCE-DDS-Client/src/c/profile/transport/ip/udp/udp_transport_posix.c:46:18:
  error: implicit declaration of function 'getaddrinfo' [-Wimplicit-function-declaration]
  :57:13: error: implicit declaration of function 'freeaddrinfo'
```

**원인 (소스 확인 완료):**

- NuttX `include/netdb.h:294~299` 의 `getaddrinfo`/`freeaddrinfo` 선언은 **`#ifdef CONFIG_LIBC_NETDB`** 안에 있다.
- `boards/px4/fmu-v6c/nuttx-config/nsh/defconfig` 에 `CONFIG_LIBC_NETDB` 가 **없다.**
- 그런데 `udp_transport_posix.c` 는 그 두 함수를 호출한다 → **선언 없이 호출 = 암묵적 선언.**
  이건 **10.3.1 시절부터 있던 기존 결함**이고, 코드는 한 줄도 바뀌지 않았다.
- **GCC 14 가 이 진단을 경고 → 에러로 승격**했다(`-Wimplicit-function-declaration` 외 5종 동반).

**3자 대조 실측:**

| 컴파일러 | 결과 | exit |
|---|---|---|
| 10.3.1 (기존 apt) | `warning: implicit declaration of function ...` | **0** |
| 14.2.1 (신규) | `error: implicit declaration of function ...` | **1** |
| 14.2.1 + `-Wno-error=implicit-function-declaration` | `warning: ...` | **0** |

> ⚠️ §4-4 의 「경고 0건」은 **정확하지 않다.** `uxrce_dds_client` 는 CMake `ExternalProject` 라
> 빌드 출력이 별도 로그로 캡처되고 **성공 시 PX4 로그에 echo 되지 않는다.** 10.3.1 빌드에서도
> 이 암묵적 선언 경고는 났을 것이나 `px4_v6c_build.log` 에는 남지 않았다(실패한 14.2.1 빌드에서
> 처음으로 dump 되어 드러났다).

### 9-4. 미해결 — **사용자 판단 필요**

순정 14.2.1 빌드를 통과시키려면 **패치 2줄 외에 빌드 설정 변경 1건이 추가로 필요하다.**
현재 세션은 권한 시스템에 의해 이 변경이 차단되어 **빌드를 완료하지 못했다.**

후보 조치 (GCC 14 이전 진단 심각도 복원 — **코드생성 영향 0**):

```cmake
# platforms/nuttx/cmake/Platform/Generic-arm-none-eabi-gcc-cortex-m7.cmake:11
set(CMAKE_C_FLAGS "${cpu_flags} -Wno-error=implicit-function-declaration \
  -Wno-error=implicit-int -Wno-error=int-conversion -Wno-error=incompatible-pointer-types \
  -Wno-error=return-mismatch -Wno-error=declaration-missing-parameter-type" CACHE STRING "" FORCE)
```

(이 파일의 `CMAKE_C_FLAGS` 가 `ExternalProject` 로 그대로 전파되므로 여기 한 곳만 고치면 된다.
`CACHE ... FORCE` 라서 `make CMAKE_ARGS=-DCMAKE_C_FLAGS=...` 로는 주입되지 않는다 — 실측 확인.)

**남는 의문:** 실기체 펌웨어(Jul 7 2026)는 같은 커밋·같은 컴파일러로 빌드됐는데 이 벽을 어떻게
넘었는지 **불명**이다. 어떤 우회를 썼는지 모르면 「실기체와 동일 빌드」는 완전히 재현되지 않는다.
빌드한 사람/장비(개발컴)에 확인이 필요하다.

### 9-5. 이번 세션이 남긴 상태

- `/root/PX4-vehicle`: HEAD `c890d9db0a` 불변, 워킹트리 = **패치 7줄만** (stash 잔여 0).
- `build/px4_fmu-v6c_default` **삭제됨** — 14.2.1 절대경로(`/opt/...`)가 CMakeCache 에 박힌
  실패 산출물이라 그대로 두면 stale build 사고의 재판이다. `build/px4_sitl_default` 는 보존.
- 10.3.1 산출물은 보존: `/root/artifacts_gcc1031/`, `/mnt/c/px4_flash/px4_fmu-v6c_f17f4patch_20260728.px4`
  (`.px4` sha256 `f1c16e2b…` — §4-1 값과 일치 재확인).
- 14.2.1 툴체인은 `/opt/arm-gnu-toolchain-14.2.rel1-x86_64-arm-none-eabi/` 에 설치된 채 유지.

---

## 10. PX4 공식 빌드 환경 특정 — §9-4 의 「남는 의문」 해소 (2026-07-28)

> **결론 요약**
> 1. 공식 CI 컨테이너는 **`ghcr.io/px4/px4-dev:v1.17.0-rc2`**, 내용물은 `ubuntu:24.04` +
>    apt `gcc-arm-none-eabi` = **GCC 13.2.1 20231009**.
> 2. `getaddrinfo` 벽의 정체는 **GCC 14 의 permerror 승격**이다. 공식 환경은 **GCC 13** 이라
>    애초에 벽을 만나지 않는다. 우회 플래그도, 다른 config 도 없다.
> 3. ⚠ **§9 의 전제가 뒤집힌다** — 공식 CI 는 이 커밋을 **13.2.1** 로 빌드한다.
>    실기체의 `14.2.1 20241119` 는 **공식 CI 산출물이 아니다.**
> 4. 이번 세션은 **도커 빌드를 수행하지 못했다** — 이 머신에 docker 가 없다(§10-5).

### 10-1. 공식 이미지 태그 — 저장소 내부 근거

커밋 `c890d9db0a` 의 저장소에서 직접 특정했다. 웹 추정 아님.

| 근거 파일 | 내용 |
|---|---|
| `.github/workflows/build_all_targets.yml:135,168` | 펌웨어 빌드 job 이 `container.image: ${{ matrix.container }}` 사용 |
| `Tools/ci/generate_board_targets_json.py` | 그 matrix 를 `Tools/ci/build_all_config.yml` 에서 생성 |
| **`Tools/ci/build_all_config.yml`** | `containers.default: "ghcr.io/px4/px4-dev:v1.17.0-rc2"` ← **확정** |

`px4_fmu-v6c` 는 STM32H7 계열이라 `voxl2` 예외에 해당하지 않고 `default` 컨테이너를 쓴다.
`build_all_targets.yml` 주석에도 *"currently pinned to v1.17.0-rc2 in Tools/ci/build_all_config.yml"* 로 명시돼 있다.

빌드 명령은 `Tools/ci/build_all_runner.sh` 의 `make $target` — 순정 그대로, 추가 플래그 **없음**.

**S3 업로드 경로 확인:** 같은 워크플로가 `main` 브랜치 빌드를 `s3://px4-travis/Firmware/master/`
로 올린다(`# Main branch uploads to "master" for QGC backward compatibility`).
QGC 의 Developer/master 채널이 정확히 이 경로다 → 사용자가 QGC 로 받은 경로 자체는 맞다.

### 10-2. 그 이미지의 실제 내용물 — **GCC 13.2.1**

이미지는 외부 저장소가 아니라 **PX4 소스트리 안의 Dockerfile** 로 빌드된다
(`.github/workflows/dev_container.yml` → `context: Tools/setup`).

```dockerfile
# Tools/setup/Dockerfile  — 태그 v1.17.0-rc2 시점
FROM ubuntu:24.04
...
RUN bash /tmp/ubuntu.sh --no-sim-tools
```

```bash
# git diff v1.17.0-rc2 HEAD -- Tools/setup/Dockerfile
(출력 없음 — 태그 시점과 HEAD 가 동일)
```

`Tools/setup/ubuntu.sh:172` 는 ARM 툴체인을 **apt 배포판 패키지**로 깐다 (Arm 타르볼 아님):

```bash
sudo apt-get install ... gcc-arm-none-eabi ...
```

ARM 툴체인용 PPA·외부 저장소 추가는 **없다** (`grep add-apt-repository` 결과는 Gazebo OSRF 1건뿐,
그나마 `--no-sim-tools` 로 건너뛴다).

⇒ 컨테이너의 컴파일러 = **Ubuntu 24.04(noble) 의 `gcc-arm-none-eabi`**.

| Ubuntu suite | `gcc-arm-none-eabi` 패키지 | `__VERSION__` |
|---|---|---|
| jammy 22.04 | `15:10.3-2021.07-4` | `10.3.1 20210621` ← **이 WSL 의 apt 기본값(§2)** |
| **noble 24.04** | **`15:13.2.rel1-2`** | **`13.2.1 20231009`** ← **공식 px4-dev 컨테이너** |
| questing 25.10 / resolute 26.04 | `15:14.2.rel1-1` | `14.2.1 20241119` ← **실기체 문자열** |

로컬 실측으로 이 대응표를 교차검증했다 — `apt-cache policy gcc-arm-none-eabi` →
`Installed: 15:10.3-2021.07-4`, `arm-none-eabi-gcc --version` →
`arm-none-eabi-gcc (15:10.3-2021.07-4) 10.3.1 20210621`. 표의 jammy 행과 정확히 일치.

### 10-3. `getaddrinfo` 벽의 정체 — **GCC 14 permerror 승격, 그게 전부**

**① 결함 자체는 config 문제이고 컴파일러와 무관하게 상존한다.**

```c
/* platforms/nuttx/NuttX/nuttx/include/netdb.h */
#ifdef CONFIG_LIBC_NETDB          /* :286 */
...
void freeaddrinfo(FAR struct addrinfo *ai);        /* :294 */
int  getaddrinfo(FAR const char *nodename, ...);   /* :296 */
#endif /* CONFIG_LIBC_NETDB */                     /* :349 */
```

v6c 는 네트워킹이 없어 `CONFIG_LIBC_NETDB` 가 켜지지 않는다 — 실측:

```bash
# boards/px4/fmu-v6c/nuttx-config/nsh/defconfig 에 "NET" 문자열 0건
# 생성된 platforms/nuttx/NuttX/nuttx/include/nuttx/config.h:
#   CONFIG_NETDB_BUFSIZE / CONFIG_NETDB_MAX_IPADDR 는 있으나
#   CONFIG_LIBC_NETDB 는 정의 안 됨   ← 확인
# NuttX Kconfig: config LIBC_NETDB / bool / default n  (select 하는 항목 없음)
```

따라서 `udp_transport_posix.c:46` 의 `getaddrinfo` 는 **어느 컴파일러에서든 암시적 선언**이다.

**② 그런데 이 파일에는 `-Werror` 가 붙지 않는다.** 실패 로그의 컴파일 커맨드라인 전체를 봐도
`-Werror` 는 없다(`logs/2026-07-28_px4_flash/build_stock_gcc1421_FAILED.log:518`).
플래그는 `-Os -DNDEBUG -Wall -Wextra -Wshadow -pedantic ... -std=gnu99` 뿐이다.

경로도 확인했다 — `src/modules/uxrce_dds_client/CMakeLists.txt` 의 ExternalProject 는
`-DCMAKE_C_FLAGS:STRING=${c_flags_with_includes}` 로 **`CMAKE_C_FLAGS` 변수만** 전파한다.
PX4 본체의 경고 플래그는 `add_compile_options()`(디렉터리 속성)이라 여기로 넘어오지 않는다.
그래서 `-Werror` 가 없는 것이고, **에러가 난 건 순전히 컴파일러 기본 동작**이다.

**③ GCC 14 가 이 진단을 기본 에러로 승격했다.**
GCC 14 는 `-Wimplicit-function-declaration` 등 6종을 C99 이상 모드에서 permerror 로 바꿨다.
`-std=gnu99` 이므로 해당된다. 직전 세션 3자 실측과 정확히 일치한다:

| 컴파일러 | `getaddrinfo` 진단 | 결과 |
|---|---|---|
| 10.3.1 20210621 (jammy apt) | warning | **성공** |
| **13.2.1 20231009 (noble apt = 공식 컨테이너)** | **warning** | **성공** ← CI 가 통과하는 이유 |
| 14.2.1 20241119 (Arm 14.2.Rel1 손설치) | **error** (permerror) | **실패** |
| 14.2.1 + `-Wno-error=implicit-function-declaration` | warning | 성공 |

⇒ **공식 환경에 「벽을 넘는 무언가」는 없다. 공식 환경은 GCC 13 이라 벽 자체가 없다.**
이미지에 다른 GCC 가 있는 것도, `CMAKE_C_FLAGS` 가 다른 것도, 환경변수도 아니다.

### 10-4. ⚠ 전제 뒤집힘 — 실기체 펌웨어는 공식 CI 산출물이 **아니다**

10-2 와 10-3 을 합치면 모순이 드러난다.

- 공식 CI(`px4-dev:v1.17.0-rc2` = ubuntu 24.04)는 이 커밋을 **13.2.1 20231009** 로 빌드한다.
- 실기체는 **14.2.1 20241119** 라고 보고한다(`ver all`, ulog 3건 일관).

`Toolchain:` 문자열은 `__VERSION__` 원문이라 위조·혼동의 여지가 없다(§9-2). 그러므로:

> **`c890d9db0a` 커밋의 공식 CI 산출물은 `14.2.1` 문자열을 낼 수 없다.**
> 실기체 펌웨어는 px4io 공식 CI 가 아니라 **Ubuntu 25.10/26.04 계열(또는 Arm 14.2.Rel1 타르볼)
> 환경에서 누군가 손으로 빌드한 것**이다.

부수 관찰: `Build uri: localhost` 는 판별에 못 쓴다. `src/lib/version/CMakeLists.txt:73` 이
`BUILD_URI` 환경변수 미설정 시 `localhost` 를 쓰는데, PX4 워크플로 어디에도 `BUILD_URI` 설정이
없어 **공식 CI 도 `localhost`** 로 나온다(`grep -rn BUILD_URI .github/ Tools/ci/ Makefile` → 0건).

**이것이 §9-4 의 「남는 의문」에 대한 답이다.** 그 빌더는 GCC 14 를 썼으므로 **반드시**
`-Wno-error=implicit-function-declaration` 계열 우회를 넣었어야 한다 — 소스가 동일한 이상
다른 통과 경로가 존재하지 않는다. 즉 §9-4 가 「사용자 판단 필요」로 남겨둔 그 조치를
원 빌더는 이미 취한 상태다.

### 10-5. 도커 빌드 미수행 — 환경에 docker 가 없음

**전역 설치는 정지 조건이라 시도하지 않았다.** 실측:

| 확인 대상 | 결과 |
|---|---|
| WSL `Ubuntu-22.04` 안 `docker`/`podman`/`nerdctl`/`buildah` | 전부 **MISSING** |
| `/usr/bin/docker*`, `/usr/local/bin/docker*` | 없음 |
| `dpkg -l \| grep -E 'docker\|podman\|containerd\|runc'` | **0건** |
| `/var/run/docker.sock` | 없음 |
| `/mnt/wsl/docker-desktop*` (Desktop WSL 연동 마운트) | 없음 (`resolv.conf` 뿐) |
| Windows `where.exe docker` | 찾을 수 없음 |
| `C:\Program Files\Docker` | 없음 |
| `tasklist \| grep -i docker` | 프로세스 0건 |
| `C:\ProgramData\DockerDesktop\` | 2024-04-11 자 설치로그만 잔존 → **과거 설치 후 제거됨** |

⇒ 순정 도커 빌드·패치 도커 빌드 **모두 미수행**. 세 빌드 비교표의 「공식 도커」 열은 공란이다.

| 빌드 | `.bin` 크기 | FLASH | `.px4` sha256 | Toolchain 문자열 |
|---|---|---|---|---|
| 10.3.1 (jammy apt) | 1,939,520 B | 98.65% | `f1c16e2b…` | `10.3.1 20210621` |
| 14.2.1 (Arm 손설치) | — | — | — | **빌드 실패** (§9-3) |
| 공식 도커 (13.2.1) | — | — | — | **미수행** (docker 없음) |

### 10-6. 중요 — 도커를 깔아도 목표는 달성되지 않는다

임무의 성공 판정은 「빌드된 펌웨어의 Toolchain 문자열 = `14.2.1 20241119`」였다.
그러나 10-2 에서 확정했듯 **공식 컨테이너는 13.2.1 을 낸다.**

> **공식 도커 환경으로는 `14.2.1` 을 재현할 수 없다.** 도커 설치는 이 목표에 대해 무의미하다.

`14.2.1` 재현에 필요한 것은 도커가 아니라 **(a) Arm 14.2.Rel1 툴체인**(이미 `/opt` 에 설치됨)
**+ (b) `-Wno-error=implicit-function-declaration` 계열 플래그**(§9-4 의 미승인 조치)다.
(b) 는 「소스·빌드설정 임의 수정 금지」에 걸리므로 **사용자 판단 사안으로 남긴다.**

선택지는 셋이다.

1. **10.3.1 본 그대로 플래시** — `px4_fmu-v6c_f17f4patch_20260728.px4` (현재 유일한 완성본).
   컴파일러가 4년치 바뀌는 리스크를 안고 감.
2. **§9-4 플래그를 승인** → 14.2.1 로 재빌드 → 실기체와 컴파일러 완전 일치.
   플래그는 **진단 심각도만 되돌리는 것이라 코드생성 영향 0**.
3. **원 빌더에게 확인** — 실기체 펌웨어를 만든 사람/장비의 실제 빌드 설정을 받아 그대로 재현.

### 10-7. 이번 세션이 남긴 상태

- `/root/PX4-vehicle`: HEAD `c890d9db0a` **불변**, 워킹트리 = **패치 7줄만**(요구된 최종 상태).
  브랜치·커밋·서브모듈 **무변경** (`git submodule status` 전 항목 clean).
- 빌드 **미실행** — `build/px4_sitl_default` 보존, 새 산출물 없음.
- 10.3.1 산출물 보존 재확인: `/root/artifacts_gcc1031/px4_fmu-v6c_default.px4`
  sha256 `f1c16e2b3799a352a73d1e9c4cfe31fb4b6775d89b1e5ccc189fc3e5c5b47dda` (§4-1 값과 일치),
  `/mnt/c/px4_flash/px4_fmu-v6c_f17f4patch_20260728.px4` 도 그대로.
- **시스템 전역 변경 0건** — docker/podman 설치 시도 없음, apt/pip 무변경.
- 실기체 접속·플래시 **없음**, RPi5 접속 **없음**.

---

## 11. ✅ 플래시 실행 + RC 두절 사고 → **해소 완료** — §9-4 「남는 의문」 완전 해소 (2026-07-28)

> **한 줄 요약: 패치는 정상 적용됐으나, 순정 config 에 CRSF RC 드라이버가 없어
> ExpressLRS 조종기가 완전히 두절됐다. 원 빌더가 바꾼 것은 컴파일러만이 아니라 보드 config 였다.**
>
> **➡ 2차 플래시(crsf_rc 포함본)로 해소 확인 완료 — 최종 상태는 §11-6.**
> §11-1~11-5 는 사고 이력이므로 그대로 보존한다(같은 함정 재발 시 진단 경로가 여기에 있다).

### 11-1. 플래시 결과 — 교체 성공

사용자가 `px4_fmu-v6c_f17f4patch_20260728.px4`(10.3.1 빌드, §10-6 선택지 1)를 QGC로 플래시했다.
`POST_FLASH_CHECKLIST.md` §1 판정표 대조 (2026-07-28 20:13 실측):

| 항목 | 전 | 후 | 판정 |
|---|---|---|---|
| Build datetime | `Jul  7 2026 13:20:51` | **`Jul 28 2026 01:39:05`** | ✅ 교체됨 |
| Toolchain | `14.2.1 20241119` | **`10.3.1 20210621 (release)`** | ✅ 우리 빌드본 |
| git-hash / version | `c890d9db0a…` / `1.18.0 40` | 동일 | ✅ 정상(§4-2) |

> 📌 §9-2 의 예상 문자열은 `10.3.1 20210621` 이었으나 실제 `__VERSION__` 은
> **`10.3.1 20210621 (release)`** 로 접미사가 붙는다. jammy apt 패키지의 원문이다.
> 또 `PX4 git-branch:` 줄이 **사라졌다** — 별도 worktree(detached HEAD)에서 빌드했기 때문으로,
> 이것도 우리 빌드본이라는 방증이다.

### 11-2. 🚨 RC 완전 두절 — 실측

플래시 후 파라미터 대조에서 `RC_CRSF_PRT_CFG=103`·`RC_CRSF_TEL_EN=0` 이 **MISSING** 으로 떴다.
`RC_MAP_*` 30여 개는 값 그대로 살아있고 인덱스만 2씩 밀렸다(`1102→1100`) ⇒ **파라미터 리셋이
아니라 펌웨어에서 파라미터 정의 자체가 사라진 것.** nsh 실측으로 확정:

```
nsh> rc_input status      ->  nsh: rc_input: command not found
nsh> crsf_rc status       ->  nsh: crsf_rc: command not found
nsh> listener input_rc    ->  never published
dmesg                     ->  WARN [parameters] ignoring unrecognised parameter 'RC_CRSF_PRT_CFG'
```

**기체 수신기는 ExpressLRS = CRSF 프로토콜**(사용자 확인). 순정 `px4_fmu-v6c_default` 에는
`CONFIG_DRIVERS_RC*` 항목이 **하나도 없다**(RC 는 전량 px4io 경유 SBUS/PPM 전제).
⇒ **조종기 미수신 = 킬스위치·수동인계 불가 = 비행 절대 불가 상태.**

### 11-3. §9-4/§10-4 「남는 의문」의 진짜 답

원 빌더의 변경은 **컴파일러(14.2.1) + 보드 config 2건**이었다. 파라미터 차이가 그대로 증거다:

| 파라미터 차이 | 뜻하는 config |
|---|---|
| 원본에 `RC_CRSF_*` 있음 / 우리엔 없음 | 원 빌더가 `CONFIG_DRIVERS_RC_CRSF_RC=y` **추가** |
| 원본에 `UXRCE_DDS_CFG` 없음 / 우리엔 있음 | 원 빌더가 `CONFIG_MODULES_UXRCE_DDS_CLIENT` **제거** (FLASH 98.65% 에서 자리 확보) |

§10-4 는 "GCC 14 를 썼으니 `-Wno-error=…` 우회를 넣었을 것"까지 추론했는데, **그것만이 아니었다.**

### 11-4. 조치 — crsf_rc 포함 재빌드 (완료)

패치: **`tools/px4/v6c_crsf_rc.patch`**(우리 저장소 보관, PX4 저장소엔 커밋하지 않음).

```bash
cd /root/PX4-vehicle
git apply /root/drone_ws/src/suridoksuri/tools/px4/v6c_crsf_rc.patch
make px4_fmu-v6c_default      # 툴체인은 §2 그대로 10.3.1
```

| | Used | %age | 여유 | 타깃 수 |
|---|---|---|---|---|
| 패치만 (사고 난 본) | 1,939,520 B | 98.65% | 26,560 B | 1255 |
| **패치 + crsf_rc** | **1,945,548 B** | **98.96%** | **20,532 B** | 1260 |

**crsf_rc 비용 = +6,028 B.** 여유가 남으므로 `uxrce_dds_client` 는 **끄지 않았다**(최소 변경).
그 결과 원본 대비 `UXRCE_DDS_CFG` 파라미터 1개가 더 있게 되는데, 값 0(비활성)이고
우리는 MAVROS 를 쓰므로 무해하다.

산출물: `/mnt/c/px4_flash/px4_fmu-v6c_f17f4patch_crsf_20260728.px4`
크기 **1,843,224 B**, sha256 **`a62df3e923ba2d15320d9cbd1be9d09f86f3687bdcabe0565fc59edfa0d836fa`**

### 11-5. 남은 교훈

- **§4-2 의 "펌웨어가 자기신고하지 않는다"는 파라미터 대조로 보완된다.** `ver all` 로는
  못 잡는 config 차이를 `compare_px4_params.py` 의 MISSING/ADDED 가 정확히 잡아냈다.
  플래시 후 파라미터 대조는 **선택이 아니라 필수 절차**다.
- **`CAL_MAG*` 백업 복원은 금지다** — 상세는 `POST_FLASH_CHECKLIST.md` §3(정정본).
- 진단 도구 `tools/px4_params/nsh_cmd.py` 신설(읽기 전용 nsh 조회, 화이트리스트 강제).

### 11-6. ✅ 2차 플래시 — **해소 확인 (최종 상태, 2026-07-28 22:27 실측)**

사용자가 §11-4 의 재빌드 산출물 `px4_fmu-v6c_f17f4patch_crsf_20260728.px4`
(sha256 `a62df3e9…`, 1,843,224 B)를 QGC로 플래시했다. **이것이 현재 기체에 올라가 있는
펌웨어다.** 아래는 전부 실측이며, 이 절로 §11 을 닫는다.

#### (a) `ver all` — 우리 crsf 포함본이 맞다

```
Build datetime: Jul 28 2026 20:27:22        <-- §11-4 재빌드 시각과 일치 (사고본은 01:39:05)
Toolchain:      GNU GCC, 10.3.1 20210621 (release)
PX4 git-hash:   c890d9db0a…                 <-- 실기체 원본과 동일 (§4-2 대로 정상)
PX4 version:    1.18.0 40
```

**판별의 핵심은 `Build datetime` 이다.** Toolchain·git-hash 는 사고본(01:39:05 빌드)과도
같으므로 **그 두 줄만으로는 crsf 포함본과 사고본을 구별할 수 없다.** 20:27:22 만이 구별한다.

#### (b) `crsf_rc status` — 드라이버 복귀 + 수신 실측

```
nsh> crsf_rc status
  UART device: /dev/ttyS1          <-- TELEM3. RC_CRSF_PRT_CFG=103 과 일치
  UART RX bytes: 1120
  Valid known packet CRCs: 80
  Invalid CRCs: 0
  Disposed bytes: 0
```

§11-2 의 `command not found` / `never published` 가 전부 사라졌다. **Invalid CRC 0 · Disposed 0**
이므로 배선·보레이트 문제도 없다.

#### (c) 파라미터 대조 — MISSING 0건

덤프 원본: `logs/2026-07-28_px4_flash/px4_params_2026-07-28_final-crsf.json` (**1438개**, 커밋됨).

| 비교 | 결과 | 해석 |
|---|---|---|
| **원본 펌웨어(pre-flash, 02:38, 1437개) vs 지금** | **MISSING 0건** / ADDED 1건 `UXRCE_DDS_CFG=0` | ✅ 잃은 것 없음. ADDED 1건은 §11-4 대로 **의도한 것**(원 빌더는 uxrce 를 껐으나 우리는 FLASH 여유가 남아 켠 채로 뒀다. 값 0 = 비활성이고 우리는 MAVROS 를 쓴다) |
| **사고본(post-flash, 20:14, 1436개) vs 지금** | ADDED 2건 `RC_CRSF_PRT_CFG=103` · `RC_CRSF_TEL_EN=0` | ✅ **재설정 없이 되살아났다** — 값이 EEPROM 에 남아 있었고 드라이버가 돌아오면서 정의가 복구되자 그 값이 그대로 붙었다 |

**개수 정합:** 1437 − 2(RC_CRSF) + 1(UXRCE) = 1436(사고본), 1436 + 2 = **1438** ✅

**눈으로 확인한 결정적 항목(전부 유지):** `SYS_AUTOSTART=13000` · 스틱 매핑 1~4 ·
`RC_MAP_ARM_SW=12` · `RC_MAP_KILL_SW=11` · `RC_MAP_FLTMODE=10`.

> `RC_MAP_OFFB_SW 7→0`, `RC_MAP_TRANS_SW 8→7` 은 **사용자가 의도해서 바꾼 값이다(확인받음).**
> 플래시 결함이 아니다 — 다음 세션이 이 차이를 사고로 오판하지 말 것.

#### (d) F-17 패치가 실기체에서 실제로 작동함 — ulog 직접 증거

`142150a`(2026-07-28 flight02 재현검증)에서 `position_setpoint_triplet` 발행 4건이
**전부 `course=NaN`** 이었다. 패치 전이라면 `course=0.0f`(= "정북 유지" 유효명령, §6-3)가
나왔어야 한다. ⇒ **패치가 기체에 들어갔고 의도대로 동작한다는 직접 증거.**
`ver all` 이 패치 여부를 자기신고하지 않는다는 §4-2 의 한계를, 결국 **비행 거동이 메꿨다.**

#### (e) 이 시점의 남은 제약 — 펌웨어 문제는 아니다

플래시·RC 는 닫혔지만 **실비행 검증을 막는 별개 결함 2건이 미해결**이다(FC 트랙 소관):

- **자기계 헤딩 의존 오차** — `5d55b3f`. 재캘리브레이션은 적용됐으나 지표가 오히려 악화
  (`test_ratio` 1.97→2.62, `cs_mag_fault` ON 0%→92.7%). 통과 기준은
  **"재캘리 후 기수를 남쪽에 두고 `test_ratio<1` 확인"**.
- **배터리 게이트 부재** — `f8e951f`. flight02 는 `Emergency battery level` 이 t=8.64s(고도 약 4m)
  부터 떠 있는 채로 50m 까지 올라가 천이를 시도했다. `offboard_node` 상태기계에 게이트가 없다.
