# pitot_bench — 피토 차압 영점 열드리프트 벤치 계측 도구

`docs/fc_pitot_thermal_experiment.md` 실험(가설 T/W/M 판별)의 **계측 장비**다.
PX4 USB MAVLink 로 **raw 차압 + 온도 3채널**을 CSV 로 연속기록한다.

**전부 읽기 전용이다.** 파라미터 쓰기·ARM·캘리브레이션·재부팅을 하지 않는다.
`MAV_CMD_SET_MESSAGE_INTERVAL`(런타임 스트림 요청)만 보내며, 이는 FC 재부팅 시 사라진다.

---

## 왜 ulog 대신 이걸 쓰나

절차서 초판은 `SDLOG_MODE=2` + FC 재부팅으로 ulog 를 남기게 했다. 이 로거는 그게 필요 없다.

| | ulog (`SDLOG_MODE=2`) | **이 CSV 로거 (권장)** |
|---|---|---|
| 파라미터 변경 | **필요** (`SDLOG_MODE` 0→2, 끝나고 원복) | **불필요** |
| FC 재부팅 | **필요** | **불필요** |
| 표본율 | 0.98 Hz | **10 Hz** (`--hz`) |
| 온도 채널 | 다이 1채널 | **3채널** (다이 / FC 내장 기압계 / HIGHRES_IMU) |
| 환경 변경 마커 | 없음 | **있음** (`mark.sh`, 같은 CSV 에 event 행) |
| 중간 확인 | 회수 후에만 | 돌아가는 중에 `dpres_stats.py` |
| 회수 | SD 카드 뽑거나 `pull_ulog.py` | 파일이 이미 호스트에 있음 |

ulog 경로는 **폐기하지 않았다** — 절차서 §4 에 대안으로 남아 있다.
FC 를 어차피 재부팅해야 하거나 USB 를 못 붙이는 상황이면 그쪽을 쓴다.

---

## 데이터 출처 (2026-07-31 실기체 실측 확인)

| CSV 컬럼 | MAVLink 출처 | 의미 |
|---|---|---|
| `dp_raw_pa` | `SCALED_PRESSURE.press_diff` × 100 | **`SENS_DPRES_OFF` 적용 전 RAW 차압**. ulog `differential_pressure.differential_pressure_pa` 와 같은 값 |
| `dp_corr_pa` | `dp_raw_pa − SENS_DPRES_OFF` | PX4 `airspeed_selector` 와 같은 식. 무풍이면 0 이어야 한다 |
| `dp_die_temp_c` | `SCALED_PRESSURE.temperature_press_diff` / 100 | **MS4525DO 다이 온도** (분해능 0.0977 °C). ulog `differential_pressure.temperature` 와 같은 값 |
| `hr_imu_temp_c` | `HIGHRES_IMU.temperature` | 이 빌드에선 다이 온도와 **동일**(40초 대조 확인). 확장필드 누락 시 대체채널로 쓴다 |
| `baro_temp_c` | `SCALED_PRESSURE.temperature` / 100 | FC 내장 기압계 온도 ≈ **FC 케이스 내부 온도** |
| `airspeed_vfr_ms` | `VFR_HUD.airspeed` | PX4 최종 IAS (오프셋 적용 後). 정합 확인용 |
| `event` | `mark.sh` | 비어있지 않은 행 = 사용자 마커 |

`SENS_DPRES_OFF` 는 시작 시 `PARAM_REQUEST_READ` 로 **읽어서** CSV 헤더에 박아둔다(쓰지 않는다).

---

## 사용

```bash
# RPi 에서 (컨테이너 밖 호스트. pymavlink 필요)
tools/pitot_bench/dpres_start.sh "Phase2 저온 안정구간 시작"

# 환경을 바꿀 때마다 다른 터미널에서 — 같은 CSV 에 event 행이 박힌다
tools/pitot_bench/mark.sh "에어컨 OFF, 창문 개방"
tools/pitot_bench/mark.sh "외기 평형 도달, 기상청 31.8C"

# 돌아가는 중에 중간 확인 (읽기만 하므로 로거를 안 건드린다)
python3 tools/pitot_bench/dpres_stats.py ~/dpres_bench/latest.csv -22.7882

# 종료 (SIGTERM -> CSV 에 stopped_ 푸터를 쓰고 닫는다. kill -9 금지)
tools/pitot_bench/dpres_stop.sh
```

`dpres_start.sh` 는 `setsid` + `nohup` + `SIGHUP` 무시로 띄운다 — **SSH 가 끊겨도 계속 돈다.**
CSV 는 `$HOME/dpres_bench/`(또는 `$DPRES_BENCH_DIR`)에 쌓이고 `latest.csv` 심볼릭 링크가 걸린다.
**CSV 는 저장소로 회수해 커밋한다** (`logs/2026-07-31_pitot_thermal/` 등).

### 정찰용 (실험 전 1회면 충분)

```bash
python3 tools/pitot_bench/probe_mav.py /dev/ttyACM0 12     # 어떤 메시지에 뭐가 실리는지
python3 tools/pitot_bench/probe_temp_sources.py /dev/ttyACM0 40  # 온도 채널 4종 실측 대조
python3 tools/pitot_bench/ulog_dpres_temp.py "logs/2026-07-*/*.ulg" 10  # 기존 ulog 회귀
```

> `ulog_dpres_temp.py` 는 **로그당 1점**을 뽑는 조잡한 정찰판이다. 본격 분석은
> `tools/flight_logs/pitot_zero_temp.py`(표본 120개, 전류·대지속도 게이트 포함)를 쓴다.
> 정찰판의 `-1.9 Pa/°C` 가 중간 온도 표본 결손 때문에 과대추정이었다는 게
> `docs/fc_pitot_temp_drift.md` §9.1 의 결론이다.

---

## ⚠️ 운용 함정 (실측으로 물린 것만)

1. **`tools/px4_params/nsh_cmd.py`(및 `ver_all.py`·`dump_px4_params.py`)와 동시 사용 불가.**
   둘 다 `/dev/ttyACM0` 을 열어 읽으므로 MAVLink 바이트스트림이 갈려 양쪽 다 깨진다.
   FC 셸을 써야 하면 `dpres_stop.sh` → 조회 → `dpres_start.sh` 순으로 한다
   (CSV 가 두 파일로 갈리지만 시각이 남으므로 이어붙일 수 있다).
   같은 이유로 **mavros/컨테이너가 `/dev/ttyACM0` 을 잡고 있으면 안 된다.**

2. **`temperature_press_diff` 는 MAVLink 확장 필드다.** PX4 가 후행 0 바이트를 잘라 보내면
   pymavlink 메시지에 **속성 자체가 없다.** `getattr` 로 받아 `HIGHRES_IMU.temperature`
   로 대체한다 — v2 첫 기동에서 `AttributeError` 로 즉사한 실측 이력이 있다.

3. **`kill -9` 금지.** SIGTERM 이라야 핸들러가 `stopped_local`·`samples=` 푸터를 쓴다.
   (CSV 는 라인버퍼 + 10초 `fsync` 라 강제종료해도 데이터 자체는 거의 안 날아간다.)

4. **다이 온도는 자기발열로 외기보다 4~8 °C 높다.** `baro_temp_c` 와 함께 봐야
   "센서가 더운 건지 방이 더운 건지"가 갈린다. 07-30 실비행에서 다이가
   37.8 → 30.8 °C 로 식은 것은 **기류에 의한 자기발열 감소**였다.

---

## 검증 이력

- **2026-07-31, RPi 실기체 USB 직결.** `dpres_log.py` 가 **5,000행 이상** 연속기록으로
  가동 검증됨(상온 베이스라인 축적). 재접속·스트림 재요청·마커 경로 모두 실동작 확인.
- 같은 날 `probe_temp_sources.py` 40초 대조로
  `SCALED_PRESSURE.temperature_press_diff` == `HIGHRES_IMU.temperature` 확인.
- `probe_mav.py` 로 `SCALED_PRESSURE.press_diff` 가 **오프셋 적용 전 RAW** 임을 확인
  (`VFR_HUD.airspeed` 와 `SENS_DPRES_OFF` 관계로 역산).

> **이 코드는 실기체에서 동작이 검증된 것이다. 리팩터링하지 말 것.**
> 저장소로 들여오면서 바꾼 것은 경로 해석 3줄뿐이다:
> `dpres_start.sh` 의 로거 경로(`$HOME/tmp` → 스크립트 옆, `$DPRES_LOG_PY` 로 덮어쓰기 가능),
> `mark.sh`·`dpres_stop.sh` 의 `$DPRES_BENCH_DIR` 지원. 로직은 무변경이다.
