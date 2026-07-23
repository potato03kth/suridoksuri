# 매몰비용(Sunk Cost) 재점검 진단 리포트

> 대상: `suridoksuri` VTOL 시뮬레이터 전체
> 성격: **진단 전용.** 이 문서는 어떤 코드도 수정하지 않는다. 각 항목의 권고 처리 방향은 "격리 + 표시"(삭제 아님, git 히스토리 보존)이며, 실제 실행은 이 리포트 승인 후 별도 작업으로 남긴다.
> 방법: 4개 독립 코드베이스 스윕(코어 알고리즘 · 아키텍처 · git 히스토리 · Claude-facing 아티팩트) + 핵심 파일 직접 확인으로 교차검증.

---

## 0. 왜 이걸 보는가 — "멍청해진 클로드" 문제

이 레포는 git상 **2주간 단일 저자 집중 버스트**(42 커밋, 2026-04-17~04-30)로 만들어졌다. 그 과정에서 "이미 만들었으니 남겨둔다"는 **매몰비용 판단**이 코드·설계·문서에 누적됐다.

핵심은 이게 단순 기술부채가 아니라 **다음 Claude 세션을 멍청하게 만드는 피드백 루프**라는 점이다:

```
거짓 헤더 / 낡은 문서 / 중복 레지스트리
        │  (새 Claude가 읽고 신뢰)
        ▼
가짜 기반(스텁) 위에서 작업  또는  엉뚱한 파일을 수정
        │
        ▼
v3 → v3.1 → v3.2 처럼 레이어를 하나 더 쌓음
        │
        ▼
매몰비용이 더 커지고, 다음 세션은 더 헷갈린다  ──┐
        ▲                                        │
        └────────────────────────────────────────┘
```

**"내 클로드를 살려줘"의 실질 = 이 루프를 끊는 것.** 그래서 이 리포트는 §A(Claude-facing)를 최우선으로 둔다.

### 매몰비용 판정 기준 (렌즈)

어떤 아티팩트가 아래 **3가지를 모두** 만족하면 "매몰비용 판단"으로 표시:

1. 트리 / 레지스트리 / 문서에 **여전히 존재**하고,
2. **현재 가치가 없음** — 죽음(dead) · 가짜(fake) · 상위판으로 대체됨(superseded) · 중복(duplicate) 중 하나, 그리고
3. 살아남은 유일한 이유가 **과거 투자**로 보임 — 작동하는 대체물 옆에 나란히 유지되거나, 작동 안 하는데 작동하는 것처럼 광고됨.

**반례 가드:** 의도된 벤치마크 다양성(작동하는 여러 플래너를 비교 목적으로 병렬 유지)은 병렬이어도 매몰비용이 **아니다**. §C3에서 "버려진 반복"과 "의도된 다양성"을 명시적으로 분류한다.

**심각도 표기:** 🔴 치명(가짜를 진짜로 광고 / 새 세션 오도) · 🟠 중대(죽은 코드·중복·가드레일 부재) · 🟡 정리(위생).

---

## A. Claude-facing 매몰비용 — 최우선

새 세션을 오도하는 문서·헤더·레지스트리 함정. **이 섹션이 "클로드 살리기"의 본체다.**

### 🔴 A1. `path_planning/CLAUDE.md` 디렉터리 맵이 eta3 계열 전체를 누락
- **증거:** `CLAUDE.md:9–23`의 디렉터리 구조 표에 `eta3clothoid_planner.py`, `eta3clothoid_stage2_planner.py`, `eta3clothoid_v3_1_planner.py`, `piecewise_clothoid_planner.py` **4개가 없다.** 정작 git 로그를 지배한 가장 최근·가장 복잡한 플래너들이다.
- **왜 매몰비용:** 옛 구조를 그대로 둔 채 신규 파일만 늘렸다. 지도(map)를 갱신하는 비용을 안 치른 것.
- **클로드 영향:** 새 세션은 자기 작업 가이드만 읽고는 eta3 계열의 **존재조차 모른다.** 스스로 파일을 발견하면 어느 게 스텁/구버전인지 경고를 못 받는다.
- **권고(격리+표시):** 맵을 갱신하되 **상태 컬럼**을 추가 — 각 플래너에 `실동 / PSEUDO / DEPRECATED` 명시. 특히 `eta3clothoid`=PSEUDO, `eta3clothoid2`=휴리스틱(v2), `eta3clothoid3`=**실동(권장 시작점)**.

### 🔴 A2. `eta3clothoid_planner.py`는 스텁인데 "✓ G2 보장"이라 광고하고, crash 없이 가짜 경로를 낸다
- **증거:** 헤더(`:1–19`)는 "보장 항목: ✓ WP 완전 통과 / ✓ κ_max / ✓ G2 연속성"이라 선언. 그런데 `_eta3_g2_residual`은 `:166`에서 `raise NotImplementedError("PSEUDO: η³ G2 조건 …")`. `plan()`은 `:353`에서 "NotImplementedError를 발생시킴"을 주석으로 인정하고 `:361–362`에서 **PSEUDO 대체 초기값**(`_eta3_initial_guess`, 내부 곡률 0)으로 우회한다.
- **결과:** 플래너는 **크래시하지 않고** degenerate 경로(내부 곡률 뭉갬, heading=단순 bisector)를 반환한다. `--planner eta3clothoid`로 등록됨(`run_scenario.py:69–70`, argparse choices `:114`).
- **왜 매몰비용:** 미완성 골격에 "코드 흐름 확인용" 우회를 붙여 **실행만 되게** 만들어 놓고, 헤더의 보장 문구는 지우지 않았다.
- **클로드 영향:** "eta3 clothoid 플래너 테스트해줘"를 받은 세션이 그럴듯하지만 수학적으로 텅 빈 출력을 얻고, docstring은 "보장됨"이라 말하니 **버그로 인지하지 못한다.** 최악의 오도.
- **권고:** `_deprecated/`로 격리 + 레지스트리에서 등록 해제 + 헤더를 `PSEUDO — 실행되지만 가짜 경로. eta3clothoid_v3_1_planner.py 사용`으로 교체.

### 🔴 A3. 동명 클래스 `Eta3ClothoidPlanner`가 두 파일에 존재 (사일런트 풋건)
- **증거:** `eta3clothoid_planner.py`(v1/스텁)와 `eta3clothoid_stage2_planner.py`(v2)가 **둘 다 `class Eta3ClothoidPlanner`**. `run_scenario.py:29`가 `import ... as Eta3ClothoidPhase2Planner` alias로만 구분한다.
- **왜 매몰비용:** stage2를 새 파일로 만들면서 클래스명을 재사용(복붙 혈통). 이름을 정리하는 비용을 안 냄.
- **클로드 영향:** `from path_planning.eta3clothoid_planner import Eta3ClothoidPlanner`와 stage2 모듈에서의 동일 import가 **전혀 다른 구현**을 가져온다 — 조용히 잘못된 걸 쓰게 된다.
- **권고:** 표시 우선(맵/헤더에 명기). 개명은 후속(등록 해제 시 자연 해소).

### 🔴 A4. 설계 문서 4건이 편집을 **죽은 스텁 쪽으로 유도**
- **증거:**
  - `ETA3CLOTHOID_STAGE2_REDESIGN.md` — 헤더가 `대상 파일: eta3clothoid_planner.py`. 본문은 `_check_and_insert_wps()`·`_global_stage2_nr()`를 "✅ 완료(2026-04-29)"로 표기하고 스모크 테스트 결과까지 싣지만, **그 코드는 named 파일이 아니라 별도 파일 `eta3clothoid_stage2_planner.py`에 있다.** named 파일은 아직 `NotImplementedError` 상태.
  - `CONSTRAINED_OPTIMIZATION_KAPPA.md` / `WP_INSERTION_KAPPA.md` — 둘 다 대상 파일을 스텁(`eta3clothoid_planner.py`)으로 지목. 이미 v2/v3가 폐기했거나 이미 구현한 로직의 TODO.
  - `eta3_clothoid_planner_v3.md` — import 예제가 **존재하지 않는 모듈** `from eta3_clothoid_planner_v3 import ...` 를 참조(실제 파일: `eta3clothoid_v3_1_planner.py`). 복붙 시 ImportError 또는 새 중복 파일 생성 유발.
- **왜 매몰비용:** 리팩터를 새 파일로 옮기면서 옛 문서의 "대상 파일" 헤더를 갱신 안 함. 문서 자산에 투자했으니 남겨둔 것.
- **클로드 영향:** 이 문서를 읽고 named 파일을 열면 "작업이 사라졌다/회귀했다"고 오판하거나, **이미 다른 파일에 있는 걸 재구현**한다.
- **권고:** 각 문서 상단에 배너 — `⚠ 이 문서는 <파일>의 과거/미완 상태를 기술함. 현재 실동 구현은 eta3clothoid_v3_1_planner.py` — 또는 `docs/legacy/`로 격리.

### 🟠 A5. 과대광고 vs 코드 — "fully implemented" / "구조적 보장"이 자기모순
- **증거:**
  - `eta3clothoid_stage2_planner.py:1` "v2 (Stage 1 fully implemented)". 하지만 실제 잔차는 `_kappa_natural`(0.5·Menger + 0.25·이웃 스무딩)을 쓰는 **휴리스틱 대체**이고, 같은 `REDESIGN.md`가 "3순위: 실제 Stage 1 η³ G2 잔차 ⬜ 미구현"이라 인정.
  - `ETA3CLOTHOID_EXPLAINED.md`의 "구조적 보장 / 위반 불가"(G2·κ_max)는 `eta3clothoid_planner_CHANGES_v3_2.md`의 ⚠ 노트(affine 보정 시 G2 보장 상실·κ_max 초과 가능, "NR 잔차 충분히 작을 때만" 성립)와 모순.
- **왜 매몰비용:** 각 세대의 홍보 문구를 지우지 않고 다음 세대로 넘어감.
- **클로드 영향:** "구조적 보장"을 믿고 실제 실패 모드를 스트레스 테스트하지 않는다.
- **권고:** EXPLAINED의 보장 표현을 "NR 수렴 시 성립(조건부)"으로 완화 표기.

### 🟠 A6. `piecewise_clothoid_planner.py` — 미등록 고아 PSEUDO인데 docstring은 작동하는 척
- **증거:** 헤더는 "[PSEUDO CODE]"이고 `_g3_subdivide`는 `NotImplementedError`. 그런데 **클래스 docstring은 PSEUDO 딱지를 떼고** 정상 글로벌-NR 플래너처럼 서술. `run_scenario.py` 어디에도 import/등록 안 됨(고아). 더불어 `CLOTHOID_LOOP_PLANNER_PSEUDO.md`는 **존재하지 않는 .py**(`clothoid_loop_planner.py`)를 설계하며 스스로 "WP 통과 보장 불가"라 명시.
- **클로드 영향:** 자신만만한 클래스 docstring이 "이거 마저 완성해줘" 유혹 → 이미 작동하는 `eta3clothoid3` 옆에 **또 하나의 병렬 미완성**을 만들게 한다.
- **권고:** 고아 파일 → `_deprecated/` 격리 + 헤더 정정. 없는 파일용 설계 문서 → `docs/legacy/`.

---

## B. 구조 / 시스템 매몰비용

### 🟠 B1. 죽은 `ModeManager` (HOVER/TRANSITION/CRUISE 상태기계 전체가 데드)
- **증거:** `simulator.py:24` import, `:193` `self.mode_mgr = ModeManager(...)` 인스턴스화. 그러나 `update_mode` / `quick_start_cruise` / `initialize`가 **시뮬 루프에서 한 번도 호출되지 않음**(grep 결과 인스턴스화만 존재). 시뮬은 항상 순수 CRUISE.
- **왜 매몰비용:** VTOL 전이 모드를 위해 만든 하위 시스템을, 크루즈만 쓰는 현 시나리오에서도 "이미 만들었으니" 배선만 남겨둠.
- **권고:** `dynamics/mode_manager.py`를 `_deprecated/`로 격리 + `simulator.py`의 미사용 인스턴스화 제거는 후속.

### 🟠 B2. 스테일 중복 레지스트리 — `compare_algorithms.py`가 `run_scenario.py`의 옛 좁은 복사본
- **증거:** `compare_algorithms.py:51–55` `COMBINATIONS`는 **dubins/spline만** 안다. `build_planner`(`:59`)·`build_controller`(`:67`)는 `run_scenario.py`의 12-플래너 버전에서 갈라진 뒤 갱신 안 된 축소 복제.
- **왜 매몰비용:** 배치 러너를 복붙으로 만든 뒤 단일 소스로 합치는 비용을 안 냄.
- **클로드 영향:** 새 플래너를 `run_scenario`에만 등록하면 `compare_algorithms`에선 조용히 빠진다. 두 진실 소스.
- **권고:** 표시(문서에 "레지스트리 2곳, 단일화 필요" 명기). 실제 통합은 후속.

### 🟠 B3. `run_scenario.py` 처닝 핫스팟 — 드리프트를 만드는 메커니즘
- **증거:** git상 8회 수정(최다). 플래너 하나 추가 시 **import + `build_planner` 분기 + argparse `choices`** 세 곳을 손대야 함(`:19–30`, `:50–75`, `:114`).
- **왜 매몰비용/영향:** 이 3중 수기 등록 패턴이 곧 B2(스테일 복제)를 낳는 구조적 원인. B2와 함께 다뤄야 근본 해결.
- **권고:** 진단으로만 기록(레지스트리 딕셔너리化는 후속 리팩터 후보).

### 🟡 B4. 죽은 유틸 / 영구 비활성 기능
- `utils/delay_buffer.py:53` `VariableDelayBuffer` — 정의만 있고 레포 전체에서 미사용(정의 파일 외 참조 0). `DelayBuffer`만 실사용.
- `utils/geodetic.py` — 시뮬 본체가 아니라 테스트에서만 사용.
- MPC 곡률 피드포워드 — `mpc_controller.py`에서 `D_vec` 경로가 "Phase 3에서"라는 주석과 함께 사실상 비활성(문서화된 d_k 피드포워드가 실제 계산에서 제외됨).
- **권고:** 격리 대상 목록에 기록. "Phase 3" 주석은 로드맵인지 죽은 flag인지 저자 확인 후 정리.

### 🟡 B5. bspline 4파일이 모두 `BSplinePlanner` 동명 export
- **증거:** `bspline_planner.py` / `bspline_2_planner.py` / `hermite_bspline.py` / `Qhermite_bspline.py` 전부 `class BSplinePlanner`. `run_scenario.py:21–24`에서 alias로만 구분(`as BSpline2Planner` 등).
- **왜 매몰비용:** 복붙으로 변형 플래너를 찍어내며 클래스명을 안 바꿈. A3와 동일 패턴.
- **권고:** 맵/헤더에 표시. 개명은 후속.

### 🟡 B6. 생성 산출물이 VCS에 커밋됨
- **증거:** `results/`에 ~50개 PNG. 옛 `strftime` 버그로 파일명에 `time.struct_time(...)` repr이 박힌 것 포함.
- **권고:** `.gitignore`에 `results/` 추가 + 기존 아티팩트 정리는 후속.

### 🟠 B7. 처닝·버그 집중 레이어에 테스트 0 — A2를 잡을 가드레일 부재
- **증거:** 유일한 테스트 `tests/test_checkpoint1.py`(15개)는 전부 dynamics/util/estimator 저수준. **플래너 · 컨트롤러 · 오케스트레이션(simulator/run_scenario/compare)에 대한 테스트는 없다.** `CLAUDE.md`의 체크리스트(s 단조 · wp_index 유일 · 곡률 부호 · χ 범위)는 **문서로만 존재, 강제하는 테스트 없음.**
- **왜 이게 매몰비용의 결과:** 가드레일이 없으니 A2의 degenerate 스텁 출력이 "성공"으로 통과한다. 검증을 커밋된 PNG 눈대중에 의존.
- **권고:** 진단으로 기록. `CLAUDE.md` 체크리스트를 강제하는 planner 계약 테스트가 최우선 후속 후보(모든 등록 플래너에 s단조/wp_index/부호/χ 자동 검사).

### 🟡 B8. 테스트 파일 내 복붙 데드 블록
- **증거:** `test_checkpoint1.py`의 `test_composite_score_consistency`(Test 14) 본문 뒤에 `[Test 12]` SimLog 블록이 중복 append됨(무해하나 죽은 복붙).
- **권고:** 정리 목록에 기록.

---

## C. 설계 / 알고리즘 매몰비용

### 🔴 C1. eta3 3세대가 전부 등록된 채 공존
- **증거:** `run_scenario.py`가 `eta3clothoid`(v1/가짜) · `eta3clothoid2`(v2/휴리스틱) · `eta3clothoid3`(v3/실동)을 모두 등록.
- **왜 매몰비용:** 작동하는 v3가 있는데도 v1·v2를 등록 유지 — 판정 기준 3요소를 정확히 충족하는 **교과서적 매몰비용.**
- **권고:** v3만 등록 유지. v1·v2는 `_deprecated/` 격리(히스토리 보존).

### 🟡 C2. 수치 헬퍼 바이트 단위 복붙
- **증거:** `_unit`·`_wrap`·`_clothoid_sample`·`_fresnel_endpoint`가 eta3 파일들에 동일 복사. `_quintic_hermite`가 `D_iterpin_planner.py`/`iterpin` 등 다수에 재구현(18개 basis 도함수 수기).
- **권고:** 공용 `path_planning/_clothoid_math.py`로 추출은 후속. 지금은 목록화만.

### 🟡 C3. 12 병렬 플래너 — "의도된 다양성" vs "버려진 반복" 분류 (과잉정리 방지)

| 플래너 | 등록명 | 판정 | 근거 |
| --- | --- | --- | --- |
| dubins | `dubins` | **유지 (기준선)** | 실동, 벤치마크 baseline |
| spline | `spline` | **유지** | 실동, Cubic C2 |
| bspline / bspline2 | `bspline`/`bspline2` | **유지(검토)** | 실동 변형. 근접하나 degree/정제 방식 상이 |
| hermite / qhermite | `hermite`/`qhermite` | **유지(검토)** | 실동 변형. 저자 의도 확인 권장 |
| iterpin | `iterpin` | **유지** | 실동 |
| D_iterpin | `diterpin` | **유지** | 실동, 최고 복잡도(우회 WP) |
| clothoid | `clothoid` | **유지** | 실동, 곡률 구조 보장 |
| **eta3clothoid (v1)** | `eta3clothoid` | **격리** | 가짜(A2) |
| **eta3clothoid2 (v2)** | `eta3clothoid2` | **격리** | v3로 대체된 휴리스틱(A5) |
| eta3clothoid3 (v3) | `eta3clothoid3` | **유지 (실동 권장)** | 유일한 실동 eta3 |
| piecewise_clothoid | (미등록) | **격리** | 고아 PSEUDO(A6) |

> **원칙:** spline 변형군(bspline/hermite/qhermite)은 "작동하는 벤치마크 다양성"일 가능성이 높다 — 삭제/격리 판단은 **저자에게 남긴다.** 이 리포트는 명백한 가짜/대체분(v1·v2·piecewise)만 격리 권고한다.

### 🟡 C4. `_1` 접미사 — 삭제된 중복 트리의 흔적
- **증거:** 커밋 `16348ad "delete deprecated module"`이 시뮬 전체 사본 `vtol_sim_checkpoint1/`(23파일 1336줄)를 삭제. 살아남은 디렉터리명 `vtol_sim_checkpoint1_1`의 `_1`이 그 매몰비용의 네이밍 상처.
- **권고:** 표시만(개명은 경로 참조 다수라 후속·별건).

---

## D. 권고 요약 — 격리 + 표시 (실행은 승인 후 별도)

| 분류 | 항목 | 처리 |
| --- | --- | --- |
| 가짜/죽은/고아 코드 | A2(v1), A6(piecewise), B1(ModeManager), B4(VariableDelayBuffer), C1(v1·v2) | `_deprecated/`(또는 `path_planning/legacy/`)로 이동 + 헤더에 `DEPRECATED/PSEUDO — 사용 금지, 실동 대체 명시` + 레지스트리 등록 해제. git 히스토리로 복구 가능. |
| 오해 유발 문서 | A4, A5, A6(loop doc) | 상단 배너(현재 실동 구현 지목) 추가 또는 `docs/legacy/` 격리. |
| 살아있는 함정(표시) | A1, A3, B2, B5 | `CLAUDE.md` 디렉터리 맵 갱신(+상태 컬럼), 동명 클래스·중복 레지스트리 경고 명기. |
| 위생 | B6, B8, C2, C4 | `.gitignore`, 데드 블록 제거, 헬퍼 추출, 네이밍 — 후속 위생 작업. |
| 저자 판단 필요 | C3(spline 변형군) | 유지/격리 분류표만 제시. 삭제 판단은 사용자. |
| 가드레일(핵심 후속) | B7 | 모든 등록 플래너에 `CLAUDE.md` 계약(s단조/wp_index/부호/χ)을 강제하는 테스트 — 이게 있었으면 A2가 애초에 안 통과. |

---

## E. 재검증 커맨드 (누구나 값싸게 재확인)

```bash
cd vtol_sim_checkpoint1_1/vtol_sim

# 가짜/스텁 목록
rg -n "NotImplementedError|PSEUDO" path_planning

# A2: 가짜 경로가 crash 없이 나오는지 (내부 곡률이 0으로 뭉개짐)
python run_scenario.py basic --planner eta3clothoid --seed 42 --no-plot

# 레지스트리 vs 실제 파일 / 두 러너 간 드리프트(B2)
rg -n "choices=|if name ==" run_scenario.py
rg -n "COMBINATIONS|if name ==" compare_algorithms.py

# 죽은 ModeManager(B1): 인스턴스화만 있고 호출 없음
rg -n "mode_mgr|update_mode|quick_start_cruise" simulator.py

# 동명 클래스(A3, B5)
rg -n "class (Eta3ClothoidPlanner|BSplinePlanner)\b" path_planning

# 문서-코드 모순(A4): 각 md "대상 파일" 헤더가 가리키는 .py의 실제 상태
rg -n "대상 파일" path_planning
```

**리포트 성공 판정:** *새 Claude 세션이 이 문서만 읽고 eta3 작업을 시작할 때, 올바른 파일 `eta3clothoid_v3_1_planner.py`에서 시작하고 v1 스텁을 만지지 않는가?* — 그렇다면 루프가 끊긴 것이다.

---

## F. 이번 범위 밖 (의도적 제외)

- 실제 코드 이동/삭제/개명, 레지스트리 수정, 문서 배너 삽입 → 전부 승인 후 별도.
- 새 테스트 작성(B7은 진단으로만 기록).
- η³ 실제 Stage 1 등 알고리즘 정확도 재구현.
