#!/usr/bin/env python3
"""PX4 파라미터 덤프 2개를 비교해 차이 나는 항목만 출력한다.

플래시 전 백업 vs 플래시 후 재덤프 대조용.

사용법:
    python3 tools/px4_params/compare_px4_params.py <before.json> <after.json>
    python3 tools/px4_params/compare_px4_params.py <before.json> <after.json> --critical-only

종료 코드:
    0 = 완전 일치 (플래시가 파라미터를 보존했다)
    1 = 차이 있음 (CHANGED/MISSING/ADDED 목록 확인 필요)
    2 = 사용법 오류
"""
import json
import math
import sys

# 날아가면 재캘리브레이션·재세팅이 필요한 결정적 항목들
CRITICAL_PREFIXES = (
    "CAL_MAG", "CAL_ACC", "CAL_GYRO", "CAL_BARO",
    "VT_", "FW_AIRSPD", "MPC_THR_HOVER", "SYS_AUTOSTART",
    "MAV_", "COM_", "SENS_BOARD", "EKF2_MAG", "CA_", "PWM_",
)

# float 비교 허용 오차 (MAVLink REAL32 왕복 오차 흡수)
REL_TOL = 1e-6
ABS_TOL = 1e-9


def is_critical(name):
    return any(name.startswith(p) for p in CRITICAL_PREFIXES)


def same(a, b):
    if isinstance(a, float) or isinstance(b, float):
        try:
            return math.isclose(float(a), float(b),
                                rel_tol=REL_TOL, abs_tol=ABS_TOL)
        except (TypeError, ValueError):
            return a == b
    return a == b


def load(path):
    d = json.load(open(path))
    return d, {n: p["value"] for n, p in d["params"].items()}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    crit_only = "--critical-only" in sys.argv
    if len(args) < 2:
        print(__doc__)
        return 2

    db, before = load(args[0])
    da, after = load(args[1])

    print(f"BEFORE  {args[0]}")
    print(f"        captured {db['captured_utc']}  count {len(before)}")
    print(f"AFTER   {args[1]}")
    print(f"        captured {da['captured_utc']}  count {len(after)}")
    print()

    missing = sorted(set(before) - set(after))   # 플래시 후 사라진 것
    added = sorted(set(after) - set(before))     # 새로 생긴 것
    changed = sorted(n for n in (set(before) & set(after))
                     if not same(before[n], after[n]))

    if crit_only:
        missing = [n for n in missing if is_critical(n)]
        added = [n for n in added if is_critical(n)]
        changed = [n for n in changed if is_critical(n)]
        print("(--critical-only: CAL_*/VT_*/FW_AIRSPD*/MPC_THR_HOVER/"
              "SYS_AUTOSTART/MAV_*/COM_*/CA_*/PWM_* 만 표시)\n")

    if changed:
        print(f"### CHANGED ({len(changed)})  <- 값이 달라진 항목")
        w = max(len(n) for n in changed)
        for n in changed:
            mark = " *CRITICAL*" if is_critical(n) else ""
            print(f"  {n:<{w}}  {before[n]!r:>18}  ->  {after[n]!r}{mark}")
        print()

    if missing:
        print(f"### MISSING ({len(missing)})  <- 백업엔 있는데 플래시 후 없음")
        for n in missing:
            mark = " *CRITICAL*" if is_critical(n) else ""
            print(f"  {n} = {before[n]!r}{mark}")
        print()

    if added:
        print(f"### ADDED ({len(added)})  <- 플래시 후 새로 생김")
        for n in added:
            print(f"  {n} = {after[n]!r}")
        print()

    ncrit = sum(1 for n in changed + missing if is_critical(n))
    total_diff = len(changed) + len(missing) + len(added)
    if total_diff == 0:
        print("=> 차이 없음. 파라미터가 그대로 보존되었다.")
        return 0
    print(f"=> 차이 {total_diff}건 (CHANGED {len(changed)} / "
          f"MISSING {len(missing)} / ADDED {len(added)}), "
          f"그중 CRITICAL {ncrit}건")
    if ncrit:
        print("   CRITICAL 이 1건이라도 있으면 복구 절차를 밟는다 "
              "(logs/2026-07-28_px4_flash/POST_FLASH_CHECKLIST.md §4).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
