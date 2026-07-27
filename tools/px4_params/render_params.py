#!/usr/bin/env python3
"""dump_px4_params.py 가 만든 JSON 을 사람이 읽는 형식과 QGC 복구용으로 변환한다.

사용법:
    python3 tools/px4_params/render_params.py <dump.json> <출력_스템>
        -> <출력_스템>.txt     정렬된 사람이 읽는 텍스트
        -> <출력_스템>.csv     표 계산용
        -> <출력_스템>.params  QGC "Tools > Load from file" 로 그대로 복원 가능
"""
import csv
import json
import sys


def fmt(p):
    v = p["value"]
    if isinstance(v, float):
        # float 은 왕복 가능한 표현으로
        return repr(v)
    return str(v)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    src, stem = sys.argv[1], sys.argv[2]
    d = json.load(open(src))
    params = d["params"]
    names = sorted(params)

    with open(stem + ".txt", "w") as f:
        f.write(f"# PX4 parameter dump\n")
        f.write(f"# captured_utc : {d['captured_utc']}\n")
        f.write(f"# device       : {d['device']}\n")
        f.write(f"# sysid/compid : {d['target_system']}/{d['target_component']}\n")
        f.write(f"# count        : {d['param_count_received']} "
                f"(FC reported {d['param_count_reported']})\n")
        f.write("#\n# <NAME> <TYPE> <VALUE>\n\n")
        w = max(len(n) for n in names)
        for n in names:
            p = params[n]
            f.write(f"{n:<{w}}  {p['type_name']:<7} {fmt(p)}\n")

    with open(stem + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "type", "value", "index", "raw_float"])
        for n in names:
            p = params[n]
            w.writerow([n, p["type_name"], fmt(p), p["index"], repr(p["raw_float"])])

    # QGroundControl 파라미터 파일 (탭 구분: sysid compid name value MAV_PARAM_TYPE)
    sysid = d.get("target_system", 1)
    compid = d.get("target_component", 1)
    with open(stem + ".params", "w") as f:
        f.write("# Onboard parameters for Vehicle %s\n#\n" % sysid)
        f.write("# Stack: PX4\n")
        f.write("# Vehicle: VTOL\n")
        f.write("# Captured(UTC): %s\n" % d["captured_utc"])
        f.write("# Count: %d\n#\n" % len(names))
        f.write("# Vehicle-Id Component-Id Name Value Type\n")
        for n in names:
            p = params[n]
            f.write("%d\t%d\t%s\t%s\t%d\n"
                    % (sysid, compid, n, fmt(p), p["type"]))

    print(f"wrote {stem}.txt / .csv / .params ({len(names)} params)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
