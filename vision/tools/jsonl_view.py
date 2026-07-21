"""
JSONL 블랙박스 뷰어/플롯 최소본 (vision_plan.md §7.9 "지금 당장 할 일" 6번).

`BlackBoxLogger`(vision/utils/blackbox.py)가 프레임별로 쌓는 JSONL은 "쌓기만 하면 안 보인다"
(§7.9) — 이 스크립트는 그 JSONL을 그대로(새 스키마 정의 없이) 읽어 시간축 위에
검출점수·latency·state를 플롯해 PNG로 저장하는 최소 뷰어다.

읽는 스키마 (blackbox.py의 실제 출력 그대로):
  {"type": "frame", "ts", "frame_id", "alt", "attitude",
   "detections": [{"bbox": [...], "confidence": float, ...}, ...],
   "chosen": {"bbox": [...], "confidence": float} | null,
   "command", "state", "latency"}
  {"type": "rejection", "ts", "frame_id", "reason", "meta"}

score는 "chosen"(TemporalFusion이 확정한 결과)의 confidence를 우선 쓰고, chosen이 없으면
그 프레임 detections 중 최고 confidence를 쓴다(둘 다 없으면 None → 플롯에서 구멍으로 표시).
거절(rejection) 레코드는 score 서브플롯에 세로 점선으로 표시한다.

과설계 금지 — 이 스크립트는 §7.9 5번(라이브 스트림)·7번(골든셋)과 무관하다. Foxglove 연동
같은 건 범위 밖.

CLI 사용:
    python vision/tools/jsonl_view.py results/logs/vision.jsonl
    python vision/tools/jsonl_view.py results/logs/vision.jsonl --output out.png --x-axis frame_id

라이브러리로 사용(테스트 등):
    from vision.tools.jsonl_view import load_records, build_figure, save_figure
"""
import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # headless-safe — GUI 강제 호출 금지 (드론/CI에 디스플레이 없음)
import matplotlib.pyplot as plt


@dataclass
class FrameRow:
    frame_id: Optional[int]
    ts: Optional[float]
    score: Optional[float]
    latency: Optional[float]
    state: Optional[str]
    n_detections: int


def _best_score(rec: dict) -> Optional[float]:
    """chosen(confirmed) confidence 우선, 없으면 detections 중 최고 confidence."""
    chosen = rec.get("chosen")
    if isinstance(chosen, dict):
        for key in ("confidence", "score"):
            if key in chosen and chosen[key] is not None:
                return chosen[key]
    detections = rec.get("detections") or []
    scores = []
    for d in detections:
        if not isinstance(d, dict):
            continue
        for key in ("confidence", "score"):
            if key in d and d[key] is not None:
                scores.append(d[key])
                break
    return max(scores) if scores else None


def _to_frame_row(rec: dict) -> FrameRow:
    detections = rec.get("detections") or []
    return FrameRow(
        frame_id=rec.get("frame_id"),
        ts=rec.get("ts"),
        score=_best_score(rec),
        latency=rec.get("latency"),
        state=rec.get("state"),
        n_detections=len(detections),
    )


def load_records(jsonl_path) -> tuple[list[FrameRow], list[float]]:
    """JSONL 파일을 (frame_rows, rejection_ts) 로 분리해 반환.

    blackbox.py가 실제로 쓰는 스키마를 그대로 읽는다 — 새 스키마를 만들지 않는다.
    손상된 줄(예: 로테이션 도중 잘린 마지막 줄)은 건너뛰고 stderr에 경고만 남긴다.
    """
    frame_rows: list[FrameRow] = []
    rejection_ts: list[float] = []
    path = Path(jsonl_path)
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                print(f"경고: {path}:{lineno} JSON 파싱 실패 — 건너뜀", file=sys.stderr)
                continue
            rtype = rec.get("type")
            if rtype == "frame":
                frame_rows.append(_to_frame_row(rec))
            elif rtype == "rejection":
                rejection_ts.append(rec.get("ts"))
    return frame_rows, rejection_ts


def build_figure(
    frame_rows: list[FrameRow],
    rejection_ts: Optional[list[float]] = None,
    *,
    x_field: str = "ts",
    title: Optional[str] = None,
):
    """frame_rows(+선택적 rejection_ts)로 3단 시간축 플롯(score/latency/state) Figure를 만든다."""
    if not frame_rows:
        raise ValueError("frame_rows가 비어 있음 — 플롯할 데이터가 없다")
    rejection_ts = rejection_ts or []

    t0 = frame_rows[0].ts

    def _x(row: FrameRow):
        if x_field == "frame_id":
            return row.frame_id
        if t0 is None:
            # 전체 행에 ts가 하나도 없으면 일관되게 frame_id 축으로 폴백(혼용 아님).
            return row.frame_id
        if row.ts is None:
            # 일부 행만 ts 결측 — frame_id로 새면 나머지 ts-t0(경과초) 스케일과 섞여
            # 시간축이 튀어 보인다. score/latency와 같은 nan-gap 철학으로 라인만 끊는다.
            return float("nan")
        return row.ts - t0

    xs = [_x(r) for r in frame_rows]

    fig, (ax_score, ax_latency, ax_state) = plt.subplots(3, 1, sharex=True, figsize=(10, 7))

    # 결측(None)은 nan으로 채워 넣는다 — 필터링해 이어붙이면 "검출 0인 프레임"이
    # 옆 프레임과 매끄럽게 연결된 것처럼 보여 오해를 부른다. nan은 matplotlib이
    # 자동으로 선을 끊어준다. 그 결과 각 라인의 포인트 수(len(xdata))는 항상
    # len(frame_rows) — 즉 JSONL의 type=frame 행 수와 정확히 같다.
    scores = [r.score if r.score is not None else float("nan") for r in frame_rows]
    if any(r.score is not None for r in frame_rows):
        ax_score.plot(xs, scores, marker="o", ms=3, lw=1, color="tab:blue", label="score (chosen/best confidence)")
        ax_score.legend(loc="upper right", fontsize=8)
    for rt in rejection_ts:
        if rt is None:
            continue
        rx = (rt - t0) if (x_field == "ts" and t0 is not None) else rt
        ax_score.axvline(rx, color="tab:red", ls="--", lw=0.6, alpha=0.5)
    ax_score.set_ylabel("score")
    ax_score.set_ylim(-0.05, 1.05)
    ax_score.set_title(title or "vision blackbox JSONL")

    # --- latency ---
    latencies = [r.latency * 1000.0 if r.latency is not None else float("nan") for r in frame_rows]
    if any(r.latency is not None for r in frame_rows):
        ax_latency.plot(xs, latencies, marker="o", ms=3, lw=1, color="tab:orange")
    ax_latency.set_ylabel("latency (ms)")

    # --- state ---
    unique_states = sorted({r.state for r in frame_rows if r.state is not None})
    if unique_states:
        state_to_y = {s: i for i, s in enumerate(unique_states)}
        state_ys = [state_to_y[r.state] if r.state is not None else float("nan") for r in frame_rows]
        ax_state.step(xs, state_ys, where="post", marker="o", ms=3, color="tab:green")
        ax_state.set_yticks(range(len(unique_states)))
        ax_state.set_yticklabels(unique_states)
    else:
        ax_state.text(
            0.5, 0.5,
            "no state data (state machine not wired yet, vision_plan.md #5.1)",
            ha="center", va="center", transform=ax_state.transAxes, fontsize=9, color="gray",
        )
        ax_state.set_yticks([])
    ax_state.set_ylabel("state")
    ax_state.set_xlabel("elapsed time (s)" if x_field == "ts" else "frame_id")

    fig.tight_layout()
    return fig


def save_figure(fig, output_path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="vision blackbox JSONL 뷰어/플롯 최소본 (vision_plan.md §7.9 항목6)"
    )
    parser.add_argument("jsonl", help="BlackBoxLogger가 남긴 .jsonl 파일 경로")
    parser.add_argument(
        "--output", default=None,
        help="출력 PNG 경로 (기본: <jsonl 파일명>.png, 같은 디렉터리)",
    )
    parser.add_argument("--x-axis", choices=["ts", "frame_id"], default="ts", help="시간축 기준")
    parser.add_argument("--title", default=None, help="플롯 제목 (기본: jsonl 파일명)")
    args = parser.parse_args(argv)

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        print(f"Error: JSONL 파일을 찾을 수 없음 — {jsonl_path}", file=sys.stderr)
        return 1

    frame_rows, rejection_ts = load_records(jsonl_path)
    if not frame_rows:
        print(f"Error: '{jsonl_path}'에 type=frame 레코드가 없음 — 플롯할 데이터 없음", file=sys.stderr)
        return 1

    output_path = Path(args.output) if args.output else jsonl_path.with_suffix(".png")
    fig = build_figure(frame_rows, rejection_ts, x_field=args.x_axis, title=args.title or jsonl_path.name)
    save_figure(fig, output_path)

    n_scored = sum(1 for r in frame_rows if r.score is not None)
    print(f"프레임 {len(frame_rows)}개, 거절 {len(rejection_ts)}건 → {output_path}")
    print(f"  score 있는 프레임: {n_scored}/{len(frame_rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
