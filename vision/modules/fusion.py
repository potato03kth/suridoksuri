import numpy as np
from vision.core.state import VisionState, Detection


class TemporalFusion:
    """
    프레임 간 detection을 누적해 신뢰도를 높인다.
    같은 위치에 min_frames 회 이상 등장한 detection을 state.confirmed에 확정.

    `count`의 정의: **그 후보가 연속으로 관측된 프레임 수.**
    한 프레임에 관측되면 +1(그 프레임에 겹치는 검출이 몇 개든 +1), 관측되지 않은
    프레임마다 -1로 감쇠하고 0이 되면 후보를 버린다. 그래서 `min_frames`는 이름
    그대로 "프레임 수" 단위이고, 깜빡이는(한 프레임 걸러 나타나는) 검출은 +1/-1이
    상쇄돼 영원히 승격되지 않는다 — 이 억제가 이 모듈이 존재하는 이유다.
    """

    def __init__(self, min_frames: int = 5, iou_threshold: float = 0.4):
        self.min_frames = min_frames
        self.iou_threshold = iou_threshold
        # {detection, count, last_seen} — last_seen은 이 후보를 마지막으로 관측한 프레임 번호
        self._candidates: list[dict] = []
        self._frame_index = 0

    def __call__(self, state: VisionState) -> VisionState:
        self._frame_index += 1

        if not state.detections:
            # 관측된 후보가 하나도 없는 프레임 → 전부 감쇠(소실 시 확정이 풀리는 경로)
            self._decay()
            return state

        for det in state.detections:
            matched = self._match(det)
            if matched is not None:
                # 한 프레임에 같은 후보와 겹치는 검출이 여러 개 와도 count는 한 번만 오른다.
                # count가 "프레임 수"라는 정의를 지키는 지점 — 아래 설계 노트 (b) 참조.
                if matched["last_seen"] != self._frame_index:
                    matched["count"] += 1
                    matched["last_seen"] = self._frame_index
                matched["detection"] = det
            else:
                self._candidates.append(
                    {"detection": det, "count": 1, "last_seen": self._frame_index}
                )

        self._decay()

        confirmed = [c for c in self._candidates if c["count"] >= self.min_frames]
        if confirmed:
            best = max(confirmed, key=lambda c: c["count"])
            state.confirmed = best["detection"]
            state.meta["fusion"] = {"confirmed_count": best["count"]}

        return state

    def _match(self, det: Detection) -> dict | None:
        for c in self._candidates:
            if self._iou(det.bbox, c["detection"].bbox) >= self.iou_threshold:
                return c
        return None

    def _decay(self):
        """이번 프레임에 **관측되지 않은** 후보만 감쇠시킨다.

        ── 설계 노트 (2026-07-28, 구조적 결함 수정) ─────────────────────────
        예전 구현은 `__call__`이 방금 `count += 1` 한 후보까지 무조건 -1 했다.
        그래서 프레임당 검출이 1개면 순증이 0 → count가 0이 돼 후보가 목록에서
        통째로 제거됐고, **같은 자리에 몇 프레임을 연속으로 나타나든 영원히
        승격되지 않았다**(클래스 docstring의 계약과 정면 충돌). `main.py`의
        "confirmed 없으면 첫 detection 사용" 폴백 때문에 크래시는 안 나고,
        흔들림 억제만 조용히 통째로 무효인 채 돌고 있었다.

        (a) 채택: 이번 프레임에 관측된 후보를 감쇠 대상에서 제외한다.
            관측 프레임 +1 / 미관측 프레임 -1 → count가 곧 "연속 관측 프레임 수"가
            되어 min_frames 계약이 그대로 성립하고, 소실 시 감쇠로 확정이 풀리는
            동작(`test_confirmation_lapses_...`)도 그대로 남는다.

        (b) 함께 채택: 프레임당 최대 +1 (`last_seen` 게이트, `__call__` 참조).
            (a)만 하고 한 프레임에 겹치는 검출 N개를 그대로 +N 하면 count가
            "프레임 수"가 아니라 "검출 수"가 된다. 실측으로 확인한 결과 그 변형은
            **깜빡임 억제를 깨뜨렸다** — 프레임당 +2, 빈 프레임 -1이면 순증이 +1이라
            한 프레임 걸러 나타나는 검출도 결국 승격된다(모듈의 존재 이유가 무너짐).

        (c) 기각: "감쇠를 확정검사 뒤로 이동". 실제로 구현해 돌려본 결과 결함이
            전혀 고쳐지지 않았다 — 프레임당 1검출은 여전히 count가 0으로 떨어져
            제거되므로 영원히 미확정이다. 덤으로 확정 시점만 한 프레임 앞당겨
            기존 산술을 흔든다. 증상 위치가 아니라 원인이 아니었다.
        """
        for c in self._candidates:
            if c["last_seen"] == self._frame_index:
                continue
            c["count"] = max(0, c["count"] - 1)
        self._candidates = [c for c in self._candidates if c["count"] > 0]

    @staticmethod
    def _iou(a: tuple, b: tuple) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
        iy = max(0, min(ay + ah, by + bh) - max(ay, by))
        inter = ix * iy
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0
