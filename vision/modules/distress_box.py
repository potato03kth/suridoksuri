import math

import cv2
import numpy as np
from vision.core.state import VisionState

# 매트 bbox 모서리의 정규 순회 순서 — 이 순서가 곧 등거리 축퇴 시의 tie-break 우선순위다.
# (기존 구현의 `max()`가 첫 최댓값을 돌려주던 동작과 동일한 순서를 유지한다.)
CORNER_NAMES = ("tl", "tr", "br", "bl")


class WhiteBoxDetector:
    """
    조난자 구역(② `docs/vision_plan.md` §5.3) fine 단계: 초록 매트 내부 정중앙 흰 박스(고명도
    blob, 실측 20cm — §2 "정중앙 20cm 흰 사무박스")를 확인한다.

    **핵심 설계 포인트: 착륙 목표는 박스가 아니라 "박스 옆 빈 초록면"이다**(§5.3 "fine(≤~15m):
    중앙 흰 박스 → 박스 옆 빈 초록면 착륙"). 박스는 랜드마크일 뿐이므로, 이 모듈은 박스를
    확인하는 데서 그치지 않고 착륙점(landing point)까지 계산해 `detection.meta`에 싣는다.

    coarse 단계(`distress_coarse.yaml` — `ColorFilter(mode=color)` + `RectDetector`)가 낸 초록
    매트 `detections`를 ROI로 읽어 그 안에서 흰 박스를 찾는 캐스케이드 패턴이다
    (`vertiport_v.py`/`vertiport_ring.py`와 동일 구조 — 이전 단계 detections를 ROI로 읽고
    자기 결과로 덮어쓴다). `color_filter`가 mask 밖 픽셀을 지워버린 `current` 대신 원본 색상이
    보존된 `original`을 읽는다(`vision/CLAUDE.md` "주의" 절 — ColorFilter 함정).

    **착륙점 산출 방법("가장 먼 모서리를 안쪽으로 당기기"):**
    1. 매트 bbox의 네 모서리 중 흰 박스 중심에서 유클리드 거리가 가장 먼 모서리를 고른다
       (박스에서 최대한 멀어지는 방향 — "박스 옆"의 최소 요건).
    2. 그 모서리를 매트 중심 쪽으로 `interior_margin_ratio`만큼 당겨, 매트의 **물리적** 가장자리
       (실측 0.105m 라이즈드 플랫폼 — 페인트 선이 아니라 실제 구조물이므로 가장자리 이탈이
       진짜 낙하 위험이다)에서 안전마진을 확보한다.
    이 방법을 고른 이유: 박스가 매트 중심에서 벗어나 배치되는 경우(실측이 아직 확정 전인 다른
    시나리오)에도 별도 분기 없이 동일 공식이 자연스럽게 "박스 반대편"을 가리킨다.
    **"박스 옆"의 정확한 방향·거리는 대회측 미회신 상태**(`vision_plan.md` §9 "중요" 각주)이므로
    이 구현은 잠정 합리적 기본값이다 — 규정이 확정되면 재검토 대상.

    ## ⚠️ 등거리 축퇴와 그 완화 (2026-07-28)

    **실측 스펙상 흰 박스는 매트 정중앙**이라, 매트 bbox 네 모서리가 박스 중심에서 **이론상
    등거리**가 된다. 정확히 등거리이면 `max()`가 첫 최댓값을 돌려줘 결정론(§7.5) 자체는 안
    깨지지만, **실전 입력은 절대 정확히 등거리가 아니다** — mp4 압축/컨투어 양자화로 박스 중심이
    1px만 흔들려도 "가장 먼 모서리"가 TL↔BR로 뒤집힌다. 실측 재현(2026-07-28): 1px 흔들림만으로
    착륙점이 **2.97m 점프**하고 한 시퀀스 안에서 네 모서리를 전부 방문했다(정지 이미지=TL,
    같은 장면 mp4=BR). 이 착륙점은 `modules/distress_mat.py`를 거쳐 **실제 유도 좌표로 나가므로**
    프레임마다 매트 반대편을 지시하면 기체가 진동한다.

    완화는 **선택의 안정성만** 손댄다(`interior_margin_ratio`나 "박스 옆" 규약은 무변경):

    1. **동률 허용오차(`tie_tolerance_ratio`)** — 최대거리에서 `tie_tolerance_ratio × 매트 bbox
       대각선` 이내인 모서리를 전부 "동률 후보"로 보고, 그 안에서 `CORNER_NAMES` 정규 순서의
       첫 번째를 고른다. 픽셀 잡음이 허용오차보다 작으면 선택이 아예 흔들리지 않는다.
       `0.0`을 주면 **기존 동작과 정확히 동일**(정확 동률만 첫 모서리로 해소)하다.
    2. **프레임 간 히스테리시스(`corner_hysteresis`)** — 직전 프레임에 고른 모서리가 아직 동률
       후보 안에 있으면 그대로 유지한다. 이 둘의 조합은 **폭 `2 × tol`의 슈미트 트리거**가 된다
       (직전 모서리를 버리려면 tol 이상 뒤처져야 하고, 되돌아오려면 반대로 tol 이상 앞서야 한다).
       덕분에 허용오차 경계에 딱 걸친 배치에서도 **한 번 전환한 뒤 눌러앉지, 왕복 진동하지
       않는다.** 직전 프레임과의 대응은 매트 bbox IoU로 찾는다(`modules/fusion.py::TemporalFusion`
       과 같은 패턴 — 새 추적기를 발명하지 않는다).

    **결정론은 유지된다**(§7.5): 히스테리시스는 wall-clock/난수가 아니라 *입력 시퀀스*만의
    함수라, 같은 프레임열은 같은 착륙점열을 낸다. 상태는 인스턴스 단위이므로 새 인스턴스는 항상
    같은 초기 선택(정규 순서 첫 동률 후보)에서 출발한다 — `modules/tracker.py`/`fusion.py`와
    동일한 stateful 모듈 관례.

    매직넘버 금지 — 임계값은 전부 `__init__` 파라미터(→ preset yaml에서 조정 가능).
    """

    def __init__(
        self,
        val_min: int = 180,
        sat_max: int = 60,
        min_area_ratio: float = 0.0015,
        max_area_ratio: float = 0.02,
        min_solidity: float = 0.85,
        max_aspect_ratio: float = 1.4,
        roi_margin: float = 0.05,
        interior_margin_ratio: float = 0.3,
        tie_tolerance_ratio: float = 0.02,
        corner_hysteresis: bool = True,
        hysteresis_iou_min: float = 0.3,
    ):
        self.val_min = val_min
        self.sat_max = sat_max
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.min_solidity = min_solidity
        self.max_aspect_ratio = max_aspect_ratio
        self.roi_margin = roi_margin
        self.interior_margin_ratio = interior_margin_ratio
        # 기본값 0.02 = 매트 bbox 대각선의 2%. 300px 매트(=3.0m)에서 tol≈8.5px, 슈미트 폭
        # 2·tol≈17px(≈17cm)로 실측 잡음 1px 대비 약 17배 여유. 반대로 이 대역을 벗어나려면
        # 박스가 대각선 방향으로 약 3px(≈3cm)만 치우치면 되므로, 규정이 확정돼 박스가 실제로
        # 편심 배치되는 시나리오의 기존 동작(=박스 반대편 지시)은 그대로 살아 있다.
        self.tie_tolerance_ratio = float(tie_tolerance_ratio)
        self.corner_hysteresis = bool(corner_hysteresis)
        self.hysteresis_iou_min = float(hysteresis_iou_min)
        # 프레임 간 히스테리시스 상태: [(매트 bbox, 선택한 모서리 인덱스), ...]
        self._prev_corner_choices: list[tuple[tuple[int, int, int, int], int]] = []

    def reset(self) -> None:
        """프레임 간 히스테리시스 상태를 지운다(새 시퀀스 시작 시)."""
        self._prev_corner_choices = []

    @staticmethod
    def _iou(a: tuple, b: tuple) -> float:
        """`modules/fusion.py::TemporalFusion._iou`와 동일한 bbox IoU (얇게 재사용)."""
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        x0, y0 = max(ax, bx), max(ay, by)
        x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        inter = max(0, x1 - x0) * max(0, y1 - y0)
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0

    def _previous_corner_index(self, mat_bbox: tuple) -> int | None:
        """직전 프레임에서 같은 매트로 볼 수 있는 항목의 모서리 인덱스(없으면 None)."""
        if not self.corner_hysteresis:
            return None
        best_iou, best_idx = 0.0, None
        for prev_bbox, corner_idx in self._prev_corner_choices:
            iou = self._iou(mat_bbox, prev_bbox)
            if iou >= self.hysteresis_iou_min and iou > best_iou:
                best_iou, best_idx = iou, corner_idx
        return best_idx

    def _select_far_corner(
        self, corners: list, box_cx: float, box_cy: float, diag: float, prev_idx: int | None
    ) -> tuple[int, int, bool]:
        """박스에서 가장 먼 매트 모서리를 고른다 → (모서리 인덱스, 동률 후보 수, 히스테리시스 사용 여부).

        축퇴 완화의 전부가 여기 있다 — 클래스 docstring "등거리 축퇴와 그 완화" 참조.
        """
        dists = [math.hypot(cx - box_cx, cy - box_cy) for cx, cy in corners]
        d_max = max(dists)
        tol = self.tie_tolerance_ratio * diag
        tied = [i for i, d in enumerate(dists) if d >= d_max - tol]
        if prev_idx is not None and prev_idx in tied:
            return prev_idx, len(tied), True
        # 정규 순서(CORNER_NAMES)의 첫 동률 후보 — tol=0이면 기존 `max()` 동작과 동일하다.
        return tied[0], len(tied), False

    def _white_mask(self, roi_bgr: np.ndarray) -> np.ndarray:
        """고명도·저채도(흰색) 게이팅. 초록 매트는 채도가 높아(합성 기준 sat=200) sat_max로
        자연히 배제된다 — 별도 색상 배제 로직 불필요."""
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 0, self.val_min], dtype=np.uint8)
        upper = np.array([180, self.sat_max, 255], dtype=np.uint8)
        return cv2.inRange(hsv, lower, upper)

    def __call__(self, state: VisionState) -> VisionState:
        confirmed = []
        rejected = 0
        reject_reasons: list[str] = []
        corner_choices: list[tuple[tuple[int, int, int, int], int]] = []

        for det in state.detections:
            x, y, w, h = det.bbox
            mat_area = float(w * h)
            if mat_area <= 0:
                rejected += 1
                reject_reasons.append("empty_mat_bbox")
                continue

            mx, my = int(w * self.roi_margin), int(h * self.roi_margin)
            x0, y0 = max(0, x - mx), max(0, y - my)
            x1 = min(state.original.shape[1], x + w + mx)
            y1 = min(state.original.shape[0], y + h + my)
            roi = state.original[y0:y1, x0:x1]
            if roi.size == 0:
                rejected += 1
                reject_reasons.append("empty_roi")
                continue

            mask = self._white_mask(roi)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                rejected += 1
                reject_reasons.append("no_white_pixels")
                continue

            best = None  # (area, bx, by, bw, bh, solidity)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                ratio = area / mat_area
                if not (self.min_area_ratio <= ratio <= self.max_area_ratio):
                    continue
                hull_area = cv2.contourArea(cv2.convexHull(cnt))
                solidity = area / hull_area if hull_area > 0 else 0.0
                if solidity < self.min_solidity:
                    continue
                bx, by, bw, bh = cv2.boundingRect(cnt)
                aspect = max(bw, bh) / max(1, min(bw, bh))
                if aspect > self.max_aspect_ratio:
                    continue
                if best is None or area > best[0]:
                    best = (area, bx, by, bw, bh, solidity)

            if best is None:
                rejected += 1
                reject_reasons.append("no_qualifying_contour")
                continue

            area, bx, by, bw, bh, solidity = best
            # ROI 좌표 -> 원본 이미지 좌표
            box_x0, box_y0 = x0 + bx, y0 + by
            box_cx = box_x0 + bw / 2.0
            box_cy = box_y0 + bh / 2.0

            mat_cx, mat_cy = x + w / 2.0, y + h / 2.0
            corners = [
                (float(x), float(y)),
                (float(x + w), float(y)),
                (float(x + w), float(y + h)),
                (float(x), float(y + h)),
            ]
            corner_idx, tie_count, from_hysteresis = self._select_far_corner(
                corners,
                box_cx,
                box_cy,
                math.hypot(float(w), float(h)),
                self._previous_corner_index(det.bbox),
            )
            far_corner = corners[corner_idx]
            corner_choices.append((det.bbox, corner_idx))

            landing_x = far_corner[0] + self.interior_margin_ratio * (mat_cx - far_corner[0])
            landing_y = far_corner[1] + self.interior_margin_ratio * (mat_cy - far_corner[1])

            det.meta["white_box_detector"] = {
                "box_bbox": [int(box_x0), int(box_y0), int(bw), int(bh)],
                "box_center_px": [float(box_cx), float(box_cy)],
                "area_ratio": round(float(area / mat_area), 5),
                "solidity": round(float(solidity), 4),
                "landing_point_px": [float(landing_x), float(landing_y)],
                # 축퇴 진단(§7.4 블랙박스 포렌식) — tie_count>1이면 그 프레임이 실제로 등거리
                # 축퇴 상황이었다는 뜻이고, corner_from_hysteresis가 True면 정규 순서 대신
                # 직전 선택이 유지된 것이다.
                "landing_corner": CORNER_NAMES[corner_idx],
                "corner_tie_count": int(tie_count),
                "corner_from_hysteresis": bool(from_hysteresis),
            }
            confirmed.append(det)

        self._prev_corner_choices = corner_choices
        state.detections = confirmed
        state.meta["white_box_detector"] = {
            "confirmed": len(confirmed),
            "rejected": rejected,
            "reject_reasons": reject_reasons,
        }
        return state
