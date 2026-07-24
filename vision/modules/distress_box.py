import cv2
import numpy as np
from vision.core.state import VisionState


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
    이 방법을 고른 이유: 박스가 매트 정중앙에 있는 실측 스펙(현재 확정 배치)에서는 네 모서리가
    이론상 등거리인 축퇴 상황이 되는데, 그래도 고정된 모서리 순회 순서 덕에 항상 같은 모서리가
    선택되어 결정론(§7.5)이 깨지지 않는다. 박스가 매트 중심에서 벗어나 배치되는 경우(실측이
    아직 확정 전인 다른 시나리오)에도 별도 분기 없이 동일 공식이 자연스럽게 "박스 반대편"을
    가리킨다. **"박스 옆"의 정확한 방향·거리는 대회측 미회신 상태**(`vision_plan.md` §9 "중요"
    각주)이므로 이 구현은 잠정 합리적 기본값이다 — 규정이 확정되면 재검토 대상.

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
    ):
        self.val_min = val_min
        self.sat_max = sat_max
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.min_solidity = min_solidity
        self.max_aspect_ratio = max_aspect_ratio
        self.roi_margin = roi_margin
        self.interior_margin_ratio = interior_margin_ratio

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
            far_corner = max(
                corners, key=lambda c: (c[0] - box_cx) ** 2 + (c[1] - box_cy) ** 2
            )
            landing_x = far_corner[0] + self.interior_margin_ratio * (mat_cx - far_corner[0])
            landing_y = far_corner[1] + self.interior_margin_ratio * (mat_cy - far_corner[1])

            det.meta["white_box_detector"] = {
                "box_bbox": [int(box_x0), int(box_y0), int(bw), int(bh)],
                "box_center_px": [float(box_cx), float(box_cy)],
                "area_ratio": round(float(area / mat_area), 5),
                "solidity": round(float(solidity), 4),
                "landing_point_px": [float(landing_x), float(landing_y)],
            }
            confirmed.append(det)

        state.detections = confirmed
        state.meta["white_box_detector"] = {
            "confirmed": len(confirmed),
            "rejected": rejected,
            "reject_reasons": reject_reasons,
        }
        return state
