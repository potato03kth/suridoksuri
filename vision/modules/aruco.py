import cv2
import numpy as np
from vision.core.state import VisionState, Detection


class ArucoDetector:
    """
    ArUco 브랜치 Phase 2(docs/vision_aruco_branch.md §Phase 2): 정밀착륙 타겟 마커 디코드.
    cv2.aruco.ArucoDetector(DICT_4X4_50)로 코너+ID를 검출하고, ID 화이트리스트(기본 [23],
    확정 전제 §1)만 통과시킨다. 다른 ID는 후보에서 제거되고 거절 이유로 로깅된다
    (vertiport_v.BlackVMatcher의 confirmed/rejected 카운트 패턴 재사용).

    solvePnP(물리 크기 50cm×50cm 사용)는 Phase 3 범위 — 이 모듈은 디코드만 한다.
    """

    def __init__(self, valid_ids: tuple[int, ...] = (23,)):
        self.valid_ids = set(valid_ids)
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        params = cv2.aruco.DetectorParameters()
        self._detector = cv2.aruco.ArucoDetector(dictionary, params)

    def __call__(self, state: VisionState) -> VisionState:
        # ColorFilter 등 앞 단계가 current를 마스크 밖 픽셀이 지워진 상태로 남길 수 있어
        # (VisionState 필드 사용 규칙) 마커 패턴이 훼손되지 않은 original을 읽는다.
        gray = cv2.cvtColor(state.original, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)

        confirmed: list[Detection] = []
        rejected = 0

        if ids is not None:
            for marker_corners, marker_id in zip(corners, ids.flatten()):
                marker_id = int(marker_id)
                if marker_id not in self.valid_ids:
                    rejected += 1
                    continue

                pts = marker_corners.reshape(-1, 2).astype(np.float32)
                x0, y0 = pts.min(axis=0)
                x1, y1 = pts.max(axis=0)
                bbox = (int(round(x0)), int(round(y0)), int(round(x1 - x0)), int(round(y1 - y0)))

                confirmed.append(Detection(
                    bbox=bbox,
                    corners=pts,
                    meta={"aruco_id": marker_id},
                ))

        state.detections = confirmed
        state.meta["aruco"] = {"confirmed": len(confirmed), "rejected": rejected}
        return state
