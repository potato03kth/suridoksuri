import cv2
import numpy as np
from vision.core.state import VisionState


class KalmanTracker:
    """
    단일 타깃 Kalman 추적. 가장 신뢰도 높은 detection을 측정치로 사용.
    detections가 없는 프레임에서는 예측값을 meta에 기록하고 통과시킨다.
    """

    def __init__(self, process_noise: float = 0.01, measurement_noise: float = 0.1):
        # 상태: [cx, cy, vx, vy], 측정: [cx, cy]
        kf = cv2.KalmanFilter(4, 2)
        kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                          [0, 1, 0, 0]], dtype=np.float32)
        kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                         [0, 1, 0, 1],
                                         [0, 0, 1, 0],
                                         [0, 0, 0, 1]], dtype=np.float32)
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        self._kf = kf
        self._initialized = False

    def __call__(self, state: VisionState) -> VisionState:
        if not state.detections:
            if self._initialized:
                pred = self._kf.predict()
                state.meta["kalman"] = {"predicted": (int(pred[0]), int(pred[1])), "source": "predict"}
            return state

        best = max(state.detections, key=lambda d: d.confidence)
        cx, cy = best.center
        measurement = np.array([[np.float32(cx)], [np.float32(cy)]])

        if not self._initialized:
            self._kf.statePre = np.array([[cx], [cy], [0], [0]], dtype=np.float32)
            self._initialized = True

        self._kf.correct(measurement)
        pred = self._kf.predict()
        state.meta["kalman"] = {"predicted": (int(pred[0]), int(pred[1])), "source": "correct"}
        return state
