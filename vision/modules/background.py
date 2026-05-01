import cv2
import numpy as np
from vision.core.state import VisionState


class BackgroundSubtractor:
    """
    MOG2 배경 차감으로 정적 배경 제거. 영상 파이프라인 전용.
    첫 history 프레임 동안 배경 모델을 학습하므로 초반 결과는 노이즈가 많다.
    """

    def __init__(self, history: int = 500, var_threshold: float = 16.0):
        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=False,
        )

    def __call__(self, state: VisionState) -> VisionState:
        fg_mask = self._subtractor.apply(state.current)

        if state.mask is not None:
            fg_mask = cv2.bitwise_and(fg_mask, state.mask)

        state.mask = fg_mask
        return state
