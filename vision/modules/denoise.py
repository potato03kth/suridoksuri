import cv2
from vision.core.state import VisionState


class DenoiseModule:
    """
    노이즈 제거 후 그레이스케일 변환.
    mode='gaussian': 속도 우선.
    mode='bilateral': 엣지 보존 우선 (느림).
    결과는 state.current에 그레이스케일 BGR로 저장됨.
    """

    def __init__(self, mode: str = "gaussian", kernel_size: int = 5, sigma: float = 1.0):
        self.mode = mode
        self.kernel_size = kernel_size | 1  # 홀수 강제
        self.sigma = sigma

    def __call__(self, state: VisionState) -> VisionState:
        gray = cv2.cvtColor(state.current, cv2.COLOR_BGR2GRAY)

        if self.mode == "bilateral":
            smoothed = cv2.bilateralFilter(gray, self.kernel_size, self.sigma * 10, self.sigma * 10)
        else:
            smoothed = cv2.GaussianBlur(gray, (self.kernel_size, self.kernel_size), self.sigma)

        state.current = cv2.cvtColor(smoothed, cv2.COLOR_GRAY2BGR)
        return state
