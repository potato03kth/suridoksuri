import cv2
from vision.core.state import VisionState


class IlluminationModule:
    """CLAHE로 조명 불균일 보정. 역광·저조도 환경에서 유효."""

    def __init__(self, clip_limit: float = 2.0, tile_size: int = 8):
        self._clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=(tile_size, tile_size),
        )

    def __call__(self, state: VisionState) -> VisionState:
        gray = cv2.cvtColor(state.current, cv2.COLOR_BGR2GRAY)
        enhanced = self._clahe.apply(gray)
        state.current = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        return state
