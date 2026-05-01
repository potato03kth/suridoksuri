from vision.modules import (
    ColorFilter,
    IlluminationModule,
    DenoiseModule,
    EdgeDetector,
    MorphologyModule,
    RectDetector,
    BackgroundSubtractor,
    KalmanTracker,
    TemporalFusion,
)

# 이름 → 클래스 매핑. 새 모듈 추가 시 이 dict에만 등록하면 된다.
MODULES: dict[str, type] = {
    "color_filter":          ColorFilter,
    "illumination":          IlluminationModule,
    "denoise":               DenoiseModule,
    "edge_detector":         EdgeDetector,
    "morphology":            MorphologyModule,
    "rect_detector":         RectDetector,
    "background_subtractor": BackgroundSubtractor,
    "kalman_tracker":        KalmanTracker,
    "temporal_fusion":       TemporalFusion,
}
