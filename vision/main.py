"""
Landing zone detector — CLI 진입점.

사용 예:
  # 정지 이미지
  python -m vision.main image.jpg
  python -m vision.main image.jpg --preset presets/low_light.yaml --output results/out.jpg

  # 영상
  python -m vision.main flight.mp4 --preset presets/video.yaml --output results/out.mp4
"""
import argparse
import sys
from pathlib import Path

import cv2

from vision.core.runner import Pipeline
from vision.utils.image_loader import load_image
from vision.utils.visualize import save_result, draw_detections


_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}


def _run_image(pipeline: Pipeline, input_path: Path, output: str | None) -> None:
    image = load_image(str(input_path))
    state = pipeline.run(image)

    print(f"Detections: {len(state.detections)}")
    for i, d in enumerate(state.detections):
        print(f"  [{i}] bbox={d.bbox}  confidence={d.confidence:.3f}")
    if state.confirmed:
        print(f"Confirmed: bbox={state.confirmed.bbox}")

    if output:
        save_result(state, output)
        print(f"Saved: {output}")


def _run_video(pipeline: Pipeline, input_path: Path, output: str | None) -> None:
    from vision.utils.video_reader import VideoReader

    writer = None
    with VideoReader(str(input_path)) as reader:
        for frame in reader:
            state = pipeline.run(frame)
            annotated = draw_detections(state.original, state.detections, state.confirmed)

            if output:
                if writer is None:
                    h, w = annotated.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(output, fourcc, reader.fps or 30, (w, h))
                writer.write(annotated)

            cv2.imshow("Landing Zone Detector", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if writer:
        writer.release()
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Landing zone object detector")
    parser.add_argument("input", help="Input image or video file path")
    parser.add_argument(
        "--preset",
        default=str(Path(__file__).parent / "presets" / "single_frame.yaml"),
        help="Pipeline preset yaml (default: presets/single_frame.yaml)",
    )
    parser.add_argument("--output", default=None, help="Output file path (optional)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found — {input_path}", file=sys.stderr)
        sys.exit(1)

    pipeline = Pipeline.from_config(args.preset)

    if input_path.suffix.lower() in _VIDEO_SUFFIXES:
        _run_video(pipeline, input_path, args.output)
    else:
        _run_image(pipeline, input_path, args.output)


if __name__ == "__main__":
    main()
