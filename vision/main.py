"""
Landing zone detector — CLI 진입점.

--display 모드 (헤드리스 안전 — vision_plan.md §7.9 항목 1):
  none    창/스트림 없음. 파이프라인 + 로그(+ --output 파일)만. **드론 기본값**.
  window  cv2 GUI 창 표시. 디스플레이 있는 데스크톱 전용.
  file    창 없음, --output 필수 — 주석(annotated) 프레임을 파일로 기록.
  stream  라이브 저해상 스트림 (§7.9 항목 5 — 미구현).

⚠️ 드론은 디스플레이가 없다. GUI 호출(imshow/waitKey)은 반드시 --display window
   뒤로만 실행된다. 기본값 none 은 어떤 GUI 함수도 호출하지 않는다(헤드리스 크래시 방지).

사용 예:
  # 데스크톱에서 영상 보며 디버깅
  python -m vision.main flight.mp4 --preset presets/video.yaml --display window
  # 드론/헤드리스 배치 (기본값 none — 창 없음, 파일만)
  python -m vision.main flight.mp4 --preset presets/video.yaml --output results/out.mp4
  # 정지 이미지
  python -m vision.main image.jpg --preset presets/low_light.yaml --output results/out.jpg
"""
import argparse
import sys
from pathlib import Path

import cv2

from vision.core.runner import Pipeline
from vision.utils.image_loader import load_image
from vision.utils.visualize import save_result, draw_detections


_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}
_WINDOW_NAME = "Landing Zone Detector"


def _show_window(annotated, *, wait: int) -> bool:
    """cv2 GUI 창에 한 프레임 표시.

    헤드리스 OpenCV(디스플레이 없음)에서는 imshow가 예외를 던지므로 방어한다.
    반환: 계속 진행하면 True, 사용자가 'q'로 종료 요청하면 False.
    """
    try:
        cv2.imshow(_WINDOW_NAME, annotated)
        return (cv2.waitKey(wait) & 0xFF) != ord("q")
    except cv2.error as e:  # 헤드리스 빌드/디스플레이 부재
        print(
            f"Error: --display window 사용 불가 (디스플레이 없음/headless OpenCV): {e}\n"
            "       헤드리스 환경에서는 --display none (기본) 또는 --output 을 쓰세요.",
            file=sys.stderr,
        )
        sys.exit(2)


def _run_image(pipeline: Pipeline, input_path: Path, output: str | None, display: str) -> None:
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

    if display == "window":
        annotated = draw_detections(state.original, state.detections, state.confirmed)
        _show_window(annotated, wait=0)  # 키 입력까지 대기
        cv2.destroyAllWindows()


def _run_video(pipeline: Pipeline, input_path: Path, output: str | None, display: str) -> None:
    from vision.utils.video_reader import VideoReader

    writer = None
    frame_count = 0
    with VideoReader(str(input_path)) as reader:
        for frame in reader:
            state = pipeline.run(frame)
            annotated = draw_detections(state.original, state.detections, state.confirmed)
            frame_count += 1

            if output:
                if writer is None:
                    h, w = annotated.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(output, fourcc, reader.fps or 30, (w, h))
                writer.write(annotated)

            if display == "window" and not _show_window(annotated, wait=1):
                break

    if writer:
        writer.release()
        print(f"Saved: {output}  ({frame_count} frames)")
    if display == "window":
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
    parser.add_argument(
        "--display",
        choices=["none", "window", "file", "stream"],
        default="none",
        help="라이브 뷰 모드. none=헤드리스 안전(기본) · window=데스크톱 GUI · "
             "file=--output 필수 · stream=미구현(§7.9)",
    )
    args = parser.parse_args()

    if args.display == "stream":
        print(
            "Error: --display stream 은 아직 미구현입니다 "
            "(vision_plan.md §7.9 항목 5: compute_tap → MJPEG/ROS image).\n"
            "       현재는 none / window / file 을 사용하세요.",
            file=sys.stderr,
        )
        sys.exit(2)

    if args.display == "file" and not args.output:
        print("Error: --display file 은 --output 경로가 필요합니다.", file=sys.stderr)
        sys.exit(2)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found — {input_path}", file=sys.stderr)
        sys.exit(1)

    pipeline = Pipeline.from_config(args.preset)

    if input_path.suffix.lower() in _VIDEO_SUFFIXES:
        _run_video(pipeline, input_path, args.output, args.display)
    else:
        _run_image(pipeline, input_path, args.output, args.display)


if __name__ == "__main__":
    main()
