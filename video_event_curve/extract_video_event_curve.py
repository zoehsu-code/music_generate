"""Run with python -m video_event_curve.extract_video_event_curve."""
from __future__ import annotations

import argparse
from .pipeline import extract_video_event_curve


def optional_length(value):
    return None if value.lower() == "none" else int(value)


def main():
    parser = argparse.ArgumentParser(description="DINOv2-L video-event curves (V2M-ZERO Section 3.2)")
    parser.add_argument("--video", dest="video_path", required=True)
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--frame_offset", type=int, default=1)
    parser.add_argument("--hann_window", type=int, default=31)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--target_length", type=optional_length, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--feature_pooling", choices=("mean_patch", "cls"), default="mean_patch")
    parser.add_argument("--save_features", action="store_true")
    parser.add_argument("--compare_offsets", action="store_true",
                        help="Also save separate offset 1/15/60 curves; requires T >= 61")
    parser.add_argument("--cache_dir", default=".cache/huggingface")
    parser.add_argument("--local_files_only", action="store_true")
    args = parser.parse_args()
    try:
        result = extract_video_event_curve(**vars(args))
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"Extraction failed: {exc}\n")
    print(f"Saved: {result['output_path']}")
    print(f"Features: {result['features'].shape}")
    print(f"Raw: {result['curves']['raw_event_curve'].shape}; "
          f"final: {result['curves']['event_curve'].shape}")
    print(f"Device: {result['metadata']['processing_device']}")


if __name__ == "__main__":
    main()
