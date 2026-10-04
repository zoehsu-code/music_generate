"""Extract MusicFM curve and optionally compare with an existing video result."""
import argparse
import json
from pathlib import Path
import time
import shutil
import tempfile
import numpy as np

from music_event_curve.audio_io import load_audio
from music_event_curve.encoder import extract_musicfm_features
from video_event_curve.event_curve import curves_from_features, hann_smooth
from video_event_curve.visualization import plot_event_curve


def compare_curves(music, video_dir, output, audio_start_pts):
    import matplotlib.pyplot as plt
    video_dir = Path(video_dir)
    meta = json.loads((video_dir / "metadata.json").read_text())
    stem = Path(meta["video_path"]).stem
    vt = np.load(video_dir / f"{stem}_native_timestamps.npy")
    vn = np.load(video_dir / f"{stem}_normalized_event_curve.npy")
    mt = music["native_timestamps"] + audio_start_pts - meta["first_frame_pts_seconds"]
    # Compare on the existing video-time grid within shared observed support.
    # Interpolate standardized curves BEFORE smoothing both with the same kernel.
    grid = vt[(vt >= mt[0]) & (vt <= mt[-1])]
    if len(grid) < 2:
        raise ValueError("Audio and video curves have no usable shared time interval")
    video = hann_smooth(np.interp(grid, vt, vn), 31)
    audio = hann_smooth(np.interp(grid, mt, music["normalized_event_curve"]), 31)
    np.savez(output / "video_music_comparison.npz", timestamps=grid, video=video, music=audio)
    np.savetxt(output / "video_music_comparison.csv", np.column_stack([grid, video, audio]),
               delimiter=",", header="time_seconds,video_event_curve,music_event_curve", comments="")
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    axes[0].plot(grid, video, color="#2563eb", label="VIDEO — DINOv2-L")
    axes[1].plot(grid, audio, color="#e76f00", label="MUSIC / AUDIO — MusicFM")
    axes[2].plot(grid, video, color="#2563eb", label="Video")
    axes[2].plot(grid, audio, color="#e76f00", label="Music / audio")
    for ax in axes:
        ax.legend(loc="upper right")
        ax.grid(alpha=0.2)
        ax.set_ylabel("Normalized change\n(Hann smoothed)")
    # Use identical scales to make the separate panels comparable.
    lo, hi = min(video.min(), audio.min()), max(video.max(), audio.max())
    margin = max((hi-lo)*0.1, 0.1)
    for ax in axes:
        ax.set_ylim(lo-margin, hi+margin)
    axes[0].set_title(f"{stem}: video and music event curves")
    axes[2].set_xlabel("Video time (seconds)")
    axes[2].set_xlim(0, meta["duration"])
    fig.savefig(output / "video_music_comparison.png", dpi=160)
    plt.close(fig)
    return {"video_result_dir": str(video_dir.resolve()), "grid": "video pair timestamps within shared support",
            "hann_window": 31, "length": len(grid), "audio_to_video_time_offset": audio_start_pts-meta["first_frame_pts_seconds"],
            "order": "per-modality standardization, shared-grid interpolation, Hann smoothing",
            "note": "Comparison recomputes both curves on shared support; native outputs are preserved"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True, help="Audio file or video containing the paired audio")
    parser.add_argument("--video_results", help="Optional existing video extraction directory")
    parser.add_argument("--output_dir", default="outputs-music")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--layer_ix", type=int, default=7)
    parser.add_argument("--local_files_only", action="store_true")
    args = parser.parse_args()
    stem = Path(args.audio).stem
    root = Path(args.output_dir).resolve(); root.mkdir(parents=True, exist_ok=True)
    output = root / stem
    if output.exists():
        parser.error(f"Output exists: {output}; choose a new --output_dir")
    start = time.perf_counter()
    waveform, audio_meta = load_audio(args.audio)
    features, times, encoder_meta = extract_musicfm_features(waveform, args.device, args.layer_ix,
                                                           local_files_only=args.local_files_only)
    curves = curves_from_features(features, times, frame_offset=1, hann_window=31)
    temporary = Path(tempfile.mkdtemp(prefix=f".{stem}-", dir=root))
    try:
        for key, array in {**curves, "musicfm_features": features, "feature_timestamps": times}.items():
            np.save(temporary / f"{stem}_{key}.npy", array, allow_pickle=False)
        plot_event_curve(curves, temporary / f"{stem}_music_event_curve.png",
                         f"MUSIC / AUDIO — MusicFM — {stem}", audio_meta["duration"])
        np.savetxt(temporary / f"{stem}_music_curves.csv", np.column_stack([
            curves["timestamps"], curves["raw_event_curve"], curves["normalized_event_curve"], curves["event_curve"]]),
            delimiter=",", header="time_seconds,raw_dissimilarity,standardized_dissimilarity,smoothed_event_curve", comments="")
        norm = curves["normalized_event_curve"]
        metadata = {**audio_meta, **encoder_meta, "frame_offset": 1, "hann_window": 31,
                    "target_length": None, "raw_curve_length": len(norm),
                    "normalized_mean": float(norm.mean()), "normalized_std": float(norm.std()),
                    "audio_content": "entire embedded track; no source separation",
                    "timestamp_origin": "first decoded audio sample; comparison uses container PTS",
                    "standardization": "per track, population std, eps=1e-8",
                    "smoothing": "symmetric Hann, centered, edge padding"}
        if args.video_results:
            metadata["comparison"] = compare_curves(curves, args.video_results, temporary,
                                                      audio_meta["audio_start_pts_seconds"])
        metadata["elapsed_seconds"] = time.perf_counter()-start
        (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(metadata, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
