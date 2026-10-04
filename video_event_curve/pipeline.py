"""High-level video-only extraction and reproducible numerical output."""
from __future__ import annotations

import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil
import tempfile

import numpy as np

from .event_curve import curves_from_features, positive_integer
from .video_io import load_video


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_outputs(result, output_dir, save_features=False, compare_offsets=False):
    """Write into a temporary sibling directory; refuse to overwrite prior runs."""
    from .visualization import plot_event_curve, plot_offset_comparison

    parent = Path(output_dir).expanduser().resolve()
    parent.mkdir(parents=True, exist_ok=True)
    stem = Path(result["metadata"]["video_path"]).stem
    destination = parent / stem
    if destination.exists():
        raise FileExistsError(f"Output exists: {destination}. Choose a different --output_dir")
    temporary = Path(tempfile.mkdtemp(prefix=f".{stem}-", dir=parent))
    try:
        arrays = {**result["curves"], "frame_timestamps": result["frame_timestamps"]}
        if save_features:
            arrays["dinov2_features"] = result["features"]
        for name, array in arrays.items():
            np.save(temporary / f"{stem}_{name}.npy", array, allow_pickle=False)
        metadata = result["metadata"].copy()
        metadata["outputs"] = {name: {"file": f"{stem}_{name}.npy", "shape": list(array.shape),
                                      "dtype": str(array.dtype)} for name, array in arrays.items()}
        plot_event_curve(result["curves"], temporary / f"{stem}_event_curve.png",
                         stem, metadata["duration"])
        if compare_offsets:
            comparisons = {}
            for offset in (1, 15, 60):
                curves = curves_from_features(result["features"], result["frame_timestamps"],
                                               offset, metadata["hann_window"], metadata["target_length"])
                comparisons[offset] = curves
                path = temporary / f"offset_{offset}"
                path.mkdir()
                for name, array in curves.items():
                    np.save(path / f"{stem}_{name}.npy", array, allow_pickle=False)
            plot_offset_comparison(comparisons, temporary / f"{stem}_offset_comparison.png",
                                   f"{stem}: separate offset curves", metadata["duration"])
            metadata["offset_comparison"] = {
                "offsets": [1, 15, 60], "standardization": "independent for each offset",
                "median_pair_separation_seconds": {
                    str(k): float(np.median(result["frame_timestamps"][k:] -
                                             result["frame_timestamps"][:-k])) for k in comparisons}}
        with open(temporary / "metadata.json", "w") as handle:
            json.dump(metadata, handle, indent=2, allow_nan=False)
            handle.write("\n")
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    result["metadata"] = metadata
    result["output_path"] = str(destination)
    return destination


def extract_video_event_curve(video_path, frame_offset=1, hann_window=31,
                              target_length=None, batch_size=32, device="auto",
                              feature_pooling="mean_patch", output_dir=None,
                              save_features=False, compare_offsets=False,
                              cache_dir=None, local_files_only=False):
    """Return features [T,1024], timestamps [T], curves, metadata, and output path.

    Every decoded frame is encoded once. Only a batch of RGB frames is kept in
    memory; features remain in CPU RAM. No training or event detection occurs.
    """
    positive_integer(frame_offset, "frame_offset")
    positive_integer(batch_size, "batch_size")
    positive_integer(hann_window, "hann_window")
    if hann_window % 2 == 0:
        raise ValueError("hann_window must be odd")
    if target_length is not None:
        positive_integer(target_length, "target_length")
    if feature_pooling not in ("mean_patch", "cls"):
        raise ValueError("feature_pooling must be mean_patch or cls")
    video = load_video(video_path)
    if output_dir is not None and (Path(output_dir).expanduser() / video.path.stem).exists():
        raise FileExistsError("Output already exists; choose a different --output_dir")
    from .dinov2_encoder import (MODEL_ID, MODEL_REVISION, load_dinov2,
                                 extract_dinov2_features)
    model, processor, selected = load_dinov2(device, cache_dir, local_files_only)
    features, times = extract_dinov2_features(video.extract_frames(), model, processor,
                                              selected, batch_size, feature_pooling)
    if compare_offsets and len(times) <= 60:
        raise ValueError("Offset comparison requires at least 61 decoded frames")
    curves = curves_from_features(features, times, frame_offset, hann_window, target_length)
    raw, normalized = curves["raw_event_curve"], curves["normalized_event_curve"]
    metadata = {
        **video.metadata, "input_sha256": _sha256(video.path),
        "paper": "https://arxiv.org/html/2603.11042v2#S3.SS2",
        "encoder": MODEL_ID, "encoder_revision": MODEL_REVISION,
        "encoder_implementation": "transformers.AutoModel", "feature_pooling": feature_pooling,
        "feature_representation": "last_hidden_state: mean of patch tokens excluding CLS"
            if feature_pooling == "mean_patch" else "last_hidden_state: CLS token at index 0",
        "feature_dimension": features.shape[1], "feature_shape": list(features.shape),
        "frame_offset": frame_offset, "raw_curve_length": len(raw), "target_length": target_length,
        "final_curve_length": len(curves["event_curve"]), "hann_window": hann_window,
        "hann_definition": "symmetric, sum-normalized, centered, edge-replicated padding",
        "hann_window_units": "samples on final curve grid, not original frames after resampling",
        "timestamp_convention": "midpoint of compared frames; first decoded frame is t=0",
        "resampling": "linear in PTS midpoint time, endpoint-aligned; L=1 uses midpoint",
        "standardization": {"scope": "per_video", "ddof": 0, "eps": 1e-8,
                            "raw_mean": float(raw.mean()), "raw_std": float(raw.std()),
                            "normalized_mean": float(normalized.mean()),
                            "normalized_std": float(normalized.std()),
                            "epsilon_limited": bool(raw.std() < 1e-8)},
        "processing_device": str(selected), "requested_device": device,
        "batch_size": batch_size, "inference_dtype": "float32", "processor": processor.to_dict(),
        "versions": {name: version(name) for name in
                     ("numpy", "av", "torch", "transformers", "Pillow", "matplotlib")},
        "python_version": platform.python_version(),
        "implementation_choices_not_specified_by_paper": [
            "exact token pooling and checkpoint processor/revision", "population std and epsilon floor",
            "time-domain interpolation and endpoints", "symmetric Hann and edge padding",
            "PTS midpoint timestamps and duration convention"],
    }
    result = {"features": features, "frame_timestamps": times, "curves": curves, "metadata": metadata}
    if output_dir is not None:
        save_outputs(result, output_dir, save_features, compare_offsets)
    return result
