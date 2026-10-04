"""Pure NumPy implementation of Section 3.2; no event detectors."""
from __future__ import annotations

import numpy as np


def _curve(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Expected a nonempty, finite 1D curve")
    return values


def positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def compute_cosine_dissimilarity(features, frame_offset=1):
    positive_integer(frame_offset, "frame_offset")
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] <= frame_offset or x.shape[1] == 0:
        raise ValueError("features must be [T,D], with T > frame_offset and D > 0")
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite features")
    # Implementation choice (not specified by paper): float64 arithmetic,
    # rescale before L2 normalization to avoid overflow; reject undefined cosine.
    scale = np.max(np.abs(x), axis=1, keepdims=True)
    if np.any(scale == 0):
        raise ValueError("Zero feature vector has undefined cosine similarity")
    x = x / scale
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    similarities = np.einsum("ij,ij->i", x[:-frame_offset], x[frame_offset:])
    return 1.0 - np.clip(similarities, -1.0, 1.0)


def standardize_curve(raw_curve, eps=1e-8):
    x = _curve(raw_curve)
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    # Implementation choice: population std (ddof=0), epsilon floor.
    return (x - x.mean()) / max(float(x.std(ddof=0)), eps)


def compute_timestamps(frame_timestamps, frame_offset=1):
    times = _curve(frame_timestamps)
    positive_integer(frame_offset, "frame_offset")
    if len(times) <= frame_offset or np.any(np.diff(times) <= 0):
        raise ValueError("Need T > offset and strictly increasing frame timestamps")
    # Implementation choice: pair midpoints, in seconds from first decoded frame.
    return (times[:-frame_offset] + times[frame_offset:]) / 2.0


def resample_curve(curve, timestamps, target_length=None):
    x, times = _curve(curve), _curve(timestamps)
    if x.shape != times.shape or np.any(np.diff(times) <= 0):
        raise ValueError("Curve and increasing timestamps must have identical shapes")
    if target_length is None:
        return x.copy(), times.copy()
    positive_integer(target_length, "target_length")
    # Implementation choice: linear interpolation in actual video time, endpoints
    # included. L=1 uses the midpoint; one input sample is repeated if L>1.
    new_times = (np.array([(times[0] + times[-1]) / 2]) if target_length == 1
                 else np.linspace(times[0], times[-1], target_length))
    return np.interp(new_times, times, x), new_times


def hann_smooth(curve, hann_window=31):
    x = _curve(curve)
    positive_integer(hann_window, "hann_window")
    if hann_window % 2 == 0:
        raise ValueError("hann_window must be odd to avoid a half-sample shift")
    if hann_window == 1:
        return x.copy()
    # Implementation choice: symmetric (not periodic) Hann and edge replication.
    # Explicit padding + valid convolution retains length even for tiny videos.
    kernel = np.hanning(hann_window)
    kernel /= kernel.sum()
    radius = hann_window // 2
    return np.convolve(np.pad(x, (radius, radius), mode="edge"), kernel, mode="valid")


def curves_from_features(features, frame_timestamps, frame_offset=1,
                         hann_window=31, target_length=None):
    if len(features) != len(frame_timestamps):
        raise ValueError("Feature count does not match timestamp count")
    raw = compute_cosine_dissimilarity(features, frame_offset)
    native_times = compute_timestamps(frame_timestamps, frame_offset)
    normalized = standardize_curve(raw)
    resampled, final_times = resample_curve(normalized, native_times, target_length)
    result = {"raw_event_curve": raw, "normalized_event_curve": normalized,
              "native_timestamps": native_times, "timestamps": final_times,
              "event_curve": hann_smooth(resampled, hann_window)}
    if target_length is not None:
        result["resampled_event_curve"] = resampled
    return result
