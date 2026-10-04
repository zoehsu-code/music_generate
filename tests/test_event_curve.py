import numpy as np
import pytest

from video_event_curve.event_curve import (
    compute_cosine_dissimilarity, compute_timestamps, standardize_curve,
    resample_curve, hann_smooth, curves_from_features,
)


def test_known_cosines_and_scale_invariance():
    features = np.array([[1, 0], [2, 0], [0, 3], [0, -5]], dtype=float)
    np.testing.assert_allclose(compute_cosine_dissimilarity(features), [0, 1, 2])
    np.testing.assert_allclose(compute_cosine_dissimilarity(features, 2), [1, 1])
    np.testing.assert_allclose(compute_cosine_dissimilarity(features * 1e200), [0, 1, 2])


def test_per_video_standardization_and_constant():
    for values in ([0, 1, 2, 5], [100, 110, 120]):
        result = standardize_curve(values)
        assert abs(result.mean()) < 1e-12
        assert abs(result.std() - 1) < 1e-12
    np.testing.assert_array_equal(standardize_curve([2, 2, 2]), [0, 0, 0])
    assert standardize_curve([1]).item() == 0


def test_midpoints_and_time_interpolation():
    times = compute_timestamps([0, 0.1, 0.4, 1.0])
    np.testing.assert_allclose(times, [0.05, 0.25, 0.7])
    curve, times = resample_curve([0, 10, 0], [0, 1, 4], 5)
    np.testing.assert_allclose(curve, [0, 10, 20/3, 10/3, 0])
    np.testing.assert_allclose(times, [0, 1, 2, 3, 4])
    curve, times = resample_curve([0, 2], [0, 2], 1)
    np.testing.assert_allclose([curve.item(), times.item()], [1, 1])
    curve, times = resample_curve([3], [0.5], 5)
    np.testing.assert_allclose(curve, 3)
    np.testing.assert_allclose(times, 0.5)


def test_hann_alignment_and_short_boundaries():
    impulse = np.zeros(101)
    impulse[50] = 1
    smoothed = hann_smooth(impulse, 31)
    assert len(smoothed) == 101 and smoothed.argmax() == 50
    np.testing.assert_allclose(smoothed[35:66], np.hanning(31) / 15)
    np.testing.assert_allclose(hann_smooth([4], 31), [4])
    np.testing.assert_allclose(hann_smooth([4, 4], 31), [4, 4])
    np.testing.assert_array_equal(hann_smooth([1, 2], 1), [1, 2])


@pytest.mark.parametrize("offset", [1, 15, 60])
@pytest.mark.parametrize("target", [None, 394])
def test_pipeline_shapes(offset, target):
    features = np.random.default_rng(42).normal(size=(90, 12))
    curves = curves_from_features(features, np.arange(90)/30, offset, 31, target)
    assert curves["raw_event_curve"].shape == (90 - offset,)
    assert curves["event_curve"].shape == (target or 90 - offset,)
    assert curves["timestamps"].shape == curves["event_curve"].shape
    assert np.all(np.diff(curves["timestamps"]) > 0)
    assert all(np.isfinite(array).all() for array in curves.values())


@pytest.mark.parametrize("call", [
    lambda: compute_cosine_dissimilarity([[0, 0], [1, 1]]),
    lambda: compute_cosine_dissimilarity([[1, 1]], 1),
    lambda: compute_cosine_dissimilarity([[1, 1], [1, 1]], 0),
    lambda: compute_cosine_dissimilarity([[1, 1], [np.nan, 1]]),
    lambda: standardize_curve([]),
    lambda: hann_smooth([1, 2], 2),
    lambda: compute_timestamps([0, 0]),
    lambda: resample_curve([1, 2], [0, 1], 0),
])
def test_invalid_inputs(call):
    with pytest.raises(ValueError):
        call()
