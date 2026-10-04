import json
from pathlib import Path
import wave
import numpy as np
import pytest

from music_event_curve.audio_io import load_audio
from video_event_curve.event_curve import curves_from_features
from extract_music_event_curve import compare_curves


def test_decode_audio_resample_and_mono(tmp_path):
    path = tmp_path / "stereo.wav"
    sample_rate = 48000
    samples = np.arange(sample_rate) / sample_rate
    tone = (np.sin(2*np.pi*440*samples)*10000).astype('<i2')
    stereo = np.column_stack([tone, -tone])
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2); wav.setsampwidth(2); wav.setframerate(sample_rate)
        wav.writeframes(stereo.tobytes())
    waveform, metadata = load_audio(path)
    assert waveform.shape == (24000,)
    assert metadata["duration"] == pytest.approx(1)
    assert metadata["original_channels"] == 2
    np.testing.assert_allclose(waveform, 0, atol=1e-7)


def test_comparison_preserves_offset_and_shared_grid(tmp_path):
    video = tmp_path / "video"; video.mkdir()
    (video / "metadata.json").write_text(json.dumps({"video_path":"clip.mp4",
        "first_frame_pts_seconds":10.0,"duration":4.0}))
    vt = np.arange(0.02, 4, 0.04)
    np.save(video / "clip_native_timestamps.npy", vt)
    np.save(video / "clip_normalized_event_curve.npy", np.sin(vt))
    features = np.random.default_rng(1).normal(size=(75, 8))
    music = curves_from_features(features,np.arange(75)/25)
    output = tmp_path / "out"; output.mkdir()
    meta = compare_curves(music,video,output,audio_start_pts=10.5)
    data = np.load(output / "video_music_comparison.npz")
    assert meta["audio_to_video_time_offset"] == 0.5
    assert data['timestamps'][0] >= music['native_timestamps'][0]+0.5
    assert data['timestamps'][-1] <= music['native_timestamps'][-1]+0.5
    assert data['timestamps'].shape == data['video'].shape == data['music'].shape
    assert (output / "video_music_comparison.png").stat().st_size>1000
