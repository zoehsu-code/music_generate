"""Offline integration with real video decoding and controlled model outputs.

These tests do NOT claim validation of pretrained DINO semantics.
"""
from fractions import Fraction
import json
from types import SimpleNamespace

import av
import numpy as np
from PIL import Image
import pytest
import torch

from video_event_curve.video_io import load_video
from video_event_curve import dinov2_encoder as encoder
from video_event_curve.pipeline import extract_video_event_curve


def make_video(path, count=65, variable_rate=False):
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=30)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        stream.time_base = Fraction(1, 30)
        pts = 0
        for i in range(count):
            pixels = np.zeros((48, 64, 3), dtype=np.uint8)
            pixels[:] = [20, 40, 80] if i < count // 2 else [180, 30, 50]
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts, frame.time_base = pts, Fraction(1, 30)
            pts += 2 if variable_rate and i % 2 else 1
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


class Inputs(dict):
    def to(self, device):
        return Inputs({key: value.to(device) for key, value in self.items()})


class Processor:
    def __call__(self, images, return_tensors):
        pixels = np.stack([np.asarray(image).mean(axis=(0, 1)) for image in images])
        return Inputs(pixel_values=torch.tensor(pixels, dtype=torch.float32))

    def to_dict(self):
        return {"test_double": True}


class Model:
    def __call__(self, pixel_values):
        assert not torch.is_grad_enabled()
        return SimpleNamespace(last_hidden_state=torch.stack(
            [pixel_values * 0 + 1, pixel_values, pixel_values], dim=1))


@pytest.mark.parametrize("vfr", [False, True])
def test_decode_preserves_frames_and_pts(tmp_path, vfr):
    path = tmp_path / "input.mp4"
    make_video(path, 65, vfr)
    reader = load_video(path)
    frames = list(reader.extract_frames())
    assert len(frames) == reader.metadata["num_frames"] == 65
    assert all(image.mode == "RGB" for image, _ in frames)
    assert reader.timestamps[0] == 0
    if vfr:
        assert len(np.unique(np.round(np.diff(reader.timestamps), 5))) == 2
    else:
        np.testing.assert_allclose(reader.timestamps, np.arange(65)/30)
        assert reader.metadata["fps"] == pytest.approx(30)
        assert reader.metadata["duration"] == pytest.approx(65/30)


def test_pooling_and_partial_batch():
    tokens = torch.tensor([[[99., 99.], [1., 2.], [3., 4.]]])
    np.testing.assert_allclose(encoder.pool_tokens(tokens, "mean_patch"), [[2, 3]])
    np.testing.assert_allclose(encoder.pool_tokens(tokens, "cls"), [[99, 99]])
    frames = [(Image.new("RGB", (8, 8), (50, 20, 30)), i/30) for i in range(5)]
    features, times = encoder.extract_dinov2_features(iter(frames), Model(), Processor(), "cpu", 2)
    assert features.shape == (5, 3) and times.shape == (5,)


def test_real_transformers_interface_without_downloading_weights():
    # Verify the actual processor/model API with a tiny randomly initialized
    # DINOv2. This checks compatibility, not pretrained semantic behavior.
    from transformers import BitImageProcessor, Dinov2Config, Dinov2Model
    processor = BitImageProcessor(size={"shortest_edge": 32},
                                 crop_size={"height": 28, "width": 28})
    model = Dinov2Model(Dinov2Config(hidden_size=32, num_hidden_layers=1,
                                   num_attention_heads=4, image_size=28, patch_size=14))
    model.eval().requires_grad_(False)
    frames = [(Image.new("RGB", (40, 30), (80, 100, 120)), i/30) for i in range(3)]
    features, times = encoder.extract_dinov2_features(iter(frames), model, processor, "cpu", 2)
    assert features.shape == (3, 32)
    np.testing.assert_allclose(features[0], features[1], atol=1e-6)
    assert np.isfinite(features).all()


def test_output_bundle_and_cli(tmp_path, monkeypatch):
    path = tmp_path / "input.mp4"
    make_video(path)
    monkeypatch.setattr(encoder, "load_dinov2", lambda *a: (Model(), Processor(), torch.device("cpu")))
    result = extract_video_event_curve(path, output_dir=tmp_path / "outputs", batch_size=8,
                                      target_length=394, save_features=True, compare_offsets=True)
    output = tmp_path / "outputs" / "input"
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["feature_shape"] == [65, 3]
    assert np.load(output / "input_event_curve.npy").shape == (394,)
    assert np.load(output / "input_native_timestamps.npy").shape == (64,)
    assert (output / "input_event_curve.png").stat().st_size > 1000
    assert (output / "input_offset_comparison.png").stat().st_size > 1000
    for offset in (1, 15, 60):
        assert np.load(output / f"offset_{offset}/input_raw_event_curve.npy").shape == (65-offset,)
    with pytest.raises(FileExistsError):
        extract_video_event_curve(path, output_dir=tmp_path / "outputs")
    from video_event_curve.extract_video_event_curve import main
    monkeypatch.setattr("sys.argv", ["extract", "--video", str(path), "--output_dir",
                                    str(tmp_path / "cli"), "--target_length", "None"])
    main()
    assert (tmp_path / "cli/input/metadata.json").is_file()
