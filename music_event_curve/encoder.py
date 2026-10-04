"""Official MusicFM model, with CPU spectrograms for MPS compatibility."""
from pathlib import Path
import sys
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from video_event_curve.dinov2_encoder import resolve_device

SOURCE_REVISION = "b83ebedb401bcef639b26b05c0c8bee1dc2dfe71"
WEIGHT_REVISION = "4513b38bc25ad1d227b1980819b9691ba97f4d87"
CONFIG_REVISION = "6b36ef01c6443c67ae7ed0822876d091ab50e4aa"


def extract_musicfm_features(waveform, device="mps", layer_ix=7,
                             cache_dir=".cache/huggingface", local_files_only=False):
    if not 0 <= layer_ix <= 12:
        raise ValueError("layer_ix must be in [0,12]")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party"))
    from musicfm.model.musicfm_25hz import MusicFM25Hz
    options = dict(cache_dir=cache_dir, local_files_only=local_files_only)
    stats = hf_hub_download("minzwon/MusicFM", "msd_stats.json", revision=WEIGHT_REVISION, **options)
    weights = hf_hub_download("minzwon/MusicFM", "pretrained_msd.pt", revision=WEIGHT_REVISION, **options)
    config = hf_hub_download("facebook/wav2vec2-conformer-rope-large-960h-ft", "config.json",
                            revision=CONFIG_REVISION, **options)
    selected = resolve_device(device)
    print(f"Loading MusicFM-MSD on {selected}", flush=True)
    model = MusicFM25Hz(is_flash=False, stat_path=stats, model_path=None, config_path=config)
    # Official prefix conversion; CPU map supports checkpoints saved on CUDA.
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    state = checkpoint["state_dict"]
    if not all(key.startswith("model.") for key in state):
        raise ValueError("Unexpected MusicFM checkpoint key prefix")
    model.load_state_dict({key[6:]: value for key, value in state.items()}, strict=True)
    del state, checkpoint
    model.eval().requires_grad_(False)
    # STFT uses complex tensors; run the unchanged official preprocessing on CPU.
    with torch.inference_mode():
        mel = model.normalize(model.preprocessing(torch.from_numpy(waveform)[None], ["melspec_2048"]))
        model.to(selected)
        print(f"Encoding {len(waveform)/24000:.2f} seconds of audio", flush=True)
        _, hidden = model.encoder(mel["melspec_2048"].to(selected))
        features = hidden[layer_ix][0].float().cpu().numpy()
    # Centered STFT at 100 Hz; two stride-2, padding-1 convolution stages
    # have centers at 0, 4, 8,... spectrogram frames => 25 Hz without a shift.
    times = np.arange(len(features), dtype=np.float64) / 25
    if features.ndim != 2 or features.shape[1] != 1024 or not np.isfinite(features).all():
        raise ValueError("Unexpected MusicFM feature output")
    return features, times, {"encoder": "MusicFM-MSD", "source_revision": SOURCE_REVISION,
        "weight_revision": WEIGHT_REVISION, "config_revision": CONFIG_REVISION,
        "layer_ix": layer_ix, "feature_rate_hz": 25, "feature_shape": list(features.shape),
        "device": str(selected), "dtype": "float32", "preprocessing_device": "cpu",
        "timestamp_convention": "centered STFT and padded convolution centers; pair midpoints",
        "implementation_choice": "MSD checkpoint and layer 7 follow MusicFM quick-start; V2M-ZERO does not specify these"}
