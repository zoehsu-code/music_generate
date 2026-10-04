"""Frozen Meta DINOv2-L/14 via its standard Transformers implementation."""
from __future__ import annotations

import warnings
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel

from .event_curve import positive_integer

MODEL_ID = "facebook/dinov2-large"
# Pin both processor and weights to the same published revision.
MODEL_REVISION = "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c"


def resolve_device(device="auto"):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected = torch.device(device)
    if selected.type not in ("cpu", "cuda", "mps"):
        raise ValueError("device must be auto, cpu, cuda[:index], or mps")
    if selected.type == "cuda" and not torch.cuda.is_available():
        warnings.warn("CUDA unavailable; falling back to CPU", stacklevel=2)
        return torch.device("cpu")
    if selected.type == "mps" and not torch.backends.mps.is_available():
        warnings.warn("MPS unavailable; falling back to CPU", stacklevel=2)
        return torch.device("cpu")
    return selected


def load_dinov2(device="auto", cache_dir=None, local_files_only=False):
    selected = resolve_device(device)
    kwargs = dict(revision=MODEL_REVISION, cache_dir=cache_dir,
                  local_files_only=local_files_only)
    # Use the checkpoint's own processor, without custom normalization/cropping.
    processor = AutoImageProcessor.from_pretrained(MODEL_ID, use_fast=False, **kwargs)
    model = AutoModel.from_pretrained(MODEL_ID, use_safetensors=True, **kwargs)
    model.eval().requires_grad_(False).to(selected)
    return model, processor, selected


def pool_tokens(tokens, feature_pooling):
    # Implementation choice: final layer-normalized spatial patch tokens;
    # token zero is CLS. This checkpoint has no register tokens.
    if feature_pooling == "mean_patch":
        return tokens[:, 1:, :].mean(dim=1)
    if feature_pooling == "cls":
        return tokens[:, 0, :]
    raise ValueError("feature_pooling must be mean_patch or cls")


def extract_dinov2_features(frames, model, processor, device, batch_size=32,
                            feature_pooling="mean_patch"):
    positive_integer(batch_size, "batch_size")
    if feature_pooling not in ("mean_patch", "cls"):
        raise ValueError("feature_pooling must be mean_patch or cls")
    batches, times, images = [], [], []

    def infer(batch):
        inputs = processor(images=batch, return_tensors="pt").to(device)
        with torch.inference_mode():
            tokens = model(**inputs).last_hidden_state
            return pool_tokens(tokens, feature_pooling).float().cpu().numpy()

    try:
        for image, timestamp in frames:
            images.append(image)
            times.append(timestamp)
            if len(images) == batch_size:
                batches.append(infer(images))
                images.clear()
        if images:
            batches.append(infer(images))
    except torch.OutOfMemoryError as exc:
        raise RuntimeError("DINOv2-L ran out of memory; retry with --batch_size 1 or 4") from exc
    if not batches:
        raise ValueError("No frames were decoded")
    features = np.concatenate(batches, axis=0)
    if features.shape[0] != len(times) or not np.isfinite(features).all():
        raise ValueError("Invalid extracted features")
    return features, np.asarray(times, dtype=np.float64)
