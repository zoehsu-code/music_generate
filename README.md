# Video event curves — V2M-ZERO baseline

## Reproduce from GitHub (video + music)

```bash
git clone https://github.com/zoehsu-code/music_generate.git
cd music_generate
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-music.txt
python scripts/download_weights.py
# Put your own input video in video-data/example.mp4
python extract_video_event_curve.py --video video-data/example.mp4 --device mps --batch_size 4 --save_features --local_files_only
python extract_music_event_curve.py --audio video-data/example.mp4 --video_results outputs/example --device mps --local_files_only
```

Use `--device cuda` on NVIDIA GPUs or `--device cpu` without an accelerator.
The repository includes source, dependencies, tests, vendored MusicFM inference
code/license, and `weights-manifest.json` with immutable revisions, file sizes
and SHA-256 hashes of all six required model/config assets. The download script
retrieves and verifies the exact weights used locally (about 2.3 GB total).
Weight binaries remain on the original model hosts rather than in Git history;
GitHub's normal Git file limit is 100 MiB. Input videos, generated results,
virtual environments and model caches are local and excluded from Git.
To use an offline machine, copy `.cache/huggingface/` after downloading, then run
`python scripts/download_weights.py --local_files_only` to verify the cache.

Video-only implementation of [V2M-ZERO, arXiv:2603.11042v2, Section 3.2](https://arxiv.org/html/2603.11042v2#S3.SS2).
Every decoded frame → frozen DINOv2-L → cosine dissimilarity at offset 1 →
per-video standardization → optional resampling → Hann smoothing.
No training, scene-cut detection, optical flow, or peak detection.
The original video baseline remains independent. A subsequent MusicFM extension
adds music/audio curves and same-time-grid comparisons (see below).

## Install

Use Python 3.10–3.12 (tested with 3.12). From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

For CUDA, install the matching PyTorch wheel for your GPU/driver using the
[official PyTorch instructions](https://pytorch.org/get-started/locally/).
PyAV wheels include FFmpeg libraries; a separate ffmpeg executable is unnecessary.
The first extraction downloads Meta's pretrained DINOv2-L weights (about 1.2 GB).
The checkpoint and processor revision are pinned in `dinov2_encoder.py`.
Use `--cache_dir .cache/huggingface`; once cached, `--local_files_only` runs offline.

## One-video run

Put a video in `video-data/` (created for input data; videos are Git-ignored).

```bash
python extract_video_event_curve.py \
  --video video-data/example.mp4 \
  --output_dir outputs \
  --frame_offset 1 --hann_window 31 --batch_size 32 \
  --feature_pooling mean_patch --save_features
```

`--device auto` selects CUDA when available, otherwise CPU. An unavailable
explicit CUDA request warns and falls back to CPU. `--device mps` is optional
on supported Macs. Reduce `--batch_size` to 4 or 1 if memory is insufficient.
CPU DINOv2-L inference can be slow. Every frame is preserved regardless of device.

Native resolution is the default (`--target_length None`). For explicit resampling:

```bash
python extract_video_event_curve.py --video video-data/example.mp4 \
  --output_dir outputs-resampled --target_length 394 --batch_size 4 --save_features
```

Existing video output directories are never overwritten; choose a new output root
for another setting. `python -m video_event_curve.extract_video_event_curve` is
an equivalent entry point. `--help` lists all arguments.

## Representation and reproduction choices

The paper specifies spatially pooled frame features and DINOv2-L but does not
identify the exact token, processor resolution, interpolation convention, std
degrees of freedom, or padding. This is a documented baseline, not a claim of
bitwise reproduction of the authors' implementation.

- **Encoder:** Meta [`facebook/dinov2-large`](https://huggingface.co/facebook/dinov2-large),
  DINOv2-L/14 without register tokens, standard PyTorch/Transformers implementation.
  Frozen weights, `eval()`, `inference_mode()`, float32 inference.
- **Default feature:** mean of the final layer-normalized patch tokens,
  `last_hidden_state[:, 1:, :].mean(dim=1)`, producing 1024 dimensions per frame.
  This is our literal interpretation of spatial pooling. Optional `cls` uses
  `last_hidden_state[:, 0, :]` and is explicitly recorded in metadata.
- **Preprocessing:** the checkpoint's own `AutoImageProcessor(use_fast=False)`;
  RGB, shortest-edge resize 256, bicubic interpolation, 224×224 center crop,
  rescaling and normalization from the published processor configuration.
  No custom normalization. The complete processor config is saved in metadata.
  See [Meta's processor config](https://huggingface.co/facebook/dinov2-large/blob/main/preprocessor_config.json)
  and [Transformers DINOv2 documentation](https://huggingface.co/docs/transformers/model_doc/dinov2).
- **Numerics:** float64 curve arithmetic; L2-normalized vectors, cosine clipped
  to [-1, 1] for roundoff; zero feature vectors and nonfinite inputs raise errors.
  Standardization uses population std (`ddof=0`) with `max(std, 1e-8)`.
  Constant signals become zero; near-constant signals may have std below 1.
- **Time:** PyAV decodes every frame of the first video stream, retaining actual
  PTS and subtracting the first frame's time. Pair timestamps are midpoints.
  Missing/non-increasing PTS raise errors rather than silently inventing FPS timing.
  Reported FPS is the stream average; decoded count is authoritative. Duration
  includes the last frame's duration, estimated from the last interval if missing;
  both this convention and container-reported duration/count are recorded.
- **Resampling:** optional linear interpolation in actual midpoint time, with
  endpoints included. This also respects variable-frame-rate timing. Length 1
  samples the midpoint. A single valid pair is repeated for larger lengths;
  its repeated timestamps indicate no temporal extent, not distinct observations.
- **Smoothing:** symmetric, sum-normalized Hann, centered convolution, replicated
  endpoints; odd positive window lengths only (1 disables smoothing). This keeps
  length even when the curve is shorter than the kernel and adds no fixed delay.
  Smoothing can still merge/move asymmetric peaks; it does not preserve every maximum.
  Native VFR smoothing is in sample space, not a fixed width in seconds.

**31 is a number of samples on the final grid.** At native 30 FPS this spans
about 1 second; at 394 points over ~32 seconds it spans about 2.4 seconds between
kernel endpoints. Keeping 31 with native resolution follows the requested
analysis mode but differs in physical smoothing scale from the paper's latent
grid. We do not force arbitrary videos to 394 samples. The final curve is not
standardized again after interpolation or smoothing.

## Output shapes and files

Let T be decoded frames, D=1024, k=frame offset, N=T−k, and L=target length
if provided, otherwise N. Require T>k. All files live in `outputs/<video_name>/`.

| File suffix / stage | Shape | Time axis |
| --- | --- | --- |
| `_dinov2_features.npy` (with `--save_features`) | [T, D] | frame timestamps |
| `_frame_timestamps.npy` | [T] | elapsed seconds |
| `_raw_event_curve.npy` | [N] | native timestamps |
| `_normalized_event_curve.npy` | [N] | native timestamps |
| `_native_timestamps.npy` | [N] | pair midpoints |
| `_resampled_event_curve.npy` (when requested) | [L] | final timestamps |
| `_event_curve.npy` | [L] | final timestamps |
| `_timestamps.npy` | [L] | seconds for final curve |
| `_event_curve.png` | three-panel diagnostic | seconds |

`metadata.json` records input SHA-256, FPS, decoded/reported frame counts,
duration, encoder and revision, pooling, dimensions, offset, normalization
statistics, target length, Hann settings, device, processor, dependency versions,
and output shapes. Raw and normalized plots always use native timestamps; the
final plot uses its own timestamps. Axes cover the decoded video duration, with
no extrapolation beyond valid pair midpoints.

## Validation

```bash
python -m pytest -q
```

Offline tests cover analytical cosine values, per-video standardization, constant
signals, centered impulse smoothing, short videos, time-domain interpolation,
native/resampled shapes, actual CFR/VFR video decoding, batching, token pooling,
CLI arguments and saved output bundles. Integration tests use a controlled test
encoder and **do not validate pretrained DINOv2-L semantics**.

After uploading one real video, run the one-video command above, then inspect
the diagnostic figure. Confirm T features, N raw samples, finite arrays, normalized
mean ≈0/std ≈1 (except constant/epsilon-limited cases), final length L, increasing
timestamps, and the correct duration. Look for plausible responses to obvious
visual changes without assuming every motion produces a peak. Real-video DINOv2
validation has been completed on `746_3_29540_30708.mp4`: 281 frames,
281×1024 DINOv2 features, 280 curve samples, MPS inference. Numerical checks
passed; first run took 124 seconds including downloading the checkpoint.

## Offset experiment (after the first video works)

```bash
python extract_video_event_curve.py --video video-data/example.mp4 \
  --output_dir outputs-offsets --save_features --compare_offsets --batch_size 4
```

This reuses the same features for offsets 1, 15, 60, standardizes each separately,
and saves separate numerical arrays under `offset_1/`, `offset_15/`, `offset_60/`
plus `_offset_comparison.png`. At least 61 frames are required. Their actual
median time separations are recorded; ~33 ms/500 ms/2 s applies only near 30 FPS.
Curves are never combined.

## Python API and layout

```python
from video_event_curve import extract_video_event_curve

result = extract_video_event_curve(
    "video-data/example.mp4", frame_offset=1, hann_window=31,
    target_length=None, batch_size=4, device="auto",
    output_dir="outputs-api", save_features=True,
)
curve = result["curves"]["event_curve"]
times = result["curves"]["timestamps"]
```

`video_io.py` streams frames; `dinov2_encoder.py` batches inference;
`event_curve.py` implements the mathematics; `pipeline.py` assembles/saves
results; `visualization.py` creates plots. Only one RGB batch is retained in
memory, plus the full feature matrix. Implementation choices absent from the
paper are marked in comments and metadata. Optional video+curve overlay export
is not included in this first version; numerical extraction and diagnostic PNGs
are the deliverable.

## Music/audio curves (added after the video baseline)

```bash
python -m pip install -r requirements-music.txt
python extract_music_event_curve.py \
  --audio video-data/746_3_29540_30708.mp4 \
  --video_results outputs/746_3_29540_30708 \
  --device mps
```

The audio argument may be an audio file or a video containing audio. The first
audio stream is decoded, channels averaged, and resampled to 24 kHz. It processes
the entire track, including any dialogue/sound effects, without source separation.
AAC padding is trimmed to the declared stream duration, and discontinuous audio
PTS are rejected. The original container start time is used for video alignment;
for separately edited audio files, ensure their container timestamps represent
the intended alignment before comparing.

We use the [official MusicFM implementation](https://github.com/minzwon/musicfm),
vendored under `third_party/musicfm` with its MIT license. Source, MSD weights,
statistics, and Conformer config revisions are pinned. The MSD checkpoint and
`hidden_states[7]` follow the MusicFM quick-start; they are explicit reproduction
choices because V2M-ZERO does not specify the checkpoint variant or hidden layer.
`--layer_ix` is configurable. There is no temporal pooling: output is [T,1024]
at 25 Hz. Inference uses eval mode, float32 and no gradients; unchanged official
spectrogram preprocessing runs on CPU for MPS compatibility and the encoder runs
on the requested device. The wrapper processes the whole clip in one pass;
long recordings can require substantial memory (no chunk-boundary artifacts are
introduced). First run downloads the MusicFM weights; `--local_files_only` uses
the cache on subsequent runs.

Native feature timestamps are 0, .04, .08, ... seconds, from the centered STFT
and two padded stride-2 convolutions. These are nominal receptive-field centers;
the Conformer is contextual, so a token is not limited to a 40 ms audio segment.
Adjacent-pair midpoints give T−1 curve timestamps. The same cosine dissimilarity,
per-track standardization and centered 31-point Hann operations as the video
baseline are used. Native audio smoothing therefore has a 1.2-second endpoint
span. Saved `.npy` arrays include features, feature timestamps, raw, standardized,
smoothed curves and curve timestamps; CSV, metadata and a diagnostic PNG are
saved under `outputs-music/<name>/`.

With `--video_results`, both standardized signals are interpolated onto the
video pair timestamps within their shared observed support, then both are Hann
smoothed with 31 points. This gives the same physical smoothing scale for the
comparison while preserving the native results. Outputs:

- `video_music_comparison.png`: video (blue), music/audio (orange), and overlay;
  separate panels share the same amplitude limits.
- `video_music_comparison.csv` / `.npz`: common timestamps and both curves.
- `metadata.json`: explicit audio/video PTS offset, common-grid convention,
  checkpoint, layer and normalization details.

This is an event-curve comparison only; it does not generate music, detect beats,
compute correlation, shift curves to improve agreement, or train a model.
