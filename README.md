# Video event curves — V2M-ZERO baseline

## Analysis Results

[Read the English 50-pair OSSL video–music alignment report](reports/ossl-test-50/README.md) — ranked scores, all 50 comparison plots, and downloadable curve data.

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

## OSSL-v2 video-only batch extraction on Lighthouse

`extract_ossl_batch.py` reads `splits/train.txt` and `splits/test.txt`, then
opens `<split>/video/<movie_id>/<clip_id>.mp4`. Audio files are not used.
Each process loads DINOv2 once and reuses it across clips. Extraction preserves
all frames and computes offset-1 cosine dissimilarity, with no temporal
resampling or smoothing. The batch outputs are:

```
RESULT_ROOT/
  train/<movie_id>/<clip_id>/
    <clip_id>_raw_event_curve.npy
    <clip_id>_native_timestamps.npy
    <clip_id>_frame_timestamps.npy
    metadata.json
  test/<movie_id>/<clip_id>/...
  _logs/shard-<index>-<unique-run-id>.jsonl
  _locks/<split>/<movie_id>/<clip_id>.lock
```

The raw curve and its native timestamps are the authoritative downstream inputs.
Standardize per clip, then resample and smooth later as needed. Normalization
statistics are recorded, but normalized/smoothed arrays and plots are not written
in batch mode. `--save-features` optionally saves the much larger `[T,1024]`
feature matrix; it is unnecessary for later resampling/smoothing, but necessary
if you want to recompute other feature offsets without rerunning the encoder.
The original single-video CLI keeps its existing behavior.

### 1. Prepare storage and environment

Use your actual project scratch paths; do not put the whole dataset in home.
Obtain the authorized Slurm account via `my_accounts` and confirm the associated
GPU partition. Account and partition names are deliberately not hard-coded.

```bash
export PROJECT_DIR=/absolute/path/to/music_generate
export OSSL_ROOT=/absolute/project/scratch/OSSL-v2
export RESULT_ROOT=/absolute/project/scratch/ossl-event-curves
export HF_HOME=/absolute/project/scratch/huggingface
cd "$PROJECT_DIR"
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Use a Lighthouse-provided Python module if Python 3.12 is not already available
(`module avail`); install a CUDA-compatible PyTorch build for the cluster's
GPU/driver using the official PyTorch installer instructions if necessary.
The batch launcher refuses CPU fallback when CUDA was requested.
Do not copy the macOS virtual environment to Linux.

Download only video files and text metadata (roughly 108 GB for videos at the
pinned revision below), using a site-approved transfer node/session:

```bash
hf download McAuley-Lab/OSSL-v2 --repo-type dataset \
  --revision 3ed5a37d933084e5ec134e2138b8faa156cf021b \
  --include 'train/video/*' 'test/video/*' 'splits/*' 'metadata.csv' 'README.md' \
  --local-dir "$OSSL_ROOT"
```

Download/cache only the DINOv2 model and processor once before launching GPU
jobs. This loads the model on CPU but does not run video inference:

```bash
python - <<'PY'
import os
from video_event_curve.dinov2_encoder import load_dinov2
load_dinov2('cpu', os.path.join(os.environ['HF_HOME'], 'hub'))
PY
```

Keep the same pinned dataset snapshot for an entire run. Each saved output
also records the individual input SHA-256 and the model revision.

### 2. Inspect inputs, then run a small GPU smoke test

Listing the first four test clips needs no model weights or GPU:

```bash
python extract_ossl_batch.py --dataset-root "$OSSL_ROOT" \
  --output-root "$RESULT_ROOT" --splits test --limit 4 --dry-run
```

Create logs before submission; run `sbatch` from the repository directory.
Replace `YOUR_ACCOUNT` and `YOUR_GPU_PARTITION` with your authorized values.
The resource requests are starting points, not a statement about your quota.

```bash
mkdir -p logs
sbatch --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION \
  --array=0-0 slurm/lighthouse_ossl.sbatch --splits test --limit 4
```

Check Slurm output for CUDA use and successful per-clip completion before
launching the full set. No real Lighthouse run has been performed by this repo's
local automated tests.

### 3. Submit all clips

```bash
sbatch --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION \
  --array=0-15%2 slurm/lighthouse_ossl.sbatch
```

This example splits the sorted clip list into 16 disjoint shards, with at most
2 concurrent jobs. Each job requests 1 GPU, 4 CPU cores, 32 GB RAM and 12 hours;
adjust these requests and concurrency to the PI's hardware/permissions. This
walltime is a starting request, not a runtime prediction. Use zero-based,
contiguous arrays with step 1. `BATCH_SIZE=4` is the default; increase it after
measuring GPU memory. Keep native float32 inference for consistency.

For an already allocated interactive GPU, the equivalent one-process run is:

```bash
python -u extract_ossl_batch.py --dataset-root "$OSSL_ROOT" \
  --output-root "$RESULT_ROOT" --cache-dir "$HF_HOME/hub" \
  --local-files-only --device cuda --batch-size 4
```

### Resume and failures

Resubmit the same command to resume after a timeout or interruption. Completed
outputs are skipped only after checking extraction settings, input SHA-256,
array existence, shapes, dtypes and finite values. Files are published by an
atomic directory rename after a clip completes. A crash may leave a hidden
temporary directory; it is ignored on retry. Per-clip advisory file locks prevent
duplicate work by concurrent launchers on filesystems supporting `flock`.

An incompatible or corrupted existing output is reported as a failure and never
overwritten: move that clip's directory aside, or choose a new output root,
before retrying. Bad/missing videos are logged while subsequent clips continue.
Each run writes a unique JSONL log, a completion summary, and returns nonzero
if any assigned clip failed. Consult logs even when Slurm reports a failed job:
other clips may have completed successfully. Shards are based on sorted IDs,
not estimated duration, so job lengths can vary.

Local validation includes synthetic MP4 decoding, model reuse, corrupt-video
isolation, verified resume, rejection of changed settings/corrupt outputs, and
disjoint sharding. It does not measure actual Lighthouse GPU throughput.
