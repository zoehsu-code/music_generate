# Video-to-Music Experiment Log

Updated 2026-11-17. Dates come from saved outputs, commits, and research discussions. The project investigates whether visual and soundtrack changes align in time and whether visual event curves can condition music generation.

## Data and Artifact Locations

| Item | Location | Verification status |
| --- | --- | --- |
| OSSL-v2 dataset root on Lighthouse | `/scratch/aimusic_project_root/xvming/OSSL_Event_Alignment/data/OSSL-v2/` | The 2026-11-10 manifest audit found all test videos and audio and all train audio. The latest scan during the supplemental download found 23,773 of 31,011 train videos present and nonempty, with 7,238 missing. |
| Batch code on Lighthouse | `/home/xvming/OSSL_Event_Alignment/music_generate/` | Confirmed from `PROJECT_DIR` in the cluster-local `lighthouse_paths.sh` on 2026-11-10. |
| Batch event-curve output on Lighthouse | `/scratch/aimusic_project_root/xvming/OSSL_Event_Alignment/event_curves/` | User-provided Lighthouse shell output on 2026-11-16 reconfirmed the path and counted 31,011 Train raw curve files and 31,011 metadata files. Manifest correspondence and array integrity remain unverified. |
| Train MusicFM output on Lighthouse | `/scratch/aimusic_project_root/xvming/OSSL_Event_Alignment/music_event_curves/msd_l7_wav_run01/train/` | Configured for the successful four-clip smoke test and full array 28176562 submitted by the user on 2026-11-16. Full output counts and integrity are pending. |
| DINOv2 model cache on Lighthouse | `/scratch/aimusic_project_root/xvming/OSSL_Event_Alignment/cache/huggingface/` | Configured as `HF_HOME` in `lighthouse_paths.sh`. The batch launcher reads models from its `hub/` subdirectory. |
| Official dataset on Hugging Face | [`McAuley-Lab/OSSL-v2`](https://huggingface.co/datasets/McAuley-Lab/OSSL-v2) | Dataset card checked on 2026-11-10. It lists 31,011 train pairs and 3,332 test pairs, with one MP4 and one WAV per pair. |
| Test-50 inputs on Mac | [`analysis-data/ossl-test/`](../../music_generate/analysis-data/ossl-test/) | The older local checkout contains 50 soundtrack WAVs and 50 raw video event curves. |
| Test-50 reports and scores | [`reports/ossl-test-50/`](reports/ossl-test-50/) | This repository contains scores, configuration, figures, and aligned curves. |

## EXP-01 · October 2–4 · Single-Clip Event-Curve Baseline

- **Video.** Implemented a baseline following V2M-ZERO. DINOv2-L/14 extracts a 1,024-dimensional vector per frame by averaging the final-layer patch tokens. The raw event curve uses adjacent-frame cosine distance, `1 − cosine`, at the midpoint of each frame pair. The single-clip pipeline supports standardization, resampling, and Hann smoothing.
- **Audio.** MusicFM-MSD layer 7 extracts 1,024-dimensional features at 25 Hz from audio resampled to 24 kHz. The event curve uses cosine distance between adjacent features. Early single-clip runs used the MP4 audio track, which may contain dialogue and sound effects. Comparisons use the shared time interval.
- **Results.** Video validation passed for `746_3_29540_30708`, yielding 281 × 1,024 features and 280 curve points. Saved video, audio, and overlay outputs also exist for `746_1_6129_7728`, `747_10_151817_161145`, and `travelcut`. The first two of these have run records showing successful numerical checks.
- **Reproducibility.** Pinned model versions and weight checksums and added numerical and audiovisual integration tests. The implementation supports frame offsets of 1, 15, and 60. Results for that offset comparison have yet to be located. [Implementation and parameters](README.md) · [Video outputs](../../music_generate/outputs/) · [Audio and overlay outputs](../../music_generate/outputs-music/)

## EXP-02 · October 27 · OSSL-v2 Batch Video Extraction on Lighthouse

- Implemented a resumable Slurm batch workflow with sharding over the OSSL-v2 train and test manifests. It retains all decoded frames and saves raw curves and timestamps at a one-frame offset. Resampling and smoothing happen downstream.
- Verified that the local copy from Lighthouse contains **50 raw video curves and 50 corresponding soundtrack WAVs**. The video metadata records CUDA extraction and the original video paths. DATA-01 records the Lighthouse file audit. Full curve extraction progress remains open. [Batch script](extract_ossl_batch.py) · [Slurm script](slurm/lighthouse_ossl.sbatch) · [Local Test-50 inputs](../../music_generate/analysis-data/ossl-test/)

Slurm accounting confirms that job `27987221_0` (`ossl-events`) completed under account `aimusic_project` in partition `aimusic_project`.

## EXP-03 · October 27 · OSSL-v2 Test-50 Paired Analysis

- **Sample.** All 50 video–soundtrack pairs completed and passed the quality checks. The selected subset spans three films, with 5 clips from film 746, 23 from 747, and 22 from 748. Results describe this subset.
- **Alignment.** Assumed coincident clip starts. Standardized the two curves separately, interpolated them to a shared 25 Hz grid, and applied a symmetric 31-point Hann window with a 1.2-second endpoint span. The primary metric is zero-lag Pearson correlation. Saved sensitivity results without smoothing and with a 61-point Hann window (2.4 seconds), plus a ±2-second lag diagnostic. The lag diagnostic uses a fixed cropped interval. A positive lag indicates that soundtrack changes occur later.
- **Results.** With 1.2 seconds of smoothing, median zero-lag Pearson r was **−0.081** and ranged from **−0.600 to 0.718**. The highest-scoring clip was `747_14_340984_342216` and the lowest was `746_0_4192_5407`. Median r without smoothing was **−0.019** and with 2.4 seconds of smoothing was **−0.087**. The report lists the top and bottom 10 clips by the primary metric for comparison within this run.
- **Artifacts.** Aligned curves and plots for every pair, a score table, and an HTML report. The paired analysis script is in the older local checkout. [Analysis script](../../music_generate/analyze_ossl_pairs.py) · [Scores](reports/ossl-test-50/alignment_scores.csv) · [Analysis configuration](reports/ossl-test-50/analysis_config.json)

## EXP-04 · November 3 · Absolute Difference Metrics and Report Publication

- Used the saved zero-lag curves from EXP-03 to calculate `|video(t) − music(t)|` at each time point, then took the mean and median per clip. Smaller differences indicate closer values on the standardized and smoothed curves. Pearson measures co-variation, while absolute difference also captures amplitude differences. Both depend on the preprocessing settings.
- **Results.** Across the 50 pairs, the mean of the per-pair mean absolute differences was **0.449**. The median of the per-pair median absolute differences was **0.350**. Both are expressed in standardized curve units. The smallest differences occurred in `746_1_6129_7728`, with a mean of **0.267** and a median of **0.182**.
- **Artifacts.** Updated scores, a difference overview, and per-clip difference plots and CSVs. The calculation reused the saved curves and preserved the Pearson ranking from EXP-03. Published and verified at commits `3d22a59` and `5f738e3`. [Local report](reports/ossl-test-50/report.html) · [Summary](reports/ossl-test-50/summary.json) · [Report builder](scripts/build_ossl_report.py) · [GitHub Pages](https://zoehsu-code.github.io/music_generate/report.html)

## EXP-05 · November 10 · Test-50 Quadrant Classification

- Reused the 50 pairs from EXP-04. The thresholds were the sample medians for zero-lag Pearson r (**−0.081413**) and per-clip mean absolute difference (**0.439938**). Assignments use full precision. Values equal to a threshold enter the high-correlation or low-difference group, respectively.
- High correlation and low difference **18** pairs, high correlation and high difference **7**, low correlation and low difference **7**, and low correlation and high difference **18**. The groups are relative to this sample. Clip-by-clip review of audiovisual characteristics remains open.
- The page contains a scatter plot, four sample tables, and links to each original video and curve comparison. Verified all 50 video paths against the pinned Hugging Face revision. The first published version was verified at commit `07d26f3`. The English edition was published and verified at commit `0bf4348`. [Quadrant report](reports/ossl-test-50/quadrants.html) · [Scores](reports/ossl-test-50/alignment_scores.csv) · [GitHub Pages](https://zoehsu-code.github.io/music_generate/quadrants.html)

## EXP-06 · November 10 · Full Train Video Event-Curve Extraction

- Submitted Slurm array `28122011` with `slurm/lighthouse_ossl.sbatch --splits train` under account and partition `aimusic_project`. The array has 64 shards and allows two concurrent tasks. It extracts DINOv2 video event curves and timestamps for the Train manifest.
- The first queue snapshot showed tasks 0 and 1 running on `lh2300`, with tasks 2–63 pending under `JobArrayTaskLimit`. The most recent video audit before submission had 7,238 missing files while the supplemental download was in progress. A later stdout sample showed 10 completed clips from each of the first two shards, with no failures in the 20 displayed lines. Per-clip processing time averaged 7.5 seconds in that sample, ranging from 2.9 to 21.5 seconds. Final extraction counts are pending.
- The configured output directory is `/scratch/aimusic_project_root/xvming/OSSL_Event_Alignment/event_curves/train/`. Slurm stdout and stderr are in the Lighthouse checkout under `logs/ossl-28122011_<task>.out` and `logs/ossl-28122011_<task>.err`.
- **2026-11-16 count audit.** User-provided shell output from `lh-login2`
  reconfirmed `RESULT_ROOT` and found 31,011 `*_raw_event_curve.npy` files and
  31,011 `metadata.json` files recursively under Train. Example paths follow
  `train/<movie_id>/<clip_id>/<clip_id>_raw_event_curve.npy`. The shell reported
  `MUSIC_RESULT_ROOT` unset. These counts match the expected Train size but do
  not establish exact manifest coverage, numerical integrity or final job state.

## EXP-07 · November 10 · Temporal Shift Test

- Reused all 50 saved Test-50 curves and evaluated Pearson r at −2, −1, −0.5, 0, +0.5, +1, and +2 seconds. Each clip used the same video anchors at all shifts, trimming 2 seconds at each end and linearly interpolating at half-sample offsets. Positive shifts compare video with later music.
- Mean r peaked at +1 second (**+0.046** versus **−0.017** at zero). Peak shifts were positive for 26 clips, negative for 18, and zero for 6. The mean absolute change at ±0.5 seconds was approximately 0.15. In the original bottom 10, 7 clips gained at least 0.20 at their best tested shift, and 3 changed from negative to positive on the fixed support.
- These are descriptive results for the selected subset. Twenty-five peaks landed at the bounds of the tested range. Verified all 350 values against saved lag diagnostics or independent half-sample interpolation. Published to GitHub in commit `774ec99`. [Experiment report and outputs](reports/ossl-test-50/temporal-shift/README.md) · [Analysis script](scripts/analyze_temporal_shift.py) · [GitHub report](https://github.com/zoehsu-code/music_generate/tree/main/reports/ossl-test-50/temporal-shift#readme)

## EXP-08 · November 16 · Train MusicFM Extraction

- **Code and environment.** Published implementation at `b2851e28431b98a2dc16f211c9818d9f68178809`.
  The user confirmed the batch CLI and shell syntax checks on Lighthouse,
  torch/torchaudio 2.8.0+cu128 and einops 0.8.1. All three pinned MusicFM assets
  were downloaded into the project scratch cache and passed SHA-256/size checks.
- **Inputs and settings.** OSSL-v2 Train soundtrack WAVs selected from the
  31,011-entry manifest; dataset revision intended to remain
  `3ed5a37d933084e5ec134e2138b8faa156cf021b` (cluster snapshot not reverified).
  MusicFM-MSD layer 7, mono 24 kHz, float32 CUDA inference, 25 Hz features,
  offset-1 raw cosine curves and midpoint timestamps, no smoothing/resampling.
  Model loaded once per shard; features are not saved by default.
- **Smoke test.** User submitted job `28176555`. Supplied stdout records CUDA
  and completed=4, skipped=0, failed=0; supplied stderr was empty. Clips were
  `0_0_1824_10960`, `0_10_114691_115970`, `0_11_145432_146999`, and
  `0_12_157946_159129`. The first clip had 91.37 seconds of audio and elapsed
  6.73 seconds including model loading. Final Slurm state and independent array
  validation have not been supplied.
- **Full extraction submitted, completion pending.** User reported
  `Submitted batch job 28176562` after the full Train submission command:
  `--array=0-63%2`, account/partition `aimusic_project`, one GPU, four CPUs,
  32 GB RAM and 12 hours per task, `--splits train --expected-count 31011`.
  Follow-up user-provided summaries on 2026-11-17 cover all 64 unique shards:
  completed=30,950, skipped=4, failed=57, totaling 31,011 assignments. Thus
  30,954 clips (99.816%) were reported completed or validated/skipped; 36 shards
  reported at least one failure. Independent output count and numerical audit
  remain pending. The attachment ends at the failure grep command and does not
  include the full failure records. A subsequent attachment on 2026-11-17
  supplied all 57 unique failures: every record reports CUDA OutOfMemoryError
  on a GPU with 44.39 GiB total capacity. The largest single failed allocation
  was 145.27 GiB for `980_3_168188_365659`; larger GPU memory alone is not
  established as sufficient. Saved evidence: `musicfm-28176562-failures.json`.
  On 2026-11-17 the user explicitly deferred all 57 failures and requested
  downstream processing of available curves. Do not retry them in this phase.
- **Artifacts.** Output root is
  `/scratch/aimusic_project_root/xvming/OSSL_Event_Alignment/music_event_curves/msd_l7_wav_run01/`;
  per-clip `music_raw.npz` and `metadata.json` live under `train/<movie>/<clip>/`.
  Per-shard stdout/stderr are in the cluster checkout's
  `logs/music-28176562_<task>.out` and `.err`; unique JSONL logs are in the
  output root's `_logs/`. Completed smoke-test outputs should be validated and
  skipped by the full run. Existing video outputs are separate.

## DATA-01 · November 10 · Lighthouse Download Audit

On `lh-login2`, checked the expected MP4 and WAV paths for existence and nonzero size against the unique clip IDs in the local train and test manifests. The total and unique ID counts for both splits matched the official counts.

| Split | Unique manifest IDs | Videos present and nonempty | Videos missing or empty | Audio files present and nonempty |
| --- | ---: | ---: | ---: | ---: |
| Train | 31,011 | 19,496 | 11,515 | 31,011 |
| Test | 3,332 | 3,332 | 0 | 3,332 |

Both `metadata.csv` and `public_train_test_remapped.pkl` were present and nonempty. Media integrity and the count of extracted curves require separate checks.

The supplemental download targets train MP4s at Hugging Face revision `3ed5a37d933084e5ec134e2138b8faa156cf021b`, as pinned in the project README. The first attempt using the Lighthouse Python module and `hf` stopped with HTTP 429 at the Hugging Face Xet token endpoint. Hugging Face login subsequently succeeded and the retry began. A follow-up scan during the download found 23,773 of 31,011 train MP4s present and nonempty, with 7,238 missing and zero empty. This is a progress snapshot. The final post-download audit is pending.

## Open Research Tasks

- **Available Train paired analysis (2026-11-17, prepared, not submitted).**
  Added `analyze_saved_ossl.py`, `music_alignment_metrics.py`, and
  `slurm/lighthouse_alignment.sbatch`; instructions in
  `slurm/ALIGNMENT_SUBMISSION.md`. Analyze the 30,954 reported available pairs,
  explicitly excluding the 57 deferred OOM IDs. No encoder reruns. Match Test-50
  standardization, 25 Hz shared grid, Hann-31, zero-lag Pearson and mean/median
  absolute difference; include Hann-61 and fixed-support lag sensitivity.
  Quadrants use medians of this run's quality-ok subset. Unit tests passed (2);
  independent reproduction of three saved metrics for all 50 Test-50 pairs
  had maximum absolute error 1.11e-16. No Train scores have been computed locally
  and no analysis Slurm job has been submitted. Exclusion may bias duration mix.

- **Train MusicFM extraction (EXP-08, submitted 2026-11-16).** Check array
  `28176562`, aggregate shard summaries, and validate all 31,011 outputs against
  the manifest before paired analysis. Preparation history: added
  `extract_ossl_music_batch.py`, `slurm/lighthouse_music.sbatch`, and
  `slurm/MUSIC_SUBMISSION.md` locally. Reused the older Test-50 encoder's
  `load_musicfm`/`encoder_bundle` support. Proposed configuration: MusicFM-MSD
  layer 7, 24 kHz mono, 25 Hz features, offset-1 raw cosine curves, no smoothing,
  64 shards with two concurrent tasks. WAV is the default to match Test-50;
  explicit MP4 input is supported in a separate output root. Scratch output
  paths in the submission guide are proposed, not verified existing locations.
  Preparation initially performed no cluster changes; subsequent user-provided
  installation, verified downloads and submissions are recorded in EXP-08.
  Local validation: six audio/video batch and audio tests passed, including
  synthetic model reuse, disjoint MP4 sharding, failure isolation, compatible
  resume and corrupt-output rejection; `bash -n` passed for the new launcher.
  Tests do not establish real MusicFM CUDA performance or Lighthouse readiness.

- **Train videos and curve extraction.** Repeat the source-file audit after the supplemental download. The 2026-11-16 curve-file counts match 31,011; exact manifest coverage, timestamps and numerical integrity still require validation.
- **Visual event curves as a generation condition.** The conditioning approach has been discussed. Training, inference, and controlled comparison results are pending. [Research discussion](chatgpt-conversation://6af1f44e-5c70-83ea-967d-ae2ca18846e7)

## Entry Format

After each run, record the experiment ID, date, status, question, dataset version and sample scope, code revision and parameters, completed and failed counts, results, limitations, and artifact locations. Keep proposed work in the open tasks section. Preserve the history of retired or superseded runs and record the reason and replacement experiment ID when known. When a data or artifact location changes, update the location and verification status in the table above.
