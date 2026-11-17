# Analyze available Train curves

The user deferred the 57 CUDA OOM clips from array 28176562 on 2026-11-17.
`musicfm-28176562-failures.json` records the explicit exclusions. The analysis
reads the original 31,011-ID manifest and expects to score 30,954 available
pairs; any other missing or invalid pair is an unexpected failure, not silently
excluded. Saved curves and media are never modified.

From the Lighthouse checkout:

```bash
git pull --ff-only origin main
module load python/3.12.1
source .venv/bin/activate
mkdir -p logs
bash -n slurm/lighthouse_alignment.sbatch
sbatch --account=aimusic_project --partition=aimusic_project \
  --export=ALL slurm/lighthouse_alignment.sbatch
```

This job requests four CPUs, 16 GB RAM and six hours, with no GPU. The configured
partition must accept CPU-only jobs; if Slurm rejects this request, retain the
error for choosing an authorized CPU partition. Do not change the extraction
jobs or request a GPU for model inference: this analysis loads no models.

The default output is a new, job-specific directory under
`/scratch/aimusic_project_root/xvming/OSSL_Event_Alignment/analysis/train_saved_curves_JOBID/`.
Existing output directories are refused. Stdout/stderr are
`logs/alignment-JOBID.out` and `.err`.

Outputs: `alignment_scores.csv`, `issues.csv`, `analysis_config.json`, and
`summary.json`. Expected summary: scored=30954, excluded=57, failed=0. Numerical
quality flags may reduce the eligible count; inspect them before interpretation.
Optional `--save-aligned` stores per-pair standardized/smoothed arrays but is off
by default to avoid many extra files. Plots are not generated for every clip.

Method matches Test-50: trim padded audio centers to decoded audio duration,
standardize each native curve, interpolate onto common 25 Hz support, then apply
a centered symmetric Hann-31 kernel. Main scores are zero-lag Pearson and mean/
median absolute difference. Hann-61 sensitivity and fixed-support lag diagnostics
are included. Clip starts are assumed coincident; no fitted shift is applied to
the main scores. Quadrants use the current eligible subset's medians, not the
Test-50 medians. Near-constant, short-support and duration-mismatch pairs are
flagged for review. Excluding OOM clips may introduce duration-related selection
bias; these results do not cover the entire Train split.

The analyzer validates saved curves/timestamps and records media hashes from
metadata; it does not decode media or recompute the original media hashes.
