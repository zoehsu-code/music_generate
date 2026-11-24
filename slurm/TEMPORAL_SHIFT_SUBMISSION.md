# Train temporal shift experiment

Analyze the 30,946 quality-ok pairs from the saved 30,953-row Train score table.
The 57 deferred OOM clips and one missing scored pair remain outside this phase;
the seven quality-review rows are listed separately and not analyzed. No models
or media decoders run. The main experiment uses the full eligible subset, with
the same successful subset at every tested shift.

## Method

Shifts are -2, -1, -0.5, 0, +0.5, +1 and +2 seconds. Positive delta evaluates
`corr(video(t), music(t+delta))`: matching events in music occur later. As in
Test-50, remove padded audio centers, standardize the native curves separately,
interpolate onto shared observed 25 Hz support, then apply centered Hann-31.
Trim 2 seconds from both ends of the aligned grid and retain identical video
anchors for all shifts. Require at least 125 anchor samples. Half-second shifts
use linear interpolation at 12.5 sample offsets. No circular shifting,
extrapolation, lag-specific restandardization or lag-specific smoothing occurs.
Pearson centers and normalizes each paired vector as usual.

Each reconstructed full-support and fixed-support zero correlation must agree
with the previous score table to within 1e-10. A mismatch is recorded as failure,
not silently accepted. Movie-level clustering, best-of-seven selection effects,
and optima at the +/-2 second boundary limit interpretation. An improved best
score alone is not evidence of a true temporal delay. The original bottom 20%
group is defined before shift analysis using the original full-support ranking;
its gain is descriptive and subject to selection/regression-to-the-mean effects.

## Lighthouse commands

From the updated checkout:

```bash
cd /home/xvming/OSSL_Event_Alignment/music_generate
git pull --ff-only origin main
module load python/3.12.1
source .venv/bin/activate
mkdir -p logs
bash -n slurm/lighthouse_temporal_shift.sbatch

# Small end-to-end check; only 20 pairs:
sbatch --account=aimusic_project --partition=aimusic_project \
  --export=ALL slurm/lighthouse_temporal_shift.sbatch --limit 20
```

After the smoke test summary reports completed=20 and failed=0:

```bash
sbatch --account=aimusic_project --partition=aimusic_project \
  --export=ALL slurm/lighthouse_temporal_shift.sbatch
```

CPU-only request: four CPUs, 16 GB RAM, six hours. The prior CPU analysis used
the same account/partition. No task array is needed for these saved-curve
calculations. Original outputs are read-only. Each job writes to a distinct
`/scratch/aimusic_project_root/xvming/OSSL_Event_Alignment/analysis/train_temporal_shift_JOBID/`
directory; an existing target directory is refused. If selecting custom paths,
export `ALIGNMENT_SCORES`, `VIDEO_ROOT`, `MUSIC_RESULT_ROOT`, or `SHIFT_RESULT_ROOT`.
Do not point a smoke test and full run at the same custom output directory.

## Outputs

- `per_clip.csv`: seven correlations, fixed-support baseline, best tested shift,
  gain over fixed zero, anchor count, original quadrant and bottom-group label.
- `aggregate.csv`: mean, median, quartiles and changes at each shift, for all
  pairs, each original quadrant and the original bottom 20% group. All group
  profiles use exactly the same successful clips at every shift.
- `summary.json`: completed/failed counts, mean profile, best-shift counts,
  boundary-optimum count and maximum baseline reconstruction discrepancy.
- `analysis_config.json`: source score-table hash, preprocessing and sign rules.
- `failures.csv` and `excluded_quality.csv`: transparent analysis accounting.
- `temporal_shift.png`: one aggregate mean/median profile and between-clip IQR.
  The IQR is descriptive spread, not a confidence interval.

No per-clip PNGs, aligned arrays or model features are saved. Expected full run:
selected=30946, completed=30946, failed=0. Inspect failures if the count differs.
Stdout/stderr: `logs/shift-JOBID.out` and `.err`.
