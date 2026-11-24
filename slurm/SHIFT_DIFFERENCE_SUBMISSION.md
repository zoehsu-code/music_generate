# Train temporal-shift absolute differences

CPU-only saved-curve analysis on the exact 30,944 successful clips from job
28224289. No encoder inference. Deferred OOM, review and previous failed clips
remain excluded.

## Method

Reuse the original native standardization, shared 25 Hz grid, Hann-31, fixed
2-second endpoint trims and minimum 125 anchors. Shifts: -2, -1, -0.5, 0, +0.5,
+1, +2 seconds. Positive delta compares video(t) with music(t+delta). Half-second
shifts use linear interpolation, without wrapping, extrapolation or per-shift
normalization. Compute mean and median over time of abs(video(t)-music(t+delta)).
Lower is better. Units are standardized, smoothed curve values.

All seven Pearson scores must reconstruct the prior run within 1e-10, with
identical anchor counts. Metadata input hashes must match the original scores;
media hashes are not recomputed. Failures exclude a clip at every shift. Check
completed=30944 and failed=0 before claiming the same cohort as the old report.
Fixed-support zero differences need not equal older full-support differences.
Best-of-seven reductions are selection-biased descriptive results, not true-delay
estimates. IQR is between-clip spread, not a confidence interval.

## Submit on Lighthouse

```bash
cd /home/xvming/OSSL_Event_Alignment/music_generate
git pull --ff-only origin main
mkdir -p logs
bash -n slurm/lighthouse_shift_difference.sbatch
sbatch --account=aimusic_project --partition=aimusic_project \
  --export=ALL slurm/lighthouse_shift_difference.sbatch
```

The launcher loads Python 3.12.1 and uses the existing .venv. Resources: four
CPUs, 16 GB RAM, six hours, no GPU. Optional small run: append `--limit 20`.
Every job writes a new directory and refuses existing targets.

Default inputs under
`/scratch/aimusic_project_root/xvming/OSSL_Event_Alignment/`:

- `analysis/train_saved_curves_28185053/alignment_scores.csv`
- `analysis/train_temporal_shift_28224289/per_clip.csv`
- `event_curves/train/`
- `music_event_curves/msd_l7_wav_run01/train/`

Overrides: PROJECT_DIR, ALIGNMENT_SCORES, PRIOR_SHIFT_SCORES, VIDEO_ROOT,
MUSIC_RESULT_ROOT, SHIFT_DIFFERENCE_RESULT_ROOT. Do not reuse custom output paths.

## Outputs

New directory:
`/scratch/aimusic_project_root/xvming/OSSL_Event_Alignment/analysis/train_shift_difference_JOBID/`
Original results and curves remain unchanged.

- `per_clip.csv`: seven mean absolute differences (mae_at_...), seven median
  absolute differences (median_ae_at_...), minima, reduction from fixed zero,
  anchor count, original quadrant, Pearson reconstruction error.
- `aggregate.csv`: all clips and each original quadrant; clip-weighted mean,
  between-clip median/IQR, equal-movie mean and signed change from fixed zero.
  Movie weighting first averages clip summaries within each movie.
- `absolute_difference_shift.png`: aggregate mean/median absolute-difference
  panels, each showing clip-weighted and equal-movie means and between-clip IQR.
- `summary.json`, `analysis_config.json`, `failures.csv`: counts, source hashes,
  parameters and failures. Results are saved even with failures; exit status
  is nonzero if any clip failed. No per-clip plots or aligned arrays are saved.

```bash
# Replace JOBID with the submitted job ID.
sacct -j JOBID --format=JobID,State,Elapsed,ExitCode
cat /scratch/aimusic_project_root/xvming/OSSL_Event_Alignment/analysis/train_shift_difference_JOBID/summary.json
cat /scratch/aimusic_project_root/xvming/OSSL_Event_Alignment/analysis/train_shift_difference_JOBID/failures.csv
```
