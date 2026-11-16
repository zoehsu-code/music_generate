# Lighthouse MusicFM submission

These commands are for Lighthouse after this source update has been copied to
`/home/xvming/OSSL_Event_Alignment/music_generate`. No job has been submitted by
the local implementation or tests. Account and partition values were used in
previous project jobs; current authorization and resources must still be available.

## Prepare once

Use WAV for consistency with the Test-50 soundtrack analysis. Use MP4 only when
the intended signal is the embedded audio track (which can include dialogue and
sound effects). The two inputs have separate output roots. Do not switch the
input kind or layer when resuming a run.

```bash
module load python/3.12.1
cd /home/xvming/OSSL_Event_Alignment/music_generate
source .venv/bin/activate
export HF_HOME=/scratch/aimusic_project_root/xvming/OSSL_Event_Alignment/cache/huggingface
export INPUT_KIND=wav
export OSSL_ROOT=/scratch/aimusic_project_root/xvming/OSSL_Event_Alignment/data/OSSL-v2
export MUSIC_RESULT_ROOT=/scratch/aimusic_project_root/xvming/OSSL_Event_Alignment/music_event_curves/msd_l7_${INPUT_KIND}_run01
mkdir -p logs
tmux new -s musicfm-prepare
```

Inside that named tmux session, repeat the module/environment/export commands
above if needed, then install dependencies and verify all pinned assets:

```bash
python -m pip install -r requirements-music.txt
python scripts/download_weights.py --cache_dir "$HF_HOME/hub"
python -m pip check
python extract_ossl_music_batch.py \
  --dataset-root "$OSSL_ROOT" --output-root "$MUSIC_RESULT_ROOT" \
  --splits train --expected-count 31011 --input-kind "$INPUT_KIND" \
  --dry-run > logs/music-input-audit.jsonl
```

The audit must exit zero. It checks all manifest-selected inputs exist and are
nonempty; decoding is validated during extraction. The download utility verifies
all six pinned assets, including DINOv2 assets, without inference. Detach with
Ctrl-B then D; reattach using `tmux attach -t musicfm-prepare`.

## Smoke test, then full extraction

From the repository root with the exports above:

```bash
sbatch --account=aimusic_project --partition=aimusic_project \
  --export=ALL --array=0-0 slurm/lighthouse_music.sbatch \
  --splits train --expected-count 31011 --limit 4
```

Check `logs/music-JOBID_0.out` and `.err`. The summary must show completed=4
(or completed+skipped=4 on a retry), failed=0 and MusicFM loading on CUDA.
Measure memory and runtime on representative long clips before relying on the
initial 32 GB RAM / 12 hour request. Each whole clip is encoded in one pass.

```bash
sbatch --account=aimusic_project --partition=aimusic_project \
  --export=ALL --array=0-63%2 slurm/lighthouse_music.sbatch \
  --splits train --expected-count 31011
```

This assigns 485 clips to shards 0–34 and 484 to shards 35–63, with at most two
concurrent tasks. Every task requests one GPU and four CPUs. No GPU type is
hard-coded. These are resource starting points, not measured MusicFM capacity.

```bash
squeue -u "$USER"
# Replace JOBID with the number returned by sbatch:
sacct -j JOBID --format=JobID,State,Elapsed,ExitCode,MaxRSS
```

## Outputs and retries

Each `MUSIC_RESULT_ROOT/train/<movie>/<clip>/` contains `music_raw.npz` with
`raw` and `timestamps`, plus `metadata.json`. Curves are offset-1 cosine
dissimilarity; timestamps are adjacent feature midpoints at 25 Hz, relative to
the first decoded sample. No smoothing or resampling is applied. Input container
start PTS is retained in metadata. `--save-features` optionally stores the much
larger feature matrix; keep that setting identical on retries.

Resubmit the same full array to retry failures/timeouts. Valid completed clips
are skipped after checking the input SHA-256, model/settings, curve shape,
finite values and time grid. Corrupt/incompatible output is reported as failure
and never overwritten. Missing/bad audio is logged and subsequent clips continue.
Unique JSONL logs are stored under `MUSIC_RESULT_ROOT/_logs/`; a shard exits
nonzero if any clip fails. Final acceptance requires 31,011 validated outputs
and no unresolved failures. Slurm completion alone does not establish coverage.

For MP4, set `INPUT_KIND=mp4` and set `MUSIC_RESULT_ROOT` to a distinct MP4 run
directory before the audit and both submissions. The batch reads
`train/video/<movie>/<clip>.mp4` instead of `train/audio/<movie>/<clip>.wav`.
