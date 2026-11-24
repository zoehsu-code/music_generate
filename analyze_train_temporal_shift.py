"""Seven fixed-support shifts on saved Train curves; no encoder inference."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from video_event_curve.event_curve import standardize_curve, hann_smooth

SHIFTS = np.array([-2., -1., -.5, 0., .5, 1., 2.])
RATE = 25
ZERO = 3


def aligned_curves(vraw, vt, mraw, mt, duration):
    for raw, times in ((vraw, vt), (mraw, mt)):
        if raw.ndim != 1 or times.ndim != 1 or len(raw) != len(times) or len(raw) < 3:
            raise ValueError('Invalid curve shape')
        if not np.isfinite(raw).all() or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError('Nonfinite curve or nonincreasing timestamps')
    keep = mt < duration
    mraw, mt = mraw[keep], mt[keep]
    if len(mt) < 3:
        raise ValueError('Too few valid audio centers')
    start, end = max(vt[0], mt[0]), min(vt[-1], mt[-1])
    grid = np.arange(np.ceil(start * RATE), np.floor(end * RATE) + 1) / RATE
    if len(grid) < 3:
        raise ValueError('Insufficient common time support')
    video = hann_smooth(np.interp(grid, vt, standardize_curve(vraw)), 31)
    music = hann_smooth(np.interp(grid, mt, standardize_curve(mraw)), 31)
    return grid, video, music


def shift_scores(times, video, music):
    """Positive delta evaluates corr(video(t), music(t+delta))."""
    if len(times) != len(video) or len(times) != len(music):
        raise ValueError('Mismatched aligned lengths')
    if not np.isfinite([times, video, music]).all() or not np.allclose(np.diff(times), 1 / RATE):
        raise ValueError('Invalid aligned curves or 25 Hz grid')
    trim = int(max(abs(SHIFTS)) * RATE)
    anchors = np.arange(trim, len(times) - trim)
    if len(anchors) < 5 * RATE:
        raise ValueError('Fewer than 125 samples on fixed support')
    v = video[anchors]
    shifted = np.array([np.interp(anchors + shift * RATE, np.arange(len(times)), music) for shift in SHIFTS])
    if v.std() < 1e-8 or np.any(shifted.std(axis=1) < 1e-8):
        raise ValueError('Near-constant curve on fixed support')
    scores = np.corrcoef(np.vstack((v, shifted)))[0, 1:]
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite shift scores')
    return scores, len(anchors)


def write_csv(path, rows, fields):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def clean_json(value):
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def run(args):
    source = Path(args.scores)
    clips = list(csv.DictReader(source.open()))
    if len(clips) != len({r['clip_id'] for r in clips}) or len(clips) != args.expected_scored:
        raise ValueError('Unexpected score-table size or duplicate clip IDs')
    if any(r['split'] != 'train' for r in clips):
        raise ValueError('Train score table required')
    eligible = [r for r in clips if r['quality_flag'] == 'ok']
    eligible.sort(key=lambda r: (float(r['pearson_zero_lag']), r['clip_id']))
    low_count = int(np.ceil(.2 * len(eligible)))
    low_ids = {r['clip_id'] for r in eligible[:low_count]}
    if args.limit:
        eligible = sorted(eligible, key=lambda r: r['clip_id'])[:args.limit]
    output = Path(args.output_root).resolve()
    if output.exists():
        raise FileExistsError(f'Output exists: {output}; choose a new root')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f'.{output.name}-', dir=output.parent))
    results, failures, profiles = [], [], []
    config = dict(source_scores_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  expected_scored=args.expected_scored, eligible_before_limit=sum(r['quality_flag'] == 'ok' for r in clips),
                  selected=len(eligible), shifts_seconds=SHIFTS.tolist(), rate_hz=RATE,
                  hann_window=31, trim_each_end_seconds=2, minimum_anchor_samples=125,
                  order='trim padded audio centers; standardize native; interpolate; Hann smooth; trim anchors; shift',
                  lag_sign='positive delta: corr(video(t), music(t+delta)); music events occur later',
                  half_sample_shifts='linear interpolation in sample coordinates; no wrap or extrapolation',
                  baseline='fixed-support zero, not original full-support zero',
                  source_selection='quality_flag == ok in prior analysis; no excluded OOM recovery',
                  bottom_group='lowest ceil(20%) of original quality-ok full-support Pearson, ties by clip ID',
                  tie_break='exact ties: smallest absolute shift, then negative shift',
                  limit=args.limit, video_root=str(Path(args.video_root).resolve()),
                  music_root=str(Path(args.music_root).resolve()),
                  caveats=['Best-of-seven gain is selection-biased; not proof of true delay',
                           'Boundary peaks do not identify an optimum outside the tested range',
                           'Bottom-group gains are selected using original scores and may regress to the mean',
                           'Movie-clustered data; clip IQR is not a confidence interval'])
    fields = ['clip_id', 'movie_id', 'original_quadrant', 'original_bottom_20_percent',
              'original_full_support_r', 'fixed_zero_reference_error', 'anchor_samples',
              *(f'r_at_{shift:+g}s' for shift in SHIFTS), 'best_shift_s', 'best_r', 'gain_over_fixed_zero']
    try:
        (temporary / 'analysis_config.json').write_text(json.dumps(config, indent=2) + '\n')
        for i, row in enumerate(eligible):
            clip, movie = row['clip_id'], row['movie_id']
            try:
                vdir = Path(args.video_root) / 'train' / movie / clip
                mdir = Path(args.music_root) / 'train' / movie / clip
                mm = json.loads((mdir / 'metadata.json').read_text())
                vm = json.loads((vdir / 'metadata.json').read_text())
                if mm['clip_id'] != clip or vm['clip_id'] != clip:
                    raise ValueError('Metadata clip ID mismatch')
                if mm['input_sha256'] != row['audio_sha256'] or vm['input_sha256'] != row['video_input_sha256']:
                    raise ValueError('Input provenance differs from prior scores')
                with np.load(mdir / 'music_raw.npz', allow_pickle=False) as data:
                    mr, mt = data['raw'], data['timestamps']
                t, v, m = aligned_curves(np.load(vdir / f'{clip}_raw_event_curve.npy', allow_pickle=False),
                    np.load(vdir / f'{clip}_native_timestamps.npy', allow_pickle=False), mr, mt, mm['duration'])
                scores, n = shift_scores(t, v, m)
                error = abs(scores[ZERO] - float(row['pearson_zero_lag_fixed_support']))
                if not np.isfinite(error) or error > 1e-10:
                    raise ValueError(f'Fixed-zero mismatch versus prior run: {error}')
                full = float(np.corrcoef(v, m)[0, 1])
                if abs(full - float(row['pearson_zero_lag'])) > 1e-10:
                    raise ValueError('Full-support zero differs from prior run')
                best = max(range(len(SHIFTS)), key=lambda j: (scores[j], -abs(SHIFTS[j]), -SHIFTS[j]))
                result = dict(clip_id=clip, movie_id=movie, original_quadrant=row['quadrant'],
                    original_bottom_20_percent=int(clip in low_ids), original_full_support_r=float(row['pearson_zero_lag']),
                    fixed_zero_reference_error=error, anchor_samples=n,
                    **{f'r_at_{shift:+g}s': float(value) for shift, value in zip(SHIFTS, scores)},
                    best_shift_s=float(SHIFTS[best]), best_r=float(scores[best]),
                    gain_over_fixed_zero=float(scores[best] - scores[ZERO]))
                results.append(result); profiles.append(scores)
            except Exception as exc:
                failures.append(dict(clip_id=clip, error=f'{type(exc).__name__}: {exc}'))
            if (i + 1) % 500 == 0:
                print(json.dumps(dict(visited=i + 1, completed=len(results), failed=len(failures))), flush=True)
        write_csv(temporary / 'per_clip.csv', results, fields)
        write_csv(temporary / 'failures.csv', failures, ['clip_id', 'error'])
        score_matrix = np.array(profiles)
        aggregate = []
        if len(results):
            masks = {'all': np.ones(len(results), dtype=bool),
                     'original_bottom_20_percent': np.array([bool(r['original_bottom_20_percent']) for r in results])}
            for quadrant in sorted({r['original_quadrant'] for r in results}):
                masks[quadrant] = np.array([r['original_quadrant'] == quadrant for r in results])
            for group, mask in masks.items():
                if not mask.any():
                    continue
                subset = score_matrix[mask]
                for j, shift in enumerate(SHIFTS):
                    col = subset[:, j]
                    aggregate.append(dict(group=group, shift_s=float(shift), n=len(col), mean_r=float(col.mean()),
                        median_r=float(np.median(col)), p25_r=float(np.quantile(col, .25)), p75_r=float(np.quantile(col, .75)),
                        mean_change_from_fixed_zero=float((col - subset[:, ZERO]).mean()),
                        mean_absolute_change_from_fixed_zero=float(np.abs(col - subset[:, ZERO]).mean())))
            make_plot(score_matrix, temporary / 'temporal_shift.png')
        write_csv(temporary / 'aggregate.csv', aggregate, ['group', 'shift_s', 'n', 'mean_r', 'median_r',
            'p25_r', 'p75_r', 'mean_change_from_fixed_zero', 'mean_absolute_change_from_fixed_zero'])
        excluded = [dict(clip_id=r['clip_id'], quality_flag=r['quality_flag']) for r in clips if r['quality_flag'] != 'ok']
        write_csv(temporary / 'excluded_quality.csv', excluded, ['clip_id', 'quality_flag'])
        summary = dict(source_scored=len(clips), selected=len(eligible), quality_excluded=len(excluded),
            completed=len(results), failed=len(failures), best_shift_counts=dict(collections.Counter(str(r['best_shift_s']) for r in results)),
            boundary_best_count=sum(abs(r['best_shift_s']) == 2 for r in results),
            max_fixed_zero_reference_error=max((r['fixed_zero_reference_error'] for r in results), default=None))
        if results:
            means = score_matrix.mean(axis=0)
            summary.update(mean_profile=means.tolist(), fixed_zero_mean=float(means[ZERO]),
                mean_gain_over_fixed_zero=float(np.mean([r['gain_over_fixed_zero'] for r in results])),
                note='Descriptive seven-shift diagnostic; no per-clip curve plots or model inference.')
        (temporary / 'summary.json').write_text(json.dumps(clean_json(summary), indent=2, allow_nan=False) + '\n')
        temporary.rename(output)
        print(json.dumps(clean_json(summary)), flush=True)
        return int(bool(failures))
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def make_plot(scores, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    p25, p75 = np.quantile(scores, [.25, .75], axis=0)
    ax.fill_between(SHIFTS, p25, p75, alpha=.15, color='#0f766e', label='Between-clip middle 50% (not CI)')
    ax.plot(SHIFTS, scores.mean(axis=0), 'o-', color='#0f766e', label='Mean')
    ax.plot(SHIFTS, np.median(scores, axis=0), 's--', color='#2563eb', label='Median')
    ax.axvline(0, color='gray', linestyle=':'); ax.axhline(0, color='gray', linewidth=.7)
    ax.set(xticks=SHIFTS, xlabel='Music lag (seconds); positive = music later', ylabel='Fixed-support Pearson r',
           title=f'Temporal shift diagnostic: {len(scores):,} pairs')
    ax.legend(frameon=False); fig.savefig(path, dpi=180); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scores', required=True)
    p.add_argument('--video-root', required=True)
    p.add_argument('--music-root', required=True)
    p.add_argument('--output-root', required=True)
    p.add_argument('--expected-scored', type=int, default=30953)
    p.add_argument('--limit', type=int)
    args = p.parse_args()
    if args.limit is not None and args.limit < 1:
        p.error('--limit must be positive')
    raise SystemExit(run(args))


if __name__ == '__main__':
    main()
