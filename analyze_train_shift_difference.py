"""Absolute curve differences on the exact successful seven-shift Pearson cohort."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
from analyze_train_temporal_shift import (SHIFTS, RATE, ZERO, aligned_curves,
                                         shift_scores, write_csv)


def difference_profiles(times, video, music):
    """Return per-shift mean/median |video(t)-music(t+delta)|; no renormalization."""
    # Reuse the original support, finite-value and constant-curve checks.
    correlations, n = shift_scores(times, video, music)
    trim = int(max(abs(SHIFTS)) * RATE)
    anchors = np.arange(trim, len(times) - trim)
    shifted = np.array([np.interp(anchors + s * RATE, np.arange(len(times)), music)
                        for s in SHIFTS])
    differences = np.abs(video[anchors][None, :] - shifted)
    return differences.mean(axis=1), np.median(differences, axis=1), correlations, n


def aggregate_profiles(rows, matrices):
    """Clip summaries have equal weight; movie means give each movie equal weight."""
    aggregates = []
    groups = {'all': np.ones(len(rows), dtype=bool)}
    for q in sorted({r['original_quadrant'] for r in rows}):
        groups[q] = np.array([r['original_quadrant'] == q for r in rows])
    movies = np.array([r['movie_id'] for r in rows])
    for group, mask in groups.items():
        if not mask.any():
            continue
        mids = movies[mask]
        for metric, matrix in matrices.items():
            subset = matrix[mask]
            movie_profiles = np.array([subset[mids == m].mean(axis=0) for m in sorted(set(mids))])
            for j, shift in enumerate(SHIFTS):
                col = subset[:, j]
                aggregates.append(dict(group=group, metric=metric, shift_s=float(shift),
                    n_clips=len(col), n_movies=len(movie_profiles),
                    clip_weighted_mean=float(col.mean()), median_across_clips=float(np.median(col)),
                    p25=float(np.quantile(col, .25)), p75=float(np.quantile(col, .75)),
                    equal_movie_mean=float(movie_profiles[:, j].mean()),
                    mean_change_from_fixed_zero=float((col - subset[:, ZERO]).mean())))
    return aggregates


def make_plot(aggregates, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, metric, title in zip(axes, ('mae', 'median_ae'),
                                 ('Mean absolute difference', 'Median absolute difference')):
        rows = [r for r in aggregates if r['group'] == 'all' and r['metric'] == metric]
        ax.fill_between(SHIFTS, [r['p25'] for r in rows], [r['p75'] for r in rows],
                        color='#0f766e', alpha=.13, label='Between-clip IQR (not CI)')
        ax.plot(SHIFTS, [r['clip_weighted_mean'] for r in rows], 'o-', label='Clip-weighted mean')
        ax.plot(SHIFTS, [r['equal_movie_mean'] for r in rows], 's--', label='Equal-movie mean')
        ax.axvline(0, color='gray', linestyle=':')
        ax.set(xticks=SHIFTS, xlabel='Music lag (s); positive = later music',
               ylabel='Absolute difference (standardized curve units)', title=title + ' (lower is better)')
        ax.legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def read_rows(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    prior = read_rows(args.shift_scores)
    original = read_rows(args.scores)
    if len(prior) != args.expected_completed or len({r['clip_id'] for r in prior}) != len(prior):
        raise ValueError('Unexpected previous successful cohort size or duplicate IDs')
    if len({r['clip_id'] for r in original}) != len(original):
        raise ValueError('Duplicate original score IDs')
    originals = {r['clip_id']: r for r in original}
    for row in prior:
        ref = originals.get(row['clip_id'])
        if ref is None or ref['split'] != 'train' or ref['quality_flag'] != 'ok' or ref['movie_id'] != row['movie_id']:
            raise ValueError('Previous cohort is not a subset of original quality-ok Train scores')
        if not np.isfinite([float(row[f'r_at_{s:+g}s']) for s in SHIFTS]).all():
            raise ValueError('Nonfinite prior Pearson score')
    selected = sorted(prior, key=lambda r: r['clip_id'])
    if args.limit is not None:
        selected = selected[:args.limit]
    output = Path(args.output_root).resolve()
    if output.exists():
        raise FileExistsError(f'Output exists: {output}; choose a new root')
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f'.{output.name}-', dir=output.parent))
    rows, failures, means, medians = [], [], [], []
    max_error = 0.
    config = dict(original_scores=str(Path(args.scores).resolve()), original_scores_sha256=sha256(args.scores),
        prior_shift_scores=str(Path(args.shift_scores).resolve()), prior_shift_scores_sha256=sha256(args.shift_scores),
        expected_completed=args.expected_completed, selected=len(selected), limit=args.limit,
        video_root=str(Path(args.video_root).resolve()), music_root=str(Path(args.music_root).resolve()),
        shifts_seconds=SHIFTS.tolist(), rate_hz=RATE, hann_window=31,
        trim_each_end_seconds=2, minimum_anchor_samples=125,
        preprocessing='Remove padded audio centers; standardize native curves separately; interpolate to shared 25 Hz grid; Hann-31; fixed anchors; shift music.',
        lag_sign='positive delta compares video(t) with music(t+delta)',
        mean_absolute_difference='mean_t abs(video(t)-music(t+delta)) on fixed anchors',
        median_absolute_difference='median_t abs(video(t)-music(t+delta)) on fixed anchors',
        units='Separately standardized native-curve units after smoothing; no per-shift renormalization.',
        cohort='Exact successful IDs in prior per_clip.csv; deferred OOM/review/previous failures remain excluded.',
        baseline='Fixed-support zero; not the older full-support absolute difference.',
        validation='All seven Pearson values and anchor count must reconstruct the prior run (atol 1e-10). Metadata input hashes must match original scores; media hashes are not recomputed.',
        tie_break='Exact ties: smallest absolute shift, then negative shift.',
        caveats=['Best-of-seven reductions are selection-biased, not evidence of a true delay.',
                 'Between-clip IQR is not a confidence interval.',
                 'Median across clips and median over timestamps are different operations.'])
    try:
        (temp / 'analysis_config.json').write_text(json.dumps(config, indent=2) + '\n')
        for i, old in enumerate(selected):
            clip, movie = old['clip_id'], old['movie_id']
            try:
                ref = originals[clip]
                vd = Path(args.video_root) / 'train' / movie / clip
                md = Path(args.music_root) / 'train' / movie / clip
                vm = json.loads((vd / 'metadata.json').read_text())
                mm = json.loads((md / 'metadata.json').read_text())
                if vm['clip_id'] != clip or mm['clip_id'] != clip:
                    raise ValueError('Metadata clip ID mismatch')
                if vm['input_sha256'] != ref['video_input_sha256'] or mm['input_sha256'] != ref['audio_sha256']:
                    raise ValueError('Input provenance differs from original scores')
                with np.load(md / 'music_raw.npz', allow_pickle=False) as data:
                    mr, mt = data['raw'], data['timestamps']
                t, v, m = aligned_curves(np.load(vd / f'{clip}_raw_event_curve.npy', allow_pickle=False),
                    np.load(vd / f'{clip}_native_timestamps.npy', allow_pickle=False), mr, mt, mm['duration'])
                mae, med, corr, n = difference_profiles(t, v, m)
                saved = np.array([float(old[f'r_at_{s:+g}s']) for s in SHIFTS])
                error = float(np.max(np.abs(corr - saved)))
                if error > 1e-10 or n != int(old['anchor_samples']):
                    raise ValueError(f'Prior shift reconstruction mismatch: max Pearson error {error}, anchors {n}')
                result = dict(clip_id=clip, movie_id=movie, original_quadrant=ref['quadrant'],
                              anchor_samples=n, max_pearson_reference_error=error)
                for metric, values in (('mae', mae), ('median_ae', med)):
                    result.update({f'{metric}_at_{s:+g}s': float(value) for s, value in zip(SHIFTS, values)})
                    best = min(range(len(SHIFTS)), key=lambda j: (values[j], abs(SHIFTS[j]), SHIFTS[j]))
                    result.update({f'best_{metric}_shift_s': float(SHIFTS[best]),
                                   f'min_{metric}': float(values[best]),
                                   f'{metric}_reduction_from_fixed_zero': float(values[ZERO] - values[best])})
                rows.append(result); means.append(mae); medians.append(med)
                max_error = max(max_error, error)
            except Exception as exc:
                failures.append(dict(clip_id=clip, error=f'{type(exc).__name__}: {exc}'))
            if (i + 1) % 500 == 0:
                print(json.dumps(dict(visited=i + 1, completed=len(rows), failed=len(failures))), flush=True)
        fields = ['clip_id', 'movie_id', 'original_quadrant', 'anchor_samples', 'max_pearson_reference_error']
        for metric in ('mae', 'median_ae'):
            fields += [*(f'{metric}_at_{s:+g}s' for s in SHIFTS), f'best_{metric}_shift_s',
                       f'min_{metric}', f'{metric}_reduction_from_fixed_zero']
        write_csv(temp / 'per_clip.csv', rows, fields)
        write_csv(temp / 'failures.csv', failures, ['clip_id', 'error'])
        aggregates = aggregate_profiles(rows, {'mae': np.array(means), 'median_ae': np.array(medians)}) if rows else []
        write_csv(temp / 'aggregate.csv', aggregates, ['group', 'metric', 'shift_s', 'n_clips', 'n_movies',
            'clip_weighted_mean', 'median_across_clips', 'p25', 'p75', 'equal_movie_mean', 'mean_change_from_fixed_zero'])
        if rows:
            make_plot(aggregates, temp / 'absolute_difference_shift.png')
        summary = dict(prior_completed=len(prior), selected=len(selected), completed=len(rows), failed=len(failures),
            identical_to_selected_prior_cohort=not failures,
            max_pearson_reference_error=max_error if rows else None,
            aggregate_all=[r for r in aggregates if r['group'] == 'all'],
            best_mae_shift_counts=dict(collections.Counter(str(r['best_mae_shift_s']) for r in rows)),
            best_median_ae_shift_counts=dict(collections.Counter(str(r['best_median_ae_shift_s']) for r in rows)),
            note='Saved-curve CPU analysis only; original outputs unchanged; inspect failures before comparing cohorts.')
        (temp / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
        temp.rename(output)
        print(json.dumps(summary), flush=True)
        return int(bool(failures))
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scores', required=True)
    p.add_argument('--shift-scores', required=True)
    p.add_argument('--video-root', required=True)
    p.add_argument('--music-root', required=True)
    p.add_argument('--output-root', required=True)
    p.add_argument('--expected-completed', type=int, default=30944)
    p.add_argument('--limit', type=int)
    args = p.parse_args()
    if args.expected_completed < 1 or (args.limit is not None and args.limit < 1):
        p.error('Counts and --limit must be positive')
    raise SystemExit(run(args))


if __name__ == '__main__':
    main()
