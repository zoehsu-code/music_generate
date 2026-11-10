"""Analyze the seven temporal shifts on the saved OSSL-v2 Test-50 curves."""
import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt


SHIFTS = np.array([-2., -1., -.5, 0., .5, 1., 2.])
ZERO = 3
RATE = 25
TRIM = int(max(abs(SHIFTS)) * RATE)


def shift_scores(path):
    times, video, music = np.loadtxt(path, delimiter=',', skiprows=1, unpack=True)
    if not np.isfinite([times, video, music]).all() or not np.allclose(np.diff(times), 1 / RATE):
        raise ValueError(f'Invalid aligned curves or time grid: {path}')
    anchors = np.arange(TRIM, len(times) - TRIM)
    if len(anchors) < 5 * RATE:
        raise ValueError(f'Fewer than 125 points on the common support: {path}')
    video = video[anchors]
    # Interpolate in sample coordinates so integer shifts reuse the exact saved samples.
    shifted_music = np.array([np.interp(anchors + shift * RATE, np.arange(len(times)), music)
                              for shift in SHIFTS])
    if video.std() < 1e-8 or np.any(shifted_music.std(axis=1) < 1e-8):
        raise ValueError(f'Near-constant curve on the common support: {path}')
    scores = np.corrcoef(np.vstack((video, shifted_music)))[0, 1:]
    return scores, len(anchors)


def plot(scores, low, destination):
    with plt.rc_context({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.labelcolor': '#172033', 'text.color': '#172033'}):
        figure, (individual, aggregate) = plt.subplots(1, 2, figsize=(12.6, 5.5), layout='constrained')
        figure.suptitle('Pearson Correlation vs Temporal Shift', fontsize=20, fontweight='bold')
        for axis in (individual, aggregate):
            axis.axvline(0, color='#64748b', lw=1, linestyle='--', zorder=0)
            axis.axhline(0, color='#cbd5e1', lw=1, zorder=0)
            axis.set_xticks(SHIFTS)
            axis.set_xlabel('Music lag Δt (s)  ·  positive means music occurs later')
            axis.set_ylabel('Pearson r')
            axis.grid(axis='y', color='#e2e8f0', linewidth=.6)

        individual.set_title(f'Per-clip profiles · {len(scores)} pairs', loc='left', fontweight='bold')
        individual.set_ylim(-1, 1)
        individual.plot(SHIFTS, scores[~low].T, color='#64748b', alpha=.26, lw=1)
        individual.plot(SHIFTS, scores[low].T, color='#c2410c', alpha=.5, lw=1.15)
        individual.plot([], [], color='#c2410c', label=f'Original bottom {low.sum()}')
        individual.plot([], [], color='#64748b', label='Other pairs')
        individual.legend(loc='lower left', frameon=False)

        aggregate.set_title('Mean profiles and between-clip spread', loc='left', fontweight='bold')
        lower, upper = np.percentile(scores, [25, 75], axis=0)
        aggregate.fill_between(SHIFTS, lower, upper, color='#0f766e', alpha=.12,
                               label='Middle 50% of all pairs')
        aggregate.plot(SHIFTS, scores.mean(axis=0), 'o-', color='#0f766e', lw=2.4, label='All pairs · mean')
        aggregate.plot(SHIFTS, scores[low].mean(axis=0), 'o-', color='#c2410c', lw=2,
                       label=f'Original bottom {low.sum()} · mean')
        aggregate.legend(loc='best', frameon=False)
        figure.savefig(destination, dpi=180)
        plt.close(figure)


def write_csv(path, fieldnames, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle, lineterminator='\n')
        writer.writerow(fieldnames)
        writer.writerows(rows)


def main():
    root = Path(__file__).resolve().parents[1] / 'reports/ossl-test-50'
    with (root / 'alignment_scores.csv').open(newline='', encoding='utf-8') as handle:
        clips = list(csv.DictReader(handle))
    scores, sample_counts = zip(*(shift_scores(root / Path(clip['figure']).parent / 'aligned_curves.csv')
                                  for clip in clips))
    scores = np.asarray(scores)
    low = np.array([clip['similarity_group'] == 'low' for clip in clips])
    saved_zero = [float(clip['pearson_zero_lag_fixed_support']) for clip in clips]
    np.testing.assert_allclose(scores[:, ZERO], saved_zero, atol=1e-12, rtol=0)
    # Prefer the smallest absolute shift for exact ties, then the negative shift.
    best_indices = [max(range(len(SHIFTS)), key=lambda i: (row[i], -abs(SHIFTS[i]), -SHIFTS[i]))
                    for row in scores]
    best_shifts = SHIFTS[best_indices]
    best_scores = scores[np.arange(len(clips)), best_indices]

    output = root / 'temporal-shift'
    output.mkdir(exist_ok=True)
    write_csv(output / 'per_clip.csv',
              ['clip_id', 'movie_id', 'original_pearson_group', 'original_full_support_r', 'anchor_samples',
               *(f'r_at_{shift:+g}s' for shift in SHIFTS), 'best_shift_s', 'best_r', 'gain_over_fixed_zero'],
              [(clip['clip_id'], clip['movie_id'], clip['similarity_group'], clip['pearson_zero_lag'], n,
                *row, shift, best, best - row[ZERO])
               for clip, n, row, shift, best in zip(clips, sample_counts, scores, best_shifts, best_scores)])
    write_csv(output / 'aggregate.csv',
              ['shift_s', 'n', 'mean_r', 'median_r', 'p25_r', 'p75_r', 'original_low_group_n',
               'original_low_group_mean_r', 'mean_absolute_change_from_fixed_zero'],
              [(shift, len(clips), column.mean(), np.median(column), *np.percentile(column, [25, 75]),
                low.sum(), scores[low, i].mean(), np.abs(column - scores[:, ZERO]).mean())
               for i, (shift, column) in enumerate(zip(SHIFTS, scores.T))])
    plot(scores, low, output / 'pearson_vs_temporal_shift.png')
    print(f'Wrote {len(clips)} clips × {len(SHIFTS)} shifts to {output}')


if __name__ == '__main__':
    main()
