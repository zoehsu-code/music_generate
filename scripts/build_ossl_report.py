"""Rebuild OSSL metrics and static report from saved aligned curves only.

Run: python3 scripts/build_ossl_report.py [--report-dir reports/ossl-test-50]
No model, video, audio, or network dependencies are used.
"""
import argparse
import csv
import html
import json
import math
from pathlib import Path
import statistics

MEAN = 'mean_absolute_difference'
MEDIAN = 'median_absolute_difference'
METHOD = ('Absolute differences use the same saved, independently standardized, '
          '25 Hz aligned and 1.2-second Hann-smoothed curves as the primary Pearson score, '
          'at zero lag over the full common interval. For each sample, d(t) = |video(t) - music(t)|; '
          'mean absolute difference = mean(d), median absolute difference = median(d). '
          'Lower is better (0 means identical curves), in standardized curve units, not raw audio/video units. '
          'Pearson is higher-is-better and measures co-variation; absolute difference also reflects amplitude mismatch. '
          'Smoothing changes amplitude, so these values should only be compared with the same preprocessing. '
          'Ranks and high/low groups remain based on Pearson; neither metric measures perceptual music quality.')


def absolute_difference(video, music):
    if not video or len(video) != len(music):
        raise ValueError('Curves must be nonempty and have equal length')
    if not all(math.isfinite(x) for x in video + music):
        raise ValueError('Curves must contain only finite values')
    differences = [abs(v - m) for v, m in zip(video, music)]
    return statistics.mean(differences), statistics.median(differences), differences


def svg_plot(path, series, title, xlabel, ylabel):
    """Small dependency-free line/scatter plot with explicit labels."""
    points = [(x, y) for _, _, values in series for x, y in values]
    xmin, xmax = min(x for x, _ in points), max(x for x, _ in points)
    ymin, ymax = min(0, min(y for _, y in points)), max(y for _, y in points)
    def xy(x, y):
        return (75 + 825 * (x-xmin)/(xmax-xmin or 1), 320 - 255 * (y-ymin)/(ymax-ymin or 1))
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 410" role="img">',
             f'<title>{html.escape(title)}</title><rect width="960" height="410" fill="white"/>',
             '<g font-family="sans-serif" font-size="14" fill="#172033">',
             f'<text x="75" y="25" font-size="19">{html.escape(title)}</text>',
             '<path d="M75 55 V320 H910" fill="none" stroke="#888"/>']
    for i in range(6):
        x = xmin + (xmax-xmin)*i/5
        y = ymin + (ymax-ymin)*i/5
        px, _ = xy(x, ymin)
        _, py = xy(xmin, y)
        parts += [f'<text x="{px:.1f}" y="342" text-anchor="middle">{x:.2f}</text>',
                  f'<text x="65" y="{py:.1f}" text-anchor="end">{y:.2f}</text>']
    for i, (label, color, values) in enumerate(series):
        coords = ' '.join(f'{xy(x,y)[0]:.2f},{xy(x,y)[1]:.2f}' for x,y in values)
        parts += [f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="1.5"/>',
                  f'<text x="{100+i*380}" y="385" fill="{color}">{html.escape(label)}</text>']
    parts += [f'<text x="480" y="365" text-anchor="middle">{html.escape(xlabel)}</text>',
              f'<text transform="translate(18 200) rotate(-90)" text-anchor="middle">{html.escape(ylabel)}</text>', '</g></svg>']
    path.write_text('\n'.join(parts))


def build(root):
    scores = root / 'alignment_scores.csv'
    with scores.open() as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames)
        rows = list(reader)
    for field in (MEAN, MEDIAN):
        if field not in fields:
            fields.append(field)
    prepared = []
    # Validate all saved inputs before changing any output.
    for row in rows:
        directory = root / Path(row['figure']).parent
        with (directory / 'aligned_curves.csv').open() as f:
            curves = list(csv.DictReader(f))
        times = [float(c['time_seconds']) for c in curves]
        video = [float(c['video_curve']) for c in curves]
        music = [float(c['music_curve']) for c in curves]
        if not all(math.isfinite(t) for t in times) or any(b <= a for a,b in zip(times,times[1:])):
            raise ValueError(f'Invalid time grid: {directory}')
        mean, median, differences = absolute_difference(video, music)
        row.update({MEAN: mean, MEDIAN: median})
        prepared.append((directory, times, differences, mean, median))
    for directory, times, differences, mean, median in prepared:
        with (directory / 'absolute_difference.csv').open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time_seconds', 'absolute_difference'])
            writer.writerows(zip(times, differences))
        svg_plot(directory / 'absolute_difference.svg', [('Absolute difference', '#7c3aed', list(zip(times,differences)))],
                 f'Mean {mean:.3f} | Median {median:.3f} — lower is better', 'Time (s)', '|video - music|')
    with scores.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    svg_plot(root / 'absolute_difference_overview.svg', [
        ('Mean absolute difference', '#7c3aed', [(i, r[MEAN]) for i,r in enumerate(rows,1)]),
        ('Median absolute difference', '#059669', [(i, r[MEDIAN]) for i,r in enumerate(rows,1)])],
        'Absolute difference by Pearson rank — lower is better', 'Pearson rank (unchanged)', 'Absolute difference')
    for filename, updates in [
        ('analysis_config.json', {'absolute_difference': {'definition': METHOD, 'source': 'saved aligned_curves.csv', 'encoder_rerun': False}}),
        ('summary.json', {'mean_of_sample_mean_absolute_difference': statistics.mean(r[MEAN] for r in rows),
                          'median_of_sample_median_absolute_difference': statistics.median(r[MEDIAN] for r in rows)})]:
        path = root / filename
        data = json.loads(path.read_text())
        data.update(updates)
        path.write_text(json.dumps(data, indent=2) + '\n')
    readme = root / 'README.md'
    text = readme.read_text().split('\n## Absolute Difference')[0]
    text += '\n## Absolute Difference\n\n' + METHOD + '\n\n![Absolute difference overview](absolute_difference_overview.svg)\n\n'
    text += '| Clip ID | Mean absolute difference ↓ | Median absolute difference ↓ | Difference plot |\n| --- | ---: | ---: | --- |\n'
    for r in rows:
        plot = str(Path(r['figure']).parent / 'absolute_difference.svg')
        text += f"| {r['clip_id']} | {r[MEAN]:.6f} | {r[MEDIAN]:.6f} | [Plot]({plot}) |\n"
    text += '\nRegenerate without encoders: `python3 scripts/build_ossl_report.py` from the repository root.\n'
    readme.write_text(text)
    report = root / 'report.html'
    text = report.read_text().split('<!-- absolute-difference -->')[0]
    # Preserve the original Pearson report verbatim, and append the new metrics.
    text += '<!-- absolute-difference -->\n<h2>Absolute Difference</h2><p>' + html.escape(METHOD) + '</p>'
    text += '<img alt="Mean and median absolute difference by Pearson rank" src="absolute_difference_overview.svg">'
    text += '<div style="overflow-x:auto"><table><tr><th>Pearson rank</th><th>Clip ID</th><th>Mean absolute difference ↓</th><th>Median absolute difference ↓</th><th>Difference curve</th></tr>'
    for i,r in enumerate(rows,1):
        base = html.escape(str(Path(r['figure']).parent))
        text += f'<tr><td>{i}</td><td>{html.escape(r["clip_id"])}</td><td>{r[MEAN]:.6f}</td><td>{r[MEDIAN]:.6f}</td><td><a href="{base}/absolute_difference.svg">Plot</a> · <a href="{base}/absolute_difference.csv">CSV</a></td></tr>'
    text += '</table></div>\n'
    report.write_text(text)
    (root / 'index.html').write_text('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0; url=report.html"><a href="report.html">OSSL test-50 report</a>\n')
    print(f'Regenerated {len(rows)} samples from saved curves; no encoders run.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', type=Path, default=Path('reports/ossl-test-50'))
    build(parser.parse_args().report_dir)
