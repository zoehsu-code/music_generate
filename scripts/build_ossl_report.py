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
PEARSON = 'pearson_zero_lag'
SOURCE = ('https://huggingface.co/datasets/McAuley-Lab/OSSL-v2/blob/'
          '3ed5a37d933084e5ec134e2138b8faa156cf021b')
QUADRANTS = (
    ('high-low', 'High Pearson · Low difference', '高相关 · 小差异', '#0f766e'),
    ('high-high', 'High Pearson · High difference', '高相关 · 大差异', '#a16207'),
    ('low-low', 'Low Pearson · Low difference', '低相关 · 小差异', '#1d4ed8'),
    ('low-high', 'Low Pearson · High difference', '低相关 · 大差异', '#a21caf'),
)
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


def build_quadrants(root, rows):
    pearson_cut = statistics.median(float(row[PEARSON]) for row in rows)
    difference_cut = statistics.median(row[MEAN] for row in rows)
    groups = {key: [] for key, *_ in QUADRANTS}
    for row in rows:
        pearson = 'high' if float(row[PEARSON]) >= pearson_cut else 'low'
        difference = 'low' if row[MEAN] <= difference_cut else 'high'
        groups[f'{pearson}-{difference}'].append(row)

    # Round the chart bounds outward so the axes cover all samples.
    xmin = math.floor(min(float(row[PEARSON]) for row in rows) * 5) / 5
    xmax = math.ceil(max(float(row[PEARSON]) for row in rows) * 5) / 5
    ymax = math.ceil(max(row[MEAN] for row in rows) * 5) / 5

    def x(value):
        return 62 + 620 * (value - xmin) / (xmax - xmin or 1)

    def y(value):
        return 318 - 288 * value / (ymax or 1)

    chart = ['<svg viewBox="0 0 720 375" role="group" aria-label="Interactive scatter plot">',
             '<rect x="62" y="30" width="620" height="288" fill="#f8fafc" stroke="#cbd5e1"/>']
    for tick in range(round(xmin * 5), round(xmax * 5) + 1):
        value = tick / 5
        chart.append(f'<text x="{x(value):.1f}" y="338" text-anchor="middle">{value:g}</text>')
    for tick in range(round(ymax * 5) + 1):
        value = tick / 5
        chart.append(f'<text x="52" y="{y(value) + 4:.1f}" text-anchor="end">{value:g}</text>')
    chart += [f'<path d="M{x(pearson_cut):.2f} 30V318 M62 {y(difference_cut):.2f}H682" '
              'fill="none" stroke="#64748b" stroke-dasharray="5 5"/>',
              '<text x="372" y="364" text-anchor="middle">Zero-lag Pearson r</text>',
              '<text transform="translate(17 174) rotate(-90)" text-anchor="middle">Mean absolute difference</text>']
    links, sections = [], []
    for key, label, chinese, color in QUADRANTS:
        samples = sorted(groups[key], key=lambda row: (-float(row[PEARSON]), row['clip_id']))
        links.append(f'<a href="#{key}" style="--accent:{color}"><strong>{len(samples)}</strong> '
                     f'{label}</a>')
        table = []
        for row in samples:
            clip = html.escape(row['clip_id'])
            pearson, difference = float(row[PEARSON]), row[MEAN]
            video = html.escape(f"{SOURCE}/{row['split']}/video/{row['movie_id']}/{row['clip_id']}.mp4")
            figure = html.escape(row['figure'])
            chart.append(f'<a href="#clip-{clip}" aria-label="{clip}, Pearson {pearson:.3f}, '
                         f'mean difference {difference:.3f}"><circle cx="{x(pearson):.2f}" '
                         f'cy="{y(difference):.2f}" r="5" fill="{color}"><title>{clip} · '
                         f'r {pearson:.3f} · difference {difference:.3f}</title></circle></a>')
            table.append(f'<tr id="clip-{clip}"><th scope="row">{clip}</th><td>{pearson:.4f}</td>'
                         f'<td>{difference:.4f}</td><td><a href="{video}" target="_blank" '
                         f'rel="noopener">原视频 ↗</a></td><td><a href="{figure}">曲线对比</a></td></tr>')
        sections.append(f'<section id="{key}" style="--accent:{color}"><h2>{chinese} '
                        f'<span>· {len(samples)} 对</span></h2><div class="scroll"><table><thead><tr>'
                        '<th scope="col">Clip ID</th><th scope="col">Pearson r</th>'
                        '<th scope="col">平均绝对差异</th><th scope="col">Hugging Face</th>'
                        '<th scope="col">对齐曲线</th></tr></thead><tbody>'
                        + '\n'.join(table) + '</tbody></table></div></section>')
    chart.append('</svg>')
    (root / 'quadrants.html').write_text(f'''<!doctype html>
<html lang="zh-Hans"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OSSL-v2 Test-50 · Pearson × 平均绝对差异</title>
<style>
body{{font:16px/1.6 system-ui,sans-serif;background:#f8fafc;color:#172033;margin:0}}
main{{max-width:1000px;margin:auto;padding:28px 18px 60px}} a{{color:#1d4ed8;text-underline-offset:3px}}
h1{{font-size:clamp(1.9rem,5vw,3rem);line-height:1.16;margin:.35em 0}} p{{margin:.7em 0}}
.intro,.chart,section{{background:white;border:1px solid #dbe4ee;border-radius:14px;padding:clamp(14px,3vw,28px);margin:18px 0}}
.groups{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;margin-top:22px}}
.groups a{{color:var(--accent);padding:10px 12px;border:1px solid #dbe4ee;border-left:4px solid var(--accent);border-radius:8px}}
.groups strong{{display:block;font-size:1.6rem;line-height:1.15}} svg{{display:block;width:100%;height:auto}}
svg text{{font:13px system-ui;fill:#334155}} circle{{stroke:white;stroke-width:1.2}} a:hover circle,a:focus circle{{r:8;stroke:#172033}}
section h2{{color:var(--accent);margin-top:0}} h2 span{{font-size:.8em;color:#475569}}
.scroll{{overflow-x:auto}} table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}
th,td{{padding:10px;text-align:left;border-bottom:1px solid #e2e8f0;white-space:nowrap}}
tbody th{{font:inherit}} thead{{background:#f1f5f9}} tr:target{{background:#fef3c7}}
</style>
<main>
<a href="report.html">← 原始 Test-50 报告</a>
<header class="intro"><p>OSSL-v2 · Test-50 · {len(rows)} 对</p><h1>Pearson × 平均绝对差异</h1>
<p>用这批样本各自的中位数作分界。零延迟 Pearson ≥ <strong>{pearson_cut:.6f}</strong> 为高相关，
平均绝对差异 ≤ <strong>{difference_cut:.6f}</strong> 为小差异。归组使用未四舍五入的数值。</p>
<p>两项指标使用同一套 25 Hz 对齐曲线和 1.2 秒 Hann 平滑。高低是这 50 对样本内的相对分组。
样本来自三部电影。平均绝对差异是每个片段逐时间点绝对差的均值。</p>
<nav class="groups" aria-label="四象限分组">{''.join(links)}</nav></header>
<div class="chart"><p>点散点可跳到对应样本。表格里的原视频链接会打开 Hugging Face。</p>
{''.join(chart)}</div>
{''.join(sections)}
</main></html>
''', encoding='utf-8')


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
    build_quadrants(root, rows)
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
