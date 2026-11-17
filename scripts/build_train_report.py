"""Build an English standalone report from the complete saved Train score table."""
import argparse
import base64
import collections
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def build(source, out):
    out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader((source / 'alignment_scores.csv').open()))
    summary = json.loads((source / 'summary.json').read_text())
    config = json.loads((source / 'analysis_config.json').read_text())
    assert len(rows) == summary['scored'] == len({r['clip_id'] for r in rows})
    eligible = [r for r in rows if r['quality_flag'] == 'ok']
    assert len(eligible) == summary['eligible']
    counts = dict(collections.Counter(r['quadrant'] for r in rows))
    assert counts == summary['quadrant_counts']
    def values(key):
        a = np.array([float(r[key]) for r in eligible])
        assert np.isfinite(a).all(), key
        return a
    r, d = values('pearson_zero_lag'), values('mean_absolute_difference')
    rt, dt = float(np.median(r)), float(np.median(d))
    assert abs(rt - summary['pearson_threshold']) < 1e-12
    assert abs(dt - summary['mean_absolute_difference_threshold']) < 1e-12
    for row in eligible:
        expected = ('high_correlation' if float(row['pearson_zero_lag']) >= rt else 'low_correlation') + ('_low_difference' if float(row['mean_absolute_difference']) <= dt else '_high_difference')
        assert row['quadrant'] == expected
    assert summary['scored'] + summary['excluded'] + summary['failed'] == summary['expected']
    assert np.all((r >= -1) & (r <= 1)) and np.all(d >= 0)
    provenance = {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                  for name in ['alignment_scores.csv', 'analysis_config.json', 'summary.json']}
    for name in provenance:
        shutil.copy2(source / name, out / name)
    metrics = ['pearson_zero_lag', 'mean_absolute_difference', 'median_absolute_difference',
               'pearson_unsmoothed', 'pearson_smooth_2_4s', 'music_duration_seconds']
    stats = {}
    for key in metrics:
        a = values(key)
        stats[key] = dict(mean=float(a.mean()), minimum=float(a.min()), maximum=float(a.max()),
                          p05=float(np.quantile(a, .05)), p25=float(np.quantile(a, .25)),
                          median=float(np.median(a)), p75=float(np.quantile(a, .75)), p95=float(np.quantile(a, .95)))
    positive, strong = int(sum(r > 0)), int(sum(r >= .5))
    relation = float(np.corrcoef(r, d)[0, 1])
    movie_groups = collections.defaultdict(list)
    for row in eligible:
        movie_groups[row['movie_id']].append(float(row['pearson_zero_lag']))
    movie_medians = np.array([np.median(v) for v in movie_groups.values()])
    stats.update(positive_pearson=positive, pearson_ge_0_3=int(sum(r >= .3)), pearson_ge_0_5=strong,
                 pearson_le_minus_0_3=int(sum(r <= -.3)), movie_count=len(movie_groups),
                 equal_movie_median_of_medians=float(np.median(movie_medians)),
                 pearson_between_r_and_mean_difference=relation,
                 quality_flags=dict(collections.Counter(x['quality_flag'] for x in rows)), source_sha256=provenance)
    (out / 'report_statistics.json').write_text(json.dumps(stats, indent=2) + '\n')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.facecolor': 'white', 'axes.titleweight': 'bold'})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    axes[0].hist(r, bins=np.linspace(-1, 1, 61), color='#2563a6', edgecolor='white', linewidth=.3)
    axes[0].axvline(rt, color='#be123c', ls='--', label=f'Median {rt:.4f}')
    axes[0].set(title='Zero-lag Pearson distribution', xlabel='Pearson r (Hann 31)', ylabel='Eligible pair count')
    axes[0].legend(frameon=False)
    axes[1].hist(d, bins=60, color='#0f766e', edgecolor='white', linewidth=.3)
    axes[1].axvline(dt, color='#be123c', ls='--', label=f'Median {dt:.4f}')
    axes[1].set(title='Mean absolute difference distribution', xlabel='Mean |video − music| (standardized units)', ylabel='Eligible pair count')
    axes[1].legend(frameon=False)
    fig.savefig(out / 'distributions.png', dpi=180); plt.close(fig)
    colors = {'high_correlation_low_difference': '#0f766e', 'high_correlation_high_difference': '#b45309',
              'low_correlation_low_difference': '#2563eb', 'low_correlation_high_difference': '#be123c'}
    fig, ax = plt.subplots(figsize=(9, 5.6), constrained_layout=True)
    for q, color in colors.items():
        mask = np.array([x['quadrant'] == q for x in eligible])
        ax.scatter(r[mask], d[mask], s=4, alpha=.18, color=color, rasterized=True)
    ax.axvline(rt, color='#172033', ls='--', lw=1); ax.axhline(dt, color='#172033', ls='--', lw=1)
    ax.set(xlabel='Zero-lag Pearson r', ylabel='Mean absolute difference (standardized units)',
           title=f'All {len(eligible):,} eligible pairs · median thresholds', xlim=(-1, 1))
    fig.savefig(out / 'quadrants.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    samples = [values('pearson_unsmoothed'), r, values('pearson_smooth_2_4s')]
    axes[0].boxplot(samples, tick_labels=['No smoothing', 'Hann 31 (1.2 s)', 'Hann 61 (2.4 s)'], showfliers=False)
    axes[0].axhline(0, color='gray', lw=.8); axes[0].set(ylabel='Zero-lag Pearson r', title='Sensitivity to smoothing')
    axes[1].hexbin(r, values('pearson_smooth_2_4s'), gridsize=55, mincnt=1, bins='log', cmap='Blues')
    axes[1].plot([-1, 1], [-1, 1], '--', color='#be123c', lw=1)
    axes[1].set(xlabel='Pearson, Hann 31', ylabel='Pearson, Hann 61', title='Same clips under two smoothing scales')
    fig.savefig(out / 'sensitivity.png', dpi=180); plt.close(fig)
    def image(name):
        return '<img alt="' + name + '" src="data:image/png;base64,' + base64.b64encode((out / name).read_bytes()).decode() + '">'
    def fmt(x):
        return f'{x:.4f}'
    names = dict(pearson_zero_lag='Pearson (Hann 31)', mean_absolute_difference='Mean absolute difference',
                 median_absolute_difference='Median absolute difference', pearson_unsmoothed='Pearson (no smoothing)',
                 pearson_smooth_2_4s='Pearson (Hann 61)', music_duration_seconds='Music duration (seconds)')
    stat_rows = ''.join('<tr><td>' + names[k] + '</td>' + ''.join('<td>' + fmt(stats[k][col]) + '</td>' for col in ['mean', 'p05', 'p25', 'median', 'p75', 'p95']) + '</tr>' for k in metrics)
    labels = {q: q.replace('_', ' ').replace('correlation ', 'correlation / ') for q in colors}
    quadrant_rows = ''.join(f'<tr><td>{labels[q].capitalize()}</td><td>{counts[q]:,}</td><td>{counts[q]/len(eligible):.2%}</td></tr>' for q in colors)
    rev = config.get('dataset_revision', '3ed5a37d933084e5ec134e2138b8faa156cf021b')
    def link(row):
        return f'https://huggingface.co/datasets/McAuley-Lab/OSSL-v2/blob/{rev}/train/video/{row["movie_id"]}/{row["clip_id"]}.mp4'
    def example_table(title, selection):
        entries = ''.join(f'<tr><td><a target="_blank" rel="noopener" href="{link(x)}">{html.escape(x["clip_id"])}</a></td><td>{float(x["pearson_zero_lag"]):.4f}</td><td>{float(x["mean_absolute_difference"]):.4f}</td><td>{float(x["music_duration_seconds"]):.2f}</td></tr>' for x in selection)
        return f'<h3>{title}</h3><div class="tablewrap"><table><tr><th>Clip ID</th><th>Pearson</th><th>Mean difference</th><th>Seconds</th></tr>{entries}</table></div>'
    examples = example_table('Highest zero-lag Pearson (10 eligible pairs)', sorted(eligible, key=lambda x: -float(x['pearson_zero_lag']))[:10])
    examples += example_table('Lowest zero-lag Pearson (10 eligible pairs)', sorted(eligible, key=lambda x: float(x['pearson_zero_lag']))[:10])
    # Within-quadrant examples near the quadrant median avoid presenting only extremes.
    for q in colors:
        members = [x for x in eligible if x['quadrant'] == q]
        qr = np.median([float(x['pearson_zero_lag']) for x in members])
        qd = np.median([float(x['mean_absolute_difference']) for x in members])
        representatives = sorted(members, key=lambda x: ((float(x['pearson_zero_lag'])-qr)/.2)**2 + ((float(x['mean_absolute_difference'])-qd)/.1)**2)[:3]
        examples += example_table(labels[q].capitalize() + ' — near group medians', representatives)
    data = [[x['clip_id'], x['movie_id'], float(x['pearson_zero_lag']) if x['pearson_zero_lag'] else None,
             float(x['mean_absolute_difference']), float(x['median_absolute_difference']), x['quadrant'],
             x['quality_flag'], float(x['music_duration_seconds'])] for x in rows]
    js = '''const rows=DATA;let page=0,filtered=[];const size=50;
const $=s=>document.querySelector(s);function render(reset=true){if(reset)page=0;let q=$('#search').value.trim().toLowerCase(),g=$('#group').value;filtered=rows.filter(x=>(!q||x[0].toLowerCase().includes(q)||x[1]===q)&&(!g||x[5]===g));let sort=$('#sort').value;filtered.sort((a,b)=>sort==='low'?(a[2]??2)-(b[2]??2):sort==='diff'?a[3]-b[3]:(b[2]??-2)-(a[2]??-2));$('#body').innerHTML=filtered.slice(page*size,(page+1)*size).map(x=>`<tr><td><a target="_blank" rel="noopener" href="https://huggingface.co/datasets/McAuley-Lab/OSSL-v2/blob/REV/train/video/${x[1]}/${x[0]}.mp4">${x[0]}</a></td><td>${x[2]===null?'n/a':x[2].toFixed(4)}</td><td>${x[3].toFixed(4)}</td><td>${x[4].toFixed(4)}</td><td>${x[5].replaceAll('_',' ')}</td><td>${x[6]}</td></tr>`).join('');$('#page').textContent=`${filtered.length.toLocaleString()} pairs · Page ${page+1} / ${Math.max(1,Math.ceil(filtered.length/size))}`;$('#prev').disabled=page===0;$('#next').disabled=(page+1)*size>=filtered.length;}
['#search','#group','#sort'].forEach(s=>$(s).addEventListener('input',()=>render()));$('#prev').onclick=()=>{page--;render(false)};$('#next').onclick=()=>{page++;render(false)};render();'''.replace('DATA', json.dumps(data, separators=(',', ':'), allow_nan=False)).replace('REV', rev)
    css = '''body{font:16px/1.7 system-ui,sans-serif;color:#172033;background:#f1f4f8;margin:0}main{max-width:1100px;margin:36px auto;background:white;padding:46px;border-radius:14px}h1{font-size:43px;line-height:1.15;letter-spacing:-1px}h2{margin-top:42px;color:#183b56}h3{margin-top:28px}header p{color:#526278}.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.kpis div{background:#edf4f6;padding:20px;border-radius:8px}.kpis strong{font-size:26px}.kpis p{font-size:13px;margin:4px 0}.callout{padding:24px;border-left:4px solid #0f766e;background:#eff8f6;margin:24px 0}img{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:10px;border-bottom:1px solid #dce4ed}th{background:#f3f6fa;position:sticky;top:0}.tablewrap{overflow:auto}a{color:#12639c}input,select,button{padding:10px;border:1px solid #bccbd9;border-radius:5px;background:white;font:inherit;font-size:14px;margin:4px}button:disabled{opacity:.4}.controls{display:flex;flex-wrap:wrap;align-items:center}code{font-size:12px;overflow-wrap:anywhere}small{color:#526278}.pagebar{display:flex;align-items:center;gap:12px;margin:12px 0}@media(max-width:750px){main{padding:22px;margin:0}.kpis{grid-template-columns:repeat(2,1fr)}h1{font-size:32px}}@media print{main{margin:0;padding:0}body{background:white}.browser{display:none}h2,h3{break-after:avoid}img,tr{break-inside:avoid}}'''
    body = f'''<header><p>OSSL-v2 Train · Analysis job 28185053 · Full score-table edition</p><h1>Video–Music Event Alignment</h1><p>30,953 scored pairs across the available Train subset</p></header>
<div class="kpis"><div><strong>{len(eligible):,}</strong><p>Quality-eligible pairs</p></div><div><strong>{rt:.5f}</strong><p>Median zero-lag Pearson</p></div><div><strong>{dt:.5f}</strong><p>Median mean absolute difference</p></div><div><strong>{positive/len(eligible):.2%}</strong><p>Pairs with positive Pearson</p></div></div>
<section class="callout"><b>Main finding.</b> The clip-weighted Pearson distribution centers near zero (mean {r.mean():.5f}; median {rt:.5f}). Positive correlations account for {positive:,} of {len(eligible):,} eligible pairs ({positive/len(eligible):.2%}). There are {strong:,} pairs with r ≥ 0.5 ({strong/len(eligible):.2%}). Strong positive event-curve correspondence exists for a minority of clips, but is not typical of the analyzed subset.</section>
<h2>1. Coverage and verification</h2><p>The complete supplied CSV contains {len(rows):,} unique clip IDs. Its counts, eligible-sample medians, quadrant assignments and quadrant totals agree with summary.json. Of 31,011 Train entries, 57 OOM clips were deferred, one additional pair failed analysis, and seven scored pairs are marked for review. Main distribution statistics below use only the {len(eligible):,} quality-ok pairs from {len(movie_groups):,} movie IDs. No recovery work is included.</p>
<p>This report validates the saved score table and classification. It does not recompute curves or independently validate the original media. The full scored table, including review rows, remains available in the browser below.</p>
<h2>2. Metric distributions</h2>{image('distributions.png')}<div class="tablewrap"><table><tr><th>Metric</th><th>Mean</th><th>P05</th><th>P25</th><th>Median</th><th>P75</th><th>P95</th></tr>{stat_rows}</table></div>
<p>Pearson ranges from {r.min():.4f} to {r.max():.4f}. The middle half lies between {np.quantile(r,.25):.4f} and {np.quantile(r,.75):.4f}. Mean absolute difference is measured in standardized curve units. Lower values indicate closer curve values, not lower audio volume or better perceptual quality.</p>
<h2>3. Four-quadrant classification</h2>{image('quadrants.png')}<table><tr><th>Quadrant</th><th>Pairs</th><th>Eligible share</th></tr>{quadrant_rows}</table>
<p>Thresholds are this run’s eligible medians: r = {rt:.8f}, mean difference = {dt:.8f}. Equality enters high correlation or low difference. “High correlation” is a relative label: because the threshold is slightly negative, it must not be interpreted as strong positive alignment.</p><p>Across eligible clips, Pearson and mean absolute difference have a Pearson association of {relation:.4f}. Both are calculated from the same standardized and smoothed signals, so quadrant overlap is descriptive rather than independent validation.</p>
<h2>4. Sensitivity and sampling</h2>{image('sensitivity.png')}<p>Smoothing changes the spread of correlations. The median remains close to zero at all three scales: no smoothing {stats['pearson_unsmoothed']['median']:.5f}, Hann 31 {rt:.5f}, Hann 61 {stats['pearson_smooth_2_4s']['median']:.5f}. Higher scores after stronger smoothing are not, by themselves, evidence of better perceptual synchronization.</p>
<p>Clips are clustered within movies, and movies contribute unequal clip counts. An equal-movie descriptive summary (the median of {len(movie_groups):,} per-movie Pearson medians) is {np.median(movie_medians):.5f}. This is a sensitivity summary, not a confidence interval or a replacement for the clip-level analysis. No independence-based p-values are reported.</p>
<h2>5. Examples for audiovisual review</h2><p>These are score-based candidates, not manually verified audiovisual explanations. The three examples per quadrant are closest to that quadrant’s median Pearson and median mean difference using fixed distance scales of 0.2 and 0.1. Video links are constructed from the pinned dataset layout and have not been individually checked. Curve overlays require the original saved curves and are not fabricated from score rows.</p>{examples}
<h2>6. Full score browser</h2><section class="browser"><p>Search by clip or movie ID, filter by quadrant, and select a ranking. All {len(rows):,} scored records are embedded in this standalone HTML file.</p><div class="controls"><input id="search" aria-label="Search clip or movie" placeholder="Clip or movie ID"><select id="group" aria-label="Quadrant"><option value="">All quadrants and review</option>''' + ''.join(f'<option value="{q}">{labels[q]}</option>' for q in colors) + '''<option value="review">Review</option></select><select id="sort" aria-label="Sort"><option value="high">Highest Pearson</option><option value="low">Lowest Pearson</option><option value="diff">Lowest mean difference</option></select></div><div class="pagebar"><button id="prev">Previous</button><span id="page"></span><button id="next">Next</button></div><div class="tablewrap"><table><thead><tr><th>Clip</th><th>Pearson</th><th>Mean difference</th><th>Median difference</th><th>Quadrant</th><th>Quality</th></tr></thead><tbody id="body"></tbody></table></div></section>''' + f'''
<h2>7. Method and interpretation</h2><p>DINOv2-L mean-patch video features and MusicFM-MSD layer 7 audio features are converted to adjacent-feature cosine dissimilarities. Audio is mono 24 kHz and features are nominally 25 Hz. Audio pair midpoints outside decoded duration are removed. Curves are standardized independently on their native support (population standard deviation, epsilon 10⁻⁸), interpolated onto common observed support at 25 Hz, then smoothed with a centered symmetric 31-point Hann kernel with replicated endpoints (1.2-second endpoint span). Main metrics use coincident clip starts and zero lag, with no fitted shift. Hann 61 spans 2.4 seconds. Lag diagnostics use separate fixed cropped support and should not be directly compared with full-support primary scores.</p>
<p>A near-zero median does not establish that music is unsuitable for the video or that no local synchronized events exist. Encoder choice, pooling, contextual audio features, start alignment and smoothing affect the results. Deferred OOM samples may bias the duration mix. Conclusions apply to the quality-eligible available subset, not every Train clip.</p>
<h2>8. Sources and reproducibility</h2><p>Input files: alignment_scores.csv, analysis_config.json and summary.json downloaded from Lighthouse analysis job 28185053. Report creation reads every score row. Extraction code b2851e2; analysis code 9b56559. No model was rerun. Source SHA-256 hashes and recomputed statistics are included in report_statistics.json in the report bundle. Manifest SHA-256: <code>{config['manifest_sha256']}</code>.</p><p>Use the accompanying CSV for full numeric precision. This HTML embeds its charts and score browser for offline viewing.</p>'''
    (out / 'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>OSSL Train Event Alignment Report</title><style>' + css + '</style><main>' + body + '</main><script>' + js + '</script></html>')
    md = f'''# OSSL Train Video–Music Event Alignment\n\nFull score-table report for Lighthouse job 28185053.\n\n- Scored pairs: {len(rows):,}; quality eligible: {len(eligible):,}; movies: {len(movie_groups):,}.\n- Median Pearson: {rt:.6f}; mean Pearson: {r.mean():.6f}.\n- Positive Pearson: {positive:,} ({positive/len(eligible):.2%}); r >= 0.5: {strong:,} ({strong/len(eligible):.2%}).\n- Median per-pair mean absolute difference: {dt:.6f}.\n- 57 deferred OOM clips, one analysis failure, seven scored review cases.\n\nOpen `report.html` for embedded figures, examples and the searchable full score table. The source counts, medians and all eligible quadrant assignments were recomputed and reconciled. Original curves/media were not rerun or validated here.\n\nThe distribution centers near zero; strong positive correlations occur in a minority of clips. These are encoder-derived event correspondence metrics, not perceptual music-quality ratings. Relative quadrant labels must be interpreted against their median thresholds.\n'''
    md += '\n[Open the interactive report](https://zoehsu-code.github.io/music_generate/train/report.html) · [Download the full score table](alignment_scores.csv) · [Analysis configuration](analysis_config.json) · [Derived statistics](report_statistics.json)\n\n'
    md += '## Metric distributions\n\n![Pearson and absolute difference distributions](distributions.png)\n\n'
    md += '## Four quadrants\n\n![All eligible pairs and median thresholds](quadrants.png)\n\n'
    md += '| Quadrant | Pairs | Eligible share |\n| --- | ---: | ---: |\n'
    for q in colors:
        md += f'| {labels[q].capitalize()} | {counts[q]:,} | {counts[q]/len(eligible):.2%} |\n'
    md += '\n## Smoothing sensitivity\n\n![Sensitivity to smoothing](sensitivity.png)\n'
    (out / 'README.md').write_text(md)
    print(json.dumps(dict(rows=len(rows), eligible=len(eligible), stats=stats), indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    build(a.source, a.output)
