"""Validate all Train temporal-shift results and build a standalone HTML report."""
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

SHIFTS = np.array([-2., -1., -.5, 0., .5, 1., 2.])


def build(source, out):
    out.mkdir(parents=True, exist_ok=True)
    def read(name):
        return list(csv.DictReader((source / name).open()))
    rows, aggregates = read('per_clip.csv'), read('aggregate.csv')
    summary = json.loads((source / 'summary.json').read_text())
    config = json.loads((source / 'analysis_config.json').read_text())
    failures, exclusions = read('failures.csv'), read('excluded_quality.csv')
    n = len(rows)
    assert n == summary['completed'] == len({r['clip_id'] for r in rows})
    assert len(failures) == summary['failed'] and len(exclusions) == summary['quality_excluded']
    assert n + len(failures) == summary['selected']
    assert summary['source_scored'] == summary['selected'] + len(exclusions)
    scores = np.array([[float(r[f'r_at_{s:+g}s']) for s in SHIFTS] for r in rows])
    assert np.isfinite(scores).all() and np.all(np.abs(scores) <= 1 + 1e-12)
    zero = scores[:, 3]
    best_indices = np.array([max(range(7), key=lambda j: (x[j], -abs(SHIFTS[j]), -SHIFTS[j])) for x in scores])
    best = scores[np.arange(n), best_indices]
    best_shifts = SHIFTS[best_indices]
    gain = best - zero
    np.testing.assert_allclose(best_shifts, [float(r['best_shift_s']) for r in rows], atol=0, rtol=0)
    np.testing.assert_allclose(best, [float(r['best_r']) for r in rows], atol=1e-12, rtol=0)
    np.testing.assert_allclose(gain, [float(r['gain_over_fixed_zero']) for r in rows], atol=1e-12, rtol=0)
    np.testing.assert_allclose(scores.mean(axis=0), summary['mean_profile'], atol=1e-12, rtol=0)
    counts = {str(float(s)): int(sum(best_shifts == s)) for s in SHIFTS}
    assert counts == summary['best_shift_counts']
    assert int(sum(np.abs(best_shifts) == 2)) == summary['boundary_best_count']
    groups = {'all': np.ones(n, dtype=bool), 'original_bottom_20_percent': np.array([int(r['original_bottom_20_percent']) == 1 for r in rows])}
    for q in sorted({r['original_quadrant'] for r in rows}):
        groups[q] = np.array([r['original_quadrant'] == q for r in rows])
    for a in aggregates:
        group = scores[groups[a['group']]]
        j = int(np.where(SHIFTS == float(a['shift_s']))[0][0]); col = group[:, j]
        assert int(a['n']) == len(col)
        expected = dict(mean_r=col.mean(), median_r=np.median(col), p25_r=np.quantile(col, .25),
                        p75_r=np.quantile(col, .75), mean_change_from_fixed_zero=(col-group[:, 3]).mean(),
                        mean_absolute_change_from_fixed_zero=np.abs(col-group[:, 3]).mean())
        for key, value in expected.items():
            assert abs(float(a[key])-value) < 1e-12, (a['group'], j, key)
    assert len(aggregates) == 7 * len(groups)
    fixed_error = max(float(r['fixed_zero_reference_error']) for r in rows)
    assert fixed_error <= 1e-10
    means, medians = scores.mean(axis=0), np.median(scores, axis=0)
    peak = int(np.argmax(means)); delta = scores[:, peak]-zero
    positive = int(sum(best_shifts > 0)); negative = int(sum(best_shifts < 0)); unchanged = int(sum(best_shifts == 0))
    boundary = int(sum(np.abs(best_shifts) == 2))
    movie_profiles = collections.defaultdict(list)
    for row, profile in zip(rows, scores):
        movie_profiles[row['movie_id']].append(profile)
    movie_means = np.array([np.mean(v, axis=0) for v in movie_profiles.values()])
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir() if p.is_file()}
    stats = dict(completed=n, movie_ids=len(movie_profiles), aggregate_peak_s=float(SHIFTS[peak]),
        mean_at_peak=float(means[peak]), fixed_zero_mean=float(means[3]), peak_mean_gain=float(delta.mean()),
        fraction_improved_at_aggregate_peak=float(np.mean(delta > 0)), positive_best=positive, negative_best=negative,
        zero_best=unchanged, boundary_best=boundary, mean_best_of_seven_gain=float(gain.mean()),
        median_best_of_seven_gain=float(np.median(gain)),
        equal_movie_mean_profile=movie_means.mean(axis=0).tolist(), source_sha256=hashes)
    (out / 'report_statistics.json').write_text(json.dumps(stats, indent=2)+'\n')
    for name in ['per_clip.csv', 'aggregate.csv', 'summary.json', 'analysis_config.json', 'failures.csv', 'excluded_quality.csv']:
        shutil.copy2(source / name, out / name)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    p25, p75 = np.quantile(scores, [.25, .75], axis=0)
    axs[0].fill_between(SHIFTS, p25, p75, color='#0f766e', alpha=.15, label='Between-clip IQR (not CI)')
    axs[0].plot(SHIFTS, means, 'o-', color='#0f766e', label='Mean')
    axs[0].plot(SHIFTS, medians, 's--', color='#2563eb', label='Median')
    axs[0].set(title='Distribution spread at each tested shift', ylabel='Fixed-support Pearson r')
    axs[1].plot(SHIFTS, means, 'o-', color='#0f766e', label='Clip-weighted mean')
    axs[1].plot(SHIFTS, movie_means.mean(axis=0), 's--', color='#b45309', label='Equal-movie mean')
    axs[1].set(title='Mean profile, expanded vertical scale', ylabel='Mean fixed-support Pearson r')
    for ax in axs:
        ax.set(xticks=SHIFTS, xlabel='Music lag (s); positive = later music')
        ax.axhline(0, color='gray', lw=.8); ax.axvline(0, color='gray', ls=':', lw=.8); ax.legend(frameon=False)
    fig.savefig(out / 'profiles.png', dpi=180); plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axs[0].bar(np.arange(7), [counts[str(float(s))] for s in SHIFTS], color=['#a9512b' if abs(s)==2 else '#2563a6' for s in SHIFTS])
    axs[0].set(xticks=np.arange(7), xticklabels=[f'{s:+g}' for s in SHIFTS], xlabel='Best tested shift (seconds)', ylabel='Clip count', title='Per-clip best shift; endpoints highlighted')
    axs[1].hist(gain, bins=60, color='#0f766e', edgecolor='white', linewidth=.3)
    axs[1].set(xlabel='Best tested r minus fixed-support zero r', ylabel='Clip count', title='Best-of-seven gain (selection-biased)')
    fig.savefig(out / 'best_shifts.png', dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    for name, mask in groups.items():
        if name in ('all', 'original_bottom_20_percent'):
            continue
        ax.plot(SHIFTS, scores[mask].mean(axis=0), 'o-', label=name.replace('_', ' '))
    ax.set(xticks=SHIFTS, xlabel='Music lag (s); positive = later music', ylabel='Mean fixed-support Pearson r', title='Profiles by original zero-lag quadrant')
    ax.axhline(0, color='gray', lw=.8); ax.legend(fontsize=8, frameon=False)
    fig.savefig(out / 'group_profiles.png', dpi=180); plt.close(fig)
    def img(name):
        return f'<img alt="{name}" src="data:image/png;base64,{base64.b64encode((out/name).read_bytes()).decode()}">'
    agrows = ''.join(f'<tr><td>{s:+g}</td><td>{means[j]:+.6f}</td><td>{medians[j]:+.6f}</td><td>{means[j]-means[3]:+.6f}</td><td>{counts[str(float(s))]:,}</td></tr>' for j,s in enumerate(SHIFTS))
    low = scores[groups['original_bottom_20_percent']]
    low_text = f'The original bottom-20% group contains {len(low):,} successfully analyzed clips. Its fixed-support zero mean is {low[:,3].mean():+.4f}; its −2 s and +2 s means are {low[:,0].mean():+.4f} and {low[:,-1].mean():+.4f}. This group was selected for low original scores, so movement toward zero at other shifts can reflect selection and regression to the mean. It does not establish that these clips have a true delay.'
    top_gain = sorted(rows, key=lambda r:-float(r['gain_over_fixed_zero']))[:10]
    examples = ''.join(f'<tr><td>{html.escape(r["clip_id"])}</td><td>{float(r["r_at_+0s"]):+.4f}</td><td>{float(r["best_shift_s"]):+g}</td><td>{float(r["best_r"]):+.4f}</td><td>{float(r["gain_over_fixed_zero"]):.4f}</td></tr>' for r in top_gain)
    failure_text = ''.join(f'<li><code>{html.escape(r["clip_id"])}</code>: {html.escape(r["error"])}</li>' for r in failures)
    data = [[r['clip_id'], r['original_quadrant'], *[float(r[f'r_at_{s:+g}s']) for s in SHIFTS], float(r['best_shift_s']), float(r['gain_over_fixed_zero'])] for r in rows]
    js = '''const data=DATA;let page=0,filtered=[];const size=50,$=s=>document.querySelector(s);function render(reset=true){if(reset)page=0;let text=$('#search').value.trim(),shift=$('#shift').value;filtered=data.filter(r=>(!text||r[0].includes(text))&&(shift===''||r[9]===Number(shift)));let order=$('#order').value;filtered.sort((a,b)=>order==='gain'?b[10]-a[10]:order==='zero'?b[5]-a[5]:a[0].localeCompare(b[0]));$('#rows').innerHTML=filtered.slice(page*size,(page+1)*size).map(r=>`<tr><td>${r[0]}</td>${r.slice(2,9).map(x=>`<td>${x.toFixed(4)}</td>`).join('')}<td>${r[9]}</td><td>${r[10].toFixed(4)}</td></tr>`).join('');$('#count').textContent=`${filtered.length.toLocaleString()} clips · Page ${page+1} / ${Math.max(1,Math.ceil(filtered.length/size))}`;$('#prev').disabled=page===0;$('#next').disabled=(page+1)*size>=filtered.length;}['#search','#shift','#order'].forEach(s=>$(s).addEventListener('input',()=>render()));$('#prev').onclick=()=>{page--;render(false)};$('#next').onclick=()=>{page++;render(false)};render();'''.replace('DATA',json.dumps(data,separators=(',',':'),allow_nan=False))
    css='''body{font:16px/1.7 system-ui,sans-serif;color:#172033;background:#f1f4f8;margin:0}main{max-width:1100px;margin:32px auto;background:white;padding:44px;border-radius:14px}h1{font-size:42px;line-height:1.15}h2{margin-top:38px;color:#183b56}header p,small{color:#526278}.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.kpis div{padding:18px;background:#edf4f6;border-radius:8px}.kpis strong{font-size:25px}.kpis p{font-size:13px;margin:4px 0}.callout{padding:22px;border-left:4px solid #0f766e;background:#eff8f6;margin:24px 0}img{width:100%;height:auto}.tablewrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:10px;border-bottom:1px solid #dce4ed}th{background:#f3f6fa}input,select,button{font:inherit;font-size:14px;padding:9px;border:1px solid #bccbd9;background:white;border-radius:5px;margin:4px}button:disabled{opacity:.4}code{font-size:12px;overflow-wrap:anywhere}a{color:#12639c}@media(max-width:750px){main{margin:0;padding:20px}h1{font-size:31px}.kpis{grid-template-columns:repeat(2,1fr)}}@media print{main{margin:0;padding:0}body{background:white}.browser{display:none}h2{break-after:avoid}img,tr{break-inside:avoid}}'''
    body=f'''<header><p>OSSL-v2 Train · Temporal shift experiment · Job 28224289 · 24 November 2026</p><h1>Does shifting music improve event alignment?</h1><p>A seven-shift diagnostic on saved video and soundtrack curves</p></header>
<div class="kpis"><div><strong>{n:,}</strong><p>Completed paired profiles</p></div><div><strong>{SHIFTS[peak]:+g} s</strong><p>Highest tested aggregate mean</p></div><div><strong>{means[peak]:+.5f}</strong><p>Mean r at that shift</p></div><div><strong>{boundary/n:.2%}</strong><p>Best shifts at tested boundaries</p></div></div>
<div class="callout"><b>Main finding.</b> Among the seven tested shifts, the clip-weighted mean peaks at {SHIFTS[peak]:+g} seconds: r={means[peak]:+.5f}, versus {means[3]:+.5f} at fixed-support zero, a change of {delta.mean():+.5f}. The aggregate effect is small in absolute Pearson units. There is no evidence here for a universal one-second correction: per-clip optima are mixed and {boundary/n:.2%} fall at the tested boundaries.</div>
<h2>1. Coverage and validation</h2><p>The prior score table contains {summary['source_scored']:,} scored pairs. Seven quality-review rows were excluded, leaving {summary['selected']:,} selected pairs. This run completed {n:,} profiles with two near-constant failures, yielding {n*7:,} correlations across {len(movie_profiles):,} movie IDs. The original 57 OOM clips remain deferred. Every completed pair contributes to every tested shift.</p><p>All per-clip correlations were read. Best shifts, gains, counts and every group’s aggregate statistics were recomputed and reconciled with the supplied files. The maximum reconstructed fixed-zero discrepancy reported in the per-clip table is {fixed_error:.3g}. No encoders or raw curve reconstruction were rerun during report generation.</p>
<h2>2. Aggregate shift profile</h2>{img('profiles.png')}<div class="tablewrap"><table><tr><th>Shift (s)</th><th>Mean r</th><th>Median r</th><th>Mean change vs fixed zero</th><th>Clips best at this shift</th></tr>{agrows}</table></div><p>The left plot shows between-clip spread, not uncertainty in the mean. The expanded scale on the right makes the small mean differences visible. The equal-movie profile gives every movie equal weight instead of every clip; it is a descriptive weighting check, not a significance test.</p>
<h2>3. Per-clip best shifts and selection effects</h2>{img('best_shifts.png')}<p>Best tested shifts are positive for {positive:,} clips ({positive/n:.2%}), negative for {negative:,} ({negative/n:.2%}) and zero for {unchanged:,} ({unchanged/n:.2%}). {boundary:,} clips peak at −2 or +2 seconds. An endpoint peak does not identify the true optimum or justify extrapolating beyond the range.</p><p>Choosing the best of seven shifts separately for every clip gives an average gain of {gain.mean():.4f} and median gain of {np.median(gain):.4f}. This is an optimistic, selection-biased diagnostic, not an independently evaluated improvement. By contrast, applying the aggregate-peak shift to every clip changes the mean by only {delta.mean():.4f}; {np.mean(delta>0):.2%} of clips improve under that shift. Selecting the aggregate peak and evaluating it on the same set is also exploratory.</p>
<h2>4. Original groups</h2>{img('group_profiles.png')}<p>Quadrants were assigned before this experiment using the prior full-support zero-lag Pearson and mean absolute difference medians. Their labels are relative to that subset, not absolute synchronization classes. Group gains are not independent evidence because grouping already uses a related score.</p><p>{low_text}</p>
<h2>5. Largest diagnostic gains</h2><p>These ten clips are candidates for review, selected by best-of-seven gain. Their values do not establish that moving the soundtrack improves perceived synchronization. No per-clip time-series plots were generated.</p><div class="tablewrap"><table><tr><th>Clip</th><th>Fixed zero r</th><th>Best shift (s)</th><th>Best r</th><th>Gain</th></tr>{examples}</table></div>
<h2>6. Browse all completed profiles</h2><section class="browser"><p>Search a clip, filter its best tested shift, or sort by diagnostic gain. All {n:,} completed profiles are embedded for offline viewing.</p><input id="search" aria-label="Clip ID" placeholder="Clip ID"><select id="shift" aria-label="Best shift"><option value="">All best shifts</option>''' + ''.join(f'<option value="{s}">{s:+g} s</option>' for s in SHIFTS) + '''</select><select id="order" aria-label="Sort"><option value="gain">Largest best-of-seven gain</option><option value="zero">Highest fixed-zero r</option><option value="clip">Clip ID</option></select><p><button id="prev">Previous</button><span id="count"></span><button id="next">Next</button></p><div class="tablewrap"><table><thead><tr><th>Clip</th>''' + ''.join(f'<th>{s:+g} s</th>' for s in SHIFTS) + '''<th>Best shift</th><th>Gain</th></tr></thead><tbody id="rows"></tbody></table></div></section>''' + f'''
<h2>7. Method and sign convention</h2><p>Tested shifts: −2, −1, −0.5, 0, +0.5, +1 and +2 seconds. Positive Δ evaluates <code>corr(video(t), music(t+Δ))</code>: matching music events occur later in the original paired timeline. If independently confirmed as a real delay, compensation would move that music earlier, not later. Here this remains a diagnostic comparison.</p><p>As in Test-50: remove padded audio centers, standardize native curves independently, interpolate to common 25 Hz observed support and apply centered Hann-31 smoothing (1.2-second endpoint span). Trim two seconds at both ends of the aligned grid and use identical video anchors at all shifts, requiring at least 125 samples. Half-second shifts use linear interpolation at 12.5-sample offsets. No wrapping or extrapolation is used. Exact best-score ties prefer the smallest absolute shift, then the negative shift.</p>
<p>The proper zero baseline is the fixed-support zero column. It differs from the prior report’s full-support zero score because endpoints are removed. Comparing best shifted scores with the old untrimmed baseline would confound support and shift effects.</p>
<h2>8. Limits and next experiments</h2><ul><li>Curves measure encoder-derived changes rather than musical quality, semantics or human synchronization judgments.</li><li>The best tested shift is chosen from a sparse grid and is not a precise delay estimate. Best-of-seven gains and bottom-group gains have selection effects.</li><li>Movie-clustered clips and smoothed time points are dependent. No p-values or confidence intervals are claimed.</li><li>A stronger causal or control claim needs matched wrong-pair/time-shift controls and held-out evaluation. Learn any global correction on a movie-disjoint development set and evaluate once on held-out movies.</li><li>For event-conditioned generation, use these results to motivate controlled timing experiments, not a universal +1-second correction.</li></ul>
<h2>9. Accounting and provenance</h2><p>Two selected pairs could not be evaluated because at least one curve became near-constant on the fixed comparison support:</p><ul>{failure_text}</ul><p>Only the supplied result files were used. Source SHA-256 hashes, weighting checks and derived statistics are in report_statistics.json. Source score-table SHA-256: <code>{config['source_scores_sha256']}</code>. Experiment implementation: e98beb5. Data came from Lighthouse job 28224289.</p><p><a href="per_clip.csv">Per-clip scores</a> · <a href="aggregate.csv">Group aggregates</a> · <a href="analysis_config.json">Configuration</a> · <a href="summary.json">Run summary</a></p>'''
    (out / 'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>OSSL Train Temporal Shift Report</title><style>'+css+'</style><main>'+body+'</main><script>'+js+'</script></html>')
    markdown=f'''# OSSL Train Temporal Shift Report\n\n[Open the interactive report](https://zoehsu-code.github.io/music_generate/train/temporal-shift/report.html) · [Per-clip scores](per_clip.csv) · [Group aggregates](aggregate.csv)\n\nCompleted: {n:,} pairs, {n*7:,} correlations. Two fixed-support near-constant failures.\n\nMean r peaks at {SHIFTS[peak]:+g} s ({means[peak]:+.6f}), compared with {means[3]:+.6f} at fixed zero: gain {delta.mean():+.6f}. The aggregate effect is small. Per-clip best-of-seven gains are selection-biased; {boundary/n:.2%} of optima are at tested boundaries. This is not evidence for a universal time correction.\n\n![Shift profiles](profiles.png)\n\n![Best shifts and diagnostic gains](best_shifts.png)\n\n![Original quadrant profiles](group_profiles.png)\n\nThe HTML contains the method, full searchable table, examples and limitations. Statistics were recomputed from all supplied rows and reconciled against every aggregate group.\n'''
    (out / 'README.md').write_text(markdown)
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    args=p.parse_args();build(args.source,args.output)
