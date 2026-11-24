"""Validate downloaded absolute-difference results and build an English HTML report."""
import argparse
import base64
import collections
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_train_shift_difference import aggregate_profiles, SHIFTS, ZERO


def read(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(source, prior, out):
    rows = read(source/'per_clip.csv')
    saved = read(source/'aggregate.csv')
    summary = json.loads((source/'summary.json').read_text())
    config = json.loads((source/'analysis_config.json').read_text())
    oldrows = read(prior/'per_clip.csv')
    failures = read(source/'failures.csv')
    n = len(rows)
    assert n == summary['completed'] == len({r['clip_id'] for r in rows})
    assert len(failures) == summary['failed'] == 0
    assert n == summary['selected'] == summary['prior_completed'] == len(oldrows)
    assert summary['identical_to_selected_prior_cohort']
    assert config['prior_shift_scores_sha256'] == digest(prior/'per_clip.csv')
    old = {r['clip_id']: r for r in oldrows}
    assert set(old) == {r['clip_id'] for r in rows}
    assert all(r['movie_id'] == old[r['clip_id']]['movie_id'] and
               int(r['anchor_samples']) == int(old[r['clip_id']]['anchor_samples']) for r in rows)
    error = max(float(r['max_pearson_reference_error']) for r in rows)
    assert error == summary['max_pearson_reference_error'] and error <= 1e-10
    matrices = {metric: np.array([[float(r[f'{metric}_at_{s:+g}s']) for s in SHIFTS] for r in rows])
                for metric in ('mae','median_ae')}
    counts = {}
    for metric, x in matrices.items():
        assert np.isfinite(x).all() and (x >= 0).all()
        ix = np.array([min(range(7),key=lambda j:(a[j],abs(SHIFTS[j]),SHIFTS[j])) for a in x])
        np.testing.assert_allclose(SHIFTS[ix],[float(r[f'best_{metric}_shift_s']) for r in rows],atol=0,rtol=0)
        np.testing.assert_allclose(x[np.arange(n),ix],[float(r[f'min_{metric}']) for r in rows],atol=1e-12,rtol=0)
        np.testing.assert_allclose(x[:,ZERO]-x[np.arange(n),ix],[float(r[f'{metric}_reduction_from_fixed_zero']) for r in rows],atol=1e-12,rtol=0)
        counts[metric] = dict(collections.Counter(str(float(s)) for s in SHIFTS[ix]))
        assert counts[metric] == summary[f'best_{metric}_shift_counts']
    calculated = aggregate_profiles(rows, matrices)
    key = lambda r:(r['group'],r['metric'],float(r['shift_s']))
    lookup = {key(r):r for r in saved}
    assert len(lookup) == len(saved) == len(calculated)
    for a in calculated:
        for k,v in a.items():
            if isinstance(v,(int,float)):
                assert abs(float(lookup[key(a)][k])-v)<1e-12,(key(a),k)
    sa = {key(r):r for r in summary['aggregate_all']}
    assert len(sa)==14
    for a in calculated:
        if a['group']=='all':
            for k,v in a.items():
                if isinstance(v,(int,float)):
                    assert abs(float(sa[key(a)][k])-v)<1e-12
    corr = np.array([[float(old[r['clip_id']][f'r_at_{s:+g}s']) for s in SHIFTS] for r in rows])
    assert np.isfinite(corr).all()
    movies = np.array([r['movie_id'] for r in rows]); mids=sorted(set(movies))
    movie_corr=np.array([corr[movies==m].mean(axis=0) for m in mids]).mean(axis=0)
    out.mkdir(parents=True,exist_ok=True)
    colors=['#0f766e','#c65f09']
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for ax,metric,title in zip(axes,matrices,['Mean absolute difference','Median absolute difference']):
        a=[r for r in calculated if r['group']=='all' and r['metric']==metric]
        ax.plot(SHIFTS,[r['clip_weighted_mean'] for r in a],'o-',color=colors[0],label='Clip-weighted mean')
        ax.plot(SHIFTS,[r['equal_movie_mean'] for r in a],'s--',color=colors[1],label='Equal-movie mean')
        ax.axvline(0,color='gray',ls=':'); ax.set(xticks=SHIFTS,title=title+' (expanded scale)',
            xlabel='Music lag (s); positive = later music',ylabel='Absolute difference (lower is better)')
        ax.ticklabel_format(axis='y',style='plain',useOffset=False);ax.legend(frameon=False,fontsize=8)
    fig.savefig(out/'expanded_profiles.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for ax,values,title in zip(axes,[corr.mean(axis=0),matrices['mae'].mean(axis=0),matrices['median_ae'].mean(axis=0)],
        ['Pearson r: higher is better','Mean absolute difference: lower is better','Median absolute difference: lower is better']):
        change=values-values[ZERO]
        ax.plot(SHIFTS,change,'o-',color=colors[0]);ax.axhline(0,color='gray',lw=.8);ax.axvline(0,color='gray',ls=':')
        ax.set(xticks=SHIFTS,title=title, xlabel='Music lag (s)',ylabel='Change from fixed-support zero')
    fig.savefig(out/'metric_comparison.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4),constrained_layout=True)
    for offset,metric,label in [(-.14,'mae','Minimum mean absolute difference'),(.14,'median_ae','Minimum median absolute difference')]:
        ax.bar(np.arange(7)+offset,[counts[metric].get(str(float(s)),0) for s in SHIFTS],width=.28,label=label)
    ax.set(xticks=np.arange(7),xticklabels=[f'{s:+g}' for s in SHIFTS],xlabel='Best tested music lag (s)',ylabel='Clips',title='Per-clip minima across seven tested shifts')
    ax.legend(frameon=False);fig.savefig(out/'best_shift_counts.png',dpi=180);plt.close(fig)
    for f in source.iterdir():
        if f.is_file(): shutil.copy2(f,out/f.name)
    def picture(name,alt):
        return f'<img alt="{html.escape(alt)}" src="data:image/png;base64,{base64.b64encode((out/name).read_bytes()).decode()}">'
    def table(headers,data):
        return '<div class="scroll"><table><thead><tr>'+''.join(f'<th>{html.escape(str(h))}</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in r)+'</tr>' for r in data)+'</tbody></table></div>'
    mean=matrices['mae'].mean(axis=0);median=matrices['median_ae'].mean(axis=0)
    reduction=mean[ZERO]-mean[4]
    stats=dict(completed=n,movies=len(mids),failed=0,max_pearson_reference_error=error,
        clip_weighted_mae_zero=float(mean[ZERO]),clip_weighted_mae_plus_half=float(mean[4]),
        mae_plus_half_reduction=float(reduction),mae_plus_half_reduction_percent=float(100*reduction/mean[ZERO]),
        source_sha256={f.name:digest(f) for f in source.iterdir() if f.is_file()},
        prior_pearson_sha256=digest(prior/'per_clip.csv'),
        validation='Recomputed all groups, minima, reductions and counts from per-clip results; matched prior cohort and source hash. Raw curves were not re-read locally.')
    (out/'report_statistics.json').write_text(json.dumps(stats,indent=2)+'\n')
    content=f'''<header><p class="eyebrow">OSSL-v2 · TRAIN · JOB 28228611</p><h1>Absolute Differences Under Temporal Shifts</h1><p>Saved video and MusicFM event curves · Seven fixed-support shifts · 24 November 2026</p></header>
<section><h2>1. Verified findings</h2><div class="cards"><div><b>{n:,}</b><span>completed clips</span></div><div><b>{len(mids):,}</b><span>movies</span></div><div><b>0</b><span>failed clips</span></div><div><b>0</b><span>reported Pearson reconstruction error</span></div></div>
<p>The successful cohort exactly matches the previous Pearson shift experiment. All saved group aggregates, best-shift selections and reductions were independently recomputed from the downloaded per-clip results.</p>
<ul><li>The clip-weighted mean absolute difference is lowest at <strong>+0.5 s</strong>: {mean[4]:.8f}, versus {mean[ZERO]:.8f} at zero. The reduction is {reduction:.8f} ({100*reduction/mean[ZERO]:.4f}%).</li>
<li>The clip-weighted average of per-clip median absolute differences is lowest at <strong>0 s</strong>. Equal-movie means are lowest at +0.5 s for both absolute-difference metrics.</li>
<li>The previous clip-weighted Pearson mean peaks at <strong>+1 s</strong>. At +1 s, both clip-weighted absolute-difference summaries are higher (worse) than at zero.</li>
<li>These descriptive optima do not establish a universal music delay or a statistically reliable correction.</li></ul></section>
<section><h2>2. Absolute-difference profiles</h2><p>The overview shows between-clip interquartile ranges (middle 50%, not confidence intervals). The next figure expands the vertical scale to reveal small changes in the averages.</p>
{picture('absolute_difference_shift.png','Absolute differences and between-clip spread')}
{picture('expanded_profiles.png','Expanded mean absolute-difference profiles')}
<p>Clip-weighted means give every clip equal weight. Equal-movie means first average clip summaries within each movie, then give every movie equal weight. Median absolute difference is calculated over timestamps within each clip; averaging those values is different from taking a median across clips.</p></section>
<section><h2>3. Comparison with the previous Pearson experiment</h2><p>Positive lag compares video(t) with music(t + lag). All shifts use the same video anchors. The table uses clip-weighted means. Absolute-difference values are in separately standardized native-curve units after interpolation and smoothing.</p>
{table(['Music lag (s)','Pearson r ↑','Mean absolute difference ↓','Mean of clip median differences ↓'],[[f'{s:+g}',f'{corr[:,j].mean():.8f}',f'{mean[j]:.8f}',f'{median[j]:.8f}'] for j,s in enumerate(SHIFTS)])}
{picture('metric_comparison.png','Changes from zero in three metrics')}
<p>Pearson measures centered, normalized co-variation. Absolute difference measures pointwise numerical closeness. Their preferred shifts need not agree. No per-shift restandardization is applied. This report does not infer perceptual soundtrack quality from either metric.</p></section>
<section><h2>4. Per-clip minimum shifts</h2>{picture('best_shift_counts.png','Counts of best tested shifts')}
{table(['Lag (s)','Minimum mean difference','Minimum median difference'],[[f'{s:+g}',counts['mae'].get(str(float(s)),0),counts['median_ae'].get(str(float(s)),0)] for s in SHIFTS])}
<p>For mean absolute difference, {counts['mae']['-2.0']+counts['mae']['2.0']:,} clips ({100*(counts['mae']['-2.0']+counts['mae']['2.0'])/n:.2f}%) select a boundary shift (±2 s). Choosing the best of seven shifts is selection-biased; boundary minima do not locate an optimum beyond the tested range.</p></section>
<section><h2>5. Original quadrant groups</h2><p>Groups retain their original zero-lag classification; they are not reclassified after shifting.</p>
{table(['Original group','Clips','Lowest mean difference lag','Zero mean difference','Minimum mean difference'],[[g, next(r['n_clips'] for r in calculated if r['group']==g), f"{min([r for r in calculated if r['group']==g and r['metric']=='mae'],key=lambda r:r['clip_weighted_mean'])['shift_s']:+g}",f"{next(r['clip_weighted_mean'] for r in calculated if r['group']==g and r['metric']=='mae' and r['shift_s']==0):.8f}",f"{min(r['clip_weighted_mean'] for r in calculated if r['group']==g and r['metric']=='mae'):.8f}"] for g in sorted(set(r['original_quadrant'] for r in rows))])}</section>
<section><h2>6. Method and limits</h2><p>Remove padded audio centers; separately standardize native curves; interpolate to shared 25 Hz support; apply centered Hann-31 smoothing; trim two seconds at both ends; shift music by −2, −1, −0.5, 0, +0.5, +1 or +2 seconds. Half-second offsets use linear interpolation at 12.5 samples. No wrap, extrapolation or shift-specific normalization. Each clip retains at least 125 anchors.</p>
<p>Mean absolute difference = mean over anchors of |video(t) − music(t + lag)|. Median absolute difference replaces that mean with a median. Fixed-support zero is not the older full-support zero analysis. The original 57 OOM cases, review rows and two prior near-constant shift failures remain outside this cohort.</p>
<p>The cluster script reconstructed all seven prior Pearson scores and anchor counts; downloaded per-clip errors are zero. Local checks validate saved numerical outputs and provenance, not raw-curve or media contents. No significance test or confidence interval is reported. Movie clustering, interpolation and smoothing should be considered when interpreting the small changes.</p></section>
<section><h2>7. Files and provenance</h2><ul>'''
    for name in ['per_clip.csv','aggregate.csv','summary.json','analysis_config.json','failures.csv','report_statistics.json']:
        content+=f'<li><a href="{name}">{name}</a></li>'
    content+=f'''</ul><p>Cluster source: analysis/train_shift_difference_28228611/. Pearson reference: analysis/train_temporal_shift_28224289/per_clip.csv. SHA-256 fingerprints are recorded in report_statistics.json. All charts are embedded for offline viewing; downloads require the accompanying report folder.</p></section>'''
    css='body{margin:0;background:#f3f6f8;color:#243746;font:16px/1.65 system-ui,sans-serif}main{max-width:1120px;margin:auto;padding:36px 24px}header,section{background:white;border-radius:12px;padding:30px;margin-bottom:24px}h1{font-size:38px;line-height:1.2;color:#173d55}h2{color:#173d55}.eyebrow{font-weight:700;letter-spacing:.12em;color:#0f766e;font-size:13px}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:16px}.cards div{background:#eef7f5;padding:18px;border-radius:8px}.cards b{display:block;font-size:28px}.cards span{font-size:13px}img{width:100%;height:auto;margin:16px 0}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;text-align:left;border-bottom:1px solid #dce5ea}th{background:#eef3f6}a{color:#086e82}@media(max-width:700px){.cards{grid-template-columns:1fr 1fr}main{padding:12px}header,section{padding:18px}h1{font-size:28px}}'
    (out/'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>OSSL Train — Absolute Differences Under Temporal Shifts</title><style>'+css+'</style><main>'+content+'</main></html>')
    print(json.dumps(stats,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--prior',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();build(a.source,a.prior,a.output)
