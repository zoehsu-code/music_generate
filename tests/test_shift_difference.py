import argparse
import json
import numpy as np
import pytest
import analyze_train_shift_difference as batch
from analyze_train_temporal_shift import SHIFTS, aligned_curves, shift_scores


def test_pointwise_difference_and_half_sample_sign():
    t = np.arange(500) / 25
    v = np.sin(t) + .2 * np.sin(3*t)
    m = 2*(np.sin(t-.5) + .2*np.sin(3*(t-.5))) + 4
    mean, median, corr, n = batch.difference_profiles(t, v, m)
    anchors = np.arange(50, 450)
    expected = np.abs(v[anchors] - np.interp(anchors+12.5, np.arange(500), m))
    assert n == 400
    assert mean[4] == pytest.approx(expected.mean())
    assert median[4] == pytest.approx(np.median(expected))
    assert corr[4] > .999 and mean[4] > 3
    a, b, _, _ = batch.difference_profiles(t, v, v)
    assert a[3] == b[3] == 0


def test_movie_weighting():
    rows = [dict(movie_id='a', original_quadrant='q')]*2 + [dict(movie_id='b', original_quadrant='q')]
    row = batch.aggregate_profiles(rows, {'mae': np.array([[1.]*7, [1.]*7, [4.]*7])})[0]
    assert row['clip_weighted_mean'] == 2
    assert row['equal_movie_mean'] == 2.5


def test_batch_reconstruction_failure_and_no_overwrite(tmp_path, monkeypatch):
    t = np.arange(400)/25
    raw = np.sin(t)
    times, v, m = aligned_curves(raw, t, np.sin(t-.5), t, 16)
    corr, n = shift_scores(times, v, m)
    original, prior = [], []
    for i in range(2):
        clip = f'1_{i}_0_1'
        original.append(dict(clip_id=clip, movie_id='1', split='train', quality_flag='ok', quadrant='q', video_input_sha256='v', audio_sha256='m'))
        prior.append(dict(clip_id=clip, movie_id='1', anchor_samples=n, **{f'r_at_{s:+g}s':float(r) for s,r in zip(SHIFTS,corr)}))
        vd=tmp_path/'video/train/1'/clip; md=tmp_path/'music/train/1'/clip
        vd.mkdir(parents=True); md.mkdir(parents=True)
        (vd/'metadata.json').write_text(json.dumps(dict(clip_id=clip,input_sha256='v')))
        (md/'metadata.json').write_text(json.dumps(dict(clip_id=clip,input_sha256='m',duration=16)))
        np.save(vd/f'{clip}_raw_event_curve.npy',raw)
        np.save(vd/f'{clip}_native_timestamps.npy',t)
        np.savez(md/'music_raw.npz',raw=np.sin(t-.5),timestamps=t)
    prior[1]['r_at_+1s'] += .1
    for name,rows in [('original.csv',original),('prior.csv',prior)]:
        batch.write_csv(tmp_path/name,rows,list(rows[0]))
    monkeypatch.setattr(batch,'make_plot',lambda *a:None)
    args=argparse.Namespace(scores=tmp_path/'original.csv',shift_scores=tmp_path/'prior.csv',expected_completed=2,limit=None,output_root=tmp_path/'out',video_root=tmp_path/'video',music_root=tmp_path/'music')
    assert batch.run(args)==1
    summary=json.loads((tmp_path/'out/summary.json').read_text())
    assert (summary['completed'],summary['failed'])==(1,1)
    assert not summary['identical_to_selected_prior_cohort']
    row=batch.read_rows(tmp_path/'out/per_clip.csv')[0]
    mean,median,_,_=batch.difference_profiles(times,v,m)
    for j,s in enumerate(SHIFTS):
        assert float(row[f'mae_at_{s:+g}s'])==pytest.approx(mean[j])
        assert float(row[f'median_ae_at_{s:+g}s'])==pytest.approx(median[j])
    with pytest.raises(FileExistsError):
        batch.run(args)
