import csv
import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from analyze_train_temporal_shift import SHIFTS, aligned_curves, shift_scores
import analyze_train_temporal_shift as batch


def test_positive_lag_and_half_sample_interpolation():
    t = np.arange(500) / 25
    video = np.exp(-((t - 8) / .3) ** 2)
    music = np.exp(-((t - 8.5) / .3) ** 2)
    scores, n = shift_scores(t, video, music)
    assert SHIFTS[np.argmax(scores)] == .5
    assert scores[4] > .99
    assert n == 400
    anchor = np.arange(50, 450)
    expected = np.corrcoef(video[anchor], np.interp(anchor + 12.5, np.arange(500), music))[0, 1]
    assert abs(scores[4] - expected) < 1e-12


def test_invalid_support_and_constant_inputs():
    t = np.arange(200) / 25
    with pytest.raises(ValueError, match='125'):
        shift_scores(t, np.sin(t), np.cos(t))
    t = np.arange(300) / 25
    with pytest.raises(ValueError, match='constant'):
        shift_scores(t, np.ones(300), np.sin(t))


def test_reconstruct_saved_test50_profiles():
    root = Path(__file__).resolve().parents[1] / 'reports/ossl-test-50'
    prior = {r['clip_id']: r for r in csv.DictReader((root / 'temporal-shift/per_clip.csv').open())}
    clips = list(csv.DictReader((root / 'alignment_scores.csv').open()))
    for row in clips:
        path = root / Path(row['figure']).parent / 'aligned_curves.csv'
        times, video, music = np.loadtxt(path, delimiter=',', skiprows=1, unpack=True)
        scores, _ = shift_scores(times, video, music)
        saved = [float(prior[row['clip_id']][f'r_at_{shift:+g}s']) for shift in SHIFTS]
        np.testing.assert_allclose(scores, saved, atol=1e-12, rtol=0)


def test_batch_accounting_and_provenance_failure(tmp_path, monkeypatch):
    t = np.arange(400) / 25
    vraw, mraw = 1 + .2 * np.sin(t), 1 + .2 * np.sin(t - .5)
    times, video, music = aligned_curves(vraw, t, mraw, t, 16)
    scores, _ = shift_scores(times, video, music)
    rows = []
    for i in range(3):
        clip = f'1_{i}_0_1'
        rows.append(dict(clip_id=clip, movie_id='1', split='train', quality_flag='ok' if i < 2 else 'duration_mismatch',
            quadrant='high_correlation_low_difference', pearson_zero_lag=float(np.corrcoef(video, music)[0, 1]),
            pearson_zero_lag_fixed_support=scores[3], audio_sha256='audio', video_input_sha256='video'))
        vd = tmp_path / 'video/train/1' / clip; vd.mkdir(parents=True)
        md = tmp_path / 'music/train/1' / clip; md.mkdir(parents=True)
        (vd / 'metadata.json').write_text(json.dumps(dict(clip_id=clip, input_sha256='video')))
        (md / 'metadata.json').write_text(json.dumps(dict(clip_id=clip, duration=16,
            input_sha256='audio' if i == 0 else 'changed')))
        np.save(vd / f'{clip}_raw_event_curve.npy', vraw)
        np.save(vd / f'{clip}_native_timestamps.npy', t)
        np.savez(md / 'music_raw.npz', raw=mraw, timestamps=t)
    path = tmp_path / 'scores.csv'
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    monkeypatch.setattr(batch, 'make_plot', lambda *a: None)
    args = argparse.Namespace(scores=str(path), expected_scored=3, limit=None,
        video_root=str(tmp_path / 'video'), music_root=str(tmp_path / 'music'), output_root=str(tmp_path / 'out'))
    assert batch.run(args) == 1
    summary = json.loads((tmp_path / 'out/summary.json').read_text())
    assert (summary['completed'], summary['failed'], summary['quality_excluded']) == (1, 1, 1)
    assert len(list(csv.DictReader((tmp_path / 'out/per_clip.csv').open()))) == 1
    with pytest.raises(FileExistsError):
        batch.run(args)
