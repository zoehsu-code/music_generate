import argparse
import json
import numpy as np

from analyze_saved_ossl import run, score_pair


def test_identical_curves_and_constant_quality():
    t = np.arange(250) / 25
    raw = np.sin(t) + 1
    _, score = score_pair(raw, t, raw, t, 10, 10)
    assert abs(score['pearson_zero_lag'] - 1) < 1e-12
    assert score['mean_absolute_difference'] == 0
    _, score = score_pair(np.ones(250), t, raw, t, 10, 10)
    assert score['pearson_zero_lag'] is None
    assert score['quality_flag'] == 'near_constant'


def test_exclusion_and_unexpected_missing_output(tmp_path):
    ids = ['1_0_0_1', '1_1_1_2', '1_2_2_3']
    manifest = tmp_path / 'train.txt'; manifest.write_text('\n'.join(ids))
    excluded = tmp_path / 'excluded.json'
    excluded.write_text(json.dumps([dict(clip_id=ids[1])]))
    vd = tmp_path / 'video/train/1' / ids[0]; vd.mkdir(parents=True)
    md = tmp_path / 'music/train/1' / ids[0]; md.mkdir(parents=True)
    t = np.arange(250) / 25; raw = np.sin(t) + 1
    np.save(vd / (ids[0] + '_raw_event_curve.npy'), raw)
    np.save(vd / (ids[0] + '_native_timestamps.npy'), t)
    np.savez(md / 'music_raw.npz', raw=raw, timestamps=t)
    (vd / 'metadata.json').write_text(json.dumps(dict(clip_id=ids[0], duration=10, input_sha256='v')))
    (md / 'metadata.json').write_text(json.dumps(dict(clip_id=ids[0], duration=10, input_sha256='m',
        batch_config=dict(input_kind='wav', layer_ix=7, sample_rate=24000, frame_offset=1))))
    args = argparse.Namespace(manifest=str(manifest), exclude_failures=str(excluded), expected_count=3,
        output_root=str(tmp_path / 'out'), video_root=str(tmp_path / 'video'),
        music_root=str(tmp_path / 'music'), split='train', save_aligned=False)
    assert run(args) == 1
    summary = json.loads((tmp_path / 'out/summary.json').read_text())
    assert (summary['scored'], summary['excluded'], summary['failed']) == (1, 1, 1)
    assert summary['quadrant_counts'] == {'high_correlation_low_difference': 1}
