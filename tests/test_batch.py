import argparse
import json
import numpy as np
import torch

import extract_ossl_batch as batch
from test_video_pipeline import make_video, Model, Processor


def args(root, output, **kwargs):
    values = dict(dataset_root=str(root), output_root=str(output), splits=['train'],
                  num_shards=1, shard_index=0, batch_size=4, device='cpu',
                  cache_dir=None, local_files_only=True, save_features=False,
                  limit=None, dry_run=False)
    values.update(kwargs)
    return argparse.Namespace(**values)


def dataset(root):
    (root / 'splits').mkdir(parents=True)
    (root / 'splits/train.txt').write_text('1_0_0_10\n1_1_10_20\n1_2_20_30\n')
    folder = root / 'train/video/1'; folder.mkdir(parents=True)
    make_video(folder / '1_0_0_10.mp4', count=5)
    (folder / '1_1_10_20.mp4').write_bytes(b'broken video')
    make_video(folder / '1_2_20_30.mp4', count=5)


def test_batch_resume_and_failure_isolation(tmp_path, monkeypatch):
    root, output = tmp_path / 'data', tmp_path / 'results'
    dataset(root)
    calls = []
    def load(*a):
        calls.append(1)
        return Model(), Processor(), torch.device('cpu')
    monkeypatch.setattr(batch.dinov2_encoder, 'load_dinov2', load)
    assert batch.run(args(root, output)) == 1
    assert len(calls) == 1  # reused across the whole batch, including a bad clip
    dest = output / 'train/1/1_0_0_10'
    curve = dest / '1_0_0_10_raw_event_curve.npy'
    before = curve.stat().st_mtime_ns
    assert np.load(curve).shape == (4,)
    assert not list(dest.glob('*.png'))
    assert not list(dest.glob('*normalized*'))
    assert not list(dest.glob('*dinov2_features*'))
    assert (output / 'train/1/1_2_20_30/metadata.json').exists()
    meta = json.loads((dest / 'metadata.json').read_text())
    assert meta['hann_window'] is None and not meta['smoothing_applied']
    assert batch.run(args(root, output)) == 1
    assert curve.stat().st_mtime_ns == before
    # Changed extraction settings may not silently reuse or overwrite outputs.
    assert batch.run(args(root, output, save_features=True)) == 1
    assert curve.stat().st_mtime_ns == before
    # Corrupt output is rejected, not skipped or overwritten.
    curve.write_bytes(b'broken npy')
    assert batch.run(args(root, output)) == 1
    assert curve.read_bytes() == b'broken npy'


def test_deterministic_disjoint_shards(tmp_path):
    dataset(tmp_path)
    items = batch.discover(tmp_path, ['train'])
    shards = [items[i::2] for i in range(2)]
    assert set(shards[0]).isdisjoint(shards[1])
    assert sorted(shards[0] + shards[1]) == items
    output = tmp_path / 'out'
    assert batch.run(args(tmp_path, output, dry_run=True)) == 0
    assert not output.exists()
