import argparse
import json
import numpy as np
import pytest
import torch

import extract_ossl_music_batch as batch


def options(root, output, **changes):
    values = dict(dataset_root=str(root), output_root=str(output), splits=['train'],
                  input_kind='wav', num_shards=1, shard_index=0, expected_count=3,
                  limit=None, layer_ix=7, device='cpu', cache_dir=None,
                  local_files_only=True, save_features=False, dry_run=False)
    values.update(changes)
    return argparse.Namespace(**values)


def test_resume_failure_isolation_and_model_reuse(tmp_path, monkeypatch):
    root, output = tmp_path / 'data', tmp_path / 'out'
    (root / 'splits').mkdir(parents=True)
    ids = ['1_0_0_10', '1_1_10_20', '1_2_20_30']
    (root / 'splits/train.txt').write_text('\n'.join(ids))
    audio = root / 'train/audio/1'; audio.mkdir(parents=True)
    for clip in ids:
        (audio / (clip + '.wav')).write_bytes(clip.encode())
    loads = []
    def decode(path):
        if path.stem == ids[1]:
            raise ValueError('broken audio')
        return np.ones(24000, dtype=np.float32), dict(duration=1, audio_path=str(path))
    def load(*args):
        loads.append(1)
        return object(), torch.device('cpu')
    def encode(*args, **kwargs):
        return np.ones((25, 1024), dtype=np.float32), np.arange(25) / 25, dict(feature_shape=[25, 1024])
    monkeypatch.setattr(batch, 'load_audio', decode)
    monkeypatch.setattr(batch.encoder, 'load_musicfm', load)
    monkeypatch.setattr(batch.encoder, 'extract_musicfm_features', encode)
    assert batch.run(options(root, output)) == 1
    assert len(loads) == 1
    dest = output / 'train/1' / ids[0]
    path = dest / 'music_raw.npz'
    before = path.stat().st_mtime_ns
    assert batch.run(options(root, output)) == 1
    assert len(loads) == 1 and path.stat().st_mtime_ns == before
    assert batch.run(options(root, output, input_kind='mp4')) == 1
    assert path.stat().st_mtime_ns == before
    path.write_bytes(b'corrupt')
    assert batch.run(options(root, output)) == 1
    assert path.read_bytes() == b'corrupt'
    summary = json.loads(sorted((output / '_logs').glob('*'), key=lambda p: p.stat().st_mtime_ns)[-1].read_text().splitlines()[-1])
    assert summary['failed'] == 2 and summary['skipped'] == 1


def test_dry_run_shards_and_manifest_guard(tmp_path, capsys):
    (tmp_path / 'splits').mkdir()
    ids = ['1_0_0_10', '1_1_10_20', '1_2_20_30']
    (tmp_path / 'splits/train.txt').write_text('\n'.join(ids))
    folder = tmp_path / 'train/video/1'; folder.mkdir(parents=True)
    for clip in ids:
        (folder / (clip + '.mp4')).write_bytes(b'video')
    results = []
    for shard in range(2):
        assert batch.run(options(tmp_path, tmp_path / 'out', dry_run=True,
                                 input_kind='mp4', num_shards=2, shard_index=shard)) == 0
        results.append({json.loads(line)['source'] for line in capsys.readouterr().out.splitlines()[1:]})
    assert results[0].isdisjoint(results[1]) and len(results[0] | results[1]) == 3
    assert not (tmp_path / 'out').exists()
    with pytest.raises(ValueError, match='Expected'):
        batch.run(options(tmp_path, tmp_path / 'out', expected_count=31011, dry_run=True))
