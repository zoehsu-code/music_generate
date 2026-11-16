"""Sharded OSSL MusicFM extraction with verified resume and atomic publication."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time
import uuid

import numpy as np
import torch

from extract_ossl_batch import discover, lock
from music_event_curve.audio_io import load_audio
from music_event_curve import encoder
from video_event_curve.event_curve import compute_cosine_dissimilarity, compute_timestamps


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def complete(dest, checksum, config):
    meta = json.loads((dest / 'metadata.json').read_text())
    if meta['input_sha256'] != checksum or meta['batch_config'] != config:
        return False
    raw = np.load(dest / 'music_raw.npz', allow_pickle=False)
    with raw:
        curve, times = raw['raw'], raw['timestamps']
        n = meta['feature_shape'][0]
        if curve.shape != (n - 1,) or times.shape != curve.shape or n < 2:
            return False
        if not np.isfinite(curve).all() or not np.isfinite(times).all():
            return False
        if not np.allclose(times, (np.arange(n - 1) + 0.5) / 25, rtol=0, atol=1e-12):
            return False
        if np.any(curve < 0) or np.any(curve > 2):
            return False
    if config['save_features']:
        features = np.load(dest / 'musicfm_features.npy', mmap_mode='r', allow_pickle=False)
        if list(features.shape) != meta['feature_shape'] or not np.isfinite(features).all():
            return False
    return True


def run(args):
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError('Require 0 <= shard-index < num-shards')
    if args.limit is not None and args.limit < 1:
        raise ValueError('limit must be positive')
    if not 0 <= args.layer_ix <= 12:
        raise ValueError('layer-ix must be in [0,12]')
    root, output = Path(args.dataset_root).resolve(), Path(args.output_root).resolve()
    items = discover(root, args.splits)
    if args.expected_count is not None and len(items) != args.expected_count:
        raise ValueError(f'Expected {args.expected_count} manifest entries; found {len(items)}')
    digest = hashlib.sha256(json.dumps(items).encode()).hexdigest()
    if args.limit:
        items = items[:args.limit]
    assigned = items[args.shard_index::args.num_shards]
    config = dict(schema_version=1, source_revision=encoder.SOURCE_REVISION,
                  weight_revision=encoder.WEIGHT_REVISION, config_revision=encoder.CONFIG_REVISION,
                  input_kind=args.input_kind, layer_ix=args.layer_ix, sample_rate=24000,
                  feature_rate_hz=25, frame_offset=1, output_mode='raw_only',
                  save_features=args.save_features)
    def source_for(split, movie, clip):
        folder, suffix = ('audio', '.wav') if args.input_kind == 'wav' else ('video', '.mp4')
        return root / split / folder / movie / (clip + suffix)
    print(json.dumps(dict(total=len(items), assigned=len(assigned), manifest_sha256=digest)), flush=True)
    if args.dry_run:
        missing = 0
        for item in assigned:
            source = source_for(*item)
            ok = source.is_file() and source.stat().st_size > 0
            missing += not ok
            print(json.dumps(dict(source=str(source), present_nonempty=ok)), flush=True)
        return int(missing > 0)
    # Fail before processing if a requested GPU is unavailable; never run silently on CPU.
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    logs = output / '_logs'; logs.mkdir(parents=True, exist_ok=True)
    logfile = logs / f'shard-{args.shard_index}-{uuid.uuid4().hex}.jsonl'
    counts = dict(completed=0, skipped=0, failed=0)
    bundle = None
    with logfile.open('w', buffering=1) as log:
        log.write(json.dumps(dict(event='run', config=config, manifest_sha256=digest,
                                  args=vars(args))) + '\n')
        for split, movie, clip in assigned:
            source = source_for(split, movie, clip)
            dest = output / split / movie / clip
            started = time.monotonic()
            record = dict(split=split, clip_id=clip, source=str(source))
            try:
                with lock(output / '_locks' / split / movie / (clip + '.lock')):
                    checksum = sha256(source)
                    if dest.exists():
                        if not complete(dest, checksum, config):
                            raise ValueError(f'Incompatible or corrupt output: {dest}; nothing overwritten')
                        record['status'] = 'skipped'
                    else:
                        waveform, audio_meta = load_audio(source)
                        if bundle is None:
                            bundle = encoder.load_musicfm(args.device, args.cache_dir, args.local_files_only)
                        features, times, model_meta = encoder.extract_musicfm_features(
                            waveform, args.device, args.layer_ix, encoder_bundle=bundle)
                        curve = compute_cosine_dissimilarity(features, 1)
                        timestamps = compute_timestamps(times, 1)
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        temporary = Path(tempfile.mkdtemp(prefix=f'.{clip}-', dir=dest.parent))
                        try:
                            np.savez(temporary / 'music_raw.npz', raw=curve, timestamps=timestamps)
                            if args.save_features:
                                np.save(temporary / 'musicfm_features.npy', features, allow_pickle=False)
                            metadata = dict(**audio_meta, **model_meta, input_sha256=checksum,
                                            batch_config=config, clip_id=clip, split=split, movie_id=movie,
                                            raw_curve_length=len(curve), smoothing_applied=False,
                                            timestamp_origin='first decoded audio sample',
                                            manifest_sha256=digest)
                            (temporary / 'metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
                            temporary.rename(dest)
                        except BaseException:
                            shutil.rmtree(temporary, ignore_errors=True)
                            raise
                        del features, waveform
                        record['status'] = 'completed'
            except Exception as exc:
                record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            counts[record['status']] += 1
            record['elapsed_seconds'] = time.monotonic() - started
            log.write(json.dumps(record) + '\n')
            print(json.dumps(record), flush=True)
        summary = dict(event='summary', **counts)
        log.write(json.dumps(summary) + '\n')
        print(json.dumps(summary), flush=True)
    return int(counts['failed'] > 0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root', required=True)
    p.add_argument('--output-root', required=True)
    p.add_argument('--splits', nargs='+', choices=['train', 'test'], default=['train'])
    p.add_argument('--input-kind', choices=['wav', 'mp4'], default='wav')
    p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--expected-count', type=int)
    p.add_argument('--limit', type=int)
    p.add_argument('--layer-ix', type=int, default=7)
    p.add_argument('--device', default='cuda')
    p.add_argument('--cache-dir', default='.cache/huggingface')
    p.add_argument('--local-files-only', action='store_true')
    p.add_argument('--save-features', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    raise SystemExit(run(p.parse_args()))


if __name__ == '__main__':
    main()
