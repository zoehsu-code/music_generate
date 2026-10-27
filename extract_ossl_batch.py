"""OSSL-v2 video-only extraction; deterministic shards and verified resume."""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path
import re
import time
import uuid

import numpy as np

from video_event_curve import dinov2_encoder
from video_event_curve.pipeline import extract_video_event_curve, _sha256


def discover(root, splits):
    items = []
    for split in splits:
        ids = (root / 'splits' / f'{split}.txt').read_text().splitlines()
        seen = set()
        for value in ids:
            clip = value.strip()
            if not clip:
                continue
            if not re.fullmatch(r'\d+_\d+_\d+_\d+', clip) or clip in seen:
                raise ValueError(f'Invalid or duplicate clip ID in {split}: {clip!r}')
            seen.add(clip)
            movie = clip.split('_')[0]
            items.append((split, movie, clip))
    return sorted(items)


def signature(args):
    return dict(schema_version=1, encoder=dinov2_encoder.MODEL_ID,
                revision=dinov2_encoder.MODEL_REVISION, pooling='mean_patch',
                frame_offset=1, output_mode='raw_only', save_features=args.save_features)


def complete(destination, source, config):
    meta = json.loads((destination / 'metadata.json').read_text())
    if meta.get('batch_config') != config or meta['input_sha256'] != _sha256(source):
        return False
    stem = source.stem
    required = {f'{stem}_raw_event_curve.npy', f'{stem}_native_timestamps.npy',
                f'{stem}_frame_timestamps.npy'}
    if config['save_features']:
        required.add(f'{stem}_dinov2_features.npy')
    listed = {entry['file'] for entry in meta['outputs'].values()}
    if not required <= listed:
        return False
    for entry in meta['outputs'].values():
        array = np.load(destination / entry['file'], mmap_mode='r', allow_pickle=False)
        if list(array.shape) != entry['shape'] or str(array.dtype) != entry['dtype']:
            return False
        if not np.isfinite(array).all():
            return False
    return True


@contextmanager
def lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def run(args):
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError('Require 0 <= shard-index < num-shards')
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        raise ValueError('batch-size and limit must be positive')
    root, output = Path(args.dataset_root).resolve(), Path(args.output_root).resolve()
    items = discover(root, args.splits)
    # Limit the global list BEFORE sharding, so smoke-test shards remain disjoint.
    if args.limit is not None:
        items = items[:args.limit]
    digest = hashlib.sha256(json.dumps(items).encode()).hexdigest()
    assigned = items[args.shard_index::args.num_shards]
    print(json.dumps(dict(total=len(items), assigned=len(assigned), manifest_sha256=digest)), flush=True)
    if args.dry_run:
        for split, movie, clip in assigned:
            print(root / split / 'video' / movie / f'{clip}.mp4')
        return 0
    output.mkdir(parents=True, exist_ok=True)
    logs = output / '_logs'; logs.mkdir(exist_ok=True)
    log_path = logs / f'shard-{args.shard_index}-{uuid.uuid4().hex}.jsonl'
    config = signature(args)
    bundle = None
    counts = dict(completed=0, skipped=0, failed=0)
    with log_path.open('w', buffering=1) as log:
        log.write(json.dumps(dict(event='run', config=config, manifest_sha256=digest,
                                 args=vars(args))) + '\n')
        for split, movie, clip in assigned:
            source = root / split / 'video' / movie / f'{clip}.mp4'
            parent = output / split / movie
            destination = parent / clip
            started = time.monotonic()
            record = dict(split=split, movie_id=movie, clip_id=clip, source=str(source))
            try:
                with lock(output / '_locks' / split / movie / f'{clip}.lock'):
                    if destination.exists():
                        if not complete(destination, source, config):
                            raise ValueError(f'Existing output is incomplete or incompatible: {destination}. '
                                             'Move it aside or use another output root; nothing overwritten.')
                        record['status'] = 'skipped'
                    else:
                        if not source.is_file():
                            raise FileNotFoundError(source)
                        if bundle is None:
                            bundle = dinov2_encoder.load_dinov2(args.device, args.cache_dir,
                                                               args.local_files_only)
                            if args.device.startswith('cuda') and bundle[2].type != 'cuda':
                                bundle = None
                                raise RuntimeError('CUDA requested but unavailable; check GPU allocation and PyTorch')
                        result = extract_video_event_curve(
                            source, output_dir=None, encoder_bundle=bundle, device=args.device,
                            batch_size=args.batch_size, raw_only=True, save_plot=False)
                        result['metadata'].update(batch_config=config, dataset='McAuley-Lab/OSSL-v2',
                                                  split=split, movie_id=movie, clip_id=clip)
                        from video_event_curve.pipeline import save_outputs
                        save_outputs(result, parent, args.save_features, save_plot=False)
                        del result
                        record['status'] = 'completed'
            except Exception as exc:
                record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
                # Release unused CUDA allocator blocks after a failed clip.
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            counts[record['status']] += 1
            record['elapsed_seconds'] = time.monotonic() - started
            log.write(json.dumps(record) + '\n')
            print(json.dumps(record), flush=True)
        log.write(json.dumps(dict(event='summary', **counts)) + '\n')
    print(json.dumps(dict(log=str(log_path), **counts)), flush=True)
    return 1 if counts['failed'] else 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root', required=True)
    p.add_argument('--output-root', required=True)
    p.add_argument('--splits', nargs='+', choices=['train', 'test'], default=['train', 'test'])
    p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--device', default='cuda')
    p.add_argument('--cache-dir', default='.cache/huggingface')
    p.add_argument('--local-files-only', action='store_true')
    p.add_argument('--save-features', action='store_true')
    p.add_argument('--limit', type=int)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    raise SystemExit(run(args))


if __name__ == '__main__':
    main()
