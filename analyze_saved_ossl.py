"""Analyze saved OSSL curves without decoding media or running encoders."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np

from music_alignment_metrics import compare


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def score_pair(vraw, vt, mraw, mt, video_duration, music_duration):
    # Match Test-50: remove nominal padded audio centers beyond decoded duration.
    valid = mt < music_duration
    arrays, score = compare(vraw, vt, mraw[valid], mt[valid])
    difference = np.abs(arrays['video'] - arrays['music'])
    score.update(mean_absolute_difference=float(difference.mean()),
                 median_absolute_difference=float(np.median(difference)))
    if abs(music_duration - video_duration) > 0.1:
        score['quality_flag'] = 'duration_mismatch'
    return arrays, {key: (None if isinstance(value, float) and not np.isfinite(value) else value)
                    for key, value in score.items()}


def run(args):
    manifest = Path(args.manifest)
    ids = [x.strip() for x in manifest.read_text().splitlines() if x.strip()]
    if len(ids) != len(set(ids)) or any(not re.fullmatch(r'\d+_\d+_\d+_\d+', x) for x in ids):
        raise ValueError('Invalid or duplicate manifest IDs')
    if len(ids) != args.expected_count:
        raise ValueError(f'Expected {args.expected_count} IDs; found {len(ids)}')
    excluded_records = json.loads(Path(args.exclude_failures).read_text())
    excluded = {r['clip_id'] for r in excluded_records}
    if not excluded <= set(ids):
        raise ValueError('Exclusion list contains IDs outside this manifest')
    output = Path(args.output_root).resolve()
    if output.exists():
        raise FileExistsError(f'Output exists: {output}; select a new output root')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f'.{output.name}-', dir=output.parent))
    config = dict(manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                  rate_hz=25, hann_window=31, sensitivity_hann_window=61,
                  max_lag_seconds=2, standardization='per clip, population std, eps=1e-8',
                  time_alignment='coincident clip starts; no fitted shift',
                  music_support='midpoints strictly before decoded audio duration',
                  exclusion_reason='User deferred 57 CUDA OOM clips; duration-related selection bias possible',
                  quadrant_thresholds='medians of quality-ok pairs in this run; ties high r / low difference',
                  video_root=str(Path(args.video_root).resolve()),
                  music_root=str(Path(args.music_root).resolve()),
                  save_aligned=args.save_aligned)
    rows, issues = [], []
    try:
        write_json(temporary / 'analysis_config.json', config)
        for index, clip in enumerate(sorted(ids)):
            if clip in excluded:
                issues.append(dict(clip_id=clip, status='excluded', reason='deferred_cuda_oom'))
                continue
            movie = clip.split('_')[0]
            vdir = Path(args.video_root) / args.split / movie / clip
            mdir = Path(args.music_root) / args.split / movie / clip
            try:
                vm = json.loads((vdir / 'metadata.json').read_text())
                mm = json.loads((mdir / 'metadata.json').read_text())
                if vm.get('clip_id') != clip or mm.get('clip_id') != clip:
                    raise ValueError('Metadata clip ID mismatch')
                bc = mm['batch_config']
                if (bc['input_kind'], bc['layer_ix'], bc['sample_rate'], bc['frame_offset']) != ('wav', 7, 24000, 1):
                    raise ValueError('Incompatible audio extraction settings')
                vr = np.load(vdir / f'{clip}_raw_event_curve.npy', allow_pickle=False)
                vt = np.load(vdir / f'{clip}_native_timestamps.npy', allow_pickle=False)
                with np.load(mdir / 'music_raw.npz', allow_pickle=False) as data:
                    mr, mt = data['raw'], data['timestamps']
                if any(x.ndim != 1 for x in (vr, vt, mr, mt)):
                    raise ValueError('Expected one-dimensional curves and timestamps')
                arrays, score = score_pair(vr, vt, mr, mt, vm['duration'], mm['duration'])
                row = dict(clip_id=clip, split=args.split, movie_id=movie, **score,
                           video_duration_seconds=vm['duration'], music_duration_seconds=mm['duration'],
                           duration_difference_seconds=mm['duration'] - vm['duration'],
                           video_input_sha256=vm['input_sha256'], audio_sha256=mm['input_sha256'])
                rows.append(row)
                if args.save_aligned:
                    dest = temporary / 'clips' / args.split / movie / clip
                    dest.mkdir(parents=True)
                    np.savez_compressed(dest / 'aligned_curves.npz', **arrays)
            except Exception as exc:
                issues.append(dict(clip_id=clip, status='failed', reason=f'{type(exc).__name__}: {exc}'))
            if (index + 1) % 500 == 0:
                print(json.dumps(dict(visited=index + 1, scored=len(rows))), flush=True)
        eligible = [r for r in rows if r['quality_flag'] == 'ok' and r['pearson_zero_lag'] is not None]
        rt = float(np.median([r['pearson_zero_lag'] for r in eligible])) if eligible else None
        dt = float(np.median([r['mean_absolute_difference'] for r in eligible])) if eligible else None
        eligible_ids = {r["clip_id"] for r in eligible}
        groups = {}
        for row in rows:
            row['quadrant'] = 'review'
            if row["clip_id"] in eligible_ids:
                row['quadrant'] = ('high_correlation' if row['pearson_zero_lag'] >= rt else 'low_correlation') + ('_low_difference' if row['mean_absolute_difference'] <= dt else '_high_difference')
            groups[row['quadrant']] = groups.get(row['quadrant'], 0) + 1
        rows.sort(key=lambda r: (-(r['pearson_zero_lag'] if r['pearson_zero_lag'] is not None else -2), r['clip_id']))
        if rows:
            with (temporary / 'alignment_scores.csv').open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
        with (temporary / 'issues.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['clip_id', 'status', 'reason'])
            writer.writeheader(); writer.writerows(issues)
        summary = dict(expected=len(ids), scored=len(rows), excluded=len(excluded),
                       failed=sum(r['status'] == 'failed' for r in issues), eligible=len(eligible),
                       pearson_threshold=rt, mean_absolute_difference_threshold=dt, quadrant_counts=groups,
                       note='Saved-curve audit only; input media hashes are recorded, not recomputed.')
        write_json(temporary / 'summary.json', summary)
        temporary.rename(output)
        print(json.dumps(summary), flush=True)
        return int(summary['failed'] > 0)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--video-root', required=True)
    p.add_argument('--music-root', required=True)
    p.add_argument('--output-root', required=True)
    p.add_argument('--exclude-failures', required=True)
    p.add_argument('--expected-count', type=int, default=31011)
    p.add_argument('--split', choices=['train', 'test'], default='train')
    p.add_argument('--save-aligned', action='store_true')
    raise SystemExit(run(p.parse_args()))


if __name__ == '__main__':
    main()
