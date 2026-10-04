"""Download all pinned inference assets and verify their SHA-256 checksums."""
import argparse
import hashlib
import json
from pathlib import Path
from huggingface_hub import hf_hub_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache_dir', default='.cache/huggingface')
    parser.add_argument('--local_files_only', action='store_true')
    args = parser.parse_args()
    manifest = Path(__file__).resolve().parents[1] / 'weights-manifest.json'
    for asset in json.loads(manifest.read_text()):
        path = Path(hf_hub_download(asset['repo_id'], asset['filename'],
            revision=asset['revision'], cache_dir=args.cache_dir,
            local_files_only=args.local_files_only))
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(block)
        if path.stat().st_size != asset['size_bytes'] or digest.hexdigest() != asset['sha256']:
            raise RuntimeError(f'Checksum mismatch: {path}')
        print(f"Verified {asset['repo_id']}/{asset['filename']}", flush=True)
    print('All weights and configs are ready for --local_files_only extraction.')


if __name__ == '__main__':
    main()
