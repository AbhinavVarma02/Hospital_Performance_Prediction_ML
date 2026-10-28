"""Fetch fingerprinted source files explicitly, outside notebook execution."""
import argparse
import json
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hospital_ml.data import sha256

OFFICIAL_URL = ('https://data.chhs.ca.gov/dataset/ea0c8ca9-023e-46a3-b95b-b9d4ab8ec195/'
                'resource/81602900-aa04-4fcb-b70b-19bc0553f79b/download/'
                'selected-file_2012_2017_machinereadable.csv')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--historical-cleaned', action='store_true', help='Also fetch the original shared cleaned CSV.')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'data/source_manifest.json').read_text())
    for kind in ['raw'] + (['historical_cleaned'] if args.historical_cleaned else []):
        source = manifest[kind]
        destination = ROOT / source['path']
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            temporary = destination.with_suffix('.download')
            if kind == 'raw':
                with urllib.request.urlopen(source['url'], timeout=120) as response, open(temporary, 'wb') as stream:
                    while block := response.read(1024 * 1024):
                        stream.write(block)
            else:
                import gdown
                if gdown.download(id=source['drive_id'], output=str(temporary), use_cookies=False) is None:
                    raise RuntimeError('Historical shared file is unavailable; supply an independently verified local copy.')
            if sha256(temporary) != source['sha256']:
                temporary.unlink()
                raise ValueError('Source fingerprint changed. Verify the source version before continuing.')
            temporary.rename(destination)
        if sha256(destination) != source['sha256']:
            raise ValueError(f'Unexpected fingerprint for {destination}; verify source version.')
        print(f'Verified {source["path"]}: {source["sha256"]}')


if __name__ == '__main__':
    main()
