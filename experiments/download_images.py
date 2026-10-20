"""Download and freeze the main image of every candidate product in the task files.

Images are saved as `<asin>.jpg` (RGB, longest side <= --max-side px) so every
model sees identical stimuli even if the Amazon CDN copies change later.

Usage (repo root):
    python experiments/download_images.py data/tasks/tasks_test.jsonl data/tasks/tasks_dev.jsonl
"""
import argparse
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor
from os.path import abspath, dirname, exists, join

import requests
from PIL import Image
from tqdm import tqdm

REPO_DIR = dirname(dirname(abspath(__file__)))


def fetch(asin, url, out_dir, max_side):
    path = join(out_dir, f'{asin}.jpg')
    if exists(path):
        return asin, 'cached'
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert('RGB')
        img.thumbnail((max_side, max_side))
        img.save(path, 'JPEG', quality=90)
        return asin, 'ok'
    except Exception as e:  # noqa: BLE001
        return asin, f'error: {e}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task_files', nargs='+')
    parser.add_argument('--out-dir', default=join(REPO_DIR, 'data', 'images'))
    parser.add_argument('--max-side', type=int, default=512)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()

    urls = {}
    for path in args.task_files:
        with open(path) as f:
            for line in f:
                for c in json.loads(line)['candidates']:
                    urls[c['asin']] = c['product']['MainImage']

    os.makedirs(args.out_dir, exist_ok=True)
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(tqdm(
            pool.map(lambda kv: fetch(*kv, args.out_dir, args.max_side), urls.items()),
            total=len(urls), desc='images'))

    failures = {asin: status for asin, status in results if status.startswith('error')}
    with open(join(args.out_dir, 'failures.json'), 'w') as f:
        json.dump(failures, f, indent=2)
    print(f'{len(urls)} images: {len(urls) - len(failures)} saved, {len(failures)} failed')


if __name__ == '__main__':
    main()
