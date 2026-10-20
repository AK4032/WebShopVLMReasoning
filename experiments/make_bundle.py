"""Package everything Mercury needs into dist/mercury_bundle.zip.

Includes the runner code, model config, Slurm scripts, task files, and frozen
product images (not the 100k catalog or search indexes, which only the local
task builder and scorer use). Text files get Unix line endings so bash on
Mercury can run them.

Usage (repo root):
    python experiments/make_bundle.py
"""
import glob
import os
import zipfile
from os.path import abspath, dirname, getsize, join, relpath

REPO_DIR = dirname(dirname(abspath(__file__)))
TEXT_PATTERNS = ['experiments/*.py', 'experiments/configs/*.yaml', 'slurm/*.sh',
                 'slurm/*.sbatch', 'slurm/README.md', 'data/tasks/*.jsonl']
BINARY_PATTERNS = ['data/images/*.jpg']
EMPTY_DIRS = ['slurm/logs/', 'results/']


def main():
    os.makedirs(join(REPO_DIR, 'dist'), exist_ok=True)
    out_path = join(REPO_DIR, 'dist', 'mercury_bundle.zip')
    n_files = 0
    with zipfile.ZipFile(out_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for pattern in TEXT_PATTERNS:
            for path in sorted(glob.glob(join(REPO_DIR, pattern))):
                with open(path, 'rb') as f:
                    zf.writestr(relpath(path, REPO_DIR).replace(os.sep, '/'), f.read().replace(b'\r\n', b'\n'))
                n_files += 1
        for pattern in BINARY_PATTERNS:
            for path in sorted(glob.glob(join(REPO_DIR, pattern))):
                zf.write(path, relpath(path, REPO_DIR).replace(os.sep, '/'))
                n_files += 1
        for d in EMPTY_DIRS:
            zf.writestr(d, '')
    print(f'{n_files} files -> {out_path} ({getsize(out_path) / 1e6:.0f} MB)')


if __name__ == '__main__':
    main()
