"""Sample distractors into a spreadsheet for hand-checking that each truly violates its instruction.

Fill in `violates_instruction` (y / n / unsure) per row; the share of "n" estimates
how often a distractor is actually acceptable (a false "wrong answer").

Usage (repo root):
    python experiments/make_audit.py --tasks data/tasks/tasks_test_robust.jsonl --n 50
"""
import argparse
import csv
import json
import os
import random
from os.path import abspath, dirname, join

REPO_DIR = dirname(dirname(abspath(__file__)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', required=True)
    parser.add_argument('--n', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', default=join(REPO_DIR, 'results', 'audit', 'distractor_audit.csv'))
    args = parser.parse_args()

    with open(args.tasks) as f:
        tasks = [json.loads(line) for line in f]
    distractors = [(t, c) for t in tasks for c in t['candidates'] if c['asin'] != t['target_asin']]
    sample = random.Random(args.seed).sample(distractors, min(args.n, len(distractors)))

    os.makedirs(dirname(args.out), exist_ok=True)
    with open(args.out, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['row', 'task_id', 'instruction', 'product_title', 'price', 'options',
                         'features', 'description', 'detected_failures', 'missing_features',
                         'unavailable_options', 'violates_instruction', 'notes'])
        for i, (t, c) in enumerate(sample, start=1):
            p, check = c['product'], c['lenient_check']
            writer.writerow([
                i, t['task_id'], t['goal']['instruction_text'], p['Title'], f'{c["price"]:.2f}',
                '; '.join(f'{k}: {", ".join(v)}' for k, v in p['options'].items()),
                ' | '.join(p['BulletPoints'])[:1500], (p['Description'] or '')[:1500],
                ', '.join(check['failures']), ', '.join(check['failed_attributes']),
                ', '.join(check['unavailable_options']), '', '',
            ])
    print(f'wrote {len(sample)} distractors to {args.out}')


if __name__ == '__main__':
    main()
