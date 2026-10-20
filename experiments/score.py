"""Score model choices with WebShop's original reward function.

Reads data/tasks/tasks_<split>.jsonl and every results/raw/<split>/<model>/<effort>.jsonl,
keeps the latest successful record per (model, effort, task, sample), and writes one
row per response to results/scored/<split>.csv. Responses with an API error or an
unparseable answer get `valid = False` and no reward.

Usage (repo root, Python 3.8 `webshop` env):
    python experiments/score.py --split dev
"""
import argparse
import glob
import json
import os
import sys
from os.path import abspath, dirname, join

REPO_DIR = dirname(dirname(abspath(__file__)))
sys.path.insert(0, REPO_DIR)

import pandas as pd  # noqa: E402

from web_agent_site.engine.goal import get_reward  # noqa: E402


def load_records(raw_dir):
    """Latest record per (model, effort, task, sample), preferring successful ones."""
    best = {}
    for path in sorted(glob.glob(join(raw_dir, '*', '*.jsonl'))):
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                key = (r['model_key'], r['effort'], r['task_id'], r['sample'])
                ok = r.get('error') is None and r.get('content') is not None
                prev = best.get(key)
                if prev is None or ok or not (prev.get('error') is None and prev.get('content') is not None):
                    best[key] = r
    return list(best.values())


def score_record(r, task):
    usage = r.get('usage') or {}
    details = usage.get('completion_tokens_details') or {}
    parsed = r.get('parsed')
    row = {
        'task_id': r['task_id'],
        'sample': r['sample'],
        'model_key': r['model_key'],
        'model': r['model'],
        'effort': r['effort'],
        'provider': r.get('provider'),
        'error': r.get('error'),
        'parse_error': r.get('parse_error'),
        'finish_reason': r.get('finish_reason'),
        'valid': parsed is not None,
        'target_asin': task['target_asin'],
        'target_position': task['target_position'],
        'target_bm25_rank': task['target_bm25_rank'],
        'max_oracle_reward': task['max_oracle_reward'],
        'num_perfect_candidates': task['num_perfect_candidates'],
        'num_goal_attributes': len(task['goal']['attributes']),
        'num_goal_options': len(task['goal']['goal_options']),
        'prompt_tokens': usage.get('prompt_tokens'),
        'completion_tokens': usage.get('completion_tokens'),
        'reasoning_tokens': details.get('reasoning_tokens'),
        'cost': usage.get('cost'),
        'latency_s': r.get('latency_s'),
    }
    if parsed is None:
        return row

    candidate = task['candidates'][parsed['position'] - 1]
    reward, info = get_reward(candidate['product'], task['goal'], price=candidate['price'],
                              options=parsed['options'], verbose=True)
    row.update({
        'chosen_position': parsed['position'],
        'chosen_asin': parsed['asin'],
        'chose_target': parsed['asin'] == task['target_asin'],
        'options': json.dumps(parsed['options']),
        'num_invalid_options': len(parsed['invalid_options']),
        'num_missing_option_types': len(parsed['missing_option_types']),
        'confidence': parsed['confidence'],
        'reward': reward,
        'success': reward == 1.0,
        'chosen_oracle_reward': candidate['oracle_reward'],
        'chosen_product_perfect': candidate['oracle_reward'] == 1.0,
        'r_type': info.get('r_type'),
        'r_att': info.get('r_att'),
        'r_option': info.get('r_option'),
        'r_price': info.get('r_price'),
    })
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', required=True)
    parser.add_argument('--tasks', default=None)
    parser.add_argument('--raw-dir', default=None)
    parser.add_argument('--out-dir', default=join(REPO_DIR, 'results', 'scored'))
    args = parser.parse_args()

    tasks_path = args.tasks or join(REPO_DIR, 'data', 'tasks', f'tasks_{args.split}.jsonl')
    raw_dir = args.raw_dir or join(REPO_DIR, 'results', 'raw', args.split)
    with open(tasks_path) as f:
        tasks = {t['task_id']: t for t in map(json.loads, f)}

    records = load_records(raw_dir)
    rows = [score_record(r, tasks[r['task_id']]) for r in records if r['task_id'] in tasks]
    df = pd.DataFrame(rows).sort_values(['model_key', 'effort', 'task_id', 'sample'])

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = join(args.out_dir, f'{args.split}.csv')
    df.to_csv(out_path, index=False)
    print(f'scored {len(df)} responses from {raw_dir} -> {out_path}')
    if len(df):
        summary = df.groupby(['model_key', 'effort']).agg(
            n=('task_id', 'size'), valid=('valid', 'mean'), success=('success', 'mean'),
            reward=('reward', 'mean'), confidence=('confidence', 'mean'),
            reasoning_tokens=('reasoning_tokens', 'mean'))
        print(summary.round(3).to_string())


if __name__ == '__main__':
    main()
