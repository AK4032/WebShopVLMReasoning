"""Freeze WebShop human goals and build fixed choice-set tasks.

Goal order reproduces `SimServer` exactly (load -> get_human_goals -> seed 233
shuffle), so goal indices 0-499 are WebShop's standard test split and 500-1499
its dev split. Unlike upstream, all randomness (sampled prices for products with
a price range, and each goal's price ceiling) is seeded, so tasks are frozen.

Each task presents the goal's target product plus the top non-target BM25 hits
for the instruction text, in a seeded random order. Every candidate is also
scored with its best-matching options ("oracle"), so we know how many
candidates can earn a perfect reward.

Usage (repo root, Python 3.8 `webshop` env):
    python experiments/build_tasks.py --split test
    python experiments/build_tasks.py --split dev --limit 100
"""
import argparse
import json
import os
import random
import sys
from os.path import abspath, dirname, join

REPO_DIR = dirname(dirname(abspath(__file__)))
sys.path.insert(0, REPO_DIR)

from thefuzz import fuzz  # noqa: E402
from tqdm import tqdm  # noqa: E402

from web_agent_site.engine.engine import (  # noqa: E402
    get_top_n_product_from_keywords, init_search_engine, load_products,
)
from web_agent_site.engine.goal import get_human_goals, get_reward  # noqa: E402
from web_agent_site.engine.normalize import normalize_color  # noqa: E402
from web_agent_site.utils import DEFAULT_FILE_PATH  # noqa: E402

SPLITS = {'test': (0, 500), 'dev': (500, 1500)}
GOAL_FIELDS = ['asin', 'instruction_text', 'attributes', 'goal_options', 'price_upper',
               'name', 'query', 'product_category', 'category']
PRODUCT_FIELDS = ['asin', 'Title', 'BulletPoints', 'Description', 'options', 'MainImage',
                  'name', 'query', 'product_category', 'category', 'Attributes']


def best_options(product, goal_options):
    goal_opts = [normalize_color(g) for g in goal_options]
    chosen = {}
    for name, values in product['options'].items():
        if values:
            chosen[name] = max(values, key=lambda v: max(
                [fuzz.token_set_ratio(normalize_color(v), g) for g in goal_opts] or [0]))
    return chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=SPLITS, required=True)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--num-candidates', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out-dir', default=join(REPO_DIR, 'data', 'tasks'))
    args = parser.parse_args()

    random.seed(args.seed)
    all_products, product_item_dict, product_prices, _ = load_products(
        filepath=DEFAULT_FILE_PATH, num_products=None, human_goals=True)
    goals = get_human_goals(all_products, product_prices)
    random.seed(233)
    random.shuffle(goals)
    search_engine = init_search_engine(num_products=None)

    start, end = SPLITS[args.split]
    if args.limit is not None:
        end = min(end, start + args.limit)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = join(args.out_dir, f'tasks_{args.split}.jsonl')
    with open(out_path, 'w') as f:
        for goal_idx in tqdm(range(start, end), desc=f'{args.split} tasks'):
            goal = goals[goal_idx]
            assert isinstance(goal['goal_options'], list)
            target_asin = goal['asin']
            query = goal['instruction_text'].lower()
            hits = get_top_n_product_from_keywords(
                query.split(), search_engine, all_products, product_item_dict)
            hit_asins = [p['asin'] for p in hits]
            target_rank = hit_asins.index(target_asin) + 1 if target_asin in hit_asins else None

            distractors = []
            for asin in hit_asins:
                if asin != target_asin and asin not in distractors:
                    distractors.append(asin)
                if len(distractors) == args.num_candidates - 1:
                    break
            asins = [target_asin] + distractors
            random.Random(f'{args.seed}-{goal_idx}').shuffle(asins)

            candidates = []
            for position, asin in enumerate(asins, start=1):
                product = product_item_dict[asin]
                price = round(product_prices[asin], 2)
                options = best_options(product, goal['goal_options'])
                reward, info = get_reward(product, goal, price=price, options=options, verbose=True)
                candidates.append({
                    'position': position,
                    'asin': asin,
                    'price': price,
                    'product': {k: product[k] for k in PRODUCT_FIELDS},
                    'oracle_options': options,
                    'oracle_reward': reward,
                    'oracle_info': info,
                })

            oracle_rewards = [c['oracle_reward'] for c in candidates]
            task = {
                'task_id': f'{args.split}-{goal_idx:04d}',
                'split': args.split,
                'goal_idx': goal_idx,
                'goal': {k: goal[k] for k in GOAL_FIELDS},
                'search_query': query,
                'target_asin': target_asin,
                'target_position': asins.index(target_asin) + 1,
                'target_bm25_rank': target_rank,
                'num_candidates': len(candidates),
                'max_oracle_reward': max(oracle_rewards),
                'num_perfect_candidates': sum(r == 1.0 for r in oracle_rewards),
                'target_oracle_reward': candidates[asins.index(target_asin)]['oracle_reward'],
                'candidates': candidates,
            }
            f.write(json.dumps(task) + '\n')
    print(f'wrote {end - start} tasks to {out_path}')


if __name__ == '__main__':
    main()
