"""Freeze WebShop human goals and build fixed choice-set tasks.

Goal order reproduces `SimServer` exactly (load -> get_human_goals -> seed 233
shuffle), so goal indices 0-499 are WebShop's standard test split and 500-1499
its dev split. Unlike upstream, all randomness (sampled prices for products with
a price range, and each goal's price ceiling) is seeded, so tasks are frozen.

Every candidate is scored with its best-matching options ("oracle"). Three
distractor strategies:
  robust (default): like near_miss, but a distractor must also fail at least
      one requirement under the lenient checks in lenient.py, so products that
      satisfy the instruction but trip WebShop's string matching are excluded.
      Output: tasks_<split>_robust.jsonl
  near_miss: from a pool of the top BM25 hits plus products scraped under the
      same Amazon query, drop every product that could also earn a perfect
      reward and keep the highest-scoring same-type non-matches.
      Output: tasks_<split>_nearmiss.jsonl
  bm25: the top non-target BM25 hits for the instruction. Output: tasks_<split>.jsonl
For robust and near_miss, each task has exactly one correct product, and goals
whose target cannot reach a perfect reward are skipped.

Usage (repo root, Python 3.8 `webshop` env):
    python experiments/build_tasks.py --split test
    python experiments/build_tasks.py --split dev --limit 100
"""
import argparse
import functools
import json
import os
import random
import sys
from collections import defaultdict
from os.path import abspath, dirname, join

REPO_DIR = dirname(dirname(abspath(__file__)))
sys.path.insert(0, REPO_DIR)

from thefuzz import fuzz  # noqa: E402
from tqdm import tqdm  # noqa: E402

import lenient  # noqa: E402
import web_agent_site.engine.goal as goal_module  # noqa: E402
from web_agent_site.engine.engine import init_search_engine, load_products  # noqa: E402
from web_agent_site.engine.goal import get_human_goals, get_reward  # noqa: E402
from web_agent_site.engine.normalize import normalize_color  # noqa: E402
from web_agent_site.utils import DEFAULT_FILE_PATH  # noqa: E402

# get_type_reward re-parses the same product names many times; caching the
# spaCy parse leaves rewards unchanged and makes pool scoring fast.
goal_module.nlp = functools.lru_cache(maxsize=None)(goal_module.nlp)

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


def oracle(product, goal, price):
    options = best_options(product, goal['goal_options'])
    reward, info = get_reward(product, goal, price=price, options=options, verbose=True)
    failures = [name for name, failed in [
        ('type', info['r_type'] < 1),
        ('attributes', info['r_att'] < 1),
        ('options', info.get('r_option') is not None and info['r_option'] < 1),
        ('price', info.get('r_price') is False),
    ] if failed]
    return options, reward, info, failures, lenient.check(product, goal, price)


def bm25_asins(search_engine, query, k, product_item_dict):
    asins = []
    for hit in search_engine.search(query, k=k):
        asin = json.loads(search_engine.doc(hit.docid).raw())['id']
        if asin in product_item_dict and asin not in asins:
            asins.append(asin)
    return asins


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=SPLITS, required=True)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--distractors', choices=['robust', 'near_miss', 'bm25'], default='robust')
    parser.add_argument('--num-candidates', type=int, default=10)
    parser.add_argument('--pool-size', type=int, default=100)
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
    asins_by_query = defaultdict(list)
    for p in all_products:
        asins_by_query[p['query']].append(p['asin'])

    start, end = SPLITS[args.split]
    if args.limit is not None:
        end = min(end, start + args.limit)
    label = args.split + {'bm25': '', 'near_miss': '_nearmiss', 'robust': '_robust'}[args.distractors]
    n_distractors = args.num_candidates - 1

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = join(args.out_dir, f'tasks_{label}.jsonl')
    n_written, skipped = 0, []
    with open(out_path, 'w') as f:
        for goal_idx in tqdm(range(start, end), desc=f'{label} tasks'):
            goal = goals[goal_idx]
            assert isinstance(goal['goal_options'], list)
            target_asin = goal['asin']
            query = goal['instruction_text'].lower()
            rng = random.Random(f'{args.seed}-{goal_idx}')
            scored = {}

            def score(asin):
                if asin not in scored:
                    scored[asin] = oracle(product_item_dict[asin], goal, round(product_prices[asin], 2))
                return scored[asin]

            if args.distractors != 'bm25' and score(target_asin)[1] < 1.0:
                skipped.append(goal_idx)
                continue

            hit_asins = bm25_asins(search_engine, query, 50, product_item_dict)
            bm25_rank = {a: i + 1 for i, a in enumerate(hit_asins)}
            if args.distractors == 'bm25':
                distractors = [a for a in hit_asins if a != target_asin][:n_distractors]
            else:
                pool = bm25_asins(search_engine, query, args.pool_size, product_item_dict)
                same_query = [a for a in asins_by_query[goal['query']] if a not in pool]
                pool += rng.sample(same_query, min(len(same_query), args.pool_size))
                pool = [a for a in pool if a != target_asin and score(a)[1] < 1.0
                        and (args.distractors == 'near_miss' or not score(a)[4]['could_satisfy'])]
                pool.sort(key=lambda a: (score(a)[2]['r_type'] < 1, -score(a)[1],
                                         bm25_rank.get(a, len(bm25_rank) + 1)))
                distractors = pool[:n_distractors]

            asins = [target_asin] + distractors
            rng.shuffle(asins)
            candidates = []
            for position, asin in enumerate(asins, start=1):
                options, reward, info, failures, lenient_check = score(asin)
                candidates.append({
                    'position': position,
                    'asin': asin,
                    'price': round(product_prices[asin], 2),
                    'product': {k: product_item_dict[asin][k] for k in PRODUCT_FIELDS},
                    'oracle_options': options,
                    'oracle_reward': reward,
                    'oracle_info': info,
                    'oracle_failures': failures,
                    'lenient_check': lenient_check,
                })

            oracle_rewards = [c['oracle_reward'] for c in candidates]
            target_position = asins.index(target_asin) + 1
            task = {
                'task_id': f'{label}-{goal_idx:04d}',
                'split': label,
                'goal_idx': goal_idx,
                'distractor_strategy': args.distractors,
                'goal': {k: goal[k] for k in GOAL_FIELDS},
                'search_query': query,
                'target_asin': target_asin,
                'target_position': target_position,
                'target_bm25_rank': bm25_rank.get(target_asin),
                'num_candidates': len(candidates),
                'max_oracle_reward': max(oracle_rewards),
                'num_perfect_candidates': sum(r == 1.0 for r in oracle_rewards),
                'target_oracle_reward': candidates[target_position - 1]['oracle_reward'],
                'candidates': candidates,
            }
            f.write(json.dumps(task) + '\n')
            n_written += 1
    print(f'wrote {n_written} tasks to {out_path}; skipped {len(skipped)} goals whose '
          f'target cannot reach reward 1.0: {skipped}')


if __name__ == '__main__':
    main()
