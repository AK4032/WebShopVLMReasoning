"""Build a 100k-product WebShop catalog from the full 1.18M-product release.

The subset keeps every product that has a crowd-sourced (human) instruction,
so all human goals remain solvable, and fills the rest with a seeded random
sample of the remaining products. Product order follows the original shuffled
file. The full file is streamed, so memory use stays low.

Usage (from repo root, inside the Python 3.8 `webshop` env):
    python experiments/build_subset.py --full-dir C:/webshop_full --out-dir data
"""
import argparse
import json
import random
from os.path import join

import ijson
from tqdm import tqdm

FULL_NUM_PRODUCTS = 1181436


def is_valid_asin(asin):
    return isinstance(asin, str) and asin != 'nan' and len(asin) <= 10


def iter_products(path):
    with open(path, 'rb') as f:
        yield from ijson.items(f, 'item', use_float=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full-dir', required=True)
    parser.add_argument('--out-dir', default='data')
    parser.add_argument('--num-products', type=int, default=100_000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--suffix', default='100k')
    args = parser.parse_args()

    with open(join(args.out_dir, 'items_human_ins.json')) as f:
        human_asins = set(json.load(f))

    # Pass 1: collect unique valid ASINs in file order (mirrors load_products dedup)
    seen, order = set(), []
    for p in tqdm(iter_products(join(args.full_dir, 'items_shuffle.json')),
                  total=FULL_NUM_PRODUCTS, desc='pass 1/2 (scan)'):
        asin = p.get('asin')
        if is_valid_asin(asin) and asin not in seen:
            seen.add(asin)
            order.append(asin)

    human_in_catalog = [a for a in order if a in human_asins]
    others = [a for a in order if a not in human_asins]
    num_fill = args.num_products - len(human_in_catalog)
    if num_fill < 0:
        raise ValueError(f'{len(human_in_catalog)} human products exceed --num-products')
    rng = random.Random(args.seed)
    selected = set(human_in_catalog) | set(rng.sample(others, num_fill))

    # Pass 2: write selected products in original order
    out_products = join(args.out_dir, f'items_shuffle_{args.suffix}.json')
    written = set()
    with open(out_products, 'w') as f:
        f.write('[')
        for p in tqdm(iter_products(join(args.full_dir, 'items_shuffle.json')),
                      total=FULL_NUM_PRODUCTS, desc='pass 2/2 (write)'):
            asin = p.get('asin')
            if asin in selected and asin not in written:
                f.write((',' if written else '') + json.dumps(p))
                written.add(asin)
        f.write(']')

    with open(join(args.full_dir, 'items_ins_v2.json')) as f:
        attributes = json.load(f)
    sub_attributes = {a: attributes[a] for a in written if a in attributes}
    out_attrs = join(args.out_dir, f'items_ins_v2_{args.suffix}.json')
    with open(out_attrs, 'w') as f:
        json.dump(sub_attributes, f)

    manifest = {
        'seed': args.seed,
        'num_products': len(written),
        'num_unique_valid_asins_in_full': len(order),
        'num_human_instruction_asins': len(human_asins),
        'num_human_instruction_asins_in_catalog': len(human_in_catalog),
        'num_random_fill': num_fill,
        'num_with_attributes': len(sub_attributes),
        'products_file': out_products,
        'attributes_file': out_attrs,
    }
    with open(join(args.out_dir, f'subset_{args.suffix}_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
