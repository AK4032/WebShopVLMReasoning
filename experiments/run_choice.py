"""Query OpenRouter models on WebShop choice-set tasks.

Each request shows one human instruction and its fixed candidate products
(text + main image); the model returns the product number, its option values,
and a 0-10 confidence. One run covers one (model, effort) pair; results are
appended to results/raw/<split>/<model key>/<effort>.jsonl, and re-running
skips (task, sample) pairs that already succeeded.

Runs on Python 3.8+ with `httpx` and `pyyaml`.

Examples (repo root):
    python experiments/run_choice.py --tasks data/tasks/tasks_dev.jsonl \
        --model-key claude-opus-5.5 --effort low --limit 20
    python experiments/run_choice.py --tasks data/tasks/tasks_test.jsonl \
        --group prelim --array-index $SLURM_ARRAY_TASK_ID --limit 150
    python experiments/run_choice.py --tasks data/tasks/tasks_dev.jsonl \
        --model-key gpt-6.1-sol --effort low --limit 1 --dry-run
    python experiments/run_choice.py --group full --list-combos
"""
import argparse
import asyncio
import base64
import json
import os
import random
import re
import time
from datetime import datetime, timezone
from os.path import abspath, dirname, exists, expanduser, join

import httpx
import yaml

REPO_DIR = dirname(dirname(abspath(__file__)))
API_URL = 'https://openrouter.ai/api/v1/chat/completions'
PROMPT_VERSION = 'choice-v1'
MAX_DESCRIPTION_CHARS = 1500
MAX_ATTEMPTS = 6
RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529}

SYSTEM_PROMPT = (
    'You are a shopping assistant. A customer gives you a shopping instruction, '
    'and you must buy the single product, with the right options, that best '
    'satisfies it.'
)

ANSWER_INSTRUCTIONS = (
    'Choose exactly one product. For each of its option types, choose exactly one '
    'of the listed values. Then rate how confident you are that your chosen product '
    'and options satisfy every requirement in the customer instruction, as an integer '
    'from 0 (not at all confident) to 10 (completely confident).\n\n'
    'Respond with only a JSON object in this format:\n'
    '{"product": <product number>, "options": {"<option type>": "<value>"}, '
    '"confidence": <integer 0-10>}\n'
    'Use "options": {} if the chosen product has no options.'
)


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def combos_for(config, group):
    """(model entry, effort) pairs for a group, using each model's supported effort levels."""
    return [(m, e) for m in config['groups'][group] for e in m.get('efforts', config['efforts'])]


def resolve_run(config, args):
    """Return (model entry, effort) from --model-key/--effort or --group/--array-index."""
    if args.group is not None:
        combos = combos_for(config, args.group)
        if args.array_index is None or not 0 <= args.array_index < len(combos):
            raise SystemExit(f'--array-index must be in [0, {len(combos) - 1}] for group {args.group}')
        return combos[args.array_index]
    models = {m['key']: m for group in config['groups'].values() for m in group}
    if args.model_key not in models or args.effort is None:
        raise SystemExit('give --model-key and --effort, or --group and --array-index')
    model = models[args.model_key]
    if args.effort not in model.get('efforts', config['efforts']):
        raise SystemExit(f'{args.model_key} supports efforts {model.get("efforts", config["efforts"])}')
    return model, args.effort


def format_product(position, candidate):
    p = candidate['product']
    lines = [f'--- Product {position} ---', f'Title: {p["Title"]}', f'Price: ${candidate["price"]:.2f}']
    bullets = [b.strip() for b in p['BulletPoints'] if b and b.strip()]
    if bullets:
        lines.append('Features:')
        lines.extend(f'- {b}' for b in bullets)
    description = (p['Description'] or '').strip()
    if description:
        if len(description) > MAX_DESCRIPTION_CHARS:
            description = description[:MAX_DESCRIPTION_CHARS].rstrip() + ' ...'
        lines.append(f'Description: {description}')
    if p['options']:
        lines.append('Options:')
        lines.extend(f'  {name}: {json.dumps(values, ensure_ascii=False)}' for name, values in p['options'].items())
    else:
        lines.append('Options: none')
    return '\n'.join(lines)


def image_part(asin, images_dir):
    path = join(images_dir, f'{asin}.jpg')
    if not exists(path):
        return None
    with open(path, 'rb') as f:
        data = base64.b64encode(f.read()).decode('ascii')
    return {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{data}'}}


def build_messages(task, images_dir, use_images):
    n = task['num_candidates']
    shown = 'title, price, features, description, and selectable options'
    if use_images:
        shown += ', followed by its main image'
    content = [{'type': 'text', 'text': (
        f'Customer instruction: {task["goal"]["instruction_text"]}\n\n'
        f'Here are {n} products. Each lists its {shown}.'
    )}]
    missing_images = []
    for c in task['candidates']:
        content.append({'type': 'text', 'text': format_product(c['position'], c)})
        if use_images:
            part = image_part(c['asin'], images_dir)
            if part is None:
                missing_images.append(c['asin'])
                content.append({'type': 'text', 'text': '(image unavailable)'})
            else:
                content.append(part)
    content.append({'type': 'text', 'text': ANSWER_INSTRUCTIONS})
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': content}]
    return messages, missing_images


def extract_json(text):
    """Return the last JSON object in `text` that has a "product" key."""
    if not text:
        return None
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and 'product' in obj:
            return obj
    except ValueError:
        pass
    decoder = json.JSONDecoder()
    found = None
    for match in re.finditer(r'\{', text):
        try:
            obj, _ = decoder.raw_decode(text[match.start():])
        except ValueError:
            continue
        if isinstance(obj, dict) and 'product' in obj:
            found = obj
    return found


def parse_choice(content, task):
    """Map the model's answer onto the task's candidates. Returns (parsed, error)."""
    obj = extract_json(content)
    if obj is None:
        return None, 'no JSON object with "product" found'
    try:
        position = int(obj['product'])
    except (TypeError, ValueError):
        return None, f'invalid product: {obj.get("product")!r}'
    if not 1 <= position <= task['num_candidates']:
        return None, f'product {position} out of range'
    candidate = task['candidates'][position - 1]

    valid_options = candidate['product']['options']
    raw_options = obj.get('options') or {}
    options, invalid_options = {}, {}
    if isinstance(raw_options, dict):
        lowered = {name.lower(): name for name in valid_options}
        for name, value in raw_options.items():
            key = lowered.get(str(name).strip().lower())
            value = str(value).strip().lower()
            if key is not None and value in valid_options[key]:
                options[key] = value
            else:
                invalid_options[str(name)] = value

    confidence = obj.get('confidence')
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = None
    return {
        'position': position,
        'asin': candidate['asin'],
        'options': options,
        'invalid_options': invalid_options,
        'missing_option_types': sorted(set(valid_options) - set(options)),
        'confidence': confidence,
    }, None


async def call_openrouter(client, payload, api_key):
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
        'X-Title': 'WebShop VLM reasoning study',
    }
    last_error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            resp = await client.post(API_URL, headers=headers, json=payload)
            body = resp.json() if resp.content else {}
            if resp.status_code == 200 and 'choices' in body and body['choices']:
                return body, resp.status_code, attempt, None
            last_error = f'HTTP {resp.status_code}: {json.dumps(body.get("error", body))[:500]}'
            if resp.status_code not in RETRY_STATUS and resp.status_code != 200:
                return body, resp.status_code, attempt, last_error
        except (httpx.HTTPError, ValueError) as e:
            last_error = f'{type(e).__name__}: {e}'
        await asyncio.sleep(min(2 ** attempt, 60) + random.uniform(0, 2))
    return None, None, MAX_ATTEMPTS, last_error


async def run_one(client, semaphore, task, sample, model, effort, max_tokens, args, api_key):
    use_images = model.get('images', True)
    record = {
        'task_id': task['task_id'],
        'sample': sample,
        'model_key': model['key'],
        'model': model['openrouter'],
        'effort': effort,
        'max_tokens': max_tokens,
        'images': use_images,
        'prompt_version': PROMPT_VERSION,
    }
    async with semaphore:
        messages, record['missing_images'] = build_messages(task, args.images_dir, use_images)
        payload = {
            'model': model['openrouter'],
            'messages': messages,
            'max_tokens': max_tokens,
            'reasoning': {'enabled': True} if effort == 'default' else {'effort': effort},
            'provider': {'require_parameters': True},
        }
        record['started_at'] = datetime.now(timezone.utc).isoformat()
        t0 = time.time()
        body, status, attempts, error = await call_openrouter(client, payload, api_key)
        record['latency_s'] = round(time.time() - t0, 2)
    record.update({'http_status': status, 'attempts': attempts, 'error': error})

    if body and body.get('choices'):
        choice = body['choices'][0]
        message = choice.get('message') or {}
        record.update({
            'generation_id': body.get('id'),
            'provider': body.get('provider'),
            'finish_reason': choice.get('finish_reason'),
            'content': message.get('content'),
            'reasoning': message.get('reasoning'),
            'usage': body.get('usage'),
        })
        parsed, parse_error = parse_choice(message.get('content'), task)
        record['parsed'] = parsed
        record['parse_error'] = parse_error
        if choice.get('finish_reason') == 'error':
            record['error'] = f'provider error: {json.dumps(choice.get("error") or body.get("error"))[:500]}'
    return record


def load_done(out_path):
    done = set()
    if exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get('error') is None and r.get('content') is not None:
                    done.add((r['task_id'], r['sample']))
    return done


async def main_async(args):
    config = load_config(args.config)
    if args.list_combos:
        for i, (m, e) in enumerate(combos_for(config, args.group)):
            print(f'{i}\t{m["key"]}\t{e}')
        return
    model, effort = resolve_run(config, args)
    max_tokens = args.max_tokens or config['max_tokens']
    if not args.tasks:
        raise SystemExit('--tasks is required')

    with open(args.tasks) as f:
        tasks = [json.loads(line) for line in f]
    if args.limit is not None:
        tasks = tasks[:args.limit]
    split = tasks[0]['split']

    if args.dry_run:
        messages, missing = build_messages(tasks[0], args.images_dir, model.get('images', True))
        texts = [p['text'] for p in messages[1]['content'] if p['type'] == 'text']
        n_images = sum(p['type'] == 'image_url' for p in messages[1]['content'])
        print(f'model={model["openrouter"]} effort={effort} max_tokens={max_tokens}')
        print(f'--- SYSTEM ---\n{messages[0]["content"]}\n--- USER (text parts) ---')
        print('\n\n'.join(texts))
        print(f'--- {n_images} images attached, missing: {missing} | '
              f'~{sum(len(t) for t in texts) // 4} text tokens ---')
        return

    api_key = os.environ.get('OPENROUTER_API_KEY')
    if not api_key and exists(expanduser(args.key_file)):
        with open(expanduser(args.key_file)) as f:
            api_key = f.read().strip()
    if not api_key:
        raise SystemExit(f'set OPENROUTER_API_KEY or put the key in {args.key_file}')

    out_dir = join(args.out_dir, split, model['key'])
    os.makedirs(out_dir, exist_ok=True)
    out_path = join(out_dir, f'{effort}.jsonl')
    done = load_done(out_path)
    todo = [(t, s) for t in tasks for s in range(args.samples) if (t['task_id'], s) not in done]
    print(f'{model["key"]} / {effort}: {len(todo)} requests to run ({len(done)} already done) -> {out_path}')

    semaphore = asyncio.Semaphore(args.concurrency)
    timeout = httpx.Timeout(args.timeout, connect=30)
    n_ok = n_err = 0
    async with httpx.AsyncClient(timeout=timeout) as client:
        jobs = [run_one(client, semaphore, t, s, model, effort, max_tokens, args, api_key) for t, s in todo]
        with open(out_path, 'a') as out:
            for i, job in enumerate(asyncio.as_completed(jobs), start=1):
                record = await job
                out.write(json.dumps(record) + '\n')
                out.flush()
                if record['error'] is None and record.get('parse_error') is None:
                    n_ok += 1
                else:
                    n_err += 1
                    if n_err <= 5:
                        print(f'  {record["task_id"]}: {record["error"] or record["parse_error"]}', flush=True)
                if i % 10 == 0 or i == len(jobs):
                    print(f'  {i}/{len(jobs)} done ({n_ok} ok, {n_err} with errors)', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks')
    parser.add_argument('--list-combos', action='store_true', help='print array index -> (model, effort) for --group')
    parser.add_argument('--config', default=join(REPO_DIR, 'experiments', 'configs', 'models.yaml'))
    parser.add_argument('--model-key')
    parser.add_argument('--effort')
    parser.add_argument('--group')
    parser.add_argument('--array-index', type=int)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--samples', type=int, default=1)
    parser.add_argument('--concurrency', type=int, default=8)
    parser.add_argument('--timeout', type=float, default=600)
    parser.add_argument('--max-tokens', type=int)
    parser.add_argument('--images-dir', default=join(REPO_DIR, 'data', 'images'))
    parser.add_argument('--out-dir', default=join(REPO_DIR, 'results', 'raw'))
    parser.add_argument('--key-file', default='~/.openrouter_key')
    parser.add_argument('--dry-run', action='store_true')
    asyncio.run(main_async(parser.parse_args()))


if __name__ == '__main__':
    main()
