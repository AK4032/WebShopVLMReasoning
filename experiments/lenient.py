"""Lenient requirement matching, as a robustness check on WebShop's reward.

WebShop's reward relies on exact substrings and fuzzy string ratios, so it can
reject products that satisfy an instruction as written (e.g. "non-slip" vs the
attribute "non slip", or the option '52 x 54 inch' vs '52"w x 54"l'). The checks
here accept anything WebShop accepts, plus matches that differ only in
punctuation, hyphenation, compounding, word order, word form, a short list of
synonyms, pair/piece counts, or unit formatting (numbers must still agree). A
requested feature counts as available if one of the product's selectable options
provides it, and only the parts of a hidden goal option that the instruction
actually states are required. They err toward "satisfied", so a product that
still fails here almost certainly violates the instruction.
"""
import re

from nltk.stem import PorterStemmer
from thefuzz import fuzz

from web_agent_site.engine.normalize import normalize_color

UNIT_WORDS = {'inch', 'inches', 'in', 'cm', 'mm', 'ft', 'feet', 'oz', 'ounce', 'ounces',
              'lb', 'lbs', 'pound', 'pounds', 'ml', 'l', 'w', 'h', 'x', 'pack', 'count',
              'pcs', 'pc', 'piece', 'pieces', 'size', 'pair', 'pairs'}
STOP_WORDS = {'a', 'an', 'and', 'the', 'of', 'for', 'to', 'with', 'in', 'on', 'is', 'it', 'that'}
NUMBER = re.compile(r'\d+(?:\.\d+)?')
PAIR = re.compile(r'\bpairs?\b')
PIECE = re.compile(r'\b(?:pcs?|pieces?|count|ct)\b')

# Each group lists phrases that state the same requirement.
SYNONYM_GROUPS = [
    ['engineered wood', 'particleboard', 'particle board', 'mdf', 'plywood', 'composite wood', 'manufactured wood'],
    ['cruelty free', 'not tested on animals', 'no animal testing', 'animal testing', 'never tested on animals'],
    ['non gmo', 'gmo free', 'no gmo', 'non genetically modified'],
    ['kosher certified', 'kosher'],
    ['certified organic', 'organic', 'usda organic'],
    ['high definition', 'hd', '1080p', '4k', 'uhd'],
    ['water resistant', 'waterproof', 'water proof', 'water repellent'],
    ['zero sugar', 'sugar free', 'no sugar', '0 sugar', 'zero grams of sugar'],
    ['dairy free', 'non dairy', 'no dairy', 'lactose free'],
    ['fragrance free', 'unscented', 'no fragrance', 'without fragrance'],
    ['wireless bluetooth', 'bluetooth'],
    ['machine wash', 'machine washable', 'washable in machine'],
    ['easy clean', 'easy to clean', 'wipe clean', 'easy cleaning'],
    ['easy assemble', 'easy assembly', 'easy to assemble', 'simple assembly', 'easy to install'],
    ['light weight', 'lightweight', 'light-weight', 'ultralight', 'ultra light'],
]
_STEMMER = PorterStemmer()


def _stem(word):
    stem = _STEMMER.stem(word)
    return stem[:-3] if stem.endswith('abl') and len(stem) > 5 else stem


def tokens(text):
    """Stemmed content words of `text`, in order."""
    text = re.sub(r'[^a-z0-9.]+', ' ', text.lower().replace("''", ' ').replace('"', ' '))
    return [_stem(w) for w in text.split()
            if not NUMBER.fullmatch(w) and w.strip('.') and w not in STOP_WORDS]


def words(text):
    return set(tokens(text))


def numbers(text):
    return {float(n) for n in NUMBER.findall(text)}


def _compact(text):
    return re.sub(r'[^a-z0-9]+', '', text.lower())


def variants(phrase):
    """The phrase plus any listed synonyms."""
    for group in SYNONYM_GROUPS:
        if any(_compact(phrase) == _compact(p) for p in group):
            return group
    return [phrase]


def product_text(product):
    return ' '.join([product['Title'], ' '.join(product['BulletPoints']), product['Description'] or '',
                     ' '.join(product['Attributes'])])


def _near(phrase_words, positions, slack=2):
    """Whether every word of the phrase occurs within a window of len(phrase) + slack tokens."""
    first, rest = phrase_words[0], phrase_words[1:]
    width = len(phrase_words) + slack
    for start in positions.get(first, []):
        if all(any(abs(p - start) < width for p in positions.get(w, [])) for w in rest):
            return True
    return False


def _phrase_in(phrase, text_lower, positions, text_compact):
    # Substring matching only for longer phrases, so e.g. "hd" does not match inside "hdmi".
    if len(_compact(phrase)) >= 6 and (phrase in text_lower or _compact(phrase) in text_compact):
        return True
    phrase_words = list(dict.fromkeys(tokens(phrase)))
    return bool(phrase_words) and _near(phrase_words, positions)


def _text_context(product, extra_text):
    text_lower = (product_text(product) + ' ' + extra_text).lower()
    positions = {}
    for i, tok in enumerate(tokens(text_lower)):
        positions.setdefault(tok, []).append(i)
    return text_lower, positions, _compact(text_lower)


def attribute_satisfied(product, attribute, extra_text='', context=None):
    """WebShop's check, or the attribute (or a synonym) appears in the product text or `extra_text`.

    `context` is a precomputed `_text_context(product, extra_text)`.
    """
    if any(fuzz.token_set_ratio(a, attribute) > 85 for a in product['Attributes']):
        return True
    text_lower, positions, text_compact = context or _text_context(product, extra_text)
    if attribute in text_lower:
        return True
    return any(_phrase_in(v, text_lower, positions, text_compact) for v in variants(attribute))


def stated_part(goal_option, instruction):
    """The parts of a hidden goal option that the instruction actually states.

    WebShop's goal options come from the option the annotator bought, which can carry
    details the instruction never mentions (e.g. '52"w x 54"l' for "fifty-two inch width").
    Returns None when nothing in the option is stated.
    """
    instr_words = words(instruction)
    instr_raw = set(re.sub(r'[^a-z0-9.]+', ' ', instruction.lower()).split())
    instr_numbers = numbers(instruction)
    kept = []
    for token in re.findall(r'\d+(?:\.\d+)?|[a-z]+', goal_option.lower()):
        if NUMBER.fullmatch(token):
            if float(token) in instr_numbers:
                kept.append(token)
        elif _stem(token) in instr_words or any(w.startswith(token) for w in instr_raw if len(token) >= 2):
            kept.append(token)
        else:
            # Run-together option names such as "purplehousing": keep the stated word ("purple").
            prefix = max((w for w in instr_raw if len(w) >= 3 and token.startswith(w)), key=len, default=None)
            if prefix:
                kept.append(prefix)
    return ' '.join(kept) or None


def _counts(text):
    """Numbers in `text`, plus piece-equivalents when it counts pairs (and vice versa)."""
    nums = numbers(text)
    if PAIR.search(text):
        nums |= {2 * n for n in nums}
    if PIECE.search(text):
        nums |= {n / 2 for n in nums}
    return nums


def _option_match(g, v):
    if fuzz.token_set_ratio(v, g) > 85:
        return True
    gn, vn = numbers(g), numbers(v)
    if gn or vn:
        if not (gn and vn and (gn <= vn or vn <= gn or gn <= _counts(v) or _counts(g) & vn)):
            return False
    gw, vw = words(g) - UNIT_WORDS, words(v) - UNIT_WORDS
    return not gw or not vw or len(gw & vw) >= max(1, len(gw) / 2)


def option_matches(goal_option, value):
    """WebShop's fuzzy match, or compatible numbers and overlapping descriptive words.

    Compares both raw and colour-normalized strings: WebShop's normalize_color maps any
    string containing a colour word to that colour (e.g. "cream- french vanilla" -> "cream").
    """
    raw_g, raw_v = goal_option.lower().strip(), value.lower().strip()
    return (_option_match(normalize_color(goal_option), normalize_color(value))
            or _option_match(raw_g, raw_v))


def _option_satisfied(goal_option, instruction, values):
    stated = stated_part(goal_option, instruction)
    return (any(option_matches(goal_option, v) for v in values)
            or stated is None or any(option_matches(stated, v) for v in values))


def check(product, goal, price):
    """Which goal requirements a product cannot satisfy, even under lenient matching."""
    values = [v for vs in product['options'].values() for v in vs]
    context = _text_context(product, ' '.join(values))
    failed_attributes = [a for a in goal['attributes']
                         if not attribute_satisfied(product, a, context=context)]
    unavailable_options = [o for o in goal['goal_options']
                           if not _option_satisfied(o, goal['instruction_text'], values)]
    over_budget = goal['price_upper'] > 0 and price > goal['price_upper']
    failures = (['attributes'] if failed_attributes else []) + \
               (['options'] if unavailable_options else []) + (['price'] if over_budget else [])
    return {'could_satisfy': not failures, 'failures': failures,
            'failed_attributes': failed_attributes, 'unavailable_options': unavailable_options}


def reward(product, goal, price, options, r_type):
    """WebShop's reward formula with lenient attribute and option matching.

    Attributes are judged on the product text plus the options the model chose;
    options only on the chosen values. `r_type` is WebShop's product-type factor.
    """
    chosen = list(options.values())
    context = _text_context(product, ' '.join(chosen))
    n_attr = sum(attribute_satisfied(product, a, context=context) for a in goal['attributes'])
    n_opt = sum(_option_satisfied(g, goal['instruction_text'], chosen) for g in goal['goal_options'])
    r_price = price <= goal['price_upper'] if goal['price_upper'] > 0 else True
    total = (n_attr + n_opt + r_price) / (len(goal['attributes']) + len(goal['goal_options']) + 1)
    return total * r_type
