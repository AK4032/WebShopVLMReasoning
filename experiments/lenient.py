"""Lenient requirement matching, as a robustness check on WebShop's reward.

WebShop's reward relies on exact substrings and fuzzy string ratios, so it can
reject products that satisfy an instruction as written (e.g. "non-slip" vs the
attribute "non slip", or the option '52 x 54 inch' vs '52"w x 54"l'). The checks
here accept anything WebShop accepts, plus matches that differ only in
punctuation, hyphenation, word order, plurals, or unit formatting, provided any
numbers agree. They err toward "satisfied", so a product that still fails here
almost certainly violates the instruction.
"""
import re

from thefuzz import fuzz

from web_agent_site.engine.normalize import normalize_color

UNIT_WORDS = {'inch', 'inches', 'in', 'cm', 'mm', 'ft', 'feet', 'oz', 'ounce', 'ounces',
              'lb', 'lbs', 'pound', 'pounds', 'ml', 'l', 'w', 'h', 'x', 'pack', 'count',
              'pcs', 'piece', 'pieces', 'size'}
NUMBER = re.compile(r'\d+(?:\.\d+)?')


def _stem(word):
    if len(word) > 3 and word.endswith('es') and word[:-2].endswith(('s', 'x', 'ch', 'sh')):
        return word[:-2]
    if len(word) > 3 and word.endswith('s') and not word.endswith('ss'):
        return word[:-1]
    return word


def words(text):
    text = re.sub(r'[^a-z0-9.]+', ' ', text.lower().replace("''", ' ').replace('"', ' '))
    return {_stem(w) for w in text.split() if not NUMBER.fullmatch(w) and w.strip('.')}


def numbers(text):
    return {float(n) for n in NUMBER.findall(text)}


def product_text(product):
    return ' '.join([product['Title'], ' '.join(product['BulletPoints']), product['Description'] or '',
                     ' '.join(product['Attributes'])])


def attribute_satisfied(product, attribute, text_words=None):
    """WebShop's check, or every word of the attribute appears somewhere in the product text."""
    if any(fuzz.token_set_ratio(a, attribute) > 85 for a in product['Attributes']):
        return True
    full_text = product_text(product).lower()
    if attribute in full_text:
        return True
    attr_words = words(attribute)
    text_words = text_words if text_words is not None else words(full_text)
    return bool(attr_words) and attr_words <= text_words


def option_matches(goal_option, value):
    """WebShop's fuzzy match, or compatible numbers and overlapping descriptive words."""
    g, v = normalize_color(goal_option), normalize_color(value)
    if fuzz.token_set_ratio(v, g) > 85:
        return True
    gn, vn = numbers(g), numbers(v)
    if gn or vn:
        if not (gn and vn and (gn <= vn or vn <= gn)):
            return False
    gw, vw = words(g) - UNIT_WORDS, words(v) - UNIT_WORDS
    return not gw or not vw or bool(gw & vw)


def check(product, goal, price):
    """Which goal requirements a product cannot satisfy, even under lenient matching."""
    text_words = words(product_text(product))
    values = [v for vs in product['options'].values() for v in vs]
    failed_attributes = [a for a in goal['attributes']
                         if not attribute_satisfied(product, a, text_words)]
    unavailable_options = [o for o in goal['goal_options']
                           if not any(option_matches(o, v) for v in values)]
    over_budget = goal['price_upper'] > 0 and price > goal['price_upper']
    failures = (['attributes'] if failed_attributes else []) + \
               (['options'] if unavailable_options else []) + (['price'] if over_budget else [])
    return {'could_satisfy': not failures, 'failures': failures,
            'failed_attributes': failed_attributes, 'unavailable_options': unavailable_options}


def reward(product, goal, price, options, r_type):
    """WebShop's reward formula with lenient attribute and option matching.

    Attributes are judged on the product; options only on the values the model chose.
    `r_type` is WebShop's product-type factor, reused unchanged.
    """
    text_words = words(product_text(product))
    chosen = list(options.values())
    n_attr = sum(attribute_satisfied(product, a, text_words) for a in goal['attributes'])
    n_opt = sum(any(option_matches(g, v) for v in chosen) for g in goal['goal_options'])
    r_price = price <= goal['price_upper'] if goal['price_upper'] > 0 else True
    total = (n_attr + n_opt + r_price) / (len(goal['attributes']) + len(goal['goal_options']) + 1)
    return total * r_type
