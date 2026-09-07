'''
Food filter for the Recognize Anything Model (RAM).

RAM tags an image with everything it sees. For a food-focused pipeline we only
want the edible items, so this module keeps the tags that are foods and drops
everything else. The vocabulary lives in ``ram/data/food_tag_list.txt`` (a
curated subset of RAM's own ``ram_tag_list.txt``) so it stays in sync with what
RAM can actually predict.
'''
import os

_FOOD_LIST_PATH = os.path.join(os.path.dirname(__file__), 'data',
                               'food_tag_list.txt')


def load_food_vocab(path=_FOOD_LIST_PATH):
    """Return the set of known food tags (lower-cased)."""
    with open(path, 'r') as f:
        return {line.strip().lower() for line in f if line.strip()}


FOOD_VOCAB = load_food_vocab()


def _split_tags(tags):
    """Accept RAM's ``'a | b | c'`` string or an already-split list."""
    if isinstance(tags, str):
        return [t.strip() for t in tags.split('|') if t.strip()]
    return [str(t).strip() for t in tags if str(t).strip()]


def filter_food_tags(tags, vocab=FOOD_VOCAB):
    """Keep only the food tags from a RAM prediction.

    Args:
        tags: RAM output, either the ``'tag1 | tag2 | ...'`` string returned by
            ``inference_ram`` or a list of tag strings.
        vocab: set of allowed food tags (lower-cased). Defaults to the curated
            food vocabulary.

    Returns:
        List of food tags, preserving RAM's original casing and order, with
        duplicates removed.
    """
    foods, seen = [], set()
    for tag in _split_tags(tags):
        key = tag.lower()
        if key in vocab and key not in seen:
            foods.append(tag)
            seen.add(key)
    return foods
