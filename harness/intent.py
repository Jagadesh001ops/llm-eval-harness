"""Intent preservation for prompt rewrites (design doc §6).

Layer 1 (this module, deterministic, rejects): placeholders, label vocabulary, length cap, few-shot examples,
text leaked from dataset items. Layer 2 (intent judge) lives elsewhere and is only used once validated. Layer 3 is
measured, not enforced: lexical_novelty() below.
"""

import pathlib
import re

import yaml

LENGTH_CAP_RATIO = 3.0
LEAK_NGRAM = 8

_PLACEHOLDER = re.compile(r"\{[^{}\s]*\}")
_WORD = re.compile(r"[a-z0-9]+(?:['\-][a-z0-9]+)*")
# A line that looks like a filled-in example input ("Headline: Fed raises rates") rather than the placeholder line.
_EXAMPLE_INPUT = re.compile(r"^\s*(headline|article|text|input|question|passage|context|news)\s*:\s*(?!\{)\S", re.I | re.M)

_STOPWORDS = set("""a about above after again against all am an and any are as at be because been before being below
between both but by can could did do does doing down during each few for from further had has have having he her here
hers herself him himself his how i if in into is it its itself just me more most my myself no nor not now of off on
once only or other our ours ourselves out over own same she should so some such than that the their theirs them
themselves then there these they this those through to too under until up very was we were what when where which
while who whom why will with would you your yours yourself yourselves one only also must may might shall each every
following""".split())


def load_task_card(task):
    return yaml.safe_load(pathlib.Path(f"prompts/task_cards/{task}.yaml").read_text(encoding="utf-8"))


def words(text):
    return _WORD.findall(text.lower())


def ngrams(text, n=LEAK_NGRAM):
    w = words(text)
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def build_leak_index(item_texts, n=LEAK_NGRAM):
    """Set of every n-gram in the dataset items an optimizer could copy from (pool, dev and test)."""
    index = set()
    for t in item_texts:
        index |= ngrams(t, n)
    return index


def check_candidate(candidate, card, seed_prompt, leak_index):
    """Return a list of (rule, detail) violations; empty list = passes layer 1."""
    view = card["optimizer_view"]
    v = []
    if not candidate or not candidate.strip():
        return [("empty", "candidate is empty")]

    found = _PLACEHOLDER.findall(candidate)
    required = view["placeholders"]
    for ph in required:
        if found.count(ph) != 1:
            v.append(("placeholder", f"{ph} must appear exactly once (found {found.count(ph)})"))
    extra = sorted(set(found) - set(required))
    if extra:
        v.append(("placeholder", f"unexpected placeholders {extra}"))

    lw = set(words(candidate))
    missing = [lab for lab in view["labels"] if lab.lower() not in lw]
    if missing:
        v.append(("labels", f"label words missing: {missing}"))

    if len(candidate) > LENGTH_CAP_RATIO * len(seed_prompt):
        v.append(("length", f"{len(candidate)} chars > {LENGTH_CAP_RATIO}x seed ({len(seed_prompt)})"))

    if _EXAMPLE_INPUT.search(candidate):
        v.append(("few_shot", "contains a filled-in example input line"))
    label_alt = "|".join(re.escape(lab) for lab in view["labels"])
    if re.search(rf"^\s*(topic|answer|label|category|output|sentiment)\s*:\s*({label_alt})\b", candidate, re.I | re.M):
        v.append(("few_shot", "contains an example answer line"))

    own = ngrams(seed_prompt) | ngrams(yaml.safe_dump(view))
    leaked = (ngrams(candidate) - own) & leak_index
    if leaked:
        v.append(("leak", f"{len(leaked)} {LEAK_NGRAM}-gram(s) shared with dataset items, e.g. {' '.join(next(iter(leaked)))!r}"))
    return v


def lexical_novelty(candidate, card, seed_prompt):
    """Share of distinct content words in the candidate found in neither the seed prompt nor the optimizer-visible
    task card. Measured and reported, never used to reject (harmless synonyms also count as novel)."""
    def content(t):
        return {w for w in words(t) if len(w) >= 3 and w not in _STOPWORDS and not w.isdigit()}
    base = content(seed_prompt) | content(yaml.safe_dump(card["optimizer_view"]))
    cand = content(_PLACEHOLDER.sub(" ", candidate))
    return 0.0 if not cand else len(cand - base) / len(cand)
