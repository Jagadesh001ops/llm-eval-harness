"""Output parsers.

The legacy_* functions are byte-for-byte the logic the stored results were scored with. They are kept
so the original numbers stay reproducible; new scoring uses the strict parsers (added separately).
"""

import re

AG_LABELS = ["world", "sports", "business", "science"]

_LEGACY_AG_KEYWORDS = {
    "world":    ["world", "international", "global", "politics", "political", "foreign"],
    "sports":   ["sports", "sport", "athletic", "athletics", "game", "games", "soccer", "football", "basketball"],
    "business": ["business", "finance", "financial", "economy", "economic", "market", "stock", "trade"],
    "science":  ["science", "scientific", "technology", "tech", "sci", "research", "space", "computing"],
}


def legacy_sentiment_label(raw_output):
    """Original eval.py parser. Known flaw: 'positive' anywhere in the text wins, even inside a hedge
    ('neutral ... Sentiment: positive') or mid-reasoning, so ambiguous outputs are scored as positive."""
    text = raw_output.lower().strip()
    first_word = text.split()[0].strip(".,!?") if text else ""
    for label in ["positive", "negative"]:
        if label in text or first_word == label:
            return label
    return "unknown"


def legacy_ag_label(raw_output):
    """Original ag_news_eval.py parser (first-word keyword match, then substring scan in label order)."""
    text = raw_output.lower().strip()
    first_word = re.split(r"[\s.,!?:]", text)[0] if text else ""
    for label, keywords in _LEGACY_AG_KEYWORDS.items():
        if first_word in keywords:
            return label
    for label, keywords in _LEGACY_AG_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                return label
    return "unknown"


# ── TriviaQA (unchanged scoring; these are metric definitions, not parsing bugs) ──

def normalize_answer(s):
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def extract_trivia_answer(raw_output):
    """For CoT: pull 'Answer: X' from the last such line. Otherwise take the first line."""
    for line in reversed(raw_output.strip().split("\n")):
        if line.lower().startswith("answer:"):
            return line[7:].strip()
    return raw_output.strip().split("\n")[0].strip()


def exact_match(prediction, aliases):
    pred = normalize_answer(prediction)
    return any(normalize_answer(a) == pred for a in aliases)


def contains_answer(prediction, aliases):
    pred = normalize_answer(prediction)
    return any(normalize_answer(a) in pred for a in aliases)
