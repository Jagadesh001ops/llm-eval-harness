"""Output parsers.

The legacy_* functions are byte-for-byte the logic the stored results were scored with. They are kept
so the original numbers stay reproducible; new scoring uses strict_sentiment_label / strict_ag_label.

Strict parsing rules (documented in docs/statistics.md):
  * zero_shot / few_shot ("answer" mode): the label is the first word of the output; if the first word
    is not a label, the first line is accepted only if it names exactly one label. Text after the first
    line (models often hallucinate extra few-shot examples there) is ignored.
  * chain_of_thought ("cot" mode): the prompt asks the model to END with the label, so the last word is
    checked first. Some models answer first and explain after ("Negative. The text expresses..."), so if
    the last word is not a label, a label as the FIRST word is accepted (a label elsewhere in the first
    line is not: in CoT that is usually truncated reasoning). Reasoning truncated
    before any answer is "invalid", not guessed.
  * Outcomes: a task label; "neutral" (sentiment hedge - not an SST-2 class); "ambiguous" (the answer
    names more than one class, e.g. "science (or business, depending on the focus)"); or "invalid"
    (no parseable label). Only a single task label can be scored correct.
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


# ── Strict parsers ──────────────────────────────────────────────────────────

NEUTRAL = "neutral"
INVALID = "invalid"
AMBIGUOUS = "ambiguous"

_SENTIMENT_MAP = {"positive": "positive", "negative": "negative", "neutral": NEUTRAL, "mixed": NEUTRAL}
# AG News' official class name is "Sci/Tech", so technology words map to science. Nothing else is mapped:
# e.g. "politics" or "entertainment" are not AG classes and count as invalid rather than being guessed.
_AG_MAP = {"world": "world", "sports": "sports", "business": "business", "science": "science",
           "sci/tech": "science", "sci-tech": "science", "technology": "science", "tech": "science"}

_PREFIX = re.compile(r"^(sentiment|topic|answer|label|category|classification)\s*[:\-]\s*", re.I)
_WORD = re.compile(r"[a-z]+(?:[/\-][a-z]+)?")


def _words(text):
    return _WORD.findall(text.lower())


def _strict_label(raw_output, mapping, mode):
    text = (raw_output or "").strip()
    if not text:
        return INVALID
    if mode == "cot":
        words = _words(text)
        if words and words[-1] in mapping:
            return mapping[words[-1]]
        # fall through: answer-first style
    first_line = next((ln for ln in text.splitlines() if ln.strip()), "")
    first_line = _PREFIX.sub("", first_line.strip().lstrip("*#>\"' "))
    words = _words(first_line)
    if not words:
        return INVALID
    found = {mapping[w] for w in words if w in mapping}
    if words[0] in mapping:
        first = mapping[words[0]]
        # A leading hedge stays a hedge even if it leans ("neutral (or maybe slightly negative)").
        # A leading class label is rejected if the line also names a different class ("positive or negative").
        if first == NEUTRAL or not (found - {first, NEUTRAL}):
            return first
        return AMBIGUOUS
    if mode == "cot" or not found:
        # In CoT a label mid-sentence is usually truncated reasoning ("...reinforces the negative
        # sentiment"), not an answer, so only answer-first outputs (label as first word) are accepted.
        return INVALID
    return found.pop() if len(found) == 1 else AMBIGUOUS


def strict_sentiment_label(raw_output, prompt_variant):
    return _strict_label(raw_output, _SENTIMENT_MAP, "cot" if prompt_variant == "chain_of_thought" else "answer")


def strict_ag_label(raw_output, prompt_variant):
    return _strict_label(raw_output, _AG_MAP, "cot" if prompt_variant == "chain_of_thought" else "answer")
