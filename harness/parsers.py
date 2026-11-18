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


# ── GSM8K ───────────────────────────────────────────────────────────────────

_NUMBER = re.compile(r"-?\$?\s?\d[\d,]*(?:\.\d+)?")


def _to_float(s):
    try:
        return float(s.replace("$", "").replace(",", "").strip())
    except ValueError:
        return None


def gsm8k_gold(answer_field):
    """Gold answer from the dataset's solution text, which ends with '#### <number>'."""
    return _to_float(answer_field.split("####")[-1])


def extract_gsm8k_answer(raw_output):
    """Number after the LAST 'Answer:' in the output (the task's output contract). No 'Answer:' -> None (scored
    wrong): a reply truncated before its final line, or one ignoring the format, is not guessed from the reasoning."""
    text = raw_output or ""
    pos = text.lower().rfind("answer:")
    if pos < 0:
        return None
    m = _NUMBER.search(text[pos + len("answer:"):])
    return _to_float(m.group(0)) if m else None


def gsm8k_correct(raw_output, gold):
    pred = extract_gsm8k_answer(raw_output)
    return int(pred is not None and gold is not None and abs(pred - gold) < 1e-6)


# ── BIG-Bench Hard ──────────────────────────────────────────────────────────

_MC = re.compile(r"^\(?([A-R])\)?(?:[\s.:,)]|$)")


def normalize_bbh(ans):
    """Canonical form of a BBH answer: '(B)' for options, lowercase words otherwise ('yes', 'valid', '12')."""
    a = (ans or "").strip().strip("*`\"'").strip()
    m = _MC.match(a)
    if m:
        return f"({m.group(1)})"
    a = a.rstrip(".").strip().lower()
    return re.sub(r"\s+", " ", a)


def extract_bbh_answer(raw_output):
    """Text after the LAST 'Answer:' up to the end of that line; None if there is no such line (scored wrong)."""
    text = raw_output or ""
    pos = text.lower().rfind("answer:")
    if pos < 0:
        return None
    line = text[pos + len("answer:"):].strip().splitlines()
    return line[0].strip() if line else None


def bbh_correct(raw_output, target):
    pred = extract_bbh_answer(raw_output)
    return int(pred is not None and normalize_bbh(pred) == normalize_bbh(target))


# ── SQuAD 2.0 ───────────────────────────────────────────────────────────────

NO_ANSWER = "unanswerable"


def squad_normalize(s):
    """Official SQuAD normalisation: lowercase, drop punctuation and articles, collapse whitespace."""
    import string
    s = (s or "").lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _f1(pred, gold):
    p, g = squad_normalize(pred).split(), squad_normalize(gold).split()
    common = sum(min(p.count(w), g.count(w)) for w in set(p))
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def squad_prediction(raw_output):
    """First line of the reply, minus an 'Answer:' prefix. A reply that is just 'unanswerable' predicts no answer."""
    line = next((ln for ln in (raw_output or "").strip().splitlines() if ln.strip()), "")
    line = re.sub(r"^\s*answer\s*:\s*", "", line, flags=re.I).strip()
    return NO_ANSWER if squad_normalize(line) == NO_ANSWER else line


def squad_score(raw_output, gold_answers):
    """Per-item F1 as in the official SQuAD 2.0 metric: unanswerable items score 1 only for a no-answer prediction;
    answerable items score the max F1 over gold answers (0 for a no-answer prediction)."""
    pred = squad_prediction(raw_output)
    if not gold_answers:
        return 1.0 if pred == NO_ANSWER else 0.0
    if pred == NO_ANSWER:
        return 0.0
    return max(_f1(pred, g) for g in gold_answers)
