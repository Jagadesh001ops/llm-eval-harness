import pathlib

import pytest

from harness.intent import build_leak_index, check_candidate, load_task_card
from harness.parsers import extract_gsm8k_answer, gsm8k_correct, gsm8k_gold

CARD = load_task_card("gsm8k")
SEED = pathlib.Path("prompts/v2/gsm8k.txt").read_text(encoding="utf-8")
LEAK = build_leak_index(["Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May."])


def rules(c):
    return {r for r, _ in check_candidate(c, CARD, SEED, LEAK)}


def test_gold_parsing():
    assert gsm8k_gold("She sold 48/2 = 24 clips.\n#### 72") == 72
    assert gsm8k_gold("...\n#### 1,250") == 1250


@pytest.mark.parametrize("raw,expected", [
    ("Step 1 ... 48 + 24 = 72.\nAnswer: 72", 72),
    ("Answer: $1,250.", 1250),
    ("answer: -3.5 degrees", -3.5),
    ("First guess Answer: 10. Wait, recheck.\nAnswer: 12", 12),
    ("The total is 72.", None),            # no Answer: line -> not guessed
    ("Let me compute step by step: 48", None),
])
def test_extract_answer(raw, expected):
    assert extract_gsm8k_answer(raw) == expected


def test_correct():
    assert gsm8k_correct("...\nAnswer: 72", 72.0) == 1
    assert gsm8k_correct("...\nAnswer: 71", 72.0) == 0
    assert gsm8k_correct("72", 72.0) == 0


def test_seed_passes_intent_checks():
    assert rules(SEED) == set()


def test_answer_token_required():
    assert "labels" in rules(SEED.replace('"Answer: <number>"', '"Final: <number>"'))


def test_worked_example_rejected():
    assert "few_shot" in rules(SEED.replace("Question: {question}", "Answer: 42\n\nQuestion: {question}"))
    assert "few_shot" in rules(SEED.replace("Question: {question}",
                                            "Question: Tom has 3 apples and eats one.\n\nQuestion: {question}"))


def test_leak_rejected():
    c = SEED + "\nRemember Natalia sold clips to 48 of her friends in April."
    assert "leak" in rules(c)
