import pytest

from harness.parsers import (bbh_correct, extract_bbh_answer, normalize_bbh, squad_normalize, squad_prediction,
                             squad_score)


@pytest.mark.parametrize("raw,target,ok", [
    ("Step 1 ... so the date is 12/25/1937.\nAnswer: (B)", "(B)", 1),
    ("...\nAnswer: B", "(B)", 1),
    ("...\nAnswer: B) 12/25/1937", "(B)", 1),
    ("...\nAnswer: **(C)**", "(B)", 0),
    ("...\nAnswer: Yes.", "Yes", 1),
    ("...\nAnswer: no", "Yes", 0),
    ("The answer is (B).", "(B)", 0),          # no Answer: line -> not guessed
    ("Answer: (A)\nWait, recheck.\nAnswer: (B)", "(B)", 1),
])
def test_bbh(raw, target, ok):
    assert bbh_correct(raw, target) == ok


def test_bbh_normalize_and_extract():
    # Option letters are uppercase in BBH; lowercase is not read as an option, so words like "a lot" never become (A).
    assert normalize_bbh("a lot") == "a lot"
    assert normalize_bbh("D.") == "(D)"
    assert extract_bbh_answer("no answer line") is None


def test_squad_normalize():
    assert squad_normalize("The Normans!") == "normans"


@pytest.mark.parametrize("raw,golds,score", [
    ("Normandy", ["Normandy", "in Normandy"], 1.0),
    ("Answer: in Normandy, France", ["Normandy"], pytest.approx(2 * (1 / 3) * 1 / (1 / 3 + 1))),
    ("unanswerable", [], 1.0),
    ("Unanswerable.", [], 1.0),
    ("Normandy", [], 0.0),                       # answering an unanswerable question
    ("unanswerable", ["Normandy"], 0.0),         # abstaining on an answerable question
])
def test_squad_score(raw, golds, score):
    assert squad_score(raw, golds) == score


def test_squad_prediction_first_line_only():
    assert squad_prediction("Normandy\nThe passage says...") == "Normandy"
