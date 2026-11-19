from harness.intent import load_task_card
from harness.judge import build_judge_request, parse_verdict
from scripts.validate_judge import cohen_kappa

CARD = load_task_card("ag_news")


def test_parse_verdict():
    assert parse_verdict("<reason>Only rephrases.</reason>\n<verdict>keeps</verdict>") == ("keeps", "Only rephrases.")
    assert parse_verdict("<reason>Adds a label.</reason><verdict>CHANGES</verdict>")[0] == "changes"


def test_parse_verdict_missing_or_conflicting_is_error():
    assert parse_verdict("It keeps the intent.")[0] == "error"
    assert parse_verdict("<verdict>keeps</verdict> ... <verdict>changes</verdict>")[0] == "error"


def test_judge_sees_both_card_views_and_both_prompts():
    req = build_judge_request(CARD, "SEED {text}", "REWRITE {text}")
    assert "Sci/Tech" in req and "optimizer_view" in req and "judge_view" in req
    assert "SEED {text}" in req and "REWRITE {text}" in req


def test_cohen_kappa():
    assert cohen_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == 1.0
    assert abs(cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"])) < 1e-9
