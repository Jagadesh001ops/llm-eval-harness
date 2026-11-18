from harness.intent import load_task_card
from harness.optimizer import build_request, parse_candidates

CARD = load_task_card("ag_news")


def test_parse_candidates_tags():
    text = "<prompt>A {text}</prompt>\nnoise\n<prompt>\nB {text}\n</prompt><prompt>C {text}"
    assert parse_candidates(text) == ["A {text}", "B {text}", "C {text}"]


def test_parse_candidates_no_tags_gives_nothing():
    assert parse_candidates("Here is a better prompt: Classify {text}") == []


def test_parse_caps_at_n():
    assert len(parse_candidates("<prompt>x</prompt>" * 6, n=4)) == 4


def test_feedback_request_contains_failures_and_history():
    req = build_request(CARD, "P {text}", [("P {text}", 0.7)],
                        [{"input": "Apple chip", "output": "business", "gold": "science"}], 500)
    assert "Apple chip" in req and "accuracy 0.700" in req and "<correct_answer>science" in req
    assert "{text}" in req and "world, sports, business, science" in req


def test_control_request_has_no_failures_or_scores():
    req = build_request(CARD, "P {text}", [("P {text}", 0.7)],
                        [{"input": "Apple chip", "output": "business", "gold": "science"}], 500, feedback=False)
    assert "Apple chip" not in req and "accuracy" not in req


def test_judge_view_never_reaches_optimizer():
    req = build_request(CARD, "P {text}", [], [], 500)
    assert "Sci/Tech" not in req and "technology" not in req.lower()
