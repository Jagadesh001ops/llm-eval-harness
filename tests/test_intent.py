import pathlib

import pytest

from harness.intent import build_leak_index, check_candidate, lexical_novelty, load_task_card

CARD = load_task_card("ag_news")
SEED = pathlib.Path("prompts/v2/ag_news.txt").read_text(encoding="utf-8")
ORACLE = pathlib.Path("prompts/reference/ag_news_oracle.txt").read_text(encoding="utf-8")
ITEM = ("Fears for T N pension after talks Unions representing workers at Turner Newall say they are "
        "'disappointed' after talks with stricken parent firm Federal Mogul.")
LEAK = build_leak_index([ITEM])


def rules(candidate):
    return {r for r, _ in check_candidate(candidate, CARD, SEED, LEAK)}


def test_seed_and_oracle_pass():
    assert rules(SEED) == set()
    assert rules(ORACLE) == set()


def test_placeholder_rules():
    assert "placeholder" in rules(SEED.replace("{text}", ""))
    assert "placeholder" in rules(SEED + "\nContext: {context}")
    assert "placeholder" in rules(SEED.replace("{text}", "{text} {text}"))


def test_missing_label():
    assert "labels" in rules(SEED.replace("science", "technology"))


def test_length_cap():
    assert "length" in rules(SEED + " Think carefully." * 60)


@pytest.mark.parametrize("example", [
    "\nHeadline: Fed raises interest rates again\nTopic: business\n",
    "\nTopic: sports\n",
])
def test_few_shot_rejected(example):
    assert "few_shot" in rules(SEED.replace("Headline: {text}", example + "Headline: {text}"))


def test_leak_rejected():
    cand = SEED.replace("Reply with one word only.",
                        "Unions representing workers at Turner Newall say they are disappointed. Reply with one word only.")
    assert "leak" in rules(cand)


def test_plain_rephrase_passes():
    cand = ("Read the news article below and decide its topic. Choose exactly one of: world, sports, business, "
            "or science. Answer with that single word.\n\nArticle: {text}\nTopic:")
    assert rules(cand) == set()


def test_lexical_novelty():
    assert lexical_novelty(SEED, CARD, SEED) == 0.0
    nov = lexical_novelty(ORACLE, CARD, SEED)
    assert 0.0 < nov < 1.0  # oracle adds words such as 'technology', 'computing'
