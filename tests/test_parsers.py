"""Parser tests. Most inputs are real raw outputs (or trimmed prefixes of them) from results/*.csv."""

import pytest

from harness.parsers import (AMBIGUOUS, INVALID, NEUTRAL, legacy_ag_label, legacy_sentiment_label, strict_ag_label,
                             strict_sentiment_label)

# Mistral few-shot hedge followed by hallucinated few-shot examples (results/all_models_all_prompts.csv).
MISTRAL_HEDGE = ("neutral\nText: I absolutely love this product!\nSentiment: positive\n"
                 "Text: The customer service was unhelpful and rude.\nSentiment: negative")
# Llama CoT truncated at maxTokens=50, before any final answer.
LLAMA_TRUNCATED_COT = ('Let\'s break it down step by step:\n\n* The text describes something pleasant.\n'
                       '* "affecting" can have a positive or negative connotation, but in this context, it likely means "')


class TestLegacyFlaws:
    """Pin the legacy behaviour the re-scoring corrects, so the documented flaws stay true."""

    def test_hedge_scored_as_positive(self):
        assert legacy_sentiment_label(MISTRAL_HEDGE) == "positive"

    def test_truncated_cot_scored_from_reasoning(self):
        assert legacy_sentiment_label(LLAMA_TRUNCATED_COT) == "positive"

    def test_ag_keyword_order_bias(self):
        # "game" (sports keyword) is checked before "market" (business) in the substring scan.
        assert legacy_ag_label("This is about the video game market and its stock price.") == "sports"


class TestStrictSentimentAnswerMode:
    @pytest.mark.parametrize("raw,expected", [
        ("Positive", "positive"),
        ("Negative.", "negative"),
        ("**Positive**", "positive"),
        ("Sentiment: negative", "negative"),
        ("Neutral", NEUTRAL),
        (MISTRAL_HEDGE, NEUTRAL),
        ("neutral (or maybe slightly negative, depending on personal interpretation)", NEUTRAL),
        ("The sentiment is positive.", "positive"),
        ("positive or negative", AMBIGUOUS),
        ("neutative", INVALID),
        ("", INVALID),
        ("I cannot determine this.", INVALID),
    ])
    def test_cases(self, raw, expected):
        assert strict_sentiment_label(raw, "zero_shot") == expected
        assert strict_sentiment_label(raw, "few_shot") == expected


class TestStrictSentimentCotMode:
    @pytest.mark.parametrize("raw,expected", [
        ("Positive.", "positive"),
        ("Step 1: the reviewer praises the acting.\nStep 2: no complaints.\n\nPositive", "positive"),
        ("...so overall the answer is **negative**.", "negative"),
        (LLAMA_TRUNCATED_COT, INVALID),
        ("* The use of the word very reinforces the negative sentiment", INVALID),
        ("Overall this is neutral", NEUTRAL),
        # answer-first style (Mistral): label, then explanation
        ("Negative. The text expresses disappointment and criticism towards the waste of talented actors.", "negative"),
        ("Neutral. The text is stating a fact and does not express a positive or negative sentiment.", NEUTRAL),
    ])
    def test_cases(self, raw, expected):
        assert strict_sentiment_label(raw, "chain_of_thought") == expected


class TestStrictAg:
    @pytest.mark.parametrize("raw,variant,expected", [
        ("Business", "zero_shot", "business"),
        ("sports", "few_shot", "sports"),
        ("Technology", "zero_shot", "science"),
        ("Sci/Tech", "zero_shot", "science"),
        ("Politics", "zero_shot", INVALID),
        ("SPACE.", "zero_shot", INVALID),
        ("science (or business, depending on the focus of the article)", "few_shot", AMBIGUOUS),
        ("World (or, alternatively, Health or Science)", "zero_shot", AMBIGUOUS),
        ("Entertainment", "zero_shot", INVALID),
        ("The topic is world.", "zero_shot", "world"),
        ("world\nHeadline: Fed raises rates\nTopic: business", "few_shot", "world"),
        ("The headline mentions a launch date.\n\nScience", "chain_of_thought", "science"),
        ("which indicates a discussion about", "chain_of_thought", INVALID),
    ])
    def test_cases(self, raw, variant, expected):
        assert strict_ag_label(raw, variant) == expected


class TestLegacyReproducesStoredResults:
    """The legacy parsers must reproduce every stored predicted_label, otherwise 'original vs strict'
    comparisons would not be comparing against what was actually reported."""

    def test_sst2(self):
        import pandas as pd
        d = pd.read_csv("results/all_models_all_prompts.csv")
        assert (d.raw_output.fillna("").map(legacy_sentiment_label) == d.predicted_label).all()

    def test_ag_news(self):
        import pandas as pd
        d = pd.read_csv("results/ag_news_results.csv")
        assert (d.raw_output.fillna("").map(legacy_ag_label) == d.predicted_label).all()
