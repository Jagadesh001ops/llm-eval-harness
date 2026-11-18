# config.py

# Models we're evaluating
MODELS = {
    "llama3-8b": {
        "model_id": "meta.llama3-8b-instruct-v1:0",
        "price_per_1k_input_tokens": 0.0003,
        "price_per_1k_output_tokens": 0.0006,
    },
    "mistral-7b": {
        "model_id": "mistral.mistral-7b-instruct-v0:2",
        "price_per_1k_input_tokens": 0.00015,
        "price_per_1k_output_tokens": 0.0002,
    },
    "nova-micro": {
        "model_id": "amazon.nova-micro-v1:0",
        "price_per_1k_input_tokens": 0.000035,
        "price_per_1k_output_tokens": 0.00014,
    },
    "mixtral-8x7b": {
        "model_id": "mistral.mixtral-8x7b-instruct-v0:1",
        "price_per_1k_input_tokens": 0.00045,
        "price_per_1k_output_tokens": 0.0007,
    },
    "llama3-70b": {
        "model_id": "meta.llama3-70b-instruct-v1:0",
        "price_per_1k_input_tokens": 0.00265,
        "price_per_1k_output_tokens": 0.0035,
    },
    # ── Prompt-optimization study (docs/design_prompt_optimization.md) ──
    # Bedrock on-demand, us-east-1; prices are Bedrock list prices per 1k tokens as of 2026-11-18.
    "gemma3-4b": {"model_id": "google.gemma-3-4b-it",
                  "price_per_1k_input_tokens": 0.00004, "price_per_1k_output_tokens": 0.00008},
    "gemma3-12b": {"model_id": "google.gemma-3-12b-it",
                   "price_per_1k_input_tokens": 0.00009, "price_per_1k_output_tokens": 0.00029},
    "gemma3-27b": {"model_id": "google.gemma-3-27b-it",
                   "price_per_1k_input_tokens": 0.00023, "price_per_1k_output_tokens": 0.00038},
    "ministral-3b": {"model_id": "mistral.ministral-3-3b-instruct",
                     "price_per_1k_input_tokens": 0.0001, "price_per_1k_output_tokens": 0.0001},
    "ministral-8b": {"model_id": "mistral.ministral-3-8b-instruct",
                     "price_per_1k_input_tokens": 0.00015, "price_per_1k_output_tokens": 0.00015},
    "ministral-14b": {"model_id": "mistral.ministral-3-14b-instruct",
                      "price_per_1k_input_tokens": 0.0002, "price_per_1k_output_tokens": 0.0002},
    "mistral-large-3": {"model_id": "mistral.mistral-large-3-675b-instruct",
                        "price_per_1k_input_tokens": 0.0005, "price_per_1k_output_tokens": 0.0015},
}

# Claude models, called through the Anthropic API (not Bedrock). Prices per 1k tokens, list prices 2026-11-18.
# Thinking tokens are billed as output.
CLAUDE_MODELS = {
    "claude-opus-5.5": {"model_id": "claude-opus-5-5",
                        "price_per_1k_input_tokens": 0.004, "price_per_1k_output_tokens": 0.020},
    "claude-haiku-5.5": {"model_id": "claude-haiku-5-5",
                         "price_per_1k_input_tokens": 0.0001, "price_per_1k_output_tokens": 0.0005},
    "claude-sonnet-5.5": {"model_id": "claude-sonnet-5-5",
                          "price_per_1k_input_tokens": 0.002, "price_per_1k_output_tokens": 0.010},
}
# AWS settings
AWS_REGION = "us-east-1"
S3_BUCKET = "jagadesh-ml-projects"

# Eval settings
TASK = "sentiment"
NUM_EXAMPLES = 100  # start small, scale up later

# Prompt variants we'll test
PROMPT_TEMPLATES = {
    "zero_shot": (
        "Classify the sentiment of the following text as positive or negative. "
        "Reply with one word only.\n\nText: {text}\nSentiment:"
    ),
    "few_shot": (
        "Classify sentiment as positive or negative. Reply with one word only.\n\n"
        "Text: The food was amazing!\nSentiment: positive\n"
        "Text: Worst experience ever.\nSentiment: negative\n"
        "Text: {text}\nSentiment:"
    ),
    "chain_of_thought": (
        "Classify the sentiment of the following text as positive or negative. "
        "Think step by step, then end your answer with one word: positive or negative.\n\n"
        "Text: {text}\nAnswer:"
    ),
}

AG_NEWS_PROMPT_TEMPLATES = {
    "zero_shot": (
        "Classify the topic of the following news headline as one of: world, sports, business, or science. "
        "Reply with one word only.\n\nHeadline: {text}\nTopic:"
    ),
    "few_shot": (
        "Classify the topic as world, sports, business, or science. Reply with one word only.\n\n"
        "Headline: England win the World Cup in overtime.\nTopic: sports\n"
        "Headline: Federal Reserve raises interest rates again.\nTopic: business\n"
        "Headline: Scientists discover new exoplanet orbiting nearby star.\nTopic: science\n"
        "Headline: {text}\nTopic:"
    ),
    "chain_of_thought": (
        "Classify the topic of the following news headline as world, sports, business, or science. "
        "Think step by step, then end your answer with one word: world, sports, business, or science.\n\n"
        "Headline: {text}\nAnswer:"
    ),
}

TRIVIA_PROMPT_TEMPLATES = {
    "zero_shot": (
        "Answer the following question with a short, direct answer. "
        "Give only the answer, nothing else.\n\nQuestion: {question}\nAnswer:"
    ),
    "few_shot": (
        "Answer each question with a short, direct answer.\n\n"
        "Question: What is the capital of France?\nAnswer: Paris\n"
        "Question: Who wrote Romeo and Juliet?\nAnswer: William Shakespeare\n"
        "Question: {question}\nAnswer:"
    ),
    "chain_of_thought": (
        "Answer the following question. Think through it step by step, "
        "then give your final answer on the last line prefixed with 'Answer:'.\n\n"
        "Question: {question}"
    ),
}