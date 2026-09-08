import boto3
import re
import time
from datetime import datetime, timezone
from rouge_score import rouge_scorer as rouge_scorer_lib
from config import MODELS, AWS_REGION, PROMPT_TEMPLATES, AG_NEWS_PROMPT_TEMPLATES, TRIVIA_PROMPT_TEMPLATES

bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)
dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
cloudwatch = boto3.client("cloudwatch", region_name=AWS_REGION)
table = dynamodb.Table("llm-eval-results")

NAMESPACE = "LLMEvalHarness"

SUMMARIZE_PROMPT = (
    "Summarize the following news article in 2-3 sentences. "
    "Only include facts from the article.\n\n"
    "Article: {article}\n\nSummary:"
)

SENTIMENT_EXAMPLES = [
    {"text": "The movie was absolutely wonderful.", "true_label": "positive"},
    {"text": "I hated every minute of it.", "true_label": "negative"},
    {"text": "A brilliant and touching performance.", "true_label": "positive"},
    {"text": "Complete waste of time and money.", "true_label": "negative"},
    {"text": "One of the best films I have ever seen.", "true_label": "positive"},
    {"text": "Terrible acting and a boring plot.", "true_label": "negative"},
    {"text": "Heartwarming and beautifully directed.", "true_label": "positive"},
    {"text": "I walked out halfway through.", "true_label": "negative"},
    {"text": "Funny, clever, and endlessly entertaining.", "true_label": "positive"},
    {"text": "The worst sequel ever made.", "true_label": "negative"},
    {"text": "A masterpiece of modern cinema.", "true_label": "positive"},
    {"text": "Painfully slow and deeply disappointing.", "true_label": "negative"},
    {"text": "Charming performances from the entire cast.", "true_label": "positive"},
    {"text": "I could not follow the story at all.", "true_label": "negative"},
    {"text": "Visually stunning and emotionally gripping.", "true_label": "positive"},
    {"text": "Poorly written and badly acted.", "true_label": "negative"},
    {"text": "A delightful surprise from start to finish.", "true_label": "positive"},
    {"text": "Dull, predictable, and forgettable.", "true_label": "negative"},
    {"text": "The performances were outstanding.", "true_label": "positive"},
    {"text": "An absolute disaster of a film.", "true_label": "negative"},
]

AG_NEWS_EXAMPLES = [
    {"text": "Tensions rise between US and China over Taiwan strait military exercises", "true_label": "world"},
    {"text": "United Nations Security Council meets to discuss Gaza ceasefire proposals", "true_label": "world"},
    {"text": "Russia launches fresh missile strikes on Ukrainian infrastructure targets", "true_label": "world"},
    {"text": "European Union imposes new sanctions on Belarus over human rights violations", "true_label": "world"},
    {"text": "North Korea test-fires intercontinental ballistic missile into Sea of Japan", "true_label": "world"},
    {"text": "LeBron James scores 38 points as Lakers defeat Celtics in overtime thriller", "true_label": "sports"},
    {"text": "England qualify for Euro 2026 after defeating Ireland three goals to nil", "true_label": "sports"},
    {"text": "Novak Djokovic wins record 25th Grand Slam title at Australian Open", "true_label": "sports"},
    {"text": "Manchester City clinch Premier League title with victory over Arsenal", "true_label": "sports"},
    {"text": "Simone Biles returns to gymnastics with perfect score at US Championships", "true_label": "sports"},
    {"text": "Federal Reserve holds interest rates steady amid mixed inflation signals", "true_label": "business"},
    {"text": "Apple shares surge after record quarterly revenue tops 130 billion dollars", "true_label": "business"},
    {"text": "Amazon announces layoffs as it restructures cloud and retail divisions", "true_label": "business"},
    {"text": "Oil prices fall to six-month low on weak demand outlook from China", "true_label": "business"},
    {"text": "Tesla reports surprise profit decline despite strong vehicle delivery numbers", "true_label": "business"},
    {"text": "NASA Artemis crew completes first crewed lunar orbit in fifty years", "true_label": "science"},
    {"text": "OpenAI releases new reasoning model claiming to surpass human expert scores", "true_label": "science"},
    {"text": "Scientists confirm detection of gravitational waves from neutron star collision", "true_label": "science"},
    {"text": "SpaceX Starship completes fully successful orbital test flight and landing", "true_label": "science"},
    {"text": "Researchers develop battery technology that charges electric cars in five minutes", "true_label": "science"},
]

TRIVIA_EXAMPLES = [
    {"question": "What is the capital of Japan?", "aliases": ["Tokyo", "Tōkyō"]},
    {"question": "Who painted the Mona Lisa?", "aliases": ["Leonardo da Vinci", "Leonardo Di Vinci", "Da Vinci", "Leonardo"]},
    {"question": "What is the largest planet in our solar system?", "aliases": ["Jupiter"]},
    {"question": "In what year did World War II end?", "aliases": ["1945"]},
    {"question": "What is the chemical symbol for gold?", "aliases": ["Au"]},
    {"question": "Who wrote Pride and Prejudice?", "aliases": ["Jane Austen", "Austen"]},
    {"question": "What is the smallest country in the world by area?", "aliases": ["Vatican City", "Vatican", "Holy See"]},
    {"question": "Who invented the telephone?", "aliases": ["Alexander Graham Bell", "Graham Bell", "Alexander Bell"]},
    {"question": "What is the longest river in the world?", "aliases": ["Nile", "Nile River"]},
    {"question": "Which element has atomic number 1?", "aliases": ["Hydrogen", "H"]},
    {"question": "Who was the first person to walk on the Moon?", "aliases": ["Neil Armstrong", "Armstrong"]},
    {"question": "What is the national currency of Japan?", "aliases": ["Yen", "Japanese Yen"]},
    {"question": "How many sides does a hexagon have?", "aliases": ["6", "six"]},
    {"question": "What is the chemical formula for water?", "aliases": ["H2O", "H₂O"]},
    {"question": "Who wrote the play Hamlet?", "aliases": ["William Shakespeare", "Shakespeare"]},
]

SUMMARIZATION_EXAMPLES = [
    {
        "article": "Scientists at NASA announced a major breakthrough in space exploration on Tuesday. The team successfully tested a new propulsion system that could cut travel time to Mars in half. The system uses solar energy to power ion thrusters, making it more efficient than traditional chemical rockets. NASA plans to conduct further tests before deploying the technology on a crewed mission scheduled for 2032.",
        "reference": "NASA tested a new solar-powered ion propulsion system that could halve Mars travel time, with crewed missions planned for 2032."
    },
    {
        "article": "Global temperatures reached record highs in July 2026, with scientists warning that climate change is accelerating faster than predicted. The World Meteorological Organization reported that the past twelve months were the hottest on record. Rising sea levels and extreme weather events have displaced millions across Asia and Africa. World leaders are expected to convene an emergency summit next month to discuss accelerated emissions reduction targets.",
        "reference": "July 2026 saw record global temperatures, with WMO confirming the hottest twelve months on record and world leaders planning an emergency climate summit."
    },
    {
        "article": "Apple unveiled its latest iPhone lineup at its annual September event, featuring a new AI chip that the company claims is three times faster than its predecessor. The devices also include an upgraded camera system with real-time object recognition. Analysts expect strong holiday sales, particularly in emerging markets where Apple has been aggressively expanding its retail presence.",
        "reference": "Apple announced new iPhones with a faster AI chip and improved cameras, with analysts predicting strong holiday sales in emerging markets."
    },
    {
        "article": "Researchers at Oxford University have developed a new vaccine that shows promise against multiple strains of influenza. In clinical trials involving 5,000 participants, the vaccine demonstrated 94% efficacy across all tested strains. The team used mRNA technology similar to that used in COVID-19 vaccines. If approved, the vaccine could replace annual flu shots with a single long-lasting dose.",
        "reference": "Oxford researchers developed an mRNA flu vaccine with 94% efficacy across multiple strains that could replace annual flu shots with a single dose."
    },
    {
        "article": "The Federal Reserve announced it would hold interest rates steady at its latest policy meeting, citing mixed signals from the economy. Inflation has cooled significantly from its peak but remains above the Fed's 2% target. Unemployment remains low at 3.8% but job growth has slowed in recent months. Fed Chair Jerome Powell indicated that rate cuts could begin later in the year if inflation continues to decline.",
        "reference": "The Fed held rates steady amid mixed economic signals, with Powell hinting at possible cuts later in the year if inflation keeps falling."
    },
]

# ── Helpers ────────────────────────────────────────────────────────────────

def invoke_model(model_key, prompt, max_tokens=50):
    model_id = MODELS[model_key]["model_id"]
    start = time.time()
    response = bedrock.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": max_tokens, "temperature": 0}
    )
    latency = time.time() - start
    return {
        "output": response["output"]["message"]["content"][0]["text"].strip(),
        "input_tokens": response["usage"]["inputTokens"],
        "output_tokens": response["usage"]["outputTokens"],
        "latency_ms": round(latency * 1000),
    }

def compute_cost(input_tokens, output_tokens, model_key):
    m = MODELS[model_key]
    return (input_tokens / 1000) * m["price_per_1k_input_tokens"] + \
           (output_tokens / 1000) * m["price_per_1k_output_tokens"]

def extract_label(raw_output):
    text = raw_output.lower().strip()
    first_word = text.split()[0].strip(".,!?") if text else ""
    for label in ["positive", "negative"]:
        if label in text or first_word == label:
            return label
    return "unknown"

_AG_KEYWORDS = {
    "world":    ["world", "international", "global", "politics", "political", "foreign"],
    "sports":   ["sports", "sport", "athletic", "athletics", "game", "games", "soccer", "football", "basketball"],
    "business": ["business", "finance", "financial", "economy", "economic", "market", "stock", "trade"],
    "science":  ["science", "scientific", "technology", "tech", "sci", "research", "space", "computing"],
}

def extract_ag_label(raw_output):
    text = raw_output.lower().strip()
    first_word = re.split(r"[\s.,!?:]", text)[0] if text else ""
    for label, kws in _AG_KEYWORDS.items():
        if first_word in kws:
            return label
    for label, kws in _AG_KEYWORDS.items():
        if any(kw in text for kw in kws):
            return label
    return "unknown"

def _normalize(s):
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    return re.sub(r"\s+", " ", s).strip()

def extract_trivia_answer(raw_output):
    for line in reversed(raw_output.strip().split("\n")):
        if line.lower().startswith("answer:"):
            return line[7:].strip()
    return raw_output.strip().split("\n")[0].strip()

def exact_match(prediction, aliases):
    pred = _normalize(prediction)
    return any(_normalize(a) == pred for a in aliases)

def contains_answer(prediction, aliases):
    pred = _normalize(prediction)
    return any(_normalize(a) in pred for a in aliases)

def compute_rouge(prediction, reference):
    scorer = rouge_scorer_lib.RougeScorer(["rougeL"], use_stemmer=True)
    return round(scorer.score(reference, prediction)["rougeL"].fmeasure, 4)

def write_result(run_id, model, task, prompt_variant, example_id, metrics):
    item = {
        "run_id": run_id,
        "result_id": f"{model}#{task}#{prompt_variant}#{example_id}",
        "model": model,
        "task": task,
        "prompt_variant": prompt_variant,
        "example_id": str(example_id),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **{k: str(v) if isinstance(v, float) else v for k, v in metrics.items()}
    }
    table.put_item(Item=item)

def put_metrics(model, task, prompt_variant, latency_ms, cost_usd, tokens_in, tokens_out):
    dimensions = [
        {"Name": "Model", "Value": model},
        {"Name": "Task", "Value": task},
        {"Name": "PromptVariant", "Value": prompt_variant},
    ]
    now = datetime.now(timezone.utc)
    cloudwatch.put_metric_data(
        Namespace=NAMESPACE,
        MetricData=[
            {"MetricName": "Latency", "Dimensions": dimensions, "Timestamp": now, "Value": latency_ms, "Unit": "Milliseconds"},
            {"MetricName": "CostUSD", "Dimensions": dimensions, "Timestamp": now, "Value": cost_usd, "Unit": "None"},
            {"MetricName": "InputTokens", "Dimensions": dimensions, "Timestamp": now, "Value": tokens_in, "Unit": "Count"},
            {"MetricName": "OutputTokens", "Dimensions": dimensions, "Timestamp": now, "Value": tokens_out, "Unit": "Count"},
        ],
    )

# ── Task runners ───────────────────────────────────────────────────────────

def run_sentiment(run_id, models, num_examples):
    summary = []
    examples = SENTIMENT_EXAMPLES[:num_examples]
    for model_key in models:
        for prompt_name, template in PROMPT_TEMPLATES.items():
            correct = 0
            for i, ex in enumerate(examples):
                result = invoke_model(model_key, template.format(text=ex["text"]))
                label = extract_label(result["output"])
                cost = compute_cost(result["input_tokens"], result["output_tokens"], model_key)
                is_correct = int(label == ex["true_label"])
                correct += is_correct
                write_result(run_id, model_key, "sentiment", prompt_name, i, {
                    "true_label": ex["true_label"],
                    "predicted_label": label,
                    "correct": is_correct,
                    "latency_ms": result["latency_ms"],
                    "cost_usd": cost,
                })
                put_metrics(model_key, "sentiment", prompt_name,
                            result["latency_ms"], cost,
                            result["input_tokens"], result["output_tokens"])
            summary.append({
                "model": model_key,
                "prompt": prompt_name,
                "accuracy": round(correct / len(examples), 3),
            })
    return summary

def run_summarization(run_id, models, num_examples):
    summary = []
    examples = SUMMARIZATION_EXAMPLES[:num_examples]
    for model_key in models:
        rouge_scores = []
        for i, ex in enumerate(examples):
            prompt = SUMMARIZE_PROMPT.format(article=ex["article"])
            result = invoke_model(model_key, prompt, max_tokens=150)
            rouge = compute_rouge(result["output"], ex["reference"])
            cost = compute_cost(result["input_tokens"], result["output_tokens"], model_key)
            rouge_scores.append(rouge)
            write_result(run_id, model_key, "summarization", "default", i, {
                "summary": result["output"],
                "rouge_l": rouge,
                "faithfulness": "pending",  # filled by faithfulness_check.py
                "latency_ms": result["latency_ms"],
                "cost_usd": cost,
            })
            put_metrics(model_key, "summarization", "default",
                        result["latency_ms"], cost,
                        result["input_tokens"], result["output_tokens"])
        summary.append({
            "model": model_key,
            "avg_rouge_l": round(sum(rouge_scores) / len(rouge_scores), 3),
        })
    return summary

def run_ag_news(run_id, models, num_examples):
    summary = []
    examples = AG_NEWS_EXAMPLES[:num_examples]
    for model_key in models:
        for prompt_name, template in AG_NEWS_PROMPT_TEMPLATES.items():
            correct = 0
            for i, ex in enumerate(examples):
                result = invoke_model(model_key, template.format(text=ex["text"]))
                label = extract_ag_label(result["output"])
                cost = compute_cost(result["input_tokens"], result["output_tokens"], model_key)
                is_correct = int(label == ex["true_label"])
                correct += is_correct
                write_result(run_id, model_key, "ag_news", prompt_name, i, {
                    "true_label": ex["true_label"],
                    "predicted_label": label,
                    "correct": is_correct,
                    "latency_ms": result["latency_ms"],
                    "cost_usd": cost,
                })
                put_metrics(model_key, "ag_news", prompt_name,
                            result["latency_ms"], cost,
                            result["input_tokens"], result["output_tokens"])
            summary.append({
                "model": model_key,
                "prompt": prompt_name,
                "accuracy": round(correct / len(examples), 3),
            })
    return summary

def run_trivia(run_id, models, num_examples):
    summary = []
    examples = TRIVIA_EXAMPLES[:num_examples]
    for model_key in models:
        for prompt_name, template in TRIVIA_PROMPT_TEMPLATES.items():
            em_total, ca_total = 0, 0
            for i, ex in enumerate(examples):
                result = invoke_model(model_key, template.format(question=ex["question"]), max_tokens=100)
                answer = extract_trivia_answer(result["output"])
                cost = compute_cost(result["input_tokens"], result["output_tokens"], model_key)
                em = int(exact_match(answer, ex["aliases"]))
                ca = int(contains_answer(answer, ex["aliases"]))
                em_total += em
                ca_total += ca
                write_result(run_id, model_key, "trivia", prompt_name, i, {
                    "question": ex["question"],
                    "predicted_answer": answer,
                    "exact_match": em,
                    "contains_answer": ca,
                    "latency_ms": result["latency_ms"],
                    "cost_usd": cost,
                })
                put_metrics(model_key, "trivia", prompt_name,
                            result["latency_ms"], cost,
                            result["input_tokens"], result["output_tokens"])
            n = len(examples)
            summary.append({
                "model": model_key,
                "prompt": prompt_name,
                "exact_match": round(em_total / n, 3),
                "contains_answer": round(ca_total / n, 3),
            })
    return summary

# ── Handler ────────────────────────────────────────────────────────────────

def handler(event, context):
    """
    event = {
        "models": ["llama3-8b", "mistral-7b"],  # optional
        "tasks": ["sentiment", "summarization"], # optional
        "num_examples": 10                       # optional
    }
    """
    run_id = f"lambda-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    models = event.get("models", ["llama3-8b", "mistral-7b"])
    tasks = event.get("tasks", ["sentiment", "summarization"])
    num_examples = event.get("num_examples", 5)

    response = {"run_id": run_id, "results": {}}

    if "sentiment" in tasks:
        response["results"]["sentiment"] = run_sentiment(run_id, models, num_examples)

    if "summarization" in tasks:
        response["results"]["summarization"] = run_summarization(run_id, models, num_examples)

    if "ag_news" in tasks:
        response["results"]["ag_news"] = run_ag_news(run_id, models, num_examples)

    if "trivia" in tasks:
        response["results"]["trivia"] = run_trivia(run_id, models, num_examples)

    return {"statusCode": 200, **response}