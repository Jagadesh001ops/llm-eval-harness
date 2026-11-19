"""Task registry for the prompt-optimization study: loads frozen split items and scores model outputs."""

import json
import pathlib
from dataclasses import dataclass
from typing import Callable

from harness.parsers import (NO_ANSWER, bbh_correct, exact_match, extract_trivia_answer, gsm8k_correct,
                             gsm8k_gold, squad_score, strict_ag_label)

AG_LABELS = ["world", "sports", "business", "science"]


@dataclass
class Task:
    name: str
    max_tokens: int                      # answer budget for the target model
    fill: Callable[[str, dict], str]     # (prompt template, item) -> prompt text
    score: Callable[[str, dict], int]    # (raw output, item) -> 1 correct / 0 wrong
    show_input: Callable[[dict], str]    # item -> text shown to an optimizer in a failure example
    show_gold: Callable[[dict], str]


def _ag_items(split_name):
    from datasets import load_dataset
    spec = json.loads(pathlib.Path("data/splits/ag_news.json").read_text(encoding="utf-8"))
    part = spec[split_name]
    ds = load_dataset(spec["hf_path"], split=part["split"])
    return [{"id": f"{part['split']}:{i}", "text": ds[i]["text"], "label": AG_LABELS[ds[i]["label"]]}
            for i in part["indices"]]


def _gsm8k_items(split_name):
    from datasets import load_dataset
    spec = json.loads(pathlib.Path("data/splits/gsm8k.json").read_text(encoding="utf-8"))
    part = spec[split_name]
    ds = load_dataset(spec["hf_path"], spec["hf_config"], split=part["split"])
    return [{"id": f"{part['split']}:{i}", "question": ds[i]["question"], "gold": gsm8k_gold(ds[i]["answer"])}
            for i in part["indices"]]


def _fmt_number(x):
    return str(int(x)) if x is not None and float(x).is_integer() else str(x)


def _split_spec(task):
    return json.loads(pathlib.Path(f"data/splits/{task}.json").read_text(encoding="utf-8"))


def _bbh_items(task, split_name):
    from datasets import load_dataset
    spec = _split_spec(task)
    ds = load_dataset(spec["hf_path"], spec["hf_config"], split="test")
    return [{"id": f"test:{i}", "input": ds[i]["input"], "target": ds[i]["target"]}
            for i in spec[split_name]["indices"]]


def _squad_items(split_name):
    from datasets import load_dataset
    spec = _split_spec("squad_v2")
    part = spec[split_name]
    ds = load_dataset(spec["hf_path"], split=part["split"])
    out = []
    for i in part["indices"]:
        r = ds[i]
        golds = list(dict.fromkeys(r["answers"]["text"]))
        out.append({"id": f"{part['split']}:{i}", "context": r["context"], "question": r["question"], "golds": golds,
                    "group": "answerable" if golds else "unanswerable"})
    return out


def _trivia_items(split_name):
    from datasets import load_dataset
    spec = _split_spec("triviaqa")
    part = spec[split_name]
    ds = load_dataset(spec["hf_path"], spec["hf_config"], split=part["split"])
    out = []
    for i in part["indices"]:
        a = ds[i]["answer"]
        out.append({"id": f"{part['split']}:{i}", "question": ds[i]["question"],
                    "aliases": sorted(set(a["aliases"] + [a["value"]])), "value": a["value"]})
    return out


BBH_SUBTASKS = ["date_understanding", "logical_deduction_five_objects", "tracking_shuffled_objects_five_objects",
                "web_of_lies", "causal_judgement"]

TASKS = {
    "ag_news": Task(
        name="ag_news",
        max_tokens=50,
        fill=lambda tmpl, it: tmpl.replace("{text}", it["text"]),
        score=lambda out, it: int(strict_ag_label(out, "zero_shot") == it["label"]),
        show_input=lambda it: it["text"],
        show_gold=lambda it: it["label"],
    ),
    "gsm8k": Task(
        name="gsm8k",
        max_tokens=512,  # room to finish step-by-step reasoning (v1 CoT was truncated at 50-100 tokens)
        fill=lambda tmpl, it: tmpl.replace("{question}", it["question"]),
        score=lambda out, it: gsm8k_correct(out, it["gold"]),
        show_input=lambda it: it["question"],
        show_gold=lambda it: _fmt_number(it["gold"]),  # final number only, never the worked solution
    ),
    "triviaqa": Task(
        name="triviaqa",
        max_tokens=30,  # short direct answers; the control measures format vs knowledge
        fill=lambda tmpl, it: tmpl.replace("{question}", it["question"]),
        score=lambda out, it: int(exact_match(extract_trivia_answer(out), it["aliases"])),  # EM drives selection
        show_input=lambda it: it["question"],
        show_gold=lambda it: it["value"],
    ),
    "squad_v2": Task(
        name="squad_v2",
        max_tokens=60,
        fill=lambda tmpl, it: tmpl.replace("{context}", it["context"]).replace("{question}", it["question"]),
        score=lambda out, it: squad_score(out, it["golds"]),  # per-item F1 in [0, 1]
        show_input=lambda it: "Passage: " + it["context"] + "\nQuestion: " + it["question"],
        show_gold=lambda it: it["golds"][0] if it["golds"] else NO_ANSWER,
    ),
}

for _sub in BBH_SUBTASKS:
    TASKS[f"bbh_{_sub}"] = Task(
        name=f"bbh_{_sub}",
        max_tokens=2048,  # 768 truncated up to 39% of Ministral replies in the first headroom run
        fill=lambda tmpl, it: tmpl.replace("{input}", it["input"]),
        score=lambda out, it: bbh_correct(out, it["target"]),
        show_input=lambda it: it["input"],
        show_gold=lambda it: it["target"],
    )

_LOADERS = {"ag_news": _ag_items, "gsm8k": _gsm8k_items, "squad_v2": _squad_items, "triviaqa": _trivia_items,
            **{f"bbh_{s}": (lambda split, _t=f"bbh_{s}": _bbh_items(_t, split)) for s in BBH_SUBTASKS}}


def load_items(task, split_name):
    """split_name in {'pool', 'dev', 'test'}; items carry a stable 'id'."""
    return _LOADERS[task](split_name)
