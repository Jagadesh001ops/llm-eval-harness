"""Task registry for the prompt-optimization study: loads frozen split items and scores model outputs."""

import json
import pathlib
from dataclasses import dataclass
from typing import Callable

from harness.parsers import gsm8k_correct, gsm8k_gold, strict_ag_label

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
}

_LOADERS = {"ag_news": _ag_items, "gsm8k": _gsm8k_items}


def load_items(task, split_name):
    """split_name in {'pool', 'dev', 'test'}; items carry a stable 'id'."""
    return _LOADERS[task](split_name)
