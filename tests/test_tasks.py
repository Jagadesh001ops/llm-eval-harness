"""Every registered task must import, have a seed prompt whose placeholders it fills, and score a sample output.
(Guards against the kind of syntax/registration error that previously slipped past the parser-only tests.)"""

import pathlib
import re

import pytest

from harness.tasks import TASKS

SAMPLE_ITEMS = {
    "ag_news": {"id": "x", "text": "Apple unveils new chip", "label": "science"},
    "gsm8k": {"id": "x", "question": "2 + 2?", "gold": 4.0},
    "squad_v2": {"id": "x", "context": "The Normans lived in Normandy.", "question": "Where?", "golds": ["Normandy"],
                 "group": "answerable"},
}


def sample_item(name):
    if name.startswith("bbh_"):
        return {"id": "x", "input": "Q?\nOptions:\n(A) a\n(B) b", "target": "(B)"}
    return SAMPLE_ITEMS[name]


@pytest.mark.parametrize("name", sorted(TASKS))
def test_task_fills_seed_and_scores(name):
    task = TASKS[name]
    seed = pathlib.Path(f"prompts/v2/{name}.txt").read_text(encoding="utf-8")
    item = sample_item(name)
    filled = task.fill(seed, item)
    assert not re.search(r"\{[a-z_]+\}", filled), f"unfilled placeholder in {name}"
    assert isinstance(task.show_input(item), str) and isinstance(task.show_gold(item), str)
    assert 0 <= task.score("Answer: (B)", item) <= 1
