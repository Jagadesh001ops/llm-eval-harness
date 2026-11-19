"""Score a prompt template on a set of items with a target model, with a persistent per-item cache.

Cache key = (target model, sha256 of the exact prompt template, item id). Every condition that evaluates the same
prompt on the same items (e.g. the shared seed prompt) reuses the same outputs, so it is paid for once, and reruns
after an interruption resume instead of re-spending.
"""

import hashlib
import json
import pathlib
import threading
from concurrent.futures import ThreadPoolExecutor

from harness.bedrock import compute_cost, invoke_model

CACHE_DIR = pathlib.Path("results/opt/cache")
_lock = threading.Lock()


def prompt_sha(template):
    return hashlib.sha256(template.encode("utf-8")).hexdigest()[:16]


def cache_path(task, target, template):
    """Key = task + target + exact prompt + answer budget. Item ids are only unique within a task (BBH sub-tasks
    share ids such as 'test:31' and the same seed text), and a different max_tokens yields different outputs, so
    both must be part of the key."""
    return CACHE_DIR / task.name / target / f"{prompt_sha(template)}_mt{task.max_tokens}.jsonl"


def _load_cache(path, task):
    rows = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("task") not in (None, task.name):
                    raise RuntimeError(f"cache row for task {r.get('task')} found in {path} (expected {task.name})")
                rows[r["id"]] = r
    return rows


def score_prompt(task, target, template, items, workers=8):
    """Return dict(accuracy, n, rows, new_calls, cost_usd). rows: per item id -> {output, correct, ...}."""
    path = cache_path(task, target, template)
    path.parent.mkdir(parents=True, exist_ok=True)
    cached = _load_cache(path, task)
    sidecar = path.with_suffix(".prompt.txt")
    if not sidecar.exists():
        sidecar.write_text(template, encoding="utf-8")
    todo = [it for it in items if it["id"] not in cached]

    def run(it):
        r = invoke_model(target, task.fill(template, it), max_tokens=task.max_tokens, temperature=0)
        row = {"task": task.name, "max_tokens": task.max_tokens, "id": it["id"], "output": r["output"], "correct": task.score(r["output"], it),
               "input_tokens": r["input_tokens"], "output_tokens": r["output_tokens"],
               "stop_reason": r["stop_reason"], "latency_ms": r["latency_ms"],
               "cost_usd": compute_cost(r["input_tokens"], r["output_tokens"], target)}
        with _lock, path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        return row

    new_rows = []
    if todo:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            new_rows = list(ex.map(run, todo))
    for r in new_rows:
        cached[r["id"]] = r
    rows = {it["id"]: cached[it["id"]] for it in items}
    correct = [rows[i]["correct"] for i in rows]
    return {"accuracy": sum(correct) / len(correct) if correct else float("nan"), "n": len(correct), "rows": rows,
            "new_calls": len(new_rows), "cost_usd": sum(r["cost_usd"] for r in new_rows)}
