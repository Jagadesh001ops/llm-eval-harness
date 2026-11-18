"""Seed-prompt accuracy on the full dev split for each target: is there room for prompt optimization to help?

Uses only dev items (never test), so it can inform the choice of tasks/targets without touching reported results.
Outputs results/opt/headroom_<task>.csv with point estimate and 95% bootstrap CI per target.

    python scripts/check_headroom.py gsm8k gemma3-4b gemma3-12b gemma3-27b ministral-3b ministral-8b ministral-14b
"""

import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from harness.net import configure_huggingface  # noqa: E402
from harness.scorer import prompt_sha, score_prompt  # noqa: E402
from harness.stats import bootstrap_mean_ci  # noqa: E402
from harness.tasks import TASKS, load_items  # noqa: E402


def main():
    task_name, targets = sys.argv[1], sys.argv[2:]
    configure_huggingface()
    task = TASKS[task_name]
    seed = pathlib.Path(f"prompts/v2/{task_name}.txt").read_text(encoding="utf-8")
    dev = load_items(task_name, "dev")
    rows = []
    for t in targets:
        res = score_prompt(task, t, seed, dev)
        correct = [res["rows"][it["id"]]["correct"] for it in dev]
        trunc = sum(res["rows"][it["id"]]["stop_reason"] == "max_tokens" for it in dev)
        ci = bootstrap_mean_ci(correct)
        row = {"task": task_name, "target": t, "seed_sha": prompt_sha(seed), "split": "dev", "n": ci["n"],
               "accuracy": ci["point"], "ci_lo": ci["lo"], "ci_hi": ci["hi"], "truncated": trunc,
               "new_cost_usd": res["cost_usd"]}
        groups = sorted({it["group"] for it in dev if "group" in it})
        for g in groups:  # e.g. SQuAD 2.0 answerable vs unanswerable
            row[f"score_{g}"] = sum(c for c, it in zip(correct, dev) if it.get("group") == g) / \
                                sum(it.get("group") == g for it in dev)
        rows.append(row)
        extra = "  ".join(f"{g} {row[f'score_{g}']*100:.1f}" for g in groups)
        print(f"{t:<15} dev score {ci['point']*100:5.1f} [{ci['lo']*100:.1f}, {ci['hi']*100:.1f}]  "
              f"truncated {trunc}/{ci['n']}  {extra}  cost ${res['cost_usd']:.4f}")
    out = pathlib.Path(f"results/opt/headroom_{task_name}.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
