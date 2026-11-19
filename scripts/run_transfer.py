"""Prompt transfer (design doc §5 step 7, hypothesis H4): score every loop's final prompt, unchanged, on the test
items of the OTHER targets that run the same task. No optimization happens here, only scoring.

AG News runs on all six targets (both families), so its prompts transfer within and across families; SQuAD 2.0
and TriviaQA run on Gemma only. Work is grouped by (task, destination target) so no two workers append to the same
cache file. Prompts already scored on a destination (e.g. a final prompt identical to the seed) are cache hits.

Output: results/opt/transfer.csv, one row per (source loop, destination target):
  task, source_target, condition, run, dest_target, final_sha, final_acc, seed_acc, gain, n

    python scripts/run_transfer.py            # all finished protocol-v3 loops
    python scripts/run_transfer.py --dry-run
"""

import argparse
import json
import pathlib
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from harness.net import configure_huggingface  # noqa: E402
from harness.scorer import prompt_sha, score_prompt  # noqa: E402
from harness.tasks import TASKS, load_items  # noqa: E402

TARGETS = {
    "ag_news": ["gemma3-4b", "gemma3-12b", "gemma3-27b", "ministral-3b", "ministral-8b", "ministral-14b"],
    "squad_v2": ["gemma3-4b", "gemma3-12b", "gemma3-27b"],
    "triviaqa": ["gemma3-4b", "gemma3-12b", "gemma3-27b"],
}


def finished_loops(root):
    loops = []
    for p in sorted(pathlib.Path(root).rglob("log.json")):
        L = json.loads(p.read_text(encoding="utf-8"))
        if L.get("status") == "done" and L.get("protocol") == "v3" and "final_prompt" in L:
            loops.append(L)
    return loops


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="results/opt")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    loops = finished_loops(args.root)
    jobs = {}  # (task, dest) -> list of source loops
    for L in loops:
        for dest in TARGETS[L["task"]]:
            if dest != L["target"]:
                jobs.setdefault((L["task"], dest), []).append(L)
    n_pairs = sum(len(v) for v in jobs.values())
    print(f"finished v3 loops: {len(loops)} | transfer pairs: {n_pairs} across {len(jobs)} (task, destination) groups")
    if args.dry_run:
        return

    configure_huggingface()
    tests = {t: load_items(t, "test") for t in {k[0] for k in jobs}}
    seeds = {t: pathlib.Path(f"prompts/v2/{t}.txt").read_text(encoding="utf-8") for t in tests}

    def run(key):
        task_name, dest = key
        task, items = TASKS[task_name], tests[task_name]
        seed_res = score_prompt(task, dest, seeds[task_name], items)
        rows, spent = [], seed_res["cost_usd"]
        for L in jobs[key]:
            res = score_prompt(task, dest, L["final_prompt"], items)
            spent += res["cost_usd"]
            rows.append({"task": task_name, "source_target": L["target"], "condition": L["condition"], "run": L["run"],
                         "dest_target": dest, "final_sha": prompt_sha(L["final_prompt"]),
                         "final_acc": res["accuracy"], "seed_acc": seed_res["accuracy"],
                         "gain": res["accuracy"] - seed_res["accuracy"], "n": res["n"]})
        print(f"{task_name} -> {dest}: {len(rows)} prompts scored, new cost ${spent:.3f}", flush=True)
        return rows

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        rows = [r for group in ex.map(run, jobs) for r in group]
    out = pathlib.Path(args.root) / "transfer.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"-> {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
