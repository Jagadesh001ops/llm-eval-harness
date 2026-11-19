"""Score the 'no-judge picks' on the test split (see harness/sensitivity.py for what a pick is and its limits).

For every finished protocol-v3 loop, find the judge-rejected candidate with the best dev score if it beats the
loop's final dev score, and score that prompt on the 500 test items of the same target. Grouped by (task, target)
so no two workers share a cache file. Output: results/opt/nojudge_picks.csv (one row per loop with a pick).

    python scripts/run_sensitivity.py [--dry-run]
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
from harness.sensitivity import nojudge_pick  # noqa: E402
from harness.tasks import TASKS, load_items  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="results/opt")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    jobs, n_loops = {}, 0
    for p in sorted(pathlib.Path(args.root).rglob("log.json")):
        L = json.loads(p.read_text(encoding="utf-8"))
        if L.get("status") != "done" or L.get("protocol") != "v3":
            continue
        n_loops += 1
        pick = nojudge_pick(L)
        if pick:
            jobs.setdefault((L["task"], L["target"]), []).append((L, pick))
    n_picks = sum(len(v) for v in jobs.values())
    print(f"v3 loops: {n_loops} | loops whose no-judge pick differs from the judge-on final prompt: {n_picks}")
    if args.dry_run:
        return

    configure_huggingface()
    tests = {t: load_items(t, "test") for t in {k[0] for k in jobs}}

    def run(key):
        task_name, target = key
        rows = []
        for L, (prompt, dev, _) in jobs[key]:
            res = score_prompt(TASKS[task_name], target, prompt, tests[task_name])
            rows.append({"task": task_name, "target": target, "condition": L["condition"], "run": L["run"],
                         "pick_sha": prompt_sha(prompt), "pick_dev": dev, "final_dev": L["final_dev_accuracy"],
                         "pick_test": res["accuracy"], "final_test": L["final_test_accuracy"]})
        print(f"{task_name}/{target}: {len(rows)} picks scored", flush=True)
        return rows

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        rows = [r for grp in ex.map(run, jobs) for r in grp]
    out = pathlib.Path(args.root) / "nojudge_picks.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"-> {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
