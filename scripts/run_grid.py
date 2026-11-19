"""Run the full prompt-optimization grid (design doc §9) with resume and safe parallelism.

Loops are grouped by (task, target). Each group runs sequentially in one worker, and different groups run in
parallel, so no two processes ever append to the same per-item cache file (every loop's scoring goes to its own
target's cache). Finished loops (log status "done") are skipped, so the grid can be stopped and restarted.

    python scripts/run_grid.py --dry-run                       # print the plan and counts
    python scripts/run_grid.py --tier core --workers 4 --intent-judge
    python scripts/run_grid.py --only-task ag_news --only-target gemma3-12b --max-loops 2   # staged rollout

The intent judge should only be enabled once validated (scripts/validate_judge.py); every log records whether
it was on.
"""

import argparse
import json
import pathlib
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONDITIONS = ["strong-closed", "strong-open", "mid", "self", "control"]
LARGER_FOR = {"gemma3-4b", "gemma3-12b", "ministral-3b", "ministral-8b"}
GEMMA = ["gemma3-4b", "gemma3-12b", "gemma3-27b"]
MINISTRAL = ["ministral-3b", "ministral-8b", "ministral-14b"]
TIERS = {  # tier -> (tasks, targets, runs)   (design doc §9)
    "core": (["ag_news", "squad_v2"], GEMMA, 3),
    "replication": (["ag_news"], MINISTRAL, 2),
    "control": (["triviaqa"], GEMMA, 1),
}


def plan(tiers, only_task=None, only_target=None):
    loops = []
    for tier in tiers:
        tasks, targets, runs = TIERS[tier]
        for task in tasks:
            for target in targets:
                conds = CONDITIONS + (["larger"] if target in LARGER_FOR else [])
                for run in range(1, runs + 1):
                    for cond in conds:
                        if (only_task and task != only_task) or (only_target and target != only_target):
                            continue
                        loops.append({"tier": tier, "task": task, "target": target, "condition": cond, "run": run})
    return loops


def log_path(out_root, lp):
    return ROOT / out_root / lp["task"] / lp["target"] / lp["condition"] / f"run{lp['run']}" / "log.json"


def status(out_root, lp):
    p = log_path(out_root, lp)
    if not p.exists():
        return "todo"
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("status", "unknown")
    except json.JSONDecodeError:
        return "corrupt"


def run_group(group, args):
    task, target = group[0]["task"], group[0]["target"]
    logdir = ROOT / args.out_root / "_logs"
    logdir.mkdir(parents=True, exist_ok=True)
    done = 0
    with (logdir / f"{task}__{target}.log").open("a", encoding="utf-8") as fh:
        for lp in group:
            st = status(args.out_root, lp)
            if st == "done":
                continue
            cmd = [sys.executable, str(ROOT / "scripts" / "run_opt_loop.py"), "--task", task, "--target", target,
                   "--condition", lp["condition"], "--run", str(lp["run"]), "--out-root", args.out_root]
            if st != "todo":
                cmd.append("--force")  # rerun an interrupted or failed loop from scratch (cached scoring is reused)
            if args.intent_judge:
                cmd.append("--intent-judge")
            fh.write(f"\n=== {time.strftime('%H:%M:%S')} {lp} ===\n")
            fh.flush()
            rc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL).returncode
            done += 1
            final = status(args.out_root, lp)
            print(f"[{time.strftime('%H:%M:%S')}] {task}/{target}/{lp['condition']}/run{lp['run']}: {final} (rc={rc})",
                  flush=True)
    return done


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", nargs="+", default=list(TIERS), choices=list(TIERS))
    ap.add_argument("--only-task")
    ap.add_argument("--only-target")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-loops", type=int, default=None, help="run at most this many unfinished loops (staging)")
    ap.add_argument("--intent-judge", action="store_true")
    ap.add_argument("--out-root", default="results/opt")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    loops = plan(args.tier, args.only_task, args.only_target)
    pending = [lp for lp in loops if status(args.out_root, lp) != "done"]
    if args.max_loops is not None:
        pending = pending[: args.max_loops]
    by_tier = {t: sum(lp["tier"] == t for lp in loops) for t in args.tier}
    print(f"planned loops: {len(loops)} {by_tier} | pending this invocation: {len(pending)} | "
          f"intent judge: {'ON' if args.intent_judge else 'off'}")
    if args.dry_run:
        for lp in pending[:12]:
            print("  ", lp)
        if len(pending) > 12:
            print(f"   ... and {len(pending) - 12} more")
        return

    groups = {}
    for lp in pending:
        groups.setdefault((lp["task"], lp["target"]), []).append(lp)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        n = sum(ex.map(lambda g: run_group(g, args), groups.values()))
    print(f"finished {n} loop(s)")


if __name__ == "__main__":
    main()
