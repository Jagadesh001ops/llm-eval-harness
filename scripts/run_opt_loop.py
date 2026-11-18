"""Run one prompt-optimization loop (design doc §5) and write its full log.

    python scripts/run_opt_loop.py --task ag_news --target gemma3-12b --condition self --run 1

Conditions: strong-closed (Claude Opus 5.5, effort low), strong-open (Mistral Large 3), mid (Claude Haiku 5.5),
self (target rewrites its own prompt), control (target, no failures/scores), larger (same-family larger model).
Output: results/opt/<task>/<target>/<condition>/run<k>/log.json. Per-item model outputs are cached under
results/opt/cache/ and shared between conditions.
"""

import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from harness.bedrock import ModelUnavailable  # noqa: E402
from harness.intent import (LENGTH_CAP_RATIO, build_leak_index, check_candidate, lexical_novelty,  # noqa: E402
                            load_task_card)
from harness.net import configure_huggingface  # noqa: E402
from harness.optimizer import build_request, make_optimizer  # noqa: E402
from harness.scorer import prompt_sha, score_prompt  # noqa: E402
from harness.tasks import TASKS, load_items  # noqa: E402

LARGER = {"gemma3-4b": "gemma3-27b", "gemma3-12b": "gemma3-27b", "ministral-3b": "ministral-14b",
          "ministral-8b": "ministral-14b"}
SPLIT_SEED = 20261118
N_FAILURES = 10
HISTORY_KEEP = 8
PROTOCOL = "v2"  # v2: min 2 rounds, patience 2, diagnosis step (pilot round 1 used v1)
MIN_ROUNDS = 2
PATIENCE = 2


def optimizer_for(condition, target):
    return {
        "strong-closed": ("claude-opus-5.5", "low", True),
        "strong-open": ("mistral-large-3", None, True),
        "mid": ("claude-haiku-5.5", None, True),
        "self": (target, None, True),
        "control": (target, None, False),
        "larger": (LARGER.get(target), None, True),
    }[condition]


def sample_failures(task, target, template, pool, run, rnd):
    """Failures for this round, drawn from the pool in a fixed order per (run, round) so every condition starting
    from the same prompt sees the same examples. Scores pool items in chunks of 40 until 10 failures (max 120)."""
    order = np.random.default_rng([SPLIT_SEED, run, rnd]).permutation(len(pool))
    fails, spent = [], 0.0
    for start in range(0, 120, 40):
        chunk = [pool[i] for i in order[start:start + 40]]
        res = score_prompt(task, target, template, chunk)
        spent += res["cost_usd"]
        for it in chunk:
            row = res["rows"][it["id"]]
            if not row["correct"] and len(fails) < N_FAILURES:
                fails.append({"id": it["id"], "input": task.show_input(it), "output": row["output"],
                              "gold": task.show_gold(it)})
        if len(fails) >= N_FAILURES:
            break
    return fails, spent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--target", required=True)
    ap.add_argument("--condition", required=True,
                    choices=["strong-closed", "strong-open", "mid", "self", "control", "larger"])
    ap.add_argument("--run", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--dev-n", type=int, default=None, help="use only the first N dev items (smoke tests)")
    ap.add_argument("--skip-test", action="store_true", help="skip the final test-set evaluation (smoke tests)")
    ap.add_argument("--out-root", default="results/opt")
    ap.add_argument("--force", action="store_true", help="overwrite an existing log")
    ap.add_argument("--intent-judge", action="store_true",
                    help="layer-2 intent judge rejects rewrites it says change the intent (use only once validated)")
    args = ap.parse_args()

    configure_huggingface()
    task = TASKS[args.task]
    card = load_task_card(args.task)
    seed = pathlib.Path(f"prompts/v2/{args.task}.txt").read_text(encoding="utf-8")
    oracle_path = pathlib.Path(f"prompts/reference/{args.task}_oracle.txt")
    opt_name, effort, feedback = optimizer_for(args.condition, args.target)
    if opt_name is None:
        raise SystemExit(f"condition 'larger' is not defined for target {args.target}")

    out = pathlib.Path(args.out_root) / args.task / args.target / args.condition / f"run{args.run}" / "log.json"
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists (use --force to overwrite)")
    out.parent.mkdir(parents=True, exist_ok=True)

    pool, dev, test = (load_items(args.task, s) for s in ("pool", "dev", "test"))
    if args.dev_n:
        dev = dev[: args.dev_n]
    leak_index = build_leak_index([task.show_input(it) for it in pool + dev + test])
    max_chars = int(LENGTH_CAP_RATIO * len(seed))
    optimizer = make_optimizer(opt_name, effort=effort)
    judge = None
    if args.intent_judge:
        from harness.judge import IntentJudge
        judge = IntentJudge()

    log = {"protocol": PROTOCOL, "min_rounds": MIN_ROUNDS, "patience": PATIENCE, "task": args.task, "target": args.target, "condition": args.condition, "run": args.run,
           "optimizer": opt_name, "effort": effort, "feedback": feedback, "rounds_max": args.rounds,
           "dev_n": len(dev), "seed_prompt": seed, "seed_sha": prompt_sha(seed), "rounds": [],
           "intent_judge": bool(judge), "cost_usd": {"target_scoring": 0.0, "optimizer": 0.0, "judge": 0.0}, "status": "running",
           "started": time.strftime("%Y-%m-%dT%H:%M:%S")}

    def save():
        out.write_text(json.dumps(log, indent=1), encoding="utf-8")

    try:
        s = score_prompt(task, args.target, seed, dev)
        log["cost_usd"]["target_scoring"] += s["cost_usd"]
        current, current_dev = seed, s["accuracy"]
        log["seed_dev_accuracy"] = current_dev
        history = [(seed, current_dev)]
        streak = 0
        print(f"seed dev acc {current_dev:.3f} (n={len(dev)})")

        for rnd in range(1, args.rounds + 1):
            fails = []
            if feedback:
                fails, spent = sample_failures(task, args.target, current, pool, args.run, rnd)
                log["cost_usd"]["target_scoring"] += spent
            shown_history = sorted(history, key=lambda h: h[1])[-HISTORY_KEEP:] if feedback else []
            request = build_request(card, current, shown_history, fails, max_chars, feedback=feedback)
            resp = optimizer.propose(request)
            log["cost_usd"]["optimizer"] += resp["cost_usd"]

            cands = []
            for c in resp["candidates"]:
                viol = check_candidate(c, card, seed, leak_index)
                entry = {"prompt": c, "sha": prompt_sha(c), "violations": viol,
                         "lexical_novelty": lexical_novelty(c, card, seed), "chars": len(c)}
                if not viol and judge is not None:
                    for _attempt in range(2):  # one retry if the verdict could not be parsed
                        jv = judge.judge(card, seed, c)
                        log["cost_usd"]["judge"] += jv["cost_usd"]
                        if jv["verdict"] != "error":
                            break
                    entry["judge"] = {"verdict": jv["verdict"], "reason": jv["reason"]}
                    if jv["verdict"] != "keeps":  # 'changes' or a twice-unparseable verdict both reject
                        viol.append(("intent_judge", jv["reason"] or jv["verdict"]))
                if not viol:
                    sc = score_prompt(task, args.target, c, dev)
                    log["cost_usd"]["target_scoring"] += sc["cost_usd"]
                    entry["dev_accuracy"] = sc["accuracy"]
                    history.append((c, sc["accuracy"]))
                elif [v[0] for v in viol] == ["intent_judge"]:
                    # Shadow score: rejected ONLY by the (unvalidated) judge. Scored on dev so we can later see whether
                    # the judge blocked a would-be winner, but never selected and never shown in the history.
                    sc = score_prompt(task, args.target, c, dev)
                    log["cost_usd"]["target_scoring"] += sc["cost_usd"]
                    entry["shadow_dev_accuracy"] = sc["accuracy"]
                cands.append(entry)

            valid = [e for e in cands if "dev_accuracy" in e]
            best = max(valid, key=lambda e: e["dev_accuracy"], default=None)
            shadow = [e for e in cands if "shadow_dev_accuracy" in e]
            shadow_best = max((e["shadow_dev_accuracy"] for e in shadow), default=None)
            judge_blocked_winner = shadow_best is not None and shadow_best > max(
                current_dev, best["dev_accuracy"] if best else float("-inf"))
            improved = best is not None and best["dev_accuracy"] > current_dev
            if improved:
                current, current_dev = best["prompt"], best["dev_accuracy"]
            log["rounds"].append({
                "round": rnd, "failures_shown": [f["id"] for f in fails], "optimizer_stop_reason": resp["stop_reason"],
                "optimizer_tokens": [resp["input_tokens"], resp["output_tokens"]],
                "n_parsed": len(resp["candidates"]), "n_valid": len(valid), "candidates": cands,
                "n_judge_rejected": len(shadow), "judge_blocked_winner": judge_blocked_winner,
                "improved": improved, "current_dev_accuracy": current_dev, "optimizer_analysis": resp.get("analysis"),
                "optimizer_raw": resp["raw"]})
            save()
            streak = 0 if improved else streak + 1
            stop = rnd >= MIN_ROUNDS and streak >= PATIENCE
            print(f"round {rnd}: parsed {len(resp['candidates'])}, valid {len(valid)}, "
                  f"best {best['dev_accuracy'] if best else float('nan'):.3f}, current {current_dev:.3f}"
                  f"{' (improved)' if improved else ' (no improvement)'}{' -> stop' if stop else ''}")
            if stop:
                break

        log["final_prompt"], log["final_sha"], log["final_dev_accuracy"] = current, prompt_sha(current), current_dev
        if not args.skip_test:
            for name, tmpl in [("final", current), ("seed", seed)] + \
                              ([("oracle", oracle_path.read_text(encoding="utf-8"))] if oracle_path.exists() else []):
                sc = score_prompt(task, args.target, tmpl, test)
                log["cost_usd"]["target_scoring"] += sc["cost_usd"]
                log[f"{name}_test_accuracy"] = sc["accuracy"]
                log[f"{name}_test_sha"] = prompt_sha(tmpl)
            print(f"test: final {log['final_test_accuracy']:.3f} | seed {log['seed_test_accuracy']:.3f}"
                  + (f" | oracle {log['oracle_test_accuracy']:.3f}" if "oracle_test_accuracy" in log else ""))
        log["status"] = "done"
    except ModelUnavailable as e:
        log["status"], log["error"] = "model_unavailable", str(e)
        print(f"STOPPED: {e}")
    finally:
        log["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        save()
        c = log["cost_usd"]
        print(f"cost: target ${c['target_scoring']:.4f} + optimizer ${c['optimizer']:.4f} "
              f"+ judge ${c.get('judge', 0.0):.4f} -> {out}")


if __name__ == "__main__":
    main()
