"""Analyse the prompt-optimization grid (design doc §7-8). No model calls: reads loop logs and the per-item cache.

    python analysis/analyze_grid.py

Inputs : results/opt/<task>/<target>/<condition>/run<k>/log.json  (protocol v3, status done)
         results/opt/cache/...                                       (per-item test outputs)
         results/opt/transfer.csv                                    (optional; from scripts/run_transfer.py)
Outputs: results/opt/summary.csv       one row per (task, target, condition)
         results/opt/hypotheses.csv    one row per hypothesis test
         results/opt/summary.md        readable tables

Uncertainty: two-level bootstrap (resample runs, then test items; the same item indices for every prompt being
compared), B = 10,000, seed 20261118, 95% percentile intervals. A difference between two conditions resamples each
condition's runs independently and the shared test items jointly.
"""

import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from harness.net import configure_huggingface  # noqa: E402
from harness.parsers import contains_answer, extract_trivia_answer  # noqa: E402
from harness.scorer import cache_path  # noqa: E402
from harness.stats import DEFAULT_B, DEFAULT_SEED  # noqa: E402
from harness.tasks import TASKS, load_items  # noqa: E402

ROOT = pathlib.Path("results/opt")
CONDS = ["strong-closed", "strong-open", "mid", "larger", "self", "control"]
FAMILY = {"gemma3-4b": "gemma", "gemma3-12b": "gemma", "gemma3-27b": "gemma",
          "ministral-3b": "ministral", "ministral-8b": "ministral", "ministral-14b": "ministral"}
SIZE_ORDER = ["gemma3-4b", "gemma3-12b", "gemma3-27b", "ministral-3b", "ministral-8b", "ministral-14b"]
EQUIV_MARGIN = 0.02  # H8: |strong-open - strong-closed| within 2 points


# ── data ──────────────────────────────────────────────────────────────────────

def load_logs():
    logs = []
    for p in sorted(ROOT.rglob("log.json")):
        L = json.loads(p.read_text(encoding="utf-8"))
        if L.get("status") == "done" and L.get("protocol") == "v3":
            logs.append(L)
    return logs


_cache = {}


def item_scores(task_name, target, template, ids, field="correct"):
    key = (task_name, target, template)
    if key not in _cache:
        path = cache_path(TASKS[task_name], target, template)
        _cache[key] = {json.loads(l)["id"]: json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l}
    rows = _cache[key]
    missing = [i for i in ids if i not in rows]
    if missing:
        raise RuntimeError(f"{len(missing)} test items missing from cache for {task_name}/{target}")
    return np.array([float(rows[i][field]) for i in ids]) if field != "raw" else [rows[i]["output"] for i in ids]


# ── bootstrap ─────────────────────────────────────────────────────────────────

def _rng():
    return np.random.default_rng(DEFAULT_SEED)


def boot_mean_diff(D):
    """D: runs x items matrix of per-item differences (final - seed). Two-level bootstrap of the mean."""
    D = np.atleast_2d(D)
    R, N = D.shape
    rng = _rng()
    ri = rng.integers(0, R, size=(DEFAULT_B, R))
    ii = rng.integers(0, N, size=(DEFAULT_B, N))
    boots = np.array([D[:, ii[b]].mean(axis=1)[ri[b]].mean() for b in range(DEFAULT_B)])
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return float(D.mean()), float(lo), float(hi), boots


def boot_cond_diff(A, B):
    """A, B: runs x items matrices (per-item gains) for two conditions on the same items. Mean(A) - mean(B)."""
    A, B = np.atleast_2d(A), np.atleast_2d(B)
    N = A.shape[1]
    rng = _rng()
    ra = rng.integers(0, A.shape[0], size=(DEFAULT_B, A.shape[0]))
    rb = rng.integers(0, B.shape[0], size=(DEFAULT_B, B.shape[0]))
    ii = rng.integers(0, N, size=(DEFAULT_B, N))
    boots = np.array([A[:, ii[b]].mean(axis=1)[ra[b]].mean() - B[:, ii[b]].mean(axis=1)[rb[b]].mean()
                      for b in range(DEFAULT_B)])
    lo, hi = np.quantile(boots, [0.025, 0.975])
    p = float(min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean())))
    return float(A.mean() - B.mean()), float(lo), float(hi), p


def holm(ps):
    ps = np.asarray(ps, dtype=float)
    order, adj, run = np.argsort(ps), np.empty(len(ps)), 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - k) * ps[i]))
        adj[i] = run
    return adj


# ── per-cell summary ──────────────────────────────────────────────────────────

def mentions_tech(text):
    t = (text or "").lower()
    return any(w in t for w in ["techn", "tech ", "tech,", "tech/", "software", "computing"])


def summarise(logs):
    tests = {t: load_items(t, "test") for t in {L["task"] for L in logs}}
    ids = {t: [it["id"] for it in items] for t, items in tests.items()}
    gains, rows = {}, []
    groups = {}
    for L in logs:
        groups.setdefault((L["task"], L["target"], L["condition"]), []).append(L)
    for (task, target, cond), Ls in sorted(groups.items()):
        Ls = sorted(Ls, key=lambda x: x["run"])
        seed = item_scores(task, target, Ls[0]["seed_prompt"], ids[task])
        finals = np.vstack([item_scores(task, target, L["final_prompt"], ids[task]) for L in Ls])
        D = finals - seed
        gains[(task, target, cond)] = D
        g, lo, hi, _ = boot_mean_diff(D)
        cands = [c for L in Ls for r in L["rounds"] for c in r["candidates"]]
        n = max(len(cands), 1)
        rule_rej = sum(any(v[0] != "intent_judge" for v in c["violations"]) for c in cands)
        judge_rej = sum(any(v[0] == "intent_judge" for v in c["violations"]) for c in cands)
        dev_gain = np.mean([L["final_dev_accuracy"] - L["seed_dev_accuracy"] for L in Ls])
        oracle = Ls[0].get("oracle_test_accuracy")
        row = {
            "task": task, "target": target, "family": FAMILY[target], "condition": cond, "runs": len(Ls),
            "optimizer": Ls[0]["optimizer"], "seed_test": float(seed.mean()), "final_test": float(finals.mean()),
            "gain": g, "gain_lo": lo, "gain_hi": hi, "dev_gain": float(dev_gain), "dev_test_gap": float(dev_gain - g),
            "oracle_test": oracle, "rounds_mean": float(np.mean([len(L["rounds"]) for L in Ls])),
            "candidates": len(cands), "rule_reject_rate": rule_rej / n, "judge_reject_rate": judge_rej / n,
            "judge_blocked_winner_rounds": int(sum(r.get("judge_blocked_winner", False) for L in Ls for r in L["rounds"])),
            "lexical_novelty_mean": float(np.mean([c["lexical_novelty"] for c in cands])) if cands else np.nan,
            "final_prompt_chars_mean": float(np.mean([len(L["final_prompt"]) for L in Ls])),
            "prompt_changed_runs": int(sum(L["final_prompt"] != L["seed_prompt"] for L in Ls)),
            "cost_usd": float(sum(sum(L["cost_usd"].values()) for L in Ls)),
        }
        if task == "ag_news":
            row["diagnosed_tech_runs"] = int(sum(any(mentions_tech(r.get("optimizer_analysis")) for r in L["rounds"])
                                                 for L in Ls))
            row["fixed_tech_runs"] = int(sum(mentions_tech(L["final_prompt"]) for L in Ls))
        if task == "triviaqa":  # control: split the gain into exact match vs contains-answer
            items = tests[task]
            def ca(template):
                outs = item_scores(task, target, template, ids[task], field="raw")
                return np.array([float(contains_answer(extract_trivia_answer(o), it["aliases"])) for o, it in zip(outs, items)])
            ca_seed = ca(Ls[0]["seed_prompt"])
            Dca = np.vstack([ca(L["final_prompt"]) for L in Ls]) - ca_seed
            gc, lc, hc, _ = boot_mean_diff(Dca)
            row.update({"contains_gain": gc, "contains_gain_lo": lc, "contains_gain_hi": hc})
        rows.append(row)
    return pd.DataFrame(rows), gains


# ── hypotheses ────────────────────────────────────────────────────────────────

def verdict_greater(lo):
    return "supported" if lo > 0 else "not supported"


def hypotheses(S, gains):
    H = []
    main = S[S.task.isin(["ag_news", "squad_v2"])]
    # H1: strong-closed gain > self gain, for every target (Holm across targets x tasks)
    h1 = []
    for (task, target), _ in main.groupby(["task", "target"]):
        a, b = gains.get((task, target, "strong-closed")), gains.get((task, target, "self"))
        if a is None or b is None:
            continue
        d, lo, hi, p = boot_cond_diff(a, b)
        h1.append({"hypothesis": "H1", "task": task, "target": target,
                   "comparison": "strong-closed minus self (test gain)", "estimate": d, "lo": lo, "hi": hi, "p": p})
    for r, ph in zip(h1, holm([r["p"] for r in h1]) if h1 else []):
        r["p_holm"] = float(ph)
        r["verdict"] = "supported" if r["lo"] > 0 and ph < 0.05 else ("contradicted" if r["hi"] < 0 else "not supported")
    H += h1
    # H2: dev-test gap larger for self than for strong-closed (mean over runs; descriptive)
    for (task, target), g in main.groupby(["task", "target"]):
        s, c = g[g.condition == "self"], g[g.condition == "strong-closed"]
        if len(s) and len(c):
            d = float(s.dev_test_gap.iloc[0] - c.dev_test_gap.iloc[0])
            H.append({"hypothesis": "H2", "task": task, "target": target,
                      "comparison": "dev-test gap: self minus strong-closed", "estimate": d, "lo": np.nan, "hi": np.nan,
                      "verdict": "consistent" if d > 0 else "inconsistent"})
    # H3: self minus control gain, by target size within family
    for (task, target), _ in main.groupby(["task", "target"]):
        a, b = gains.get((task, target, "self")), gains.get((task, target, "control"))
        if a is None or b is None:
            continue
        d, lo, hi, p = boot_cond_diff(a, b)
        H.append({"hypothesis": "H3", "task": task, "target": target, "comparison": "self minus control (test gain)",
                  "estimate": d, "lo": lo, "hi": hi, "p": p, "verdict": verdict_greater(lo)})
    # H5: larger recovers a substantial part of the strong-closed gain (ratio of mean gains; descriptive + diff CI)
    for (task, target), g in main.groupby(["task", "target"]):
        a, b = gains.get((task, target, "larger")), gains.get((task, target, "strong-closed"))
        if a is None or b is None:
            continue
        d, lo, hi, p = boot_cond_diff(a, b)
        sc = float(b.mean())
        H.append({"hypothesis": "H5", "task": task, "target": target, "comparison": "larger minus strong-closed (test gain)",
                  "estimate": d, "lo": lo, "hi": hi, "p": p,
                  "share_of_strong_gain": float(a.mean() / sc) if abs(sc) > 1e-9 else np.nan,
                  "verdict": "supported" if lo > -0.01 else "not supported"})
    # H6: violation (rule + judge) rate by optimizer condition, pooled
    viol = main.assign(any_reject=main.rule_reject_rate + main.judge_reject_rate).groupby("condition").any_reject.mean()
    for cond, v in viol.items():
        H.append({"hypothesis": "H6", "task": "all", "target": "all", "comparison": f"mean rejection rate: {cond}",
                  "estimate": float(v), "lo": np.nan, "hi": np.nan, "verdict": "descriptive"})
    # H7: TriviaQA gains concentrated in exact match, not contains-answer
    for _, r in S[S.task == "triviaqa"].iterrows():
        H.append({"hypothesis": "H7", "task": "triviaqa", "target": r.target, "comparison": f"{r.condition}: EM gain vs contains gain",
                  "estimate": r.gain, "lo": r.gain_lo, "hi": r.gain_hi, "contains_gain": r.get("contains_gain"),
                  "verdict": "descriptive"})
    # H8: strong-open vs strong-closed within +/-2 points (equivalence: whole CI inside the margin)
    for (task, target), _ in main.groupby(["task", "target"]):
        a, b = gains.get((task, target, "strong-open")), gains.get((task, target, "strong-closed"))
        if a is None or b is None:
            continue
        d, lo, hi, p = boot_cond_diff(a, b)
        v = "supported" if (lo > -EQUIV_MARGIN and hi < EQUIV_MARGIN) else (
            "contradicted" if (hi < -EQUIV_MARGIN or lo > EQUIV_MARGIN) else "inconclusive")
        H.append({"hypothesis": "H8", "task": task, "target": target, "comparison": "strong-open minus strong-closed",
                  "estimate": d, "lo": lo, "hi": hi, "p": p, "verdict": v})
    # H4: transfer (if available): mean transfer gain by condition
    tpath = ROOT / "transfer.csv"
    if tpath.exists():
        T = pd.read_csv(tpath)
        for (task, cond), g in T.groupby(["task", "condition"]):
            H.append({"hypothesis": "H4", "task": task, "target": "all (transfer)", "comparison": f"mean transfer gain: {cond}",
                      "estimate": float(g.gain.mean()), "lo": np.nan, "hi": np.nan, "n_pairs": len(g), "verdict": "descriptive"})
    return pd.DataFrame(H)


# ── report ────────────────────────────────────────────────────────────────────

def pct(x):
    return "n/a" if pd.isna(x) else f"{x*100:+.1f}"


def write_md(S, H):
    out = ["# Prompt-optimization grid: results", "",
           "Generated by `python analysis/analyze_grid.py` from protocol-v3 loop logs and the per-item cache. "
           f"Gains are test-set points (final prompt minus seed prompt, same items), mean over runs, with 95% two-level "
           f"bootstrap intervals (runs, then items; B={DEFAULT_B}, seed {DEFAULT_SEED}). The intent judge was on in every "
           "loop (amendment, design doc §6.3).", ""]
    for task in ["ag_news", "squad_v2", "triviaqa"]:
        sub = S[S.task == task]
        if sub.empty:
            continue
        metric = {"ag_news": "accuracy", "squad_v2": "F1", "triviaqa": "exact match"}[task]
        out += [f"## {task} ({metric}, n=500 test items)", "",
                "| Target | Seed | " + " | ".join(CONDS) + " | Oracle |", "|---|---|" + "---|" * len(CONDS) + "---|"]
        for target in [t for t in SIZE_ORDER if t in set(sub.target)]:
            g = sub[sub.target == target].set_index("condition")
            cells = []
            for c in CONDS:
                if c in g.index:
                    r = g.loc[c]
                    cells.append(f"{pct(r.gain)} [{pct(r.gain_lo)}, {pct(r.gain_hi)}]")
                else:
                    cells.append("-")
            seed = g.seed_test.iloc[0]
            orc = g.oracle_test.dropna().iloc[0] if g.oracle_test.notna().any() else np.nan
            out.append(f"| {target} | {seed*100:.1f} | " + " | ".join(cells) + f" | {'n/a' if pd.isna(orc) else f'{orc*100:.1f}'} |")
        out.append("")
    out += ["## Process measures (pooled over runs)", "",
            "| Task | Target | Condition | Runs changed prompt | Dev gain | Dev-test gap | Rule rejects | Judge rejects | "
            "Judge blocked a winner (rounds) | Novelty | Final chars | AG: diagnosed / fixed tech |", "|" + "---|" * 12]
    for _, r in S.iterrows():
        tech = f"{int(r.diagnosed_tech_runs)}/{int(r.fixed_tech_runs)} of {r.runs}" if r.task == "ag_news" else ""
        out.append(f"| {r.task} | {r.target} | {r.condition} | {r.prompt_changed_runs}/{r.runs} | {pct(r.dev_gain)} | "
                   f"{pct(r.dev_test_gap)} | {r.rule_reject_rate:.0%} | {r.judge_reject_rate:.0%} | "
                   f"{r.judge_blocked_winner_rounds} | {r.lexical_novelty_mean:.2f} | {r.final_prompt_chars_mean:.0f} | {tech} |")
    out += ["", "## Hypotheses", "", "| Hypothesis | Task | Target | Comparison | Estimate [95% CI] | Verdict |",
            "|---|---|---|---|---|---|"]
    for _, h in H.iterrows():
        ci = "" if pd.isna(h.get("lo")) else f" [{pct(h.lo)}, {pct(h.hi)}]"
        est = f"{h.estimate:.0%}" if h.hypothesis == "H6" else f"{pct(h.estimate)}{ci}"
        out.append(f"| {h.hypothesis} | {h.task} | {h.target} | {h.comparison} | {est} | {h.verdict} |")
    out += ["", f"Total recorded cost of these loops: ${S.cost_usd.sum():.2f}.", ""]
    (ROOT / "summary.md").write_text("\n".join(out), encoding="utf-8")


def main():
    configure_huggingface()
    logs = load_logs()
    S, gains = summarise(logs)
    H = hypotheses(S, gains)
    S.to_csv(ROOT / "summary.csv", index=False)
    H.to_csv(ROOT / "hypotheses.csv", index=False)
    write_md(S, H)
    print(f"{len(logs)} loops -> {len(S)} cells, {len(H)} hypothesis rows; wrote results/opt/summary.{{csv,md}}, hypotheses.csv")


if __name__ == "__main__":
    main()
