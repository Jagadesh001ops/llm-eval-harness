"""Re-score stored results with the strict parsers and attach 95% bootstrap CIs to every metric.

Inputs (committed, per-item):  results/all_models_all_prompts.csv  (SST-2)
                               results/ag_news_results.csv          (AG News)
                               results/trivia_results.csv           (TriviaQA)
                               results/summarization_results.csv    (CNN/DailyMail)
Outputs:  results/summary_ci.csv          one row per task x model x prompt x metric x scoring
          results/paired_comparisons.csv  paired differences (models, prompts, legacy-vs-strict)
          results/summary_ci.md           readable tables
          results/rescored/*.csv          per-item rows with strict labels added

No model calls. Run:  python analysis/rescore_and_ci.py
"""

import itertools
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from harness.parsers import AMBIGUOUS, INVALID, NEUTRAL, strict_ag_label, strict_sentiment_label  # noqa: E402
from harness.stats import (DEFAULT_B, DEFAULT_SEED, bootstrap_mean_ci, bootstrap_ratio_ci,  # noqa: E402
                           holm_adjust, mcnemar_exact, paired_bootstrap_diff)

RESULTS = pathlib.Path("results")
PROMPT_ORDER = ["zero_shot", "few_shot", "chain_of_thought"]


def _row(task, model, prompt, metric, scoring, ci, note=""):
    return {"task": task, "model": model, "prompt_variant": prompt, "metric": metric, "scoring": scoring,
            "point": ci["point"], "ci_lo": ci["lo"], "ci_hi": ci["hi"],
            "half_width": (ci["hi"] - ci["lo"]) / 2, "n": ci["n"], "note": note}


def load_classification(path, task, strict_fn, class_labels):
    d = pd.read_csv(path)
    d["example_id"] = d.groupby(["model", "prompt_variant"]).cumcount()
    d["raw_output"] = d["raw_output"].fillna("")
    d["legacy_label"] = d["predicted_label"]
    d["legacy_correct"] = d["correct"].astype(int)
    d["strict_label"] = [strict_fn(o, p) for o, p in zip(d.raw_output, d.prompt_variant)]
    d["strict_correct"] = (d.strict_label == d.true_label).astype(int)
    d["strict_valid"] = d.strict_label.isin(class_labels).astype(int)
    d["task"] = task
    return d


def classification_rows(d, task, has_neutral):
    rows = []
    for (m, p), g in d.groupby(["model", "prompt_variant"]):
        rows.append(_row(task, m, p, "accuracy", "legacy", bootstrap_mean_ci(g.legacy_correct)))
        rows.append(_row(task, m, p, "accuracy", "strict", bootstrap_mean_ci(g.strict_correct)))
        if has_neutral:
            rows.append(_row(task, m, p, "neutral_rate", "strict", bootstrap_mean_ci(g.strict_label == NEUTRAL)))
        rows.append(_row(task, m, p, "ambiguous_rate", "strict", bootstrap_mean_ci(g.strict_label == AMBIGUOUS)))
        rows.append(_row(task, m, p, "invalid_rate", "strict", bootstrap_mean_ci(g.strict_label == INVALID)))
        rows.append(_row(task, m, p, "selective_accuracy", "strict",
                         bootstrap_ratio_ci(g.strict_correct * g.strict_valid, g.strict_valid),
                         "accuracy among items with a valid class label; n = number of such items"))
    return rows


def paired_rows(d, task, score_col, metric, binary):
    """Model-vs-model within each prompt, prompt-vs-prompt within each model."""
    rows = []
    wide = d.pivot_table(index="example_id", columns=["model", "prompt_variant"], values=score_col)

    def add(kind, a_key, b_key, scope):
        a, b = wide[a_key], wide[b_key]
        ok = a.notna() & b.notna()
        diff = paired_bootstrap_diff(a[ok], b[ok])
        r = {"task": task, "metric": metric, "comparison": kind, "scope": scope,
             "a": "|".join(a_key), "b": "|".join(b_key), "a_mean": a[ok].mean(), "b_mean": b[ok].mean(),
             "diff": diff["point"], "ci_lo": diff["lo"], "ci_hi": diff["hi"], "n": diff["n"]}
        if binary:
            r["a_only"], r["b_only"], r["p_mcnemar"] = mcnemar_exact(a[ok], b[ok])
        rows.append(r)

    models = sorted(d.model.unique())
    prompts = [p for p in PROMPT_ORDER if p in set(d.prompt_variant)]
    for p in prompts:
        for ma, mb in itertools.combinations(models, 2):
            add("model_vs_model", (ma, p), (mb, p), p)
    for m in models:
        for pa, pb in itertools.combinations(prompts, 2):
            add("prompt_vs_prompt", (m, pa), (m, pb), m)
    return rows


def scoring_rows(d, task):
    """Strict minus legacy accuracy for the same outputs: isolates the effect of the parser alone."""
    rows = []
    for (m, p), g in d.groupby(["model", "prompt_variant"]):
        diff = paired_bootstrap_diff(g.strict_correct, g.legacy_correct)
        a_only, b_only, pval = mcnemar_exact(g.strict_correct, g.legacy_correct)
        rows.append({"task": task, "metric": "accuracy", "comparison": "strict_vs_legacy", "scope": f"{m}|{p}",
                     "a": "strict", "b": "legacy", "a_mean": g.strict_correct.mean(),
                     "b_mean": g.legacy_correct.mean(), "diff": diff["point"], "ci_lo": diff["lo"],
                     "ci_hi": diff["hi"], "n": diff["n"], "a_only": a_only, "b_only": b_only, "p_mcnemar": pval})
    return rows


def fmt(point, lo, hi, pct=True):
    if pd.isna(point):
        return "n/a"
    if pct:
        return f"{point*100:.1f} [{lo*100:.1f}, {hi*100:.1f}]"
    return f"{point:.3f} [{lo:.3f}, {hi:.3f}]"


def main():
    (RESULTS / "rescored").mkdir(exist_ok=True)
    summary, paired = [], []

    sst = load_classification(RESULTS / "all_models_all_prompts.csv", "sst2", strict_sentiment_label,
                              {"positive", "negative"})
    ag = load_classification(RESULTS / "ag_news_results.csv", "ag_news", strict_ag_label,
                             {"world", "sports", "business", "science"})
    for d, task, has_neutral in [(sst, "sst2", True), (ag, "ag_news", False)]:
        summary += classification_rows(d, task, has_neutral)
        paired += paired_rows(d, task, "strict_correct", "accuracy_strict", binary=True)
        paired += scoring_rows(d, task)
        d.to_csv(RESULTS / "rescored" / f"{task}_items.csv", index=False)

    tq = pd.read_csv(RESULTS / "trivia_results.csv")
    tq["example_id"] = tq.groupby(["model", "prompt_variant"]).cumcount()
    for (m, p), g in tq.groupby(["model", "prompt_variant"]):
        summary.append(_row("trivia", m, p, "exact_match", "original", bootstrap_mean_ci(g.exact_match)))
        summary.append(_row("trivia", m, p, "contains_answer", "original", bootstrap_mean_ci(g.contains_answer),
                            "lenient: any gold alias appears in the answer; favours verbose answers"))
    for col in ["exact_match", "contains_answer"]:
        paired += paired_rows(tq, "trivia", col, col, binary=True)

    sm = pd.read_csv(RESULTS / "summarization_results.csv")
    sm["example_id"] = sm["article_id"]
    sm["prompt_variant"] = "default"
    faith_note = ("KNOWN FLAW: NLI premise truncated to first 800 chars of a 2000-char input; "
                  "measures support by the article's opening, not faithfulness")
    for m, g in sm.groupby("model"):
        summary.append(_row("summarization", m, "default", "rouge_l", "original", bootstrap_mean_ci(g.rouge_l)))
        summary.append(_row("summarization", m, "default", "faithfulness", "original",
                            bootstrap_mean_ci(g.faithfulness), faith_note))
    for col in ["rouge_l", "faithfulness"]:
        paired += paired_rows(sm, "summarization", col, col, binary=False)

    summary = pd.DataFrame(summary)
    paired = pd.DataFrame(paired)
    # Holm correction within each (task, metric, comparison) family.
    paired["p_holm"] = np.nan
    for _, idx in paired[paired.get("p_mcnemar").notna()].groupby(["task", "metric", "comparison"]).groups.items():
        paired.loc[idx, "p_holm"] = holm_adjust(paired.loc[idx, "p_mcnemar"].values)

    summary.to_csv(RESULTS / "summary_ci.csv", index=False)
    paired.to_csv(RESULTS / "paired_comparisons.csv", index=False)
    write_markdown(summary, paired)
    print(f"Wrote {len(summary)} summary rows and {len(paired)} paired comparisons "
          f"(B={DEFAULT_B}, seed={DEFAULT_SEED}).")


def write_markdown(summary, paired):
    s = summary.set_index(["task", "model", "prompt_variant", "metric", "scoring"])

    def cell(task, m, p, metric, scoring, pct=True):
        key = (task, m, p, metric, scoring)
        if key not in s.index:
            return "n/a"
        r = s.loc[key]
        return fmt(r.point, r.ci_lo, r.ci_hi, pct)

    out = ["# Results with 95% confidence intervals", "",
           "Generated by `python analysis/rescore_and_ci.py` from the per-item CSVs in `results/`. "
           f"Values are point estimate [95% CI], item-level percentile bootstrap, B={DEFAULT_B}, seed={DEFAULT_SEED}. "
           "Method: `docs/statistics.md`. All runs are single runs at temperature 0 on the first N items of each split.", ""]

    for task, title, has_neutral in [("sst2", "SST-2 sentiment (n=100 per cell)", True),
                                     ("ag_news", "AG News topic (n=100 per cell)", False)]:
        out += [f"## {title}", "",
                "*Legacy* = the originally reported scoring. *Strict* = re-scored from the same raw outputs "
                "(see `harness/parsers.py`). Neutral = hedged with 'neutral'; ambiguous = named more than one "
                "class; invalid = no parseable label. All three count as wrong.", ""]
        hdr = "| Model | Prompt | Accuracy (legacy) | Accuracy (strict) |" + (" Neutral rate |" if has_neutral else "") + \
              " Ambiguous rate | Invalid rate | Accuracy on valid items |"
        out += [hdr, "|" + "---|" * (hdr.count("|") - 1)]
        sub = summary[summary.task == task]
        for m in sorted(sub.model.unique()):
            for p in PROMPT_ORDER:
                line = f"| {m} | {p} | {cell(task, m, p, 'accuracy', 'legacy')} | {cell(task, m, p, 'accuracy', 'strict')} |"
                if has_neutral:
                    line += f" {cell(task, m, p, 'neutral_rate', 'strict')} |"
                line += (f" {cell(task, m, p, 'ambiguous_rate', 'strict')} | {cell(task, m, p, 'invalid_rate', 'strict')} |"
                         f" {cell(task, m, p, 'selective_accuracy', 'strict')} |")
                out.append(line)
        out.append("")

    out += ["## TriviaQA (n=100 per cell)", "",
            "| Model | Prompt | Exact match | Contains answer |", "|---|---|---|---|"]
    sub = summary[summary.task == "trivia"]
    for m in sorted(sub.model.unique()):
        for p in PROMPT_ORDER:
            out.append(f"| {m} | {p} | {cell('trivia', m, p, 'exact_match', 'original')} | "
                       f"{cell('trivia', m, p, 'contains_answer', 'original')} |")
    out += ["", "## CNN/DailyMail summarization (n=20 per model)", "",
            "| Model | ROUGE-L | Faithfulness (flawed, see note) |", "|---|---|---|"]
    for m in sorted(summary[summary.task == "summarization"].model.unique()):
        out.append(f"| {m} | {cell('summarization', m, 'default', 'rouge_l', 'original', False)} | "
                   f"{cell('summarization', m, 'default', 'faithfulness', 'original', False)} |")
    out += ["", "Faithfulness note: the NLI premise is truncated to the first 800 characters of a 2000-character "
            "input, so this measures support by the article's opening, not faithfulness. Do not cite it.", ""]

    sc = paired[paired.comparison == "strict_vs_legacy"].copy()
    out += ["## Effect of the parser alone (strict minus legacy, same outputs)", "",
            "| Task | Model / prompt | Legacy | Strict | Difference [95% CI] | Items flipped right→wrong / wrong→right |",
            "|---|---|---|---|---|---|"]
    for _, r in sc.iterrows():
        if abs(r["diff"]) < 1e-12:
            continue
        out.append(f"| {r.task} | {r.scope} | {r.b_mean*100:.1f} | {r.a_mean*100:.1f} | "
                   f"{r['diff']*100:+.1f} [{r.ci_lo*100:+.1f}, {r.ci_hi*100:+.1f}] | {int(r.b_only)} / {int(r.a_only)} |")
    out += ["", "Cells not listed scored identically under both parsers.", ""]

    mm = paired[(paired.comparison == "model_vs_model") & (paired.task == "sst2")]
    out += ["## SST-2: Llama-3-8B vs Mistral-7B (strict accuracy, paired)", "",
            "| Prompt | Llama-3-8B | Mistral-7B | Difference [95% CI] | McNemar p (Holm) |", "|---|---|---|---|---|"]
    for _, r in mm.iterrows():
        a_is_llama = r.a.startswith("llama3-8b")
        sign = 1 if a_is_llama else -1
        lo, hi = sorted([sign * r.ci_lo, sign * r.ci_hi])
        la, mi = (r.a_mean, r.b_mean) if a_is_llama else (r.b_mean, r.a_mean)
        out.append(f"| {r.scope} | {la*100:.1f} | {mi*100:.1f} | {sign*r['diff']*100:+.1f} [{lo*100:+.1f}, {hi*100:+.1f}] | "
                   f"{r.p_mcnemar:.3g} ({r.p_holm:.3g}) |")
    out += ["", "All paired comparisons (every task, model pair and prompt pair) are in `results/paired_comparisons.csv`.", ""]
    (RESULTS / "summary_ci.md").write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    main()
