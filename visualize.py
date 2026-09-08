import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

os.makedirs("results/plots", exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────
sentiment_df = pd.read_csv("results/all_models_all_prompts.csv")
summ_df = pd.read_csv("results/summarization_results.csv")

# ── Aggregate sentiment ────────────────────────────────────────────────────
sent_summary = sentiment_df.groupby(["model", "prompt_variant"]).agg(
    accuracy=("correct", "mean"),
    avg_latency_ms=("latency_ms", "mean"),
    total_cost_usd=("cost_usd", "sum"),
).reset_index()

summ_summary = summ_df.groupby("model").agg(
    avg_rouge_l=("rouge_l", "mean"),
    avg_faithfulness=("faithfulness", "mean"),
    avg_latency_ms=("latency_ms", "mean"),
    total_cost_usd=("cost_usd", "sum"),
).reset_index()

COLORS = {"llama3-8b": "#4C72B0", "mistral-7b": "#DD8452"}

# ══════════════════════════════════════════════════════════════════════════
# CHART 1: Sentiment accuracy by model + prompt variant
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5))

models = sent_summary["model"].unique()
prompts = ["zero_shot", "few_shot", "chain_of_thought"]
x = range(len(prompts))
width = 0.35

for i, model in enumerate(models):
    subset = sent_summary[sent_summary["model"] == model].set_index("prompt_variant")
    vals = [subset.loc[p, "accuracy"] if p in subset.index else 0 for p in prompts]
    offset = (i - 0.5) * width
    bars = ax.bar([xi + offset for xi in x], vals, width, label=model, color=COLORS[model], alpha=0.85)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.0%}", ha="center", va="bottom", fontsize=9)

ax.set_xticks(list(x))
ax.set_xticklabels(["Zero Shot", "Few Shot", "Chain of Thought"], fontsize=11)
ax.set_ylabel("Accuracy", fontsize=11)
ax.set_ylim(0, 1.05)
ax.set_title("Sentiment Classification Accuracy\nby Model and Prompt Strategy", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.grid(axis="y", alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("results/plots/1_sentiment_accuracy.png", dpi=150)
plt.close()
print("Chart 1 saved.")

# ══════════════════════════════════════════════════════════════════════════
# CHART 2: Cost vs Accuracy — Pareto frontier
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))

for _, row in sent_summary.iterrows():
    color = COLORS[row["model"]]
    ax.scatter(row["total_cost_usd"], row["accuracy"], color=color, s=120, zorder=3)
    label = f"{row['model']}\n{row['prompt_variant'].replace('_', ' ')}"
    ax.annotate(label, (row["total_cost_usd"], row["accuracy"]),
                textcoords="offset points", xytext=(6, 4), fontsize=7.5)

# Pareto frontier (manually highlight best points)
pareto = sent_summary.sort_values("total_cost_usd")
frontier = []
best_acc = 0
for _, row in pareto.iterrows():
    if row["accuracy"] > best_acc:
        best_acc = row["accuracy"]
        frontier.append(row)
frontier_df = pd.DataFrame(frontier)
ax.plot(frontier_df["total_cost_usd"], frontier_df["accuracy"],
        "k--", linewidth=1.2, alpha=0.5, label="Pareto frontier")

patches = [mpatches.Patch(color=c, label=m) for m, c in COLORS.items()]
ax.legend(handles=patches, fontsize=10)
ax.set_xlabel("Total Cost (USD, 100 examples)", fontsize=11)
ax.set_ylabel("Accuracy", fontsize=11)
ax.set_title("Cost vs Accuracy — Pareto Frontier\n(lower-left = cheaper, upper = more accurate)", fontsize=13, fontweight="bold")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.grid(alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("results/plots/2_pareto_frontier.png", dpi=150)
plt.close()
print("Chart 2 saved.")

# ══════════════════════════════════════════════════════════════════════════
# CHART 3: Summarization — ROUGE-L and Faithfulness side by side
# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

metrics = [("avg_rouge_l", "ROUGE-L Score", "Lexical overlap with reference summary"),
           ("avg_faithfulness", "Faithfulness Score", "Fraction of sentences supported by article")]

for ax, (col, title, subtitle) in zip(axes, metrics):
    vals = [summ_summary.loc[summ_summary["model"] == m, col].values[0] for m in COLORS]
    bars = ax.bar(list(COLORS.keys()), vals, color=list(COLORS.values()), alpha=0.85, width=0.45)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_title(f"{title}\n{subtitle}", fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(vals) * 1.3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Llama 3 8B", "Mistral 7B"], fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

plt.suptitle("Summarization Quality Comparison", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("results/plots/3_summarization_quality.png", dpi=150, bbox_inches="tight")
plt.close()
print("Chart 3 saved.")

# ══════════════════════════════════════════════════════════════════════════
# CHART 4: Latency comparison — both tasks
# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Sentiment latency by model (best prompt only = zero_shot)
zs = sent_summary[sent_summary["prompt_variant"] == "zero_shot"]
axes[0].bar(zs["model"], zs["avg_latency_ms"], color=[COLORS[m] for m in zs["model"]], alpha=0.85, width=0.45)
for ax_bar, val in zip(axes[0].patches, zs["avg_latency_ms"]):
    axes[0].text(ax_bar.get_x() + ax_bar.get_width() / 2, ax_bar.get_height() + 5,
                 f"{val:.0f}ms", ha="center", va="bottom", fontsize=11, fontweight="bold")
axes[0].set_title("Sentiment Latency\n(zero-shot, avg per call)", fontsize=11, fontweight="bold")
axes[0].set_ylabel("Latency (ms)")
axes[0].set_xticks([0, 1])
axes[0].set_xticklabels(["Llama 3 8B", "Mistral 7B"], fontsize=10)
axes[0].grid(axis="y", alpha=0.3)
axes[0].spines[["top", "right"]].set_visible(False)

# Summarization latency
axes[1].bar(summ_summary["model"], summ_summary["avg_latency_ms"],
            color=[COLORS[m] for m in summ_summary["model"]], alpha=0.85, width=0.45)
for ax_bar, val in zip(axes[1].patches, summ_summary["avg_latency_ms"]):
    axes[1].text(ax_bar.get_x() + ax_bar.get_width() / 2, ax_bar.get_height() + 5,
                 f"{val:.0f}ms", ha="center", va="bottom", fontsize=11, fontweight="bold")
axes[1].set_title("Summarization Latency\n(avg per call)", fontsize=11, fontweight="bold")
axes[1].set_xticks([0, 1])
axes[1].set_xticklabels(["Llama 3 8B", "Mistral 7B"], fontsize=10)
axes[1].grid(axis="y", alpha=0.3)
axes[1].spines[["top", "right"]].set_visible(False)

plt.suptitle("Inference Latency by Task", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("results/plots/4_latency_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Chart 4 saved.")

print("\nAll charts saved to results/plots/")
print("\n=== FINAL SCORECARD ===")
print(f"{'Metric':<30} {'Llama 3 8B':>12} {'Mistral 7B':>12}")
print("-" * 56)

llama_sent = sent_summary[(sent_summary["model"] == "llama3-8b") & (sent_summary["prompt_variant"] == "zero_shot")].iloc[0]
mistral_sent = sent_summary[(sent_summary["model"] == "mistral-7b") & (sent_summary["prompt_variant"] == "zero_shot")].iloc[0]
llama_summ = summ_summary[summ_summary["model"] == "llama3-8b"].iloc[0]
mistral_summ = summ_summary[summ_summary["model"] == "mistral-7b"].iloc[0]

print(f"{'Sentiment accuracy (zero-shot)':<30} {llama_sent['accuracy']:>11.1%} {mistral_sent['accuracy']:>11.1%}")
print(f"{'Sentiment cost (100 examples)':<30} ${llama_sent['total_cost_usd']:>10.4f} ${mistral_sent['total_cost_usd']:>10.4f}")
print(f"{'ROUGE-L (summarization)':<30} {llama_summ['avg_rouge_l']:>12.3f} {mistral_summ['avg_rouge_l']:>12.3f}")
print(f"{'Faithfulness (summarization)':<30} {llama_summ['avg_faithfulness']:>11.1%} {mistral_summ['avg_faithfulness']:>11.1%}")
print(f"{'Summ cost (20 examples)':<30} ${llama_summ['total_cost_usd']:>10.4f} ${mistral_summ['total_cost_usd']:>10.4f}")
print("-" * 56)
print(f"{'WINNER':<30} {'← accuracy':>12} {'cost →':>12}")