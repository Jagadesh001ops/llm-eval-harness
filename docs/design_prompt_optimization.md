# Design: Does prompt optimization need a stronger model, or can a small model fix its own prompt?

Status: **draft for review** — nothing in this document has been run yet. Every result slot is "not yet run".
Author date: 2026-11-18. Supersedes Phase 3 ("self-correction loop") of the improvement brief and becomes the
project's main experiment.

---

## 1. Question

Given a fixed small model (the **target**) and a task, an **optimizer** model rewrites the target's instruction
prompt using the target's failures. No weights change anywhere — only the prompt text.

> Does the optimizer need to be a frontier model, or merely larger than the target, or can the target fix its own prompt?

Terminology used throughout (and in the write-up):

| Term | Meaning here |
|---|---|
| **Self-optimization** | the target rewrites its own prompt from its own failures |
| **Cross-model prompt optimization** | a different (usually larger) model rewrites the target's prompt; knowledge reaches the target only through the prompt text |
| **Prompt transfer** | a prompt optimized for one target is evaluated on another target |

We deliberately avoid "distillation" and "transfer learning": both imply weight updates, which do not happen here.

## 2. Models

### Targets (prompt is optimized; accuracy is measured)

| Model | Why |
|---|---|
| Gemma 3 4B | small end of a clean, dense, single-family size ladder |
| Gemma 3 12B | middle; closest in size to the retired Llama-3-8B |
| Gemma 3 27B | large end of the ladder |
| Mistral-7B *(open decision, §12)* | continuity with the v1 results and the neutral-hedging analysis |

All targets are served through AWS Bedrock. Gemma weights are public, so results can be reproduced outside Bedrock.
Llama-3-8B/70B are excluded: Bedrock marks both Legacy and intermittently refuses calls (observed 2026-11-18).

### Optimizer conditions (who rewrites)

| Condition | Optimizer | Applied to |
|---|---|---|
| **Strong** | Claude Opus 5.5 | every target |
| **Mid** | Claude Haiku 5.5 | every target |
| **Self** | the target itself | its own prompt only |
| **Control** | the target itself, **without** failures or scores ("improve this prompt") | its own prompt only |
| **Same-family larger** (extension) | Gemma 3 27B | Gemma 3 4B and Gemma 3 12B |

The **control** separates "feedback helps" from "rewriting + selecting the best of many candidates on dev helps by
chance". If *self* does not beat *control*, the small model is not actually fixing anything.

The **same-family larger** extension separates "the optimizer must be frontier-class" from "the optimizer must merely be
bigger than the target". If 27B→4B approaches Opus→4B, a modest same-family teacher suffices.

### Sampling

* Answering task items (all scoring): temperature 0.
* Generating rewrites: each optimizer at its **default** sampling. Claude Opus 5.5 and Haiku 5.5 do not accept a
  temperature parameter, so Gemma/Mistral optimizers also run at their default, for parity.
* Claude calls go through the Anthropic SDK's Bedrock client; target calls through the existing `harness/bedrock.py`.

## 3. Tasks

| Role | Task | Split used for test | Skill | Primary metric |
|---|---|---|---|---|
| Main | **AG News** (4-class topic) | test | classification; contains a known label-definition flaw (class "Sci/Tech" named "science") | accuracy (strict parser) |
| Main | **GSM8K** (math word problems) | test | multi-step reasoning; standard benchmark in the prompt-optimization literature | exact numeric match |
| Main | **SQuAD 2.0** | validation (official test is hidden) | reading comprehension incl. unanswerable questions | F1 on answerable; accuracy on unanswerable; official combined F1 |
| Control | **TriviaQA** (closed-book) | validation | knowledge a prompt cannot add | exact match and contains-answer, reported separately |

Retired from the new study: **SST-2** (near ceiling; the neutral-hedging analysis stays as a supporting section using
the existing v1 data) and **CNN/DailyMail** (ROUGE-L is easy to game when optimized against; faithfulness metric is
broken). Both remain reproducible from the v1 results.

**TriviaQA as a control:** prompts cannot add facts, so large "gains" there are expected to come from answer-format
changes. Reporting them separately quantifies how much of any task's gain is format compliance rather than capability.

**Contamination caveat:** GSM8K, SQuAD and TriviaQA are widely published and may appear in training data. This can
inflate absolute scores; comparisons between prompts on the same model are much less affected.

## 4. Data splits

Fixed once with seed `20261118`, written to `data/splits/<task>.json` (item IDs only), committed, never regenerated.

| Task | Failure pool (shown to optimizers) | Dev (selects candidates) | Test (reported) |
|---|---|---|---|
| AG News | 500 from train | 200 from train, disjoint from pool | 500 from test, **excluding test[0:100]** (already inspected in v1) |
| GSM8K | 500 from train | 200 from train | 500 from test |
| SQuAD 2.0 | 500 from train | 200 from train, 50/50 answerable/unanswerable | 500 from validation, 50/50 answerable/unanswerable |
| TriviaQA | 500 from train | 200 from train | 500 from validation, **excluding validation[0:100]** |

Rules: test items are never shown to an optimizer and never used for any selection. At n=500 the 95% CI half-width
for accuracy is about ±4 points; paired differences are tighter.

## 5. Pipeline

```
 seed prompt (v2) ─► score on dev ─► failures (from pool) ─► optimizer proposes 4 rewrites
        ▲                                                            │
        │                                              intent + rule checks (§6)
        │                                                            │
        └──── keep best if it beats current ◄── score valid rewrites on dev
                    (≤ 4 rounds, early stop)
                              │
                              ▼
          final prompt ─► scored once on test ─► paired CI vs seed prompt
                       └► prompt transfer: scored on every other target's test run
```

1. **Seed prompt (v2)** per task: the v1 prompt with known harness defects fixed (AG News label named Sci/Tech;
   reasoning budget large enough to finish; explicit final-answer line). The v1 prompts are kept unchanged so v1 results
   stay reproducible. Optimization uses **instruction-only** prompts: no few-shot examples, so a rewrite cannot smuggle
   in content through examples.
2. **Score** the current prompt on dev with the target. Sample 10 failures from the failure pool (raw output + gold).
3. **Propose.** The optimizer receives the task card (§6.1), current prompt, history of previous prompts with dev
   scores, the 10 failures, and the rules. It returns 4 candidate prompts. The control condition receives the task card,
   current prompt and rules only.
4. **Check** every candidate against §6. Rejected candidates are logged with the reason and count toward that
   optimizer's violation rate; they are not replaced (equal budget across optimizers).
5. **Score** valid candidates on dev; the best replaces the current prompt only if strictly better.
6. Repeat up to **4 rounds**; stop early if a round produces no improvement.
7. **Final:** score the chosen prompt once on test. **Transfer:** score it on every other target's test run too.
8. **Seeds:** repeat each loop 3 times (independent runs; Claude models cannot be seeded, so "seed" = run index).

Budget per loop is identical for every condition: ≤ 4 rounds × 4 candidates × 200 dev items.

## 6. Intent preservation ("stay native to the intent")

A rewrite may change **how** the task is explained, never **what** the task is. It must not introduce new concepts,
labels, facts or goals that change the meaning of the prompt.

### 6.1 Task card (the fixed definition every rewrite must stay within)

One file per task, `prompts/task_cards/<task>.yaml`, written by hand before any run and frozen:

* what the input is and what is being asked;
* the exact output contract (label set / answer format / the literal token `unanswerable` for SQuAD 2.0);
* what counts as correct;
* **allowed** clarifications — e.g. for AG News, describing class *science* as covering science **and technology**
  (this matches the dataset's own Sci/Tech definition, so it clarifies intent rather than changing it);
* **forbidden** content (below).

### 6.2 Allowed vs forbidden edits

| Allowed (native to the intent) | Forbidden (changes the intent) |
|---|---|
| rephrasing, reordering, clarifying the instructions | adding, removing or renaming labels / answer options |
| defining existing labels in line with the dataset's documentation | changing the task (e.g. sentiment → emotion; QA → summarization) |
| output-format instructions consistent with the task card | adding domain facts or knowledge (would leak answers, especially on TriviaQA/GSM8K) |
| general strategies ("read the whole passage", "check your arithmetic") | text copied from any pool, dev or test item |
| guidance on edge cases the task card already defines (e.g. when a SQuAD question is unanswerable) | instructions aimed at specific items or keywords seen in the failures |
| | few-shot examples (instruction-only track) |
| | new placeholders, or removing the input placeholder |

### 6.3 Enforcement, in three layers

1. **Hard rules (code, deterministic) — reject on failure.**
   * exactly the required placeholders (e.g. `{text}`), no others;
   * every label / answer-format token in the task card appears; no other label-like token is introduced
     (checked against the task card's label vocabulary);
   * **leakage check:** no 8-word sequence shared with any pool, dev or test item;
   * length cap (e.g. ≤ 3× the seed prompt) so "longer" cannot be the only lever;
   * no few-shot examples (pattern check for input/answer pairs).
2. **Intent judge (model, rubric-based) — reject on "changes intent".**
   A model that is **not** one of the optimizers (planned: Claude Sonnet 5.5, so no optimizer judges its own work)
   compares each rewrite with the task card and answers *preserves intent / changes intent* with a reason, using a
   fixed rubric built from §6.2.
   **The judge is not used until it is validated** (the brief's rule: no unvalidated judge):
   * ~60 rewrite/task-card pairs, drawn from the pilot across all optimizers, are labelled independently by the
     project owner (blind to which optimizer wrote them and to the judge's verdict);
   * acceptance: agreement ≥ 85% **and** Cohen's κ ≥ 0.70 against the human labels; both numbers are reported;
   * if it fails, the judge is not used to reject anything; instead every **final** prompt is reviewed by a human
     against the task card, and that is stated in the results.
3. **Measured, not enforced — reported per optimizer.**
   * **lexical novelty:** share of content words in the rewrite that appear in neither the seed prompt nor the task
     card (catches "cooked-up new words" without banning harmless synonyms);
   * rule-violation rate and judge-rejection rate;
   * prompt length.

Rejections are part of the result: "the 4B self-optimizer broke the task definition in X% of rewrites" is a finding.

## 7. Metrics and statistics

Per (task, target, optimizer condition):

| Metric | Definition |
|---|---|
| **Test gain** (primary) | test score of final prompt − test score of seed prompt, same items, averaged over 3 runs |
| Dev–test gap | dev gain − test gain; large gap = overfitting to dev |
| Rounds to best | round at which the final prompt was found |
| Violation / judge-rejection rate | §6.3 |
| Lexical novelty, prompt length | §6.3 |
| Transfer gain | test gain when the final prompt is used on each other target |
| Diagnosis (AG News) | whether the final prompt defines *science* to include technology, without being told |

**Uncertainty:** two-level paired bootstrap — resample the 3 runs, then resample test items, apply the same item
indices to the final and seed prompts — B = 10,000, seed `20261118`, 95% percentile intervals. This includes both
item-sampling and optimization-run variance (extends `harness/stats.py`, method documented in `docs/statistics.md`).
Paired exact McNemar for binary per-item outcomes; Holm correction within each hypothesis family.

## 8. Pre-registered hypotheses

Written before any run; each will be reported as supported / not supported / inconclusive with its interval.

* **H1** Strong (Opus) test gain > self test gain, for every target.
* **H2** Self-optimization gains appear on dev but shrink on test (dev–test gap larger for self than for strong).
* **H3** Self beats control only for the larger targets (self-optimization ability grows with model size: 4B < 12B < 27B).
* **H4** Prompts from stronger optimizers transfer better to other targets.
* **H5** Same-family larger (27B→4B, 27B→12B) recovers a substantial part of the strong-optimizer gain.
* **H6** Smaller optimizers violate the intent rules more often (§6).
* **H7** On TriviaQA (control), gains are concentrated in exact match (format), not contains-answer (knowledge).

## 9. Size, cost and the pilot

Full grid per task and run: 4 targets × 4 conditions + 2 extension loops = 18 loops; × 3 main tasks × 3 runs =
**162 loops**, plus TriviaQA at 1 run. Each loop ≤ 16 candidates × 200 dev items = 3,200 target calls.

Cost: **not yet measured.** Rough expectation from list prices: target calls cost cents per loop on Gemma; Claude Opus
5.5 optimizer calls roughly $0.10 each (thinking is billed as output), ~$0.40–0.50 per loop. These are estimates only
and are replaced by measured numbers after the pilot.

**Pilot (run first, then decide on the grid):** AG News × Gemma 3 12B × {strong, mid, self, control, 27B→12B} × 1 run.
Outputs: measured cost, wall-clock time, rate-limit behaviour, first effect sizes, and the ~60 rewrites used to
validate the intent judge. Go/no-go for the full grid is decided with the project owner.

## 10. Outputs

| Path | Contents |
|---|---|
| `prompts/v1/`, `prompts/v2/` | frozen seed prompts |
| `prompts/task_cards/` | frozen task definitions (§6.1) |
| `data/splits/` | item-ID splits (§4) |
| `results/opt/<task>/<target>/<condition>/run<k>/` | every round: candidates, rejection reasons, dev scores, chosen prompt, final test per-item outputs |
| `results/opt/summary.csv`, `results/opt/summary.md` | §7 metrics with intervals |
| `results/opt/judge_validation.csv` | human vs judge labels, agreement, κ |

Every number in the eventual README must trace to one of these files.

## 11. Known risks

| Risk | Mitigation |
|---|---|
| Models retired mid-study (Bedrock "Legacy") | run soon; retries + explicit INCOMPLETE marking already in the harness; Gemma weights are public |
| Optimizer overfits dev | test never used for selection; dev–test gap reported |
| "Longer prompt" explains gains | length cap + length reported |
| Optimizer leaks answers/facts | 8-gram leakage check, knowledge ban in §6, TriviaQA control |
| Judge bias | judge is not an optimizer; validated against human labels before use |
| Rate limits make the grid slow | measured in the pilot; parallelism tuned before the full run |
| Contaminated benchmarks | compare prompts within a model; caveat stated |

## 12. Open decisions

1. Keep **Mistral-7B** as a fourth target (continuity), or only the three Gemma sizes?
2. Confirm **SQuAD 2.0** over RACE, and retiring **SST-2** from the new study.
3. Confirm **Claude Sonnet 5.5** as the intent judge (requires enabling it in Bedrock model access).

## 13. Relation to the original brief

| Brief phase | Status in this design |
|---|---|
| 1 Statistical rigor | done (`docs/statistics.md`); extended here with run-level resampling |
| 2 Neutral-label finding | becomes a supporting section (v1 data), not the headline |
| 3 Self-correction loop as a measured study | **replaced** by this prompt-optimization study |
| 4 Breadth (open model) | Gemma 3 on Bedrock (accepted in place of HuggingFace hosting) |
| 5 Write-up | after results |
| Stretch: validated LLM judge | the intent judge (§6.3), validated against human labels |
