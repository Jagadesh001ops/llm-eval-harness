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

| Family | Model | Role |
|---|---|---|
| **Core** | Gemma 3 4B | small end of a clean, dense, single-family size ladder |
| **Core** | Gemma 3 12B | middle; closest in size to the retired Llama-3-8B |
| **Core** | Gemma 3 27B | large end of the ladder |
| **Replication** | Ministral 3B (3.0) | second family, so findings are not specific to Gemma |
| **Replication** | Ministral 8B (3.0) | also keeps a Mistral-family model for continuity with v1 |
| **Replication** | Ministral 14B (3.0) | large end of the second ladder |

All targets are served through AWS Bedrock (on-demand, US regions). Licence and Bedrock lifecycle status (Active vs
Legacy) are to be confirmed via the Bedrock API before the pilot.

Excluded: Llama-3-8B/70B (Bedrock Legacy, intermittently refusing calls, observed 2026-11-18); Mistral-7B v0.2
(replaced by Ministral 8B to avoid the same Legacy risk); Llama 3.2 1B/3B/11B (11B is the vision model, so not a clean
text ladder; older generation; listed in one region only); Magistral (reasoning-trace model, would confound the
comparison), Devstral (code), Voxtral (audio).

### Optimizer conditions (who rewrites)

| Condition | Optimizer | Applied to |
|---|---|---|
| **Strong (closed)** | Claude Opus 5.5, effort `low` | every target |
| **Strong (open)** | Mistral Large 3 | every target |
| **Mid** | Claude Haiku 5.5 | every target |
| **Self** | the target itself | its own prompt only |
| **Control** | the target itself, **without** failures or scores ("improve this prompt") | its own prompt only |
| **Same-family larger** (extension) | Gemma 3 27B → Gemma 3 4B, 12B · Ministral 14B → Ministral 3B, 8B | smaller members of the same family |

Two strong optimizers separate "a strong optimizer" from "Claude specifically": if the open Mistral Large 3 matches
Opus, size/capability is what matters, not the vendor.

Opus 5.5 runs at effort `low` for cost (its thinking cannot be disabled and is billed as output). The pilot checks
that rewrite quality at `low` is not visibly worse than at `medium` on a small sample before the grid uses `low`.

The **control** separates "feedback helps" from "rewriting + selecting the best of many candidates on dev helps by
chance". If *self* does not beat *control*, the small model is not actually fixing anything.

The **same-family larger** extension separates "the optimizer must be frontier-class" from "the optimizer must merely be
bigger than the target". If 27B→4B approaches Opus→4B, a modest same-family teacher suffices.

### Sampling

* Answering task items (all scoring): temperature 0.
* Generating rewrites: each optimizer at its **default** sampling. Claude Opus 5.5 and Haiku 5.5 do not accept a
  temperature parameter, so Gemma/Mistral optimizers also run at their default, for parity.
* Claude calls go through the Anthropic API (the account's Bedrock access to Anthropic models required company
  details); all other models go through Bedrock via `harness/bedrock.py`.

## 3. Tasks

| Role | Task | Split used for test | Skill | Primary metric |
|---|---|---|---|---|
| Main | **AG News** (4-class topic) | test | classification; contains a known label-definition flaw (class "Sci/Tech" named "science") | accuracy (strict parser) |
| Main | **SQuAD 2.0** | validation (official test is hidden) | reading comprehension incl. unanswerable questions | per-item F1 (official combined); F1 on answerable and accuracy on unanswerable reported separately |
| Control | **TriviaQA** (closed-book) | validation | knowledge a prompt cannot add | exact match and contains-answer, reported separately |

Retired from the new study: **SST-2** (near ceiling; the neutral-hedging analysis stays as a supporting section using
the existing v1 data) and **CNN/DailyMail** (ROUGE-L is easy to game when optimized against; faithfulness metric is
broken). Both remain reproducible from the v1 results.

### Task selection: why there is no dedicated reasoning task

A reasoning task was planned and dropped because every candidate was saturated for these targets once harness
defects were fixed. All checks used dev items only (`scripts/check_headroom.py`, results in
`results/opt/headroom_*.csv`); the pre-set rule was that the seed prompt must leave room to improve (roughly
40–85% for most targets).

* **GSM8K:** seed dev accuracy 94–96% for five of six targets (Gemma 3 4B: 87%). Ministral 3B at 96% suggests the
  benchmark is in the models' training data.
* **BIG-Bench Hard**, five sub-tasks: tracking shuffled objects 87–100%; logical deduction mostly 83–97%; web of lies
  100% for all Gemma sizes; date understanding 93% for Gemma 12B/27B. Only causal judgement stayed at 51–72%, but its
  187 items leave 62 for test (CI about ±12 points) and its labels are known to be noisy.

**Finding recorded for the write-up:** with a generic seed that did not state the yes/no answer format, Gemma 3 4B
scored 1% on web of lies; with an explicit `Answer: Yes / No` line it scored 100%. Much of the apparent difficulty of
these benchmarks for small models was output-format compliance, not reasoning. This is also why gains on any task are
read against the TriviaQA control, which isolates format-driven gains.

**TriviaQA as a control:** prompts cannot add facts, so large "gains" there are expected to come from answer-format
changes. Reporting them separately quantifies how much of any task's gain is format compliance rather than capability.

**Contamination caveat:** AG News, SQuAD and TriviaQA are widely published and may appear in training data (in the
pilot, Opus wrote AG News's own class name "sci/tech" unprompted). This can
inflate absolute scores; comparisons between prompts on the same model are much less affected.

## 4. Data splits

Fixed once with seed `20261118`, written to `data/splits/<task>.json` (item IDs only), committed, never regenerated.

| Task | Failure pool (shown to optimizers) | Dev (selects candidates) | Test (reported) |
|---|---|---|---|
| AG News | 500 from train | 200 from train, disjoint from pool | 500 from test, **excluding test[0:100]** (already inspected in v1) |
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

1. **Seed prompt (v2)** per task: the v1 prompt with **harness** defects fixed only (reasoning budget large enough to
   finish; explicit final-answer line). **Prompt-quality** flaws are deliberately kept: e.g. AG News still names the
   Sci/Tech class "science", because whether an optimizer discovers that from failures is a measured outcome
   ("Diagnosis", §7). For AG News, v2 is therefore identical to the v1 zero-shot prompt. The v1 prompts are kept
   unchanged so v1 results stay reproducible. Optimization uses **instruction-only** prompts: no few-shot examples, so
   a rewrite cannot smuggle in content through examples.
   A hand-written **oracle** prompt per task (`prompts/reference/`), which fixes the known flaws using the dataset's
   own label definitions, is scored on the same test items as a human-expert reference. It is never shown to
   optimizers.
2. **Score** the current prompt on dev with the target. Sample 10 failures from the failure pool (raw output + gold).
3. **Propose.** The optimizer receives the task card (§6.1), current prompt, history of previous prompts with dev
   scores, the 10 failures, and the rules. It first writes a short **diagnosis** of what the failures have in common
   and why the current prompt causes them (in `<analysis>` tags, logged), then returns 4 candidate prompts that
   address it. The control condition receives the task card, current prompt and rules only (no failures, scores or
   diagnosis step).
4. **Check** every candidate against §6. Rejected candidates are logged with the reason and count toward that
   optimizer's violation rate; they are not replaced (equal budget across optimizers).
5. **Score** valid candidates on dev; the best replaces the current prompt only if strictly better.
6. Repeat up to **4 rounds**. Always run at least 2 rounds; stop early only after **2 consecutive rounds** without
   improvement (patience 2).

**Protocol versions.** Protocol **v2** (above) is frozen as of 2026-11-18 and used for every reported run. Pilot round
1 used v1 (stop after the first non-improving round; no diagnosis step). It is kept in `results/opt_pilot/` as
exploratory and is excluded from all reported results. The v1 → v2 changes were motivated by optimizer behaviour in
that pilot (two of four loops stopped after one round on 0.5-point dev differences; no optimizer acted on the failures
it was shown), not by test scores. Every log records its protocol version.
7. **Final:** score the chosen prompt once on test. **Transfer:** score it on every other target's test run too.
8. **Seeds:** repeat each loop 3 times (independent runs; Claude models cannot be seeded, so "seed" = run index).

Budget per loop is identical for every condition: ≤ 4 rounds × 4 candidates × 200 dev items.

## 6. Intent preservation ("stay native to the intent")

A rewrite may change **how** the task is explained, never **what** the task is. It must not introduce new concepts,
labels, facts or goals that change the meaning of the prompt.

### 6.1 Task card (the fixed definition every rewrite must stay within)

One file per task, `prompts/task_cards/<task>.yaml`, written by hand before any run and frozen. It has two parts:
an **optimizer view** (task, input, output contract, labels, placeholders), which optimizers receive, and a **judge
view** (the dataset's own label definitions, allowed and forbidden edits), which only the intent judge sees, so that
the card never hands optimizers the answer to a flaw they are supposed to discover. Together they specify:

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
| output-format instructions consistent with the task card | adding domain facts or knowledge (would leak answers, especially on TriviaQA) |
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

**Validation outcome (2026-11-18): FAILED.** Two labelling passes by the project owner gave 68.4% agreement (kappa 0.22) and 75.5% (kappa 0.36) against the 85% / 0.70 bar (`results/judge_validation/report*.md`). On the 12 constructed intent-changing items, whose answer is known by design, the judge flagged 12/12 and the human labels 5/12 in both passes; the labeller's notes suggest the labels judged prompt completeness rather than intent. Per the pre-registered rule, the judge is **not** used to reject candidates. Instead: (1) the grid runs with layer-1 checks only; (2) the judge is run afterwards on every candidate and its verdicts are reported as flags from an unvalidated judge; (3) main results are reported with and without judge-flagged final prompts (sensitivity analysis); (4) final prompts get human review, preferably by a second person.

**Amendment (2026-11-18, before any grid run): judge ON.** The project owner decided to keep the intent judge in the loop despite the failed validation, citing its 12/12 on known violations and the inconsistency of the human labels. No grid results existed when this was decided. Safeguards: (1) **shadow scoring**: a candidate rejected only by the judge is still scored on dev (never selected, never shown in the history), and each round logs whether the judge blocked a would-be winner (`judge_blocked_winner`), so the cost of judge errors is measured; (2) **adjudication**: the 13 real pilot rewrites where the owner's second pass and the judge disagreed (or the owner was unsure) are given a final call by the owner with both rationales visible (not blind; this biases towards the judge and is disclosed). If the owner overrules the judge on a consistent pattern, the judge's instructions are corrected for that pattern before the grid and re-checked on the 12 constructed items. The judge decides intent only, never completeness or quality. (3) The with/without-flagged sensitivity analysis is still reported.

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
* **H3** Self beats control only for the larger targets (self-optimization ability grows with model size within a
  family: Gemma 4B < 12B < 27B; Ministral 3B < 8B < 14B).
* **H4** Prompts from stronger optimizers transfer better to other targets.
* **H5** Same-family larger (Gemma 27B→4B/12B; Ministral 14B→3B/8B) recovers a substantial part of the
  strong-optimizer gain.
* **H6** Smaller optimizers violate the intent rules more often (§6).
* **H7** On TriviaQA (control), gains are concentrated in exact match (format), not contains-answer (knowledge).
* **H8** The open strong optimizer (Mistral Large 3) and the closed one (Opus 5.5) produce test gains whose paired
  difference is within ±2 points, i.e. "strong" is about capability, not vendor. (Equivalence claim: supported only if
  the whole 95% interval of the difference lies inside ±2 points.)
* **Replication:** H1–H6 are tested separately in each family; a finding is reported as general only if its direction
  holds in both Gemma and Ministral.

## 9. Size, cost and the pilot

Per family, per task and run: 3 targets × 5 conditions (strong-closed, strong-open, mid, self, control) + 2
same-family-larger loops = **17 loops**. Each loop ≤ 4 rounds × 4 candidates × 200 dev items = 3,200 target calls.

| Tier | Targets | Tasks | Runs | Loops |
|---|---|---|---|---|
| Core | Gemma 3 4B/12B/27B | AG News, SQuAD 2.0 | 3 | 17 × 2 × 3 = **102** |
| Replication | Ministral 3B/8B/14B | AG News | 2 | 17 × 1 × 2 = **34** |
| Control | Gemma 3 4B/12B/27B | TriviaQA | 1 | 17 |

**Cost: not yet measured.** Estimates from list prices, to be replaced by pilot measurements:

| Component | Estimate | Basis |
|---|---|---|
| Opus 5.5 optimizer (whole study) | ~$3–6 | ≤ 108 calls (27 Opus loops × 4 rounds); pilot round 2 measured about $0.03 per call at effort `low` on AG News; SQuAD failure examples include passages, so assume up to ~$0.05 |
| Mistral Large 3, Haiku 5.5 optimizers | under $1 | pilot: $0.006 (Mistral Large 3) and $0.003 (Haiku) per loop |
| Target scoring | ~$5–15 | pilot: $0.01–0.05 per AG News loop on Gemma 12B; SQuAD prompts are longer and 27B costs more per token |

Cost controls built into the harness (none of them change the design):

1. the seed prompt is scored once per target and shared across all optimizer conditions;
2. candidate scores are cached by prompt text, so identical rewrites are never re-scored;
3. early stopping when a round brings no improvement;
4. Opus 5.5 at effort `low`;
5. prompt caching on the optimizer's fixed instructions (task card + rules are byte-identical across calls).

**Pilot (run first, then decide on the grid):** AG News × Gemma 3 12B × {strong-closed, strong-open, mid, self, control,
27B→12B} × 1 run, plus a small `low` vs `medium` effort comparison for Opus.
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

Resolved: Mistral-7B dropped (Ministral 8B instead); Ministral 3B/8B/14B added as replication family; Mistral Large 3
added as open strong optimizer; Opus 5.5 at effort `low` as closed strong optimizer; Claude Sonnet 5.5 as intent judge.

Also resolved: SQuAD 2.0 (not RACE) as the second main task; SST-2 retired; no dedicated reasoning task (GSM8K and
BIG-Bench Hard saturated, §3). Nothing is open.

## 13. Relation to the original brief

| Brief phase | Status in this design |
|---|---|
| 1 Statistical rigor | done (`docs/statistics.md`); extended here with run-level resampling |
| 2 Neutral-label finding | becomes a supporting section (v1 data), not the headline |
| 3 Self-correction loop as a measured study | **replaced** by this prompt-optimization study |
| 4 Breadth (open model) | Gemma 3 on Bedrock (accepted in place of HuggingFace hosting) |
| 5 Write-up | after results |
| Stretch: validated LLM judge | the intent judge (§6.3), validated against human labels |
