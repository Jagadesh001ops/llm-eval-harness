# Can a small model fix its own prompt?

An evaluation harness and study of **automated prompt optimization**: an optimizer model rewrites a small model's
instruction prompt from its failures, and we measure on held-out data whether the rewrite helps. The question is
whether the optimizer needs to be a frontier model, merely a larger model of the same family, or whether the small
model can do it itself.

> **Status (2026-11-18): main grid in progress.** Sections marked *not yet run* will be filled from result files
> when the runs finish. Every number below links to a file that a script regenerates.

## Results at a glance

| | Result | Source |
|---|---|---|
| Main grid: test gain by optimizer, task and target | *not yet run* | `results/opt/` |
| Prompt transfer across targets | *not yet run* | |
| Self-optimization vs model size (Gemma 3 4B / 12B / 27B) | *not yet run* | |
| Pilot (exploratory, 1 run, AG News, Gemma 3 12B) | strong optimizers +5 to +8 points; self and larger-same-family 0 | `results/opt_pilot/round2/` |
| Finding: benchmark "difficulty" for small models is largely answer-format compliance | Gemma 3 4B, BBH web of lies: 1% with a generic answer instruction, 100% with an explicit one | `results/opt/headroom_bbh_*.csv` |
| Finding: the original "neutral-label bias" headline does not reproduce | the prompt behind it was never recorded | see Phase 1 |

## Approach

**Targets** (prompts are optimized, accuracy measured): Gemma 3 4B / 12B / 27B (core), Ministral 3B / 8B / 14B
(replication family). **Optimizers**: Claude Opus 5.5 (strong, closed), Mistral Large 3 (strong, open), Claude Haiku
5.5 (mid), the target itself (*self*), the target without any failure feedback (*control*), and the next larger
model of the same family (*larger*: Gemma 27B for 4B/12B, Ministral 14B for 3B/8B). No weights change anywhere.

**Loop** (protocol v3, `scripts/run_opt_loop.py`): score the seed prompt on 200 dev items → show the optimizer 10
failures from a separate pool → it writes a diagnosis and 4 rewrites → rewrites pass rule checks and an intent judge
→ the best on dev replaces the current prompt → at least 2, at most 4 rounds, stop after 2 rounds without improvement
→ the final prompt is scored once on 500 held-out test items. Every condition gets the same budget. 3 runs per
condition (core), 2 (replication), 1 (control).

**Tasks**: AG News (topic classification; its "science" label secretly means "Sci/Tech", a known flaw an optimizer
can discover), SQuAD 2.0 (reading comprehension including unanswerable questions), and TriviaQA closed-book as a
**control**: a prompt cannot add facts, so large gains there would mean format gaming or leakage.

**Statistics**: paired comparisons on identical test items; two-level bootstrap (runs, then items), B = 10,000,
fixed seed; exact McNemar; Holm correction. Method: `docs/statistics.md`. Full design and pre-registered hypotheses:
`docs/design_prompt_optimization.md`.

## Main results

*Not yet run.* The grid is 153 loops: 102 core (Gemma × AG News + SQuAD 2.0 × 6 conditions × 3 runs), 34
replication (Ministral × AG News × 2 runs), 17 control (Gemma × TriviaQA). Results will be reported with and
without final prompts flagged by the intent judge (sensitivity analysis, see "Keeping rewrites honest").

## Pilot (exploratory, not part of the results)

One run per condition, AG News, Gemma 3 12B, protocol v2, no intent judge (`results/opt_pilot/round2/`). Seed test
accuracy 80.4%; a hand-written oracle prompt that fixes the Sci/Tech definition scores 84.0%.

| Optimizer | Test vs seed [95% CI] | Diagnosed the tech→business pattern | Fixed it |
|---|---|---|---|
| self (Gemma 12B) | +0.0 | yes | no |
| control | +0.0 | (no feedback) | no |
| larger (Gemma 27B) | −1.8 [−4.0, +0.4] | yes | no |
| Mistral Large 3 | +7.4 [+4.0, +10.8] | yes | yes |
| Claude Haiku 5.5 | +5.4 [+1.6, +9.2] | yes | yes |
| Claude Opus 5.5 | +7.8 [+4.0, +11.8] | yes | yes |

The small models *diagnosed* the failure correctly and still did not *act* on it; the strong optimizers did both.
Opus's winning prompt reassigned technology companies' business news to "science", which the intent judge and the
project owner later ruled an intent change; with the judge on, that rewrite is rejected.

## Choosing tasks: saturated benchmarks

Seed-prompt dev accuracy (`scripts/check_headroom.py`; dev items only, no test data):

| Benchmark | Gemma 4B / 12B / 27B | Ministral 3B / 8B / 14B | Kept? |
|---|---|---|---|
| GSM8K | 87.0 / 95.0 / 95.5 | 96.0 / 95.0 / 94.0 | no (saturated) |
| BBH tracking shuffled objects | 96.0 / 100 / 98.7 | 86.7 / 100 / 100 | no |
| BBH logical deduction | 62.7 / 82.7 / 97.3 | 86.7 / 90.7 / 92.0 | no |
| BBH web of lies | 100 / 100 / 100 | 52.0 / 96.0 / 89.3 | no |
| BBH date understanding | 82.7 / 93.3 / 93.3 | 69.3 / 78.7 / 86.7 | no |
| BBH causal judgement | 50.7 / 60.0 / 61.3 | 62.7 / 72.0 / 68.0 | no (62 test items, noisy labels) |
| SQuAD 2.0 (F1) | 49.4 / 64.7 / 57.6 | (not used) | **yes** |
| TriviaQA (exact match) | 51.5 / 67.5 / 74.0 | (not used) | **yes (control)** |

A 3B model at 96% on GSM8K suggests the benchmark is in its training data. And with a generic seed that did not
spell out the yes/no answer format, Gemma 3 4B scored **1%** on web of lies; with `Answer: Yes / No` it scored
**100%**. For these models, much of the apparent reasoning difficulty was output-format compliance.

## Keeping rewrites honest

A rewrite may change *how* the task is explained, never *what* the task is (`harness/intent.py`, task cards in
`prompts/task_cards/`):

1. **Rule checks** (reject): placeholders intact, all labels present, length cap (larger of 3× seed and 1,000
   characters), no worked examples, no 8-word overlap with any dataset item.
2. **Intent judge** (Claude Sonnet 5.5, not used as an optimizer): rejects rewrites it judges to change the task.
3. **Measured**: lexical novelty, prompt length, violation rates per optimizer.

**The judge did not pass its pre-registered validation** (85% agreement, kappa 0.70). Two blind labelling passes by
the project owner gave 68.4% (kappa 0.22) and 75.5% (kappa 0.36). On 12 constructed rewrites that change the intent
by design, the judge flagged 12/12 and the human labels 5/12 in both passes; the labeller's notes suggest the labels
judged prompt completeness rather than intent (`results/judge_validation/`). The owner kept the judge on as a
documented amendment made before any grid result existed, with safeguards: judge-rejected rewrites are still scored
("shadow scoring") so we can see when the judge blocks a would-be winner, the 13 disputed pilot rewrites were
adjudicated (owner agreed with the judge on 9), and results are reported with and without judge-flagged prompts.

## Phase 1: the original harness, re-scored with confidence intervals

The project started as a Bedrock eval harness (Llama 3 8B vs Mistral 7B on SST-2, AG News, TriviaQA, CNN/DailyMail).
Re-scoring its stored outputs with strict parsers and 95% bootstrap CIs (`python analysis/rescore_and_ci.py` →
`results/summary_ci.md`) changed the picture:

| SST-2 (n=100) | Originally reported | Strict re-score [95% CI] | "Neutral" answers |
|---|---|---|---|
| Llama 3 8B zero-shot | 93.0 | 93.0 [88.0, 98.0] | 3% |
| Mistral 7B zero-shot | 82.0 | 72.0 [63.0, 81.0] | 24% |
| Mistral 7B few-shot | 65.0 | 53.0 [43.0, 63.0] | 44% |
| Llama 3 8B chain-of-thought | 42.0 | 0.0 (all truncated at 50 tokens) | 0% |

* The original parser scored a hedge followed by hallucinated examples ("neutral … Sentiment: positive") as
  *positive*, inflating Mistral by 10-14 points. When Mistral commits to a label it is right 96-97% of the time.
* All chain-of-thought numbers were artifacts of a 50-token limit.
* Repeated runs barely move: in 17 of 23 configurations stored in DynamoDB the score was identical across 2-5 runs
  (max range 6 points; `results/run_determinism.csv`), so uncertainty comes from item sampling, not reruns.
* The original headline, a ~12-point drop from "neutral-label prompt bias", is real in the stored data (81% vs 93%
  for Llama 3 8B on the same 100 items, fully explained by 17 "neutral" answers) but **cannot be reproduced**: the
  prompt that produced the 81% run was never recorded.

## Design decisions

* **Bedrock** for the target and open optimizer models: one API for Gemma, Ministral and Mistral Large 3, with
  on-demand pricing at cents per loop. Claude is called through the Anthropic API.
* **Instruction-only prompts**: no few-shot examples, so a rewrite cannot smuggle answers in through examples.
* **Oracle prompts**, never shown to optimizers, give a human-expert reference point.
* **Shared cache** keyed by task, model, exact prompt and token budget: identical prompts are scored once and
  interrupted runs resume.

## Limitations

* Pilot numbers are single runs on one task and target; only the grid results should be cited.
* The intent judge is unvalidated against reliable human labels (see above).
* AG News, SQuAD and TriviaQA are public; models may have seen them (Opus wrote AG News's own class name
  "sci/tech" unprompted).
* Test sets are the first-N or seeded samples of public splits; CIs assume they are representative.
* Two model families only; results may not hold for other families or sizes.

## A documented failure

**A cache-key collision silently corrupted results.** The per-item cache was keyed by model, prompt and item id.
All five BBH sub-tasks shared one seed prompt and overlapping ids (`test:31`), so four sub-tasks reused outputs to
*different questions*. It surfaced only because a debugging script crashed on an unexpected id. The key now includes
the task and token budget, rows record their task, and a mismatch raises on load (commit `8804ec2e`); the BBH numbers
were discarded and re-run. A second bug in the same week, a syntax error committed because no test imported the
module, led to tests that import and exercise every registered task.

## Reproducing

```
pip install -r requirements.txt
python analysis/rescore_and_ci.py                     # Phase 1 re-scoring, no model calls
python scripts/run_grid.py --dry-run                  # the 153-loop plan
python scripts/run_grid.py --intent-judge --workers 4 # needs AWS (Bedrock) and ANTHROPIC_API_KEY
python -m pytest
```

Behind a TLS-intercepting proxy or antivirus, set `EVAL_CA_BUNDLE` to a PEM bundle that includes its root CA.
