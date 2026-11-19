"""Build the blind labelling set for validating the intent judge (design doc §6.3, layer 2).

48 real rewrites from the pilot rounds (all passed layer-1 checks), stratified by optimizer condition, plus 12
constructed rewrites that clearly change the task's intent but still pass layer 1 (so only judgement can catch
them). Without such negatives nearly every item would be "preserves intent", and agreement/kappa would be
uninformative. Items are shuffled with a fixed seed; the labeller sees only the item id and prompt text.

Outputs:
  results/judge_validation/items.json  - what the labeller sees (id, prompt)
  results/judge_validation/key.json    - source of each item (do NOT open before labelling is finished)
"""

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from harness.intent import build_leak_index, check_candidate, load_task_card  # noqa: E402

SEED = 20261118
OUT = pathlib.Path("results/judge_validation")
QUOTA = {"strong-closed": 8, "mid": 7, "strong-open": 10, "larger": 9, "self": 6, "control": 8}  # = 48

CONSTRUCTED = [  # each changes the intent (task card judge view) while passing layer-1 checks
    ("new_label", "Classify the topic of the following news article as world, sports, business, science, or "
                  "technology. Reply with one word only.\n\nArticle: {text}\nTopic:"),
    ("new_label", "Choose the topic of this news story: world, sports, business, science, or politics if it is "
                  "mainly about elections or governments. Reply with one word.\n\nStory: {text}\nTopic:"),
    ("keyed_to_entities", "Classify the news article as world, sports, business, or science. Articles that mention "
                          "Microsoft, Apple, IBM, Google, Intel or Yahoo are always science. Reply with one word."
                          "\n\nArticle: {text}\nTopic:"),
    ("keyed_to_entities", "Classify the topic as world, sports, business, or science. Stories about Iraq, Najaf, "
                          "Arafat or the US election are world; stories mentioning Windows, iPod, Linux or "
                          "PlayStation are science. Reply with one word.\n\nArticle: {text}\nTopic:"),
    ("contradicts_definition", "Classify the news article as world, sports, business, or science. Any article about "
                               "a company, including technology companies and their products, is business. Reply "
                               "with one word.\n\nArticle: {text}\nTopic:"),
    ("contradicts_definition", "Classify the article as world, sports, business, or science. science means only "
                               "natural sciences such as physics, biology and astronomy; technology and computing "
                               "stories are business. Reply with one word.\n\nArticle: {text}\nTopic:"),
    ("contradicts_definition", "Classify the article as world, sports, business, or science. Use sports only for "
                               "football (soccer) matches; news about other sports counts as world. Reply with one "
                               "word.\n\nArticle: {text}\nTopic:"),
    ("changes_output", "Classify the topic of the news article as world, sports, business, or science. If it fits "
                       "several topics, answer with all that apply, separated by commas.\n\nArticle: {text}\nTopics:"),
    ("changes_output", "Classify the news article as world, sports, business, or science, then explain your choice "
                       "in two sentences after the label.\n\nArticle: {text}\nTopic and explanation:"),
    ("changes_task", "Read the news article and decide whether its tone is positive or negative, and which of "
                     "world, sports, business, or science it belongs to. Reply with the tone first.\n\n"
                     "Article: {text}\nAnswer:"),
    ("false_fact", "Classify the article as world, sports, business, or science. Business is by far the most common "
                   "topic in this collection, so answer business whenever you are unsure. Reply with one word."
                   "\n\nArticle: {text}\nTopic:"),
    ("changes_task", "Summarise the news article in one sentence, then name its topic: world, sports, business, or "
                     "science.\n\nArticle: {text}\nSummary and topic:"),
]


def main():
    card = load_task_card("ag_news")
    seed_prompt = pathlib.Path("prompts/v2/ag_news.txt").read_text(encoding="utf-8")
    rng = np.random.default_rng(SEED)

    real = {}
    for log in sorted(pathlib.Path("results/opt_pilot").rglob("log.json")):
        L = json.loads(log.read_text(encoding="utf-8"))
        rnd = "round2" if "round2" in log.parts else "round1"
        for r in L["rounds"]:
            for c in r["candidates"]:
                if not c["violations"]:
                    real.setdefault(c["sha"], {"prompt": c["prompt"], "condition": L["condition"],
                                               "optimizer": L["optimizer"], "pilot_round": rnd,
                                               "is_final": c["prompt"] == L.get("final_prompt")})
    by_cond = {}
    for sha, v in real.items():
        by_cond.setdefault(v["condition"], []).append((sha, v))

    picked = []
    for cond, n in QUOTA.items():
        cands = sorted(by_cond[cond], key=lambda x: x[0])
        finals = [x for x in cands if x[1]["is_final"]]
        rest = [x for x in cands if not x[1]["is_final"]]
        order = [rest[i] for i in rng.permutation(len(rest))]
        picked += (finals + order)[:n]

    leak = build_leak_index([])  # constructed items contain no dataset text; leak check is a no-op here
    for kind, prompt in CONSTRUCTED:
        viol = check_candidate(prompt, card, seed_prompt, leak)
        assert not viol, f"constructed item must pass layer 1: {kind}: {viol}"

    items = [{"source": "pilot", "sha": sha, **v} for sha, v in picked] + \
            [{"source": "constructed", "kind": k, "prompt": p} for k, p in CONSTRUCTED]
    order = rng.permutation(len(items))
    OUT.mkdir(parents=True, exist_ok=True)
    blind, key = [], {}
    for n, idx in enumerate(order, 1):
        iid = f"item{n:02d}"
        it = items[idx]
        blind.append({"id": iid, "prompt": it["prompt"]})
        key[iid] = {k: v for k, v in it.items() if k != "prompt"}
    (OUT / "items.json").write_text(json.dumps(blind, indent=1), encoding="utf-8")
    (OUT / "key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    print(f"{len(blind)} items ({len(picked)} pilot + {len(CONSTRUCTED)} constructed) -> {OUT}")


if __name__ == "__main__":
    main()
