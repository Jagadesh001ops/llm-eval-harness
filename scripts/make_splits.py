"""Create the fixed pool / dev / test splits for the prompt-optimization study (design doc §4).

Writes data/splits/<task>.json with item indices into the named HF split, plus a SHA-256 of each selected item's
text so a later run can detect if the upstream dataset changed. Run once; the outputs are committed and must not be
regenerated (the script refuses to overwrite).

    python scripts/make_splits.py ag_news gsm8k
"""

import hashlib
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from harness.net import configure_huggingface  # noqa: E402

SEED = 20261118
OUT = pathlib.Path("data/splits")

# name -> (hf path, config, pool/dev source split, test source split, text fn, test indices to exclude)
TASKS = {
    "ag_news": ("fancyzhx/ag_news", None, "train", "test", lambda r: r["text"], range(100)),
    "gsm8k": ("openai/gsm8k", "main", "train", "test", lambda r: r["question"], ()),
    "triviaqa": ("mandarjoshi/trivia_qa", "rc.nocontext", "train", "validation", lambda r: r["question"], range(100)),
}
N_POOL, N_DEV, N_TEST = 500, 200, 500

BBH_SUBTASKS = ["date_understanding", "logical_deduction_five_objects", "tracking_shuffled_objects_five_objects",
                "web_of_lies", "causal_judgement"]
BBH_POOL, BBH_DEV = 50, 75  # BBH has one split of ~250 items: pool 50, dev 75, test = the rest


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def make(task):
    from datasets import load_dataset
    path, config, train_split, test_split, text_fn, exclude = TASKS[task]
    out = OUT / f"{task}.json"
    if out.exists():
        raise SystemExit(f"{out} exists; splits are frozen. Delete it deliberately if you really mean to regenerate.")
    rng = np.random.default_rng(SEED)
    train = load_dataset(path, config, split=train_split)
    test = load_dataset(path, config, split=test_split)

    train_idx = rng.permutation(len(train))[: N_POOL + N_DEV]
    pool, dev = sorted(train_idx[:N_POOL].tolist()), sorted(train_idx[N_POOL:].tolist())
    excluded = set(exclude)
    test_candidates = np.array([i for i in range(len(test)) if i not in excluded])
    test_idx = sorted(rng.permutation(test_candidates)[:N_TEST].tolist())

    doc = {
        "task": task, "hf_path": path, "hf_config": config, "seed": SEED,
        "pool": {"split": train_split, "indices": pool, "sha": [sha(text_fn(train[i])) for i in pool]},
        "dev": {"split": train_split, "indices": dev, "sha": [sha(text_fn(train[i])) for i in dev]},
        "test": {"split": test_split, "indices": test_idx, "sha": [sha(text_fn(test[i])) for i in test_idx],
                 "excluded_indices": sorted(excluded)},
    }
    assert not set(pool) & set(dev), "pool and dev overlap"
    assert not set(test_idx) & excluded, "test contains excluded items"
    OUT.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(f"{task}: pool={len(pool)} dev={len(dev)} test={len(test_idx)} -> {out}")


def _write(task, doc):
    out = OUT / f"{task}.json"
    if out.exists():
        raise SystemExit(f"{out} exists; splits are frozen.")
    OUT.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    sizes = {k: len(doc[k]["indices"]) for k in ("pool", "dev", "test")}
    print(f"{task}: {sizes} -> {out}")


def make_bbh(sub):
    from datasets import load_dataset
    ds = load_dataset("lukaemon/bbh", sub, split="test")
    order = np.random.default_rng([SEED, sum(map(ord, sub))]).permutation(len(ds)).tolist()
    parts = {"pool": sorted(order[:BBH_POOL]), "dev": sorted(order[BBH_POOL:BBH_POOL + BBH_DEV]),
             "test": sorted(order[BBH_POOL + BBH_DEV:])}
    doc = {"task": f"bbh_{sub}", "hf_path": "lukaemon/bbh", "hf_config": sub, "seed": SEED}
    for k, idx in parts.items():
        doc[k] = {"split": "test", "indices": idx, "sha": [sha(ds[i]["input"]) for i in idx]}
    _write(f"bbh_{sub}", doc)


def make_squad():
    """Pool and dev from train, test from validation (the official test set is hidden); dev and test are balanced
    50/50 answerable/unanswerable so both behaviours are measured with equal precision."""
    from datasets import load_dataset
    rng = np.random.default_rng([SEED, 2])
    train = load_dataset("rajpurkar/squad_v2", split="train")
    val = load_dataset("rajpurkar/squad_v2", split="validation")

    def balanced(ds, n, exclude=()):
        has = np.array([len(a["text"]) > 0 for a in ds["answers"]])
        ex = set(exclude)
        ans = [i for i in rng.permutation(np.where(has)[0]).tolist() if i not in ex][: n // 2]
        una = [i for i in rng.permutation(np.where(~has)[0]).tolist() if i not in ex][: n // 2]
        return sorted(ans + una)

    dev = balanced(train, N_DEV)
    pool = balanced(train, N_POOL, exclude=dev)
    test = balanced(val, N_TEST)
    assert not set(pool) & set(dev)
    doc = {"task": "squad_v2", "hf_path": "rajpurkar/squad_v2", "hf_config": None, "seed": SEED,
           "pool": {"split": "train", "indices": pool, "sha": [sha(train[i]["question"]) for i in pool]},
           "dev": {"split": "train", "indices": dev, "sha": [sha(train[i]["question"]) for i in dev]},
           "test": {"split": "validation", "indices": test, "sha": [sha(val[i]["question"]) for i in test]}}
    _write("squad_v2", doc)


if __name__ == "__main__":
    configure_huggingface()
    for t in sys.argv[1:] or TASKS:
        if t == "bbh":
            for sub in BBH_SUBTASKS:
                make_bbh(sub)
        elif t == "squad_v2":
            make_squad()
        else:
            make(t)
