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
}
N_POOL, N_DEV, N_TEST = 500, 200, 500


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


if __name__ == "__main__":
    configure_huggingface()
    for t in sys.argv[1:] or TASKS:
        make(t)
