"""Without-judge sensitivity (design doc §6.3 amendment).

The intent judge ran inside every loop, so the loops cannot be replayed without it (later rounds depend on which
prompt each round adopted). What the logs do support is a one-step approximation: candidates that passed the rule
checks but were rejected only by the judge were still scored on dev ("shadow scoring"). If the loop's final prompt
had instead been the best-dev prompt among ALL rule-valid candidates, judge-rejected ones included, which prompt
would it be, and what would it score on test?

This is an upper-bound style approximation, not a replay: the candidate pool comes from the judge-on trajectory.
"""


def candidate_pool(log):
    """(prompt, dev_accuracy, judge_rejected) for every rule-valid candidate that has a dev score."""
    pool = []
    for r in log["rounds"]:
        for c in r["candidates"]:
            if "dev_accuracy" in c:
                pool.append((c["prompt"], c["dev_accuracy"], False))
            elif "shadow_dev_accuracy" in c:
                pool.append((c["prompt"], c["shadow_dev_accuracy"], True))
    return pool


def nojudge_pick(log):
    """The judge-rejected prompt with the highest dev score if it beats the loop's final dev score, else None.
    Ties keep the judge-on final prompt. Judge-accepted candidates cannot beat the final prompt on dev: the loop
    adopts any such candidate, so only shadow candidates can be returned."""
    best = None
    for prompt, dev, shadow in candidate_pool(log):
        if dev > log["final_dev_accuracy"] and (best is None or dev > best[1]):
            best = (prompt, dev, shadow)
    return best
