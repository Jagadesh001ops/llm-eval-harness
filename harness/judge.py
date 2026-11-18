"""Intent judge (design doc §6.3, layer 2): does a rewrite keep the task's intent as defined by the task card?

The judge is a model that is not used as an optimizer (Claude Sonnet 5.5), so no optimizer judges its own work. It
sees what the human labeller saw: the task card (optimizer and judge views), the original prompt and the rewrite.
It must not be used to reject anything until validated against human labels (scripts/validate_judge.py).

Verdicts are parsed from tags rather than free text; a missing or malformed verdict is recorded as 'error', never
guessed. Refusal fallbacks are not enabled (a different model answering would change the judge).
"""

import re

import yaml

import config

JUDGE_MODEL = "claude-sonnet-5.5"

INSTRUCTIONS = """You check whether a rewritten instruction prompt keeps the intent of a classification or question-answering task.

The task card below defines the task. A rewrite KEEPS the intent if it only changes how the task is explained: rephrasing, reordering, clarifying, defining the existing labels in line with the dataset's label definitions, output-format instructions consistent with the output contract, or general strategies.
A rewrite CHANGES the intent if it does any of the following: adds, removes or renames labels or answer options; changes what is being asked or the required answer format; gives label definitions that contradict the dataset's label definitions; adds rules tied to specific companies, people, products or events; adds facts or claims meant to steer answers; includes worked examples or text copied from dataset items.
Judge only intent, not quality: a clumsy or weaker prompt that keeps the task the same still KEEPS the intent.

<task_card>
{card}
</task_card>

<original_prompt>
{seed}
</original_prompt>

<rewritten_prompt>
{rewrite}
</rewritten_prompt>

Answer in exactly this form:
<reason>one or two sentences naming what, if anything, changes the intent</reason>
<verdict>keeps</verdict> or <verdict>changes</verdict>"""

_VERDICT = re.compile(r"<verdict>\s*(keeps|changes)\s*</verdict>", re.I)
_REASON = re.compile(r"<reason>\s*(.*?)\s*</reason>", re.S | re.I)


def build_judge_request(card, seed, rewrite):
    card_text = yaml.safe_dump({"optimizer_view": card["optimizer_view"], "judge_view": card["judge_view"]},
                               sort_keys=False, allow_unicode=True)
    return INSTRUCTIONS.format(card=card_text.strip(), seed=seed, rewrite=rewrite)


def parse_verdict(text):
    m = _VERDICT.findall(text or "")
    verdict = m[-1].lower() if len(set(v.lower() for v in m)) == 1 else "error"  # conflicting verdicts -> error
    r = _REASON.search(text or "")
    return verdict, (r.group(1).strip() if r else None)


class IntentJudge:
    def __init__(self, key=JUDGE_MODEL, max_tokens=4000):
        from harness.net import anthropic_client
        self.spec = config.CLAUDE_MODELS[key]
        self.key, self.max_tokens = key, max_tokens
        self.client = anthropic_client(max_retries=4)

    def judge(self, card, seed, rewrite):
        r = self.client.messages.create(model=self.spec["model_id"], max_tokens=self.max_tokens,
                                        messages=[{"role": "user", "content": build_judge_request(card, seed, rewrite)}])
        text = "" if r.stop_reason == "refusal" else " ".join(b.text for b in r.content if b.type == "text")
        verdict, reason = parse_verdict(text)
        u = r.usage
        cost = (u.input_tokens / 1000) * self.spec["price_per_1k_input_tokens"] + \
               (u.output_tokens / 1000) * self.spec["price_per_1k_output_tokens"]
        return {"verdict": verdict, "reason": reason, "stop_reason": r.stop_reason, "raw": text,
                "input_tokens": u.input_tokens, "output_tokens": u.output_tokens, "cost_usd": cost,
                "request_id": getattr(r, "_request_id", None)}
