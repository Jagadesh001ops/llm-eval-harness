"""Optimizers propose rewritten prompts (design doc §5, step 3).

Every optimizer receives the same instruction text; the control condition receives the same text without failures
or scores. Backends: open models on Bedrock (Converse, default sampling) and Claude via the Anthropic API.
"""

import re

import config
from harness.bedrock import compute_cost, invoke_model

N_CANDIDATES = 4

RULES = """Rules for every prompt you write:
1. Keep the input placeholder {placeholders} exactly once, and add no other placeholders.
2. The model must still answer using only these labels/answer format: {labels}. Do not add, remove or rename labels or answer options.
3. Keep the task exactly the same. You may rephrase, reorganise and clarify the instructions, but do not change what is being asked.
4. Do not add facts or knowledge about specific examples, people, companies or events, and do not copy text from the examples below.
5. Do not include worked examples (no example inputs or answers).
6. Keep each prompt under {max_chars} characters."""

FEEDBACK_TEMPLATE = """You are improving an instruction prompt for a small language model (the "target model").

Task: {description}
Input: {input}
Output contract: {output_contract}

{rules}

Current prompt:
<current_prompt>
{current}
</current_prompt>

Prompts tried so far and their accuracy on a development set (higher is better):
{history}

Examples the target model currently gets wrong with the current prompt:
{failures}

Write {n} different improved prompts that would help the target model get more of these cases right without breaking the rules.
Return each prompt inside its own <prompt></prompt> tags and nothing else."""

CONTROL_TEMPLATE = """You are improving an instruction prompt for a small language model (the "target model").

Task: {description}
Input: {input}
Output contract: {output_contract}

{rules}

Current prompt:
<current_prompt>
{current}
</current_prompt>

Write {n} different improved versions of this prompt that would help the target model perform the task better without breaking the rules.
Return each prompt inside its own <prompt></prompt> tags and nothing else."""

_TAG = re.compile(r"<prompt>\s*(.*?)\s*(?:</prompt>|$)", re.S | re.I)


def parse_candidates(text, n=N_CANDIDATES):
    """Extract up to n prompts from <prompt> tags (a final unclosed tag is accepted). No tags -> no candidates:
    an optimizer that cannot follow the output format gets no free pass (counted as unparseable)."""
    found = [c.strip() for c in _TAG.findall(text or "") if c.strip()]
    return found[:n]


def build_request(card, current, history, failures, max_chars, feedback=True, n=N_CANDIDATES):
    view = card["optimizer_view"]
    rules = RULES.format(placeholders=", ".join(view["placeholders"]), labels=", ".join(view["labels"]),
                         max_chars=max_chars)
    common = dict(description=view["description"], input=view["input"], output_contract=view["output_contract"].strip(),
                  rules=rules, current=current, n=n)
    if not feedback:
        return CONTROL_TEMPLATE.format(**common)
    hist = "\n".join(f"- accuracy {acc:.3f}:\n<tried_prompt>\n{p}\n</tried_prompt>" for p, acc in history) or "(none yet)"
    fails = "\n".join(
        f"<example>\n<input>{f['input']}</input>\n<model_output>{f['output']}</model_output>\n"
        f"<correct_answer>{f['gold']}</correct_answer>\n</example>" for f in failures) or "(none)"
    return FEEDBACK_TEMPLATE.format(history=hist, failures=fails, **common)


class BedrockOptimizer:
    """Open model on Bedrock as optimizer, at its default sampling (parity with Claude, which takes no temperature)."""

    def __init__(self, model_key, max_tokens=3000):
        self.model_key, self.max_tokens = model_key, max_tokens

    def propose(self, request):
        r = invoke_model(self.model_key, request, max_tokens=self.max_tokens, temperature=None)
        return {"raw": r["output"], "candidates": parse_candidates(r["output"]), "stop_reason": r["stop_reason"],
                "input_tokens": r["input_tokens"], "output_tokens": r["output_tokens"],
                "cost_usd": compute_cost(r["input_tokens"], r["output_tokens"], self.model_key)}


class ClaudeOptimizer:
    """Claude via the Anthropic API. Refusal fallbacks are deliberately NOT enabled: a fallback model answering in
    Opus's place would contaminate the condition. A refusal yields zero candidates and is logged."""

    def __init__(self, key, effort=None, max_tokens=16000):
        from harness.net import anthropic_client
        self.key, self.effort, self.max_tokens = key, effort, max_tokens
        self.spec = config.CLAUDE_MODELS[key]
        self.client = anthropic_client(max_retries=4)

    def propose(self, request):
        kwargs = dict(model=self.spec["model_id"], max_tokens=self.max_tokens,
                      messages=[{"role": "user", "content": request}])
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        r = self.client.messages.create(**kwargs)
        text = "" if r.stop_reason == "refusal" else " ".join(b.text for b in r.content if b.type == "text")
        u = r.usage
        cost = (u.input_tokens / 1000) * self.spec["price_per_1k_input_tokens"] + \
               (u.output_tokens / 1000) * self.spec["price_per_1k_output_tokens"]
        return {"raw": text, "candidates": parse_candidates(text), "stop_reason": r.stop_reason,
                "input_tokens": u.input_tokens, "output_tokens": u.output_tokens, "cost_usd": cost,
                "request_id": getattr(r, "_request_id", None)}


def make_optimizer(name, effort=None):
    if name in config.CLAUDE_MODELS:
        return ClaudeOptimizer(name, effort=effort)
    if name in config.MODELS:
        return BedrockOptimizer(name)
    raise ValueError(f"unknown optimizer {name}")

