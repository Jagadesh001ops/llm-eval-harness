"""Bedrock model invocation and cost accounting, shared by every eval script."""

import time

from botocore.exceptions import ClientError

from config import AWS_REGION, MODELS

_client = None

# Bedrock error codes meaning "this model can't be called from this account right now"
# (e.g. a Legacy model the account hasn't used recently). A run skips the model instead of crashing.
_UNAVAILABLE_CODES = {"ResourceNotFoundException", "AccessDeniedException"}


class ModelUnavailable(RuntimeError):
    pass


def _get_client():
    global _client
    if _client is None:
        import boto3
        _client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    return _client


def invoke_model(model_key, prompt, max_tokens=50, temperature=0, retries=3, backoff_s=(2, 5, 10)):
    """One Converse call. Bedrock's Legacy-model gate refuses calls intermittently (observed 2026-11-18
    for llama3-8b and llama3-70b), so unavailability errors are retried before ModelUnavailable is raised.
    Latency is measured for the successful attempt only."""
    model_id = MODELS[model_key]["model_id"]
    for attempt in range(retries + 1):
        start = time.time()
        try:
            response = _get_client().converse(
                modelId=model_id,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
            )
            break
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code not in _UNAVAILABLE_CODES:
                raise
            if attempt == retries:
                raise ModelUnavailable(f"{model_key} ({model_id}): {e}") from e
            time.sleep(backoff_s[min(attempt, len(backoff_s) - 1)])
    latency = time.time() - start
    return {
        "output": response["output"]["message"]["content"][0]["text"].strip(),
        "input_tokens": response["usage"]["inputTokens"],
        "output_tokens": response["usage"]["outputTokens"],
        "stop_reason": response.get("stopReason", ""),
        "latency_ms": round(latency * 1000),
    }


def compute_cost(input_tokens, output_tokens, model_key):
    m = MODELS[model_key]
    return (input_tokens / 1000) * m["price_per_1k_input_tokens"] + \
           (output_tokens / 1000) * m["price_per_1k_output_tokens"]
