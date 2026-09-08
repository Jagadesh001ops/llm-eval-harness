import boto3
from datetime import datetime, timezone
from config import AWS_REGION

cloudwatch = boto3.client("cloudwatch", region_name=AWS_REGION)

NAMESPACE = "LLMEvalHarness"

def put_invocation_metrics(model, task, prompt_variant, latency_ms, cost_usd, tokens_in, tokens_out):
    dimensions = [
        {"Name": "Model", "Value": model},
        {"Name": "Task", "Value": task},
        {"Name": "PromptVariant", "Value": prompt_variant},
    ]
    now = datetime.now(timezone.utc)

    cloudwatch.put_metric_data(
        Namespace=NAMESPACE,
        MetricData=[
            {
                "MetricName": "Latency",
                "Dimensions": dimensions,
                "Timestamp": now,
                "Value": latency_ms,
                "Unit": "Milliseconds",
            },
            {
                "MetricName": "CostUSD",
                "Dimensions": dimensions,
                "Timestamp": now,
                "Value": cost_usd,
                "Unit": "None",
            },
            {
                "MetricName": "InputTokens",
                "Dimensions": dimensions,
                "Timestamp": now,
                "Value": tokens_in,
                "Unit": "Count",
            },
            {
                "MetricName": "OutputTokens",
                "Dimensions": dimensions,
                "Timestamp": now,
                "Value": tokens_out,
                "Unit": "Count",
            },
        ],
    )