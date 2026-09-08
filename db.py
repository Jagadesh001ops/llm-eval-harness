import boto3
import uuid
from datetime import datetime, timezone
from config import AWS_REGION

dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
table = dynamodb.Table("llm-eval-results")

def write_result(run_id, model, task, prompt_variant, example_id, metrics: dict):
    item = {
        "run_id": run_id,
        "result_id": f"{model}#{task}#{prompt_variant}#{example_id}",
        "model": model,
        "task": task,
        "prompt_variant": prompt_variant,
        "example_id": str(example_id),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **{k: str(v) if isinstance(v, float) else v for k, v in metrics.items()}
    }
    table.put_item(Item=item)

def get_run_results(run_id):
    response = table.query(
        KeyConditionExpression=boto3.dynamodb.conditions.Key("run_id").eq(run_id)
    )
    return response["Items"]