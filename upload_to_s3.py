import boto3
import os
from config import S3_BUCKET, AWS_REGION

s3 = boto3.client("s3", region_name=AWS_REGION)

def upload_file(local_path, s3_key):
    s3.upload_file(local_path, S3_BUCKET, s3_key)
    print(f"  ✓ {local_path} → s3://{S3_BUCKET}/{s3_key}")

print(f"Uploading to s3://{S3_BUCKET}/llm-eval-harness/\n")

# CSVs
upload_file("results/all_models_all_prompts.csv",  "llm-eval-harness/results/all_models_all_prompts.csv")
upload_file("results/summarization_results.csv",    "llm-eval-harness/results/summarization_results.csv")
upload_file("results/boss_model_transfer.csv",      "llm-eval-harness/results/boss_model_transfer.csv")
upload_file("results/boss_model_summary.json",      "llm-eval-harness/results/boss_model_summary.json")

# Charts
for fname in os.listdir("results/plots"):
    if fname.endswith(".png"):
        upload_file(f"results/plots/{fname}", f"llm-eval-harness/plots/{fname}")

print("\nDone. Everything is in S3.")