"""Optional DynamoDB + CloudWatch persistence.

The eval scripts used to write to both services on every call, unconditionally, so they could not run
without AWS infrastructure. A Recorder with enabled=False (the scripts' --no-persist flag) skips both;
the CSV in results/ is always written either way and is the canonical record.
"""


class Recorder:
    def __init__(self, enabled=True):
        self.enabled = enabled

    def record(self, run_id, model, task, prompt_variant, example_id, metrics, result):
        if not self.enabled:
            return
        from db import write_result
        from metrics import put_invocation_metrics
        write_result(run_id, model, task, prompt_variant, example_id, metrics)
        put_invocation_metrics(model, task, prompt_variant, result["latency_ms"], metrics["cost_usd"],
                               result["input_tokens"], result["output_tokens"])


def add_common_args(parser, default_models, default_n):
    parser.add_argument("--models", nargs="+", default=default_models,
                        help=f"model keys from config.MODELS (default: {' '.join(default_models)})")
    parser.add_argument("--n", type=int, default=default_n, help=f"number of examples (default: {default_n})")
    parser.add_argument("--no-persist", action="store_true",
                        help="skip DynamoDB/CloudWatch writes; only write the results CSV")
    parser.add_argument("--out", default=None, help="output CSV path (default: the script's results/ file)")
    return parser
