"""Network setup shared by every client.

Some machines (e.g. with antivirus HTTPS scanning) re-sign TLS traffic with a local root CA that Python's certifi
bundle does not trust. Point EVAL_CA_BUNDLE (or AWS_CA_BUNDLE) at a PEM bundle that includes that root and every
client here uses it: boto3 reads AWS_CA_BUNDLE itself; HuggingFace and Anthropic clients are configured below.
With neither variable set, library defaults are used.
"""

import os


def ca_bundle():
    return os.environ.get("EVAL_CA_BUNDLE") or os.environ.get("AWS_CA_BUNDLE") or None


def configure_huggingface():
    """`datasets` ignores SSL env vars; route its HTTP client through the bundle."""
    bundle = ca_bundle()
    if bundle:
        import httpx
        import huggingface_hub
        huggingface_hub.set_client_factory(lambda: httpx.Client(verify=bundle, follow_redirects=True, timeout=60))


def anthropic_client(**kwargs):
    import anthropic
    bundle = ca_bundle()
    if bundle:
        kwargs.setdefault("http_client", anthropic.DefaultHttpxClient(verify=bundle))
    return anthropic.Anthropic(**kwargs)
