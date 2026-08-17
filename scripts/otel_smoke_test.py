"""One-off connectivity smoke test for Langfuse Cloud — Phase C.1, step 5.

Confirms LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST in .env are correct
and that a trace actually reaches Langfuse Cloud, before any real instrumentation code
is written. No application code is touched — this is deliberately standalone (see
docs/plans/phase-c/01-setup-and-foundations.md: separate plumbing problems from
instrumentation-logic problems). Delete this file once it's passed once.

    uv run python scripts/otel_smoke_test.py
"""

import os
import sys

from dotenv import load_dotenv
from langfuse import Langfuse

load_dotenv()

_REQUIRED_VARS = ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"]


def main() -> None:
    missing = [k for k in _REQUIRED_VARS if not os.environ.get(k)]
    if missing:
        print(f"Missing from .env: {', '.join(missing)}")
        sys.exit(1)

    host = os.environ["LANGFUSE_HOST"]
    client = Langfuse(
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        host=host,
    )

    print(f"Checking credentials against {host} ...")
    if not client.auth_check():
        print(
            "Auth check FAILED — check LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY/"
            "LANGFUSE_HOST in .env (host must match the project's region, e.g. "
            "https://cloud.langfuse.com or https://us.cloud.langfuse.com)"
        )
        sys.exit(1)
    print("Auth check passed.")

    with client.start_as_current_observation(
        name="otel-smoke-test",
        as_type="span",
        input="phase C.1 connectivity check",
        metadata={"purpose": "phase-c-c1-connectivity-check"},
    ) as span:
        span.update(output="smoke test span created successfully")
        trace_id = client.get_current_trace_id()

    # Spans are exported asynchronously in the background by default — flush() blocks
    # until the pending batch is actually sent, which a one-shot script needs, or it can
    # exit before the trace ever leaves the process.
    client.flush()

    print("\nTrace sent and flushed.")
    url = client.get_trace_url(trace_id=trace_id) if trace_id else None
    if url:
        print(f"View it here: {url}")
    else:
        print(f"Trace ID: {trace_id!r} — open the Langfuse Cloud UI and search for it.")


if __name__ == "__main__":
    main()
