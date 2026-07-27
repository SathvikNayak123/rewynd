"""Records one real source trace for manual MCP verification. Run once from the repo root:

    python examples/seed_demo_trace.py

Prints the trace_id to replay from — use it as `trace_id` in a `replay_from_step` call from a
real MCP client (Claude Desktop/Code), per docs/DESIGN.md "Manual verification".
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ctx_capture.capture import TraceRecorder
from ctx_capture.storage.sqlite_repo import SQLiteTraceRepository

from examples.demo_agent import AGENT_NAME, DemoClient, build_baseline_responses, record_demo_agent

DB_PATH = os.path.join(os.path.dirname(__file__), "demo_ctx_capture.db")

# Fixed, not random: the demo agent's script is itself fixed (DemoClient always returns the same
# canned responses), so pinning the id too means the README/quickstart trace_id is genuinely
# reproducible from a clean clone instead of a stale example that never matches what this script
# actually prints. `repo.save` is INSERT OR REPLACE, so re-running this script is idempotent.
DEMO_TRACE_ID = "c740c832-bb5f-402b-8223-a5c9388bc69f"


def main() -> None:
    recorder = TraceRecorder(agent_name=AGENT_NAME)
    client = DemoClient(build_baseline_responses())
    capturing_client = recorder.wrap_client(client, provider="demo-provider")
    record_demo_agent(recorder, capturing_client)
    recorder.trace.trace_id = DEMO_TRACE_ID

    repo = SQLiteTraceRepository(DB_PATH)
    repo.save(recorder.trace)

    print(f"Seeded source trace at {DB_PATH}")
    print(f"trace_id = {recorder.trace.trace_id}")
    print(f"steps = {len(recorder.trace.steps)}")
    final_answer = recorder.trace.steps[-1].model_call.response["choices"][0]["message"]["content"]
    print(f"final answer = {final_answer!r}")


if __name__ == "__main__":
    main()
