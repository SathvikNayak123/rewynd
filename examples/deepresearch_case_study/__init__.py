"""Exposes the DeepResearch replay harness as an `--entrypoints` target, so the MCP server can be
pointed at the case-study trace store the same way it's pointed at the demo agent:

    python -m trace_replay.mcp \
      --db examples/deepresearch_case_study/case_study_traces.db \
      --entrypoints examples.deepresearch_case_study:ENTRYPOINTS

The agent-name key must match the `agent_name` on the imported traces (`import_run.py` sets
"deepresearch-worker"), or `list_replayable_traces` reports the traces as not replayable and
`replay_from_step` can't find a harness for them.

Built eagerly, which is safe: `build_entrypoint` only closes over the corpus path. The corpus file
is read, and credentials are checked, inside the returned entrypoint when a replay actually runs —
so importing this module costs nothing and a server started for read-only calls
(`list_replayable_traces`, `diff_runs`) needs neither the corpus nor an API key.
"""

from __future__ import annotations

import os

from examples.deepresearch_case_study.replay_harness import build_entrypoint

AGENT_NAME = "deepresearch-worker"
CORPUS_PATH = os.environ.get(
    "DEEPRESEARCH_CORPUS",
    r"C:\Users\user\Desktop\DeepResearch\data\corpus\musique\2hop__635544_110949.json",
)

ENTRYPOINTS = {AGENT_NAME: build_entrypoint(CORPUS_PATH)}
