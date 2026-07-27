# trace-replay

Resume a recorded agent run from any step with a different prompt, model, or param — and see
what changes — without re-running the whole thing from scratch. An MCP server built on
[ctx-capture](https://github.com/SathvikNayak123/ctx-capture)'s trace schema; it doesn't define its own trace
format, it consumes ctx-capture's.

## What it is

Debugging an agent normally means re-running it end to end and hoping it fails the same way
again — slow, expensive, and non-deterministic. trace-replay freezes everything before the step
you're debugging and lets you change one thing (prompt, model, params) and run forward from
there.

**Features:**

- **Resume from any step**, with everything before it held byte-identical to the source trace —
  not re-run, not re-derived, just copied.
- **Three modes per tool call**: `MOCK` (default — serve the recorded result, free and
  deterministic), `LIVE` (actually execute it), `STUB` (serve a value you supply). Configurable
  globally, per tool, or per step.
- **Divergence handling** — if a replayed tool call doesn't match anything in the recording (the
  agent asked something new), the default is fail-fast: the replay halts and tells you exactly
  where, instead of drifting silently into an unmocked, possibly costly tail.
- **Mutation safety** — a tool never goes LIVE just because the run's default is LIVE; it needs an
  explicit, named override. Unlabeled tools are treated as mutating, not safe.
- **Replay cost accounting** — every replay reports its own cost against a comparable estimate for
  the original run.
- **Replay-of-replay** — a replay's output is just another trace, so resuming from a replay works
  with no special case.
- **5 MCP tools**: `list_replayable_traces`, `replay_from_step`, `get_replay_status`, `diff_runs`,
  `set_tool_mode` — stdio and streamable-HTTP transports, bearer auth on HTTP.

Full design rationale: [docs/DESIGN.md](docs/DESIGN.md). Real bug found with this tool, isolated
in one $0.01 replay instead of a $0.08 full re-run: [docs/CASE_STUDY.md](docs/CASE_STUDY.md).

## Setup

**Requires [ctx-capture](https://github.com/SathvikNayak123/ctx-capture)** — not on PyPI yet, so install both
from source:

```bash
git clone https://github.com/SathvikNayak123/ctx-capture && pip install -e ./ctx-capture
git clone https://github.com/SathvikNayak123/trace-replay && pip install -e "./trace-replay[dev]"
```

Your agent needs to already be instrumented with ctx-capture, and needs one small adapter — see
"Integration contract" in [docs/DESIGN.md](docs/DESIGN.md) — before trace-replay can resume it.

## How to use

Seed one real trace with the included demo agent (a tiny trip-planner that searches then
answers — see [examples/demo_agent.py](examples/demo_agent.py)):

```bash
cd trace-replay
python examples/seed_demo_trace.py
# trace_id = c740c832-bb5f-402b-8223-a5c9388bc69f
```

Run the MCP server, pointed at that trace DB and the demo agent's harness entrypoint:

```bash
python -m trace_replay.mcp \
  --db examples/demo_ctx_capture.db \
  --entrypoints examples.demo_agent:ENTRYPOINTS
```

Point an MCP client at it (`.mcp.json` in this repo already does, for Claude Code/Desktop) and
call `replay_from_step`:

```json
{"trace_id": "c740c832-...", "step_index": 2, "overrides": {"system_prompt": "Prefer budget-friendly trails."}}
```

Steps 0-1 come back byte-identical to the original; step 2 onward re-runs live with the new
prompt — the original answer was "Recommend Mount Elbert for an experienced hiker; permit not
required," the replayed one is "Recommend Bear Lake Trailhead (free, no permit) for a
budget-conscious hiker." Call `diff_runs` on the two trace ids to see that outcome change plus the
token/cost delta directly. `uvx trace-replay` will work the same way once published to PyPI.

## Limitations

- **Single-agent, linear, Python only** — no multi-agent replay, no branching UI, no non-Python
  agents in v1.
- **Divergence explosion on early-step replays** — the earlier you resume, the more of the run is
  live, and the more likely the agent asks something the recording never saw. Late-step replays
  (the common case) don't have this problem.
- **Synchronous in v1** — `replay_from_step` runs to completion before returning; no background
  job runner yet.
- **No time-travel for external world state** — a LIVE tool call hits the real, current world, not
  a reconstruction of what it looked like during the original run.
- **`set_tool_mode` is config-only** — it can't reach into an already-finished synchronous run to
  change its behavior after the fact.
- **No Postgres-backed replay-run store yet** — SQLite only.

Full list with reasoning: [CHANGELOG.md](CHANGELOG.md) and
[docs/DESIGN.md § Non-goals](docs/DESIGN.md).

## License

MIT — see [LICENSE](LICENSE).
