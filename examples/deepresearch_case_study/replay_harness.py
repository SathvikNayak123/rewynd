"""The integration-contract harness for the DeepResearch case study (docs/CASE_STUDY.md).

Replays a single research step for real: calls DeepResearch's actual `run_subagent` (real
LocalCorpusBackend, real LLM calls, real small cost) with a different sub-question, to test
whether a more targeted search finds a fact the original run's subagent didn't.

Updated for DeepResearch's post-`ef85170` topology (planner -> supervisor -> subagent -> verify
-> synthesis -> reflection). The previous version of this file called `run_worker`, which no
longer exists; the flat per-sub-question worker was replaced by a LangGraph ReAct subagent. What
that changed here, concretely:

- `deepresearch.agent.worker.run_worker` -> `deepresearch.agent.subagent.run_subagent`, which
  returns a `Finding` (not the old worker notes) and takes several dependencies the old worker
  built internally.
- The subagent needs a LangChain `chat_model` bound to `TOOLS`, because tool calls now go through
  LangGraph's `ToolNode` rather than the httpx `LLMClient`. That path is OpenAI-compatible only,
  so a replay now requires `DEEPRESEARCH_LLM_PROVIDER=openrouter` + `OPENROUTER_API_KEY` — the
  original case study ran against the Anthropic Messages API directly, which this code path no
  longer supports. Checked up front so the failure is a readable message, not a KeyError.
- `recorder` is now a required `RunRecorder`, not an optional `None`. It gets a throwaway
  instance: DeepResearch's own run-store rows aren't what this replay captures (ctx-capture's
  recorder is), and a throwaway keeps the replay from writing into a real run store.

Unchanged, and the reason this harness still works at all: the subagent's `search` tool reaches
the corpus through `AgentContext.search_backend` (via `retrieve.retrieve_chunks`, which calls
`.search()`/`.fetch()` on it). That's still the single seam where MOCK/LIVE/STUB can intercept, so
`_RoutedSearchBackend` below needed no changes.

Real integration wrinkle found while building the original version (still true, still not smoothed
over): `ToolRouter.call()` is synchronous, but DeepResearch's `SearchBackend.search`/`.fetch` are
`async`. Bridging them means the harness can't just hand `ctx.tool_router.call` an async function
as a `live_tool_fns` entry — calling it from inside a coroutine that's already running on an event
loop would try to nest event loops. The fix here: each live call runs in its own thread with a
fresh event loop (`_run_async_in_thread`), so the surrounding coroutine never has to nest loops.
Worth smoothing out with a native async tool_router in a future trace-replay version if async
agents are common.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import time
from pathlib import Path

from ctx_capture.schema import ModelCall, TokenCounts
from trace_replay.harness import ReplayContext

# Every `deepresearch` import in this module is deliberately deferred to call time rather than
# done here. Importing this module has to stay possible without DeepResearch installed, because
# `examples.deepresearch_case_study:ENTRYPOINTS` is imported at MCP server startup — and the
# server's read-only tools (list_replayable_traces, diff_runs, get_replay_status) don't touch
# DeepResearch at all. A module-level import made the server crash on any interpreter that had
# trace-replay but not DeepResearch, which is the normal case for anyone who just cloned this
# repo, and was also what broke it under the IDE's system Python.

# A syntactically valid OTel span id for the throwaway recorder's rows to reference. This replay
# doesn't nest under a real run span (there's no enclosing DeepResearch run) and nothing reads
# these rows back, but run_subagent requires the argument.
_REPLAY_SPAN_ID = "0" * 16


def _run_async_in_thread(coro):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(coro)).result()


class _RoutedSearchBackend:
    """Implements SearchBackend's interface, but every call goes through
    ctx.tool_router.call() first — that's the seam where MOCK/LIVE/STUB and recording happen."""

    def __init__(self, tool_router) -> None:
        self._router = tool_router

    async def search(self, query: str, max_results: int = 5) -> list:
        from deepresearch.schemas import SearchResult

        served = self._router.call("search", {"query": query, "max_results": max_results})
        return [SearchResult(**r) for r in served]

    async def fetch(self, url: str):
        from deepresearch.schemas import FetchResult

        served = self._router.call("fetch", {"url": url})
        return FetchResult(**served)


def _make_live_tool_fns(real_backend) -> dict:
    def live_search(query: str, max_results: int) -> list[dict]:
        results = _run_async_in_thread(real_backend.search(query, max_results=max_results))
        return [r.model_dump() for r in results]

    def live_fetch(url: str) -> dict:
        result = _run_async_in_thread(real_backend.fetch(url))
        return result.model_dump()

    return {"search": live_search, "fetch": live_fetch}


def _require_openrouter() -> None:
    """`build_chat_model` hard-requires the OpenAI-compatible provider and reads
    OPENROUTER_API_KEY straight off os.environ. Fail here, before any real work or spend, with a
    message that says what to set."""
    provider = os.getenv("DEEPRESEARCH_LLM_PROVIDER", "anthropic")
    if provider != "openrouter" or not os.getenv("OPENROUTER_API_KEY"):
        raise SystemExit(
            "This replay runs DeepResearch's ReAct subagent, whose tool-calling path is "
            "OpenAI-compatible only. Set DEEPRESEARCH_LLM_PROVIDER=openrouter and "
            "OPENROUTER_API_KEY before running.\n"
            f"  DEEPRESEARCH_LLM_PROVIDER={provider!r}, "
            f"OPENROUTER_API_KEY={'set' if os.getenv('OPENROUTER_API_KEY') else 'unset'}"
        )


def build_entrypoint(corpus_path: str):
    """`corpus_path` is the same LocalCorpusBackend JSON file the original run used — the corpus
    itself is part of the frozen world (it's data the search tool reads), not something the
    replay experiment is testing."""

    def entrypoint(ctx: ReplayContext) -> None:
        new_question = ctx.overrides.injected_context_edits.get("sub_question")
        if not new_question:
            raise ValueError("this harness requires overrides.injected_context_edits['sub_question']")

        _require_openrouter()

        from deepresearch.agent.react_agent import TOOLS
        from deepresearch.agent.subagent import run_subagent
        from deepresearch.backends.local_corpus import LocalCorpusBackend
        from deepresearch.config import RunConfig
        from deepresearch.llm.chat_model import build_chat_model
        from deepresearch.llm.client import LLMClient
        from deepresearch.schemas import SubQuestion
        from deepresearch.store.recorder import RunRecorder

        real_backend = LocalCorpusBackend.from_json_file(Path(corpus_path))
        ctx.tool_router.live_tool_fns = _make_live_tool_fns(real_backend)
        routed_backend = _RoutedSearchBackend(ctx.tool_router)

        config = RunConfig(rerank_enabled=False, cache_enabled=False, local_corpus_dir=corpus_path)
        llm = LLMClient()
        chat_model = build_chat_model(config).bind_tools(TOOLS)
        node = SubQuestion(id="replay", question=new_question)

        ctx.recorder.begin_step()
        finding, usage = asyncio.run(
            run_subagent(
                node,
                # No upstream facts substituted into the brief: this replay isolates one node's
                # own retrieval, and the step it resumes from was an independent node with no
                # resolved dependencies.
                "",
                config=config,
                chat_model=chat_model,
                llm=llm,
                search_backend=routed_backend,
                rerank_backend=None,
                recorder=RunRecorder(run_id="replay"),
                run_span_id=_REPLAY_SPAN_ID,
                source_registry={},
                started_monotonic=time.monotonic(),
            )
        )

        step = ctx.recorder.trace.steps[-1]
        step.model_call = ModelCall(
            provider="openrouter",
            model=config.worker_model,
            params={},
            messages=[{"role": "user", "content": json.dumps({"sub_question": new_question})}],
            response={"choices": [{"message": {"role": "assistant", "content": finding.model_dump_json()}}]},
            token_counts=TokenCounts(
                prompt_tokens=usage.input_tokens,
                completion_tokens=usage.output_tokens,
                total_tokens=usage.input_tokens + usage.output_tokens,
            ),
            cost_usd=usage.cost_usd,
        )
        ctx.recorder.end_step()

        print(f"\n[replay] sub_question: {new_question!r}")
        print(f"[replay] answer: {finding.answer}")
        print(f"[replay] claims found: {len(finding.claims)}")
        for claim in finding.claims:
            print(f"  - {claim.text} (source={claim.source_id}, confidence={claim.confidence})")
        print(f"[replay] open_gaps: {finding.open_gaps}")

    return entrypoint
