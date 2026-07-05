"""The integration-contract harness for the DeepResearch case study (docs/CASE_STUDY.md).

Replays a single "worker" step for real: calls DeepResearch's actual `run_worker` (real
LocalCorpusBackend, real LLMClient — genuine Anthropic API calls, real small cost) with a
different sub-question, to test whether a more targeted search finds a fact the original run's
worker didn't.

Real integration wrinkle found while building this (documented honestly, not smoothed over):
`ToolRouter.call()` is synchronous, but DeepResearch's `SearchBackend.search`/`.fetch` are
`async`. Bridging them means the harness can't just hand `ctx.tool_router.call` an async
function as a `live_tool_fns` entry — calling it from inside a coroutine that's already running
on an event loop would try to nest event loops. The fix here: each live call runs in its own
thread with a fresh event loop (`_run_async_in_thread`), so the surrounding coroutine (run_worker
awaiting our adapter's `.search()`/`.fetch()`) never has to nest loops. Worth smoothing out with
a native async tool_router in a future trace-replay version if async agents are common.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
from pathlib import Path

from deepresearch.agent.worker import run_worker
from deepresearch.backends.local_corpus import LocalCorpusBackend
from deepresearch.config import RunConfig
from deepresearch.llm.client import LLMClient
from deepresearch.schemas import FetchResult, SearchResult, SubQuestion

from ctx_capture.schema import ModelCall, TokenCounts
from trace_replay.harness import ReplayContext


def _run_async_in_thread(coro):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(coro)).result()


class _RoutedSearchBackend:
    """Implements SearchBackend's interface, but every call goes through
    ctx.tool_router.call() first — that's the seam where MOCK/LIVE/STUB and recording happen."""

    def __init__(self, tool_router) -> None:
        self._router = tool_router

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        served = self._router.call("search", {"query": query, "max_results": max_results})
        return [SearchResult(**r) for r in served]

    async def fetch(self, url: str) -> FetchResult:
        served = self._router.call("fetch", {"url": url})
        return FetchResult(**served)


def _make_live_tool_fns(real_backend: LocalCorpusBackend) -> dict:
    def live_search(query: str, max_results: int) -> list[dict]:
        results = _run_async_in_thread(real_backend.search(query, max_results=max_results))
        return [r.model_dump() for r in results]

    def live_fetch(url: str) -> dict:
        result = _run_async_in_thread(real_backend.fetch(url))
        return result.model_dump()

    return {"search": live_search, "fetch": live_fetch}


def build_entrypoint(corpus_path: str):
    """`corpus_path` is the same LocalCorpusBackend JSON file the original run used — the corpus
    itself is part of the frozen world (it's data the search tool reads), not something the
    replay experiment is testing."""

    def entrypoint(ctx: ReplayContext) -> None:
        new_question = ctx.overrides.injected_context_edits.get("sub_question")
        if not new_question:
            raise ValueError("this harness requires overrides.injected_context_edits['sub_question']")

        real_backend = LocalCorpusBackend.from_json_file(Path(corpus_path))
        ctx.tool_router.live_tool_fns = _make_live_tool_fns(real_backend)
        routed_backend = _RoutedSearchBackend(ctx.tool_router)

        config = RunConfig(rerank_enabled=False, cache_enabled=False, local_corpus_dir=corpus_path)
        llm = LLMClient()
        sub_question = SubQuestion(id="replay", question=new_question)

        ctx.recorder.begin_step()
        notes, usage = asyncio.run(
            run_worker(
                sub_question,
                search_backend=routed_backend,
                config=config,
                llm=llm,
                source_registry={},
                rerank_backend=None,
                recorder=None,
            )
        )

        step = ctx.recorder.trace.steps[-1]
        step.model_call = ModelCall(
            provider="anthropic",
            model=config.worker_model,
            params={},
            messages=[{"role": "user", "content": json.dumps({"sub_question": new_question})}],
            response={"choices": [{"message": {"role": "assistant", "content": notes.model_dump_json()}}]},
            token_counts=TokenCounts(
                prompt_tokens=usage.input_tokens,
                completion_tokens=usage.output_tokens,
                total_tokens=usage.input_tokens + usage.output_tokens,
            ),
            cost_usd=usage.cost_usd,
        )
        ctx.recorder.end_step()

        print(f"\n[replay] sub_question: {new_question!r}")
        print(f"[replay] claims found: {len(notes.claims)}")
        for claim in notes.claims:
            print(f"  - {claim.text} (source={claim.source_id}, confidence={claim.confidence})")
        print(f"[replay] open_gaps: {notes.open_gaps}")

    return entrypoint
