"""Loads ctx-capture traces. trace-replay consumes that schema; it never coerces an unsupported
major version, it refuses (see docs/DESIGN.md "Replay semantics spec", step 1).
"""

from __future__ import annotations

from ctx_capture.schema import Trace
from ctx_capture.storage.repository import TraceRepository

SUPPORTED_SCHEMA_MAJOR = "1"


class UnsupportedSchemaVersionError(Exception):
    def __init__(self, trace_id: str, schema_version: str) -> None:
        self.trace_id = trace_id
        self.schema_version = schema_version
        super().__init__(
            f"trace {trace_id} has schema_version {schema_version!r}, "
            f"trace-replay only supports major version {SUPPORTED_SCHEMA_MAJOR}.x"
        )


def _major(schema_version: str) -> str:
    return schema_version.split(".", 1)[0]


def schema_major_supported(schema_version: str) -> bool:
    """Non-raising check, for callers (like list_replayable_traces) that need a bool/reason
    rather than an exception — see check_schema_version for the refuse-outright variant."""
    return _major(schema_version) == SUPPORTED_SCHEMA_MAJOR


def check_schema_version(trace: Trace) -> None:
    if not schema_major_supported(trace.schema_version):
        raise UnsupportedSchemaVersionError(trace.trace_id, trace.schema_version)


def load_trace(repo: TraceRepository, trace_id: str) -> Trace:
    """Load a trace and refuse it outright if its major schema version isn't supported."""
    trace = repo.get_trace(trace_id)
    check_schema_version(trace)
    return trace
