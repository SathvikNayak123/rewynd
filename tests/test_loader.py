"""Loader must refuse (not coerce) traces whose major schema version isn't supported."""

from __future__ import annotations

import pytest
from ctx_capture.schema import Trace

from trace_replay.loader import UnsupportedSchemaVersionError, check_schema_version


def test_supported_major_version_passes():
    trace = Trace(agent_name="toy-agent", schema_version="1.0")
    check_schema_version(trace)  # must not raise


def test_unsupported_major_version_refused():
    trace = Trace(agent_name="toy-agent", schema_version="2.0")
    with pytest.raises(UnsupportedSchemaVersionError):
        check_schema_version(trace)
