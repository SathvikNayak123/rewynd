"""Scripted, in-process MCP client session — the same pattern ctx-capture's own MCP tests use,
so "through a real MCP client session" means the actual MCP protocol (tool listing, call_tool,
structuredContent, isError), not a bypass that calls Python functions directly.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session


@asynccontextmanager
async def mcp_client(mcp: FastMCP):
    async with create_connected_server_and_client_session(mcp) as session:
        await session.initialize()
        yield session
