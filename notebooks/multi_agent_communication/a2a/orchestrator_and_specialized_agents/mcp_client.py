from __future__ import annotations

import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPToolClient:
    def __init__(self, server_script: Path) -> None:
        self._server_script = server_script
        self._stack = AsyncExitStack()
        self._session: ClientSession | None = None

    async def connect(self) -> None:
        params = StdioServerParameters(command=sys.executable, args=[str(self._server_script)])
        read_stream, write_stream = await self._stack.enter_async_context(stdio_client(params))
        self._session = await self._stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await self._session.initialize()

    async def close(self) -> None:
        await self._stack.aclose()
        self._session = None

    async def list_tools(self) -> list[dict[str, Any]]:
        if self._session is None:
            raise RuntimeError("MCP session not connected")
        result = await self._session.list_tools()
        return [
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.inputSchema,
            }
            for tool in result.tools
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        if self._session is None:
            raise RuntimeError("MCP session not connected")
        result = await self._session.call_tool(name, arguments)
        text = [
            getattr(block, "text", str(block))
            for block in result.content
            if getattr(block, "type", "") == "text"
        ]
        if text:
            return "\n".join(text)
        return str(result.content)


