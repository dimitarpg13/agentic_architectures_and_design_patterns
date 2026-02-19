from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm_clients import LLMClient
from mcp_client import MCPToolClient


@dataclass
class SpecializedAgent:
    name: str
    role: str
    server_script: Path
    llm: LLMClient
    tool_client: MCPToolClient = field(init=False)
    capabilities: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.tool_client = MCPToolClient(self.server_script)

    async def connect(self) -> None:
        await self.tool_client.connect()

    async def close(self) -> None:
        await self.tool_client.close()

    async def discover_capabilities(self) -> list[dict[str, Any]]:
        # MCP Agent Registration & Capability Discovery via list_tools.
        self.capabilities = await self.tool_client.list_tools()
        return self.capabilities

    async def run_step(self, task: str, context: str = "") -> str:
        if not self.capabilities:
            await self.discover_capabilities()

        planner_prompt = (
            "Choose one tool for the task and return strict JSON only:\n"
            '{"tool": "...", "arguments": {...}}'
        )
        user_prompt = (
            f"Agent: {self.name}\n"
            f"Role: {self.role}\n"
            f"Task: {task}\n"
            f"Context: {context or 'None'}\n"
            f"Capabilities: {json.dumps(self.capabilities, indent=2)}"
        )
        decision = await self.llm.generate(
            system_prompt=planner_prompt,
            user_prompt=user_prompt,
        )
        tool_name, arguments = _parse_tool_call(decision, self.capabilities)
        tool_output = await self.tool_client.call_tool(tool_name, arguments)

        summarize_prompt = (
            "Turn the tool output into a concise, actionable step result for an orchestrator."
        )
        return await self.llm.generate(
            system_prompt=summarize_prompt,
            user_prompt=(
                f"Task: {task}\n"
                f"Tool: {tool_name}\n"
                f"Arguments: {json.dumps(arguments)}\n"
                f"Tool output:\n{tool_output}"
            ),
        )


def _parse_tool_call(
    raw: str, capabilities: list[dict[str, Any]]
) -> tuple[str, dict[str, Any]]:
    valid_tool_names = {item["name"] for item in capabilities}
    try:
        payload = json.loads(raw)
        tool_name = str(payload["tool"])
        arguments = payload.get("arguments", {})
        if tool_name not in valid_tool_names:
            raise ValueError("Tool does not exist in capability registry")
        if not isinstance(arguments, dict):
            raise ValueError("Arguments must be an object")
        return tool_name, arguments
    except Exception:
        return capabilities[0]["name"], {}


