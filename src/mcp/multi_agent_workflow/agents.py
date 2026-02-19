from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm import LLMClient
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
        self.capabilities = await self.tool_client.list_tools()
        return self.capabilities

    async def run_task(self, task: str, prior_context: str = "") -> str:
        if not self.capabilities:
            await self.discover_capabilities()
        available_tools = json.dumps(self.capabilities, indent=2)
        planner_prompt = (
            "Select exactly one tool and valid arguments for this task.\n"
            "Return strict JSON only with keys: tool, arguments."
        )
        user_prompt = (
            f"Agent: {self.name}\n"
            f"Role: {self.role}\n"
            f"Task: {task}\n"
            f"Prior context: {prior_context or 'None'}\n"
            f"Available tools: {available_tools}"
        )
        llm_decision = await self.llm.generate(
            system_prompt=planner_prompt,
            user_prompt=user_prompt,
        )
        tool_name, arguments = _extract_tool_call(llm_decision, self.capabilities)
        tool_output = await self.tool_client.call_tool(tool_name, arguments)

        response_prompt = (
            "You are a specialized agent. Convert tool output into a concise, useful response "
            "for the orchestrator."
        )
        final_response = await self.llm.generate(
            system_prompt=response_prompt,
            user_prompt=(
                f"Task: {task}\n"
                f"Tool called: {tool_name}\n"
                f"Arguments: {json.dumps(arguments)}\n"
                f"Tool output:\n{tool_output}"
            ),
        )
        return final_response


def _extract_tool_call(
    llm_text: str, capabilities: list[dict[str, Any]]
) -> tuple[str, dict[str, Any]]:
    tool_names = {cap["name"] for cap in capabilities}
    try:
        parsed = json.loads(llm_text)
        selected_tool = str(parsed["tool"])
        if selected_tool not in tool_names:
            raise ValueError("Unknown tool")
        args = parsed.get("arguments", {})
        if not isinstance(args, dict):
            raise ValueError("arguments must be a dict")
        return selected_tool, args
    except Exception:
        first_tool = capabilities[0]["name"]
        return first_tool, {}


