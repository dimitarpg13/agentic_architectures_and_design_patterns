from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from a2a_channel import A2AChannel
from llm_clients import LLMClient
from mcp_client import MCPToolClient


@dataclass
class ADKSpecializedAgent:
    name: str
    role: str
    model_name: str
    server_script: Path
    llm: LLMClient
    a2a_channel: A2AChannel
    tool_client: MCPToolClient = field(init=False)
    capabilities: list[dict[str, Any]] = field(default_factory=list)
    adk_agent: Any = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.tool_client = MCPToolClient(self.server_script)
        self.adk_agent = self._build_adk_agent()

    def _build_adk_agent(self) -> Any:
        try:
            from google.adk.agents import LlmAgent
        except Exception:
            return None

        init_attempts = [
            {"name": self.name, "model": self.model_name, "description": self.role},
            {"name": self.name, "model": self.model_name, "instruction": self.role},
            {"name": self.name, "model": self.model_name},
        ]
        for kwargs in init_attempts:
            try:
                return LlmAgent(**kwargs)
            except TypeError:
                continue
        return None

    async def connect(self) -> None:
        await self.tool_client.connect()

    async def close(self) -> None:
        await self.tool_client.close()

    async def discover_capabilities(self) -> list[dict[str, Any]]:
        self.capabilities = await self.tool_client.list_tools()
        return self.capabilities

    async def receive_and_process(self, from_agent: str, task_text: str, context: str) -> str:
        self.a2a_channel.send(sender=from_agent, recipient=self.name, text=task_text)

        if not self.capabilities:
            await self.discover_capabilities()

        planner_system = (
            "You are a specialized ADK agent. Pick one MCP tool and arguments for the task. "
            'Return strict JSON with keys: tool, arguments.'
        )
        planner_user = (
            f"Agent name: {self.name}\n"
            f"Role: {self.role}\n"
            f"Task: {task_text}\n"
            f"Context: {context or 'None'}\n"
            f"Capabilities: {json.dumps(self.capabilities, indent=2)}\n"
            f"ADK enabled: {self.adk_agent is not None}"
        )
        decision = await self.llm.generate(system_prompt=planner_system, user_prompt=planner_user)
        tool_name, arguments = _parse_tool_selection(decision, self.capabilities)

        tool_output = await self.tool_client.call_tool(tool_name, arguments)
        summarize_system = (
            "Summarize tool output for the orchestrator as a concise actionable result."
        )
        result = await self.llm.generate(
            system_prompt=summarize_system,
            user_prompt=(
                f"Task: {task_text}\n"
                f"Tool: {tool_name}\n"
                f"Arguments: {json.dumps(arguments)}\n"
                f"Tool output:\n{tool_output}"
            ),
        )

        self.a2a_channel.send(sender=self.name, recipient=from_agent, text=result)
        return result


def _parse_tool_selection(
    model_text: str, capabilities: list[dict[str, Any]]
) -> tuple[str, dict[str, Any]]:
    tool_names = {item["name"] for item in capabilities}
    try:
        payload = json.loads(model_text)
        tool = str(payload["tool"])
        arguments = payload.get("arguments", {})
        if tool not in tool_names:
            raise ValueError("Unknown tool selected")
        if not isinstance(arguments, dict):
            raise ValueError("Arguments are not a JSON object")
        return tool, arguments
    except Exception:
        return capabilities[0]["name"], {}


