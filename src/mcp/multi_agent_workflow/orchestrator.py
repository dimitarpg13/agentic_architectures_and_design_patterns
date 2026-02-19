from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from agents import SpecializedAgent
from llm import LLMClient


@dataclass
class AgentRegistryEntry:
    name: str
    role: str
    capabilities: list[dict[str, Any]]


class AgentRegistry:
    def __init__(self) -> None:
        self._entries: dict[str, AgentRegistryEntry] = {}

    async def register(self, agent: SpecializedAgent) -> None:
        capabilities = await agent.discover_capabilities()
        self._entries[agent.name] = AgentRegistryEntry(
            name=agent.name,
            role=agent.role,
            capabilities=capabilities,
        )

    def as_prompt_text(self) -> str:
        serializable = {
            name: {
                "role": entry.role,
                "capabilities": entry.capabilities,
            }
            for name, entry in self._entries.items()
        }
        return json.dumps(serializable, indent=2)

    def names(self) -> set[str]:
        return set(self._entries.keys())


class Orchestrator:
    def __init__(self, llm: LLMClient, agents: list[SpecializedAgent]) -> None:
        self._llm = llm
        self._agents = {agent.name: agent for agent in agents}
        self._registry = AgentRegistry()

    async def initialize(self) -> None:
        for agent in self._agents.values():
            await agent.connect()
            # MCP-based capability discovery during registration.
            await self._registry.register(agent)

    async def shutdown(self) -> None:
        for agent in self._agents.values():
            await agent.close()

    async def run(self, objective: str) -> str:
        plan = await self._build_plan(objective)
        step_outputs: list[dict[str, str]] = []

        running_context = ""
        for idx, step in enumerate(plan, start=1):
            agent_name = step["agent"]
            task = step["task"]
            agent = self._agents[agent_name]
            output = await agent.run_task(task=task, prior_context=running_context)
            step_outputs.append({"step": str(idx), "agent": agent_name, "output": output})
            running_context += f"\nStep {idx} ({agent_name}): {output}"

        return await self._synthesize(objective, step_outputs)

    async def _build_plan(self, objective: str) -> list[dict[str, str]]:
        system_prompt = (
            "You are an orchestrator. Build a short plan using registered agents only. "
            "Return strict JSON array with 2-4 steps. "
            "Each item must contain keys: agent, task."
        )
        user_prompt = (
            f"Objective: {objective}\n"
            f"Registered agents and capabilities:\n{self._registry.as_prompt_text()}"
        )
        raw = await self._llm.generate(system_prompt=system_prompt, user_prompt=user_prompt)
        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, list):
                raise ValueError("Plan is not a list")
            normalized: list[dict[str, str]] = []
            valid_names = self._registry.names()
            for item in parsed:
                name = str(item["agent"])
                task = str(item["task"])
                if name not in valid_names:
                    raise ValueError("Unknown agent in plan")
                normalized.append({"agent": name, "task": task})
            if normalized:
                return normalized
        except Exception:
            pass

        # Fallback to deterministic plan to keep example runnable.
        agent_names = list(self._registry.names())
        first = agent_names[0]
        second = agent_names[1] if len(agent_names) > 1 else agent_names[0]
        return [
            {"agent": first, "task": f"Collect core facts for: {objective}"},
            {"agent": second, "task": f"Compute or structure metrics for: {objective}"},
        ]

    async def _synthesize(self, objective: str, outputs: list[dict[str, str]]) -> str:
        system_prompt = "Synthesize a clear final answer from multi-agent step outputs."
        user_prompt = (
            f"Objective: {objective}\n"
            f"Step outputs:\n{json.dumps(outputs, indent=2)}"
        )
        return await self._llm.generate(system_prompt=system_prompt, user_prompt=user_prompt)


