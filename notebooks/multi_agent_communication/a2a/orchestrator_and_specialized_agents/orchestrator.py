from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from a2a_channel import A2AChannel
from llm_clients import LLMClient
from specialized_agents import ADKSpecializedAgent


@dataclass
class RegistryEntry:
    name: str
    role: str
    capabilities: list[dict[str, Any]]


class AgentRegistry:
    def __init__(self) -> None:
        self._entries: dict[str, RegistryEntry] = {}

    async def register(self, agent: ADKSpecializedAgent) -> None:
        capabilities = await agent.discover_capabilities()
        self._entries[agent.name] = RegistryEntry(
            name=agent.name,
            role=agent.role,
            capabilities=capabilities,
        )

    def as_json(self) -> str:
        payload = {
            name: {"role": e.role, "capabilities": e.capabilities}
            for name, e in self._entries.items()
        }
        return json.dumps(payload, indent=2)

    def names(self) -> set[str]:
        return set(self._entries.keys())


class ADKA2AOrchestrator:
    def __init__(
        self,
        *,
        llm: LLMClient,
        agents: list[ADKSpecializedAgent],
        a2a_channel: A2AChannel,
        adk_model_name: str,
    ) -> None:
        self._llm = llm
        self._a2a_channel = a2a_channel
        self._agents = {agent.name: agent for agent in agents}
        self._registry = AgentRegistry()
        self._adk_orchestrator_agent = self._build_adk_orchestrator_agent(adk_model_name)

    def _build_adk_orchestrator_agent(self, model_name: str) -> Any:
        try:
            from google.adk.agents import LlmAgent
        except Exception:
            return None

        sub_agents = [a.adk_agent for a in self._agents.values() if a.adk_agent is not None]
        attempts = [
            {
                "name": "orchestrator_agent",
                "model": model_name,
                "description": "Coordinates specialized agents over A2A.",
                "sub_agents": sub_agents,
            },
            {
                "name": "orchestrator_agent",
                "model": model_name,
                "sub_agents": sub_agents,
            },
            {"name": "orchestrator_agent", "model": model_name},
        ]
        for kwargs in attempts:
            try:
                return LlmAgent(**kwargs)
            except TypeError:
                continue
        return None

    async def initialize(self) -> None:
        for agent in self._agents.values():
            await agent.connect()
            await self._registry.register(agent)

    async def shutdown(self) -> None:
        for agent in self._agents.values():
            await agent.close()

    async def run(self, objective: str) -> dict[str, Any]:
        plan = await self._plan(objective)
        outputs: list[dict[str, str]] = []
        running_context = ""

        for idx, step in enumerate(plan, start=1):
            name = step["agent"]
            task = step["task"]
            agent = self._agents[name]

            self._a2a_channel.send(
                sender="orchestrator_agent",
                recipient=name,
                text=f"Step {idx}: {task}",
            )

            out = await agent.receive_and_process(
                from_agent="orchestrator_agent",
                task_text=task,
                context=running_context,
            )
            outputs.append({"step": str(idx), "agent": name, "task": task, "output": out})
            running_context += f"\n[{name}] {out}"

        final = await self._synthesize(objective, outputs)
        return {
            "adk_orchestrator_initialized": self._adk_orchestrator_agent is not None,
            "plan": plan,
            "outputs": outputs,
            "a2a_messages": [
                {
                    "sender": m.sender,
                    "recipient": m.recipient,
                    "text": m.text,
                    "timestamp_utc": m.timestamp_utc,
                }
                for m in self._a2a_channel.history()
            ],
            "final_answer": final,
        }

    async def _plan(self, objective: str) -> list[dict[str, str]]:
        system_prompt = (
            "You are an orchestrator. Create a 2-4 step plan in strict JSON array format. "
            'Each item must include keys: "agent" and "task". Use only registered agent names.'
        )
        user_prompt = (
            f"Objective: {objective}\n"
            f"Registered agents and MCP-discovered capabilities:\n{self._registry.as_json()}"
        )
        raw = await self._llm.generate(system_prompt=system_prompt, user_prompt=user_prompt)
        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, list):
                raise ValueError("Plan must be a list")
            names = self._registry.names()
            normalized: list[dict[str, str]] = []
            for item in parsed:
                agent = str(item["agent"])
                task = str(item["task"])
                if agent not in names:
                    raise ValueError("Unknown agent in plan")
                normalized.append({"agent": agent, "task": task})
            if normalized:
                return normalized
        except Exception:
            pass

        # deterministic fallback
        names = list(self._registry.names())
        first = names[0]
        second = names[1] if len(names) > 1 else names[0]
        return [
            {"agent": first, "task": f"Research key facts for objective: {objective}"},
            {"agent": second, "task": f"Provide KPI-style quantitative summary for: {objective}"},
        ]

    async def _synthesize(self, objective: str, outputs: list[dict[str, str]]) -> str:
        return await self._llm.generate(
            system_prompt="Synthesize a clear final response from the agent outputs.",
            user_prompt=(
                f"Objective: {objective}\n"
                f"Agent outputs:\n{json.dumps(outputs, indent=2)}"
            ),
        )


