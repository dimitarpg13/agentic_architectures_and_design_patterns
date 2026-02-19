from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from agents import SpecializedAgent
from llm_clients import LLMClient


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

    def to_prompt(self) -> str:
        payload = {
            name: {"role": entry.role, "capabilities": entry.capabilities}
            for name, entry in self._entries.items()
        }
        return json.dumps(payload, indent=2)

    def names(self) -> set[str]:
        return set(self._entries.keys())


class WorkflowState(TypedDict):
    objective: str
    registry: str
    plan: list[dict[str, str]]
    step_index: int
    current_step: dict[str, str]
    step_outputs: list[dict[str, str]]
    final_answer: str


class LangGraphOrchestrator:
    def __init__(self, llm: LLMClient, agents: list[SpecializedAgent]) -> None:
        self._llm = llm
        self._agents = {agent.name: agent for agent in agents}
        self._registry = AgentRegistry()
        self._graph = self._build_graph()

    async def initialize(self) -> None:
        for agent in self._agents.values():
            await agent.connect()
            await self._registry.register(agent)

    async def shutdown(self) -> None:
        for agent in self._agents.values():
            await agent.close()

    async def run(self, objective: str) -> str:
        initial_state: WorkflowState = {
            "objective": objective,
            "registry": self._registry.to_prompt(),
            "plan": [],
            "step_index": 0,
            "current_step": {},
            "step_outputs": [],
            "final_answer": "",
        }
        result = await self._graph.ainvoke(initial_state)
        return result["final_answer"]

    def _build_graph(self):
        builder = StateGraph(WorkflowState)
        builder.add_node("plan", self._plan_node)
        builder.add_node("next_step", self._next_step_node)
        builder.add_node("execute_step", self._execute_step_node)
        builder.add_node("synthesize", self._synthesize_node)

        builder.add_edge(START, "plan")
        builder.add_edge("plan", "next_step")
        builder.add_conditional_edges(
            "next_step",
            self._should_continue,
            {"execute": "execute_step", "done": "synthesize"},
        )
        builder.add_edge("execute_step", "next_step")
        builder.add_edge("synthesize", END)
        return builder.compile()

    async def _plan_node(self, state: WorkflowState) -> WorkflowState:
        system_prompt = (
            "You are a LangGraph orchestrator planner. Return strict JSON array with 2-4 items. "
            'Each item must have keys "agent" and "task". Use only registered agent names.'
        )
        user_prompt = (
            f"Objective: {state['objective']}\n"
            f"Agent registry with MCP capabilities:\n{state['registry']}"
        )
        raw = await self._llm.generate(system_prompt=system_prompt, user_prompt=user_prompt)
        plan = self._parse_plan(raw)
        state["plan"] = plan
        return state

    async def _next_step_node(self, state: WorkflowState) -> WorkflowState:
        idx = state["step_index"]
        if idx < len(state["plan"]):
            state["current_step"] = state["plan"][idx]
        else:
            state["current_step"] = {}
        return state

    async def _execute_step_node(self, state: WorkflowState) -> WorkflowState:
        step = state["current_step"]
        agent_name = step["agent"]
        task = step["task"]
        agent = self._agents[agent_name]
        prior = "\n".join(
            f"{item['agent']} -> {item['output']}" for item in state["step_outputs"]
        )
        output = await agent.run_step(task=task, context=prior)
        state["step_outputs"].append(
            {"agent": agent_name, "task": task, "output": output}
        )
        state["step_index"] += 1
        return state

    async def _synthesize_node(self, state: WorkflowState) -> WorkflowState:
        system_prompt = "Synthesize a final response from all step outputs."
        user_prompt = (
            f"Objective: {state['objective']}\n"
            f"Step outputs:\n{json.dumps(state['step_outputs'], indent=2)}"
        )
        state["final_answer"] = await self._llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        return state

    def _should_continue(self, state: WorkflowState) -> str:
        if state["step_index"] < len(state["plan"]):
            return "execute"
        return "done"

    def _parse_plan(self, raw: str) -> list[dict[str, str]]:
        valid_names = self._registry.names()
        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, list):
                raise ValueError("Not a list")
            normalized: list[dict[str, str]] = []
            for item in parsed:
                name = str(item["agent"])
                task = str(item["task"])
                if name not in valid_names:
                    raise ValueError("Unknown agent")
                normalized.append({"agent": name, "task": task})
            if normalized:
                return normalized
        except Exception:
            pass

        names = list(valid_names)
        first = names[0]
        second = names[1] if len(names) > 1 else names[0]
        return [
            {"agent": first, "task": "Research key points for the objective."},
            {"agent": second, "task": "Provide one KPI-style quantitative summary."},
        ]


