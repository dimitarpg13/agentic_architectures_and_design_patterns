from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from agent_l1_association import l1_association_node
from agent_l2_intervention import l2_intervention_node
from agent_l3_counterfactual import l3_counterfactual_node
from agent_orchestrator import orchestrator_node
from agent_synthesizer import synthesizer_node
from agent_validator import validator_node
from causal_state import CausalState
from llm_client import LLMClient


def _route_by_rung(state: CausalState) -> str:
    rung = state.get("ladder_rung", "L1")
    return {
        "L1": "l1_association",
        "L2": "l2_intervention",
        "L3": "l3_counterfactual",
    }.get(rung, "l1_association")


def _validation_router(state: CausalState) -> str:
    result = state.get("validation_result", {})
    status = str(result.get("status", "pass")).lower()
    if status == "re_route":
        return "re_route"
    if status == "fail":
        return "fail"
    return "pass"


def build_causal_graph(llm: LLMClient) -> Any:
    async def _orchestrator(state: CausalState) -> CausalState:
        return await orchestrator_node(state, llm)

    async def _l1(state: CausalState) -> CausalState:
        return await l1_association_node(state, llm)

    async def _l2(state: CausalState) -> CausalState:
        return await l2_intervention_node(state, llm)

    async def _l3(state: CausalState) -> CausalState:
        return await l3_counterfactual_node(state, llm)

    async def _validator(state: CausalState) -> CausalState:
        return await validator_node(state, llm)

    async def _synthesizer(state: CausalState) -> CausalState:
        return await synthesizer_node(state, llm)

    builder = StateGraph(CausalState)

    builder.add_node("orchestrator", _orchestrator)
    builder.add_node("l1_association", _l1)
    builder.add_node("l2_intervention", _l2)
    builder.add_node("l3_counterfactual", _l3)
    builder.add_node("validator", _validator)
    builder.add_node("synthesizer", _synthesizer)

    builder.add_edge(START, "orchestrator")

    builder.add_conditional_edges(
        "orchestrator",
        _route_by_rung,
        {
            "l1_association": "l1_association",
            "l2_intervention": "l2_intervention",
            "l3_counterfactual": "l3_counterfactual",
        },
    )

    for rung_node in ("l1_association", "l2_intervention", "l3_counterfactual"):
        builder.add_edge(rung_node, "validator")

    builder.add_conditional_edges(
        "validator",
        _validation_router,
        {
            "pass": "synthesizer",
            "re_route": "orchestrator",
            "fail": "synthesizer",
        },
    )

    builder.add_edge("synthesizer", END)

    return builder.compile()


async def run_causal_workflow(
    question: str,
    llm: LLMClient,
    *,
    dag: dict | None = None,
    scm: dict | None = None,
    max_iterations: int = 3,
) -> CausalState:
    graph = build_causal_graph(llm)

    initial_state: CausalState = {
        "question": question,
        "causal_query": {},
        "ladder_rung": "L1",
        "dag": dag or _default_dag(),
        "scm": scm,
        "analysis_result": {},
        "validation_result": {},
        "final_report": "",
        "iteration": 0,
        "max_iterations": max_iterations,
    }

    result = await graph.ainvoke(initial_state)
    return result


def _default_dag() -> dict:
    return {
        "nodes": ["X", "Y", "Z"],
        "edges": [["Z", "X"], ["Z", "Y"], ["X", "Y"]],
        "description": (
            "Default DAG: Z is a common cause of X and Y (confounder), "
            "and X directly causes Y."
        ),
    }
