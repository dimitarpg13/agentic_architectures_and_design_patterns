"""LangGraph graph definition for the Actor-Critic SQL generation workflow.

Graph topology:

    START → assemble_context → generate_sql → validate_sql → route_verdict
                                    ↑                            │
                                    │              ┌─────────────┼──────────────┐
                                    │              ↓             ↓              ↓
                                    │            PASS      SALVAGEABLE   NON_SALVAGEABLE
                                    │              ↓             ↓              │
                                    │          finalize   apply_correction     │
                                    │                          ↓               │
                                    │                    validate_sql          │
                                    └──────────────────────────────────────────┘
                                          (Actor re-generates with feedback)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_google_vertexai import ChatVertexAI
from langchain_google_vertexai.model_garden import ChatAnthropicVertex
from langgraph.graph import END, StateGraph

from utils.prompt_builder import PromptBuilder
from workflow.nodes.actor import ActorNode
from workflow.nodes.critic import CriticNode
from workflow.nodes.router import apply_correction, finalize, route_verdict
from workflow.state import SQLWorkflowState

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph
    from workflow.config import WorkflowConfig

logger = logging.getLogger(__name__)


def build_sql_workflow(config: WorkflowConfig) -> CompiledStateGraph:
    """Construct and compile the Actor-Critic SQL workflow.

    The returned compiled graph is a LangChain Runnable — call
    ``graph.invoke({"user_query": "...", ...})`` to execute.
    """

    # ── Initialize LLMs ────────────────────────────────────────────
    logger.info(
        "Initializing Actor LLM: %s @ %s",
        config.actor_model, config.gcp_location_claude,
    )
    actor_llm = ChatAnthropicVertex(
        model_name=config.actor_model,
        project=config.gcp_project_id,
        location=config.gcp_location_claude,
        max_tokens=4096,
    )

    logger.info(
        "Initializing Critic LLM: %s @ %s",
        config.critic_model, config.gcp_location_gemini,
    )
    critic_llm = ChatVertexAI(
        model_name=config.critic_model,
        project=config.gcp_project_id,
        location=config.gcp_location_gemini,
        max_output_tokens=4096,
    )

    # ── Prompt builder ──────────────────────────────────────────────
    prompt_builder = PromptBuilder(
        base_dir=config.base_dir,
        use_case=config.use_case,
    )

    # ── Node instances ──────────────────────────────────────────────
    actor_node = ActorNode(actor_llm)
    critic_node = CriticNode(critic_llm)

    def assemble_context(state: SQLWorkflowState) -> dict:
        """One-time context assembly — reads prompts and grounding docs."""
        logger.info("Assembling context for use-case '%s'", config.use_case)
        return {
            "actor_system_message": prompt_builder.build_actor_system_prompt(),
            "critic_system_message": prompt_builder.build_critic_system_prompt(),
            "use_case": config.use_case,
            "max_attempts": config.max_attempts,
            "attempt": 0,
            "correction_history": [],
        }

    # ── Build graph ─────────────────────────────────────────────────
    graph = StateGraph(SQLWorkflowState)

    graph.add_node("assemble_context", assemble_context)
    graph.add_node("generate_sql", actor_node)
    graph.add_node("validate_sql", critic_node)
    graph.add_node("apply_correction", apply_correction)
    graph.add_node("finalize", finalize)

    graph.set_entry_point("assemble_context")
    graph.add_edge("assemble_context", "generate_sql")
    graph.add_edge("generate_sql", "validate_sql")
    graph.add_conditional_edges(
        "validate_sql",
        route_verdict,
        {
            "pass": "finalize",
            "salvageable": "apply_correction",
            "non_salvageable": "generate_sql",
            "max_attempts": "finalize",
        },
    )
    graph.add_edge("apply_correction", "validate_sql")
    graph.add_edge("finalize", END)

    compiled = graph.compile()
    logger.info("Workflow graph compiled successfully")
    return compiled
