"""LangGraph state schema for the Actor-Critic SQL generation workflow."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class SQLWorkflowState(TypedDict, total=False):
    """Typed state that flows through every LangGraph node.

    Fields without a reducer are overwritten on each update.
    ``correction_history`` uses an add-reducer so entries accumulate
    across iterations.
    """

    # ── Input ───────────────────────────────────────────────────────
    user_query: str
    use_case: str

    # ── Assembled context (set once in assemble_context) ────────────
    actor_system_message: str
    critic_system_message: str

    # ── Actor output ────────────────────────────────────────────────
    generated_sql: str
    sql_explanation: str

    # ── Critic output ───────────────────────────────────────────────
    critic_verdict: str          # "pass" | "salvageable" | "non_salvageable"
    critic_issues: list[dict]
    critic_feedback: str
    corrected_sql: str

    # ── Loop control ────────────────────────────────────────────────
    attempt: int
    max_attempts: int
    correction_history: Annotated[list[dict], operator.add]

    # ── Final output ────────────────────────────────────────────────
    final_sql: str
    final_explanation: str
    status: str                  # "accepted" | "best_effort"
