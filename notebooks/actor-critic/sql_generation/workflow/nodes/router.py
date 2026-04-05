"""Routing, correction, and finalization nodes for the LangGraph workflow."""

from __future__ import annotations

import logging

from workflow.state import SQLWorkflowState

logger = logging.getLogger(__name__)


# ── Conditional router ──────────────────────────────────────────────

def route_verdict(state: SQLWorkflowState) -> str:
    """Decide the next node based on the Critic's verdict and the
    attempt counter.

    Returns one of: "pass", "salvageable", "non_salvageable", "max_attempts"
    """
    verdict = state.get("critic_verdict", "non_salvageable")
    attempt = state.get("attempt", 0)
    max_attempts = state.get("max_attempts", 3)

    if verdict == "pass":
        logger.info("Routing → finalize (PASS)")
        return "pass"

    if attempt >= max_attempts:
        logger.warning(
            "Routing → finalize (max attempts %d reached)", max_attempts
        )
        return "max_attempts"

    if verdict == "salvageable":
        logger.info("Routing → apply_correction (SALVAGEABLE, attempt %d/%d)",
                     attempt, max_attempts)
        return "salvageable"

    logger.info("Routing → generate_sql (NON_SALVAGEABLE, attempt %d/%d)",
                 attempt, max_attempts)
    return "non_salvageable"


# ── Apply Critic's correction ───────────────────────────────────────

def apply_correction(state: SQLWorkflowState) -> dict:
    """Replace the current SQL with the Critic's corrected version
    and increment the attempt counter to prevent infinite loops."""
    corrected = state.get("corrected_sql", "")
    attempt = state.get("attempt", 0) + 1

    logger.info("Applying Critic correction (%d chars, attempt now %d)", len(corrected), attempt)

    return {
        "generated_sql": corrected,
        "sql_explanation": f"Critic-corrected SQL (after attempt {attempt})",
        "attempt": attempt,
        "correction_history": [
            {
                "attempt": attempt,
                "source": "critic_correction",
                "sql": corrected,
            }
        ],
    }


# ── Finalize ────────────────────────────────────────────────────────

def finalize(state: SQLWorkflowState) -> dict:
    """Produce the terminal output, selecting the best available SQL."""
    verdict = state.get("critic_verdict", "")
    sql = state.get("generated_sql", "")
    explanation = state.get("sql_explanation", "")
    attempt = state.get("attempt", 0)

    if verdict == "pass":
        status = "accepted"
        logger.info("Finalized: ACCEPTED after %d attempt(s)", attempt)
    else:
        status = "best_effort"
        logger.warning(
            "Finalized: BEST_EFFORT after %d attempt(s) — last verdict: %s",
            attempt,
            verdict,
        )

    return {
        "final_sql": sql,
        "final_explanation": explanation,
        "status": status,
    }
