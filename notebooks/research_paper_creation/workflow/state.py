"""LangGraph state schema for the paper-writing workflow."""

from __future__ import annotations

from typing import TypedDict


class PaperState(TypedDict, total=False):
    # ── Input ───────────────────────────────────────────────────────
    idea_summary: str
    experimental_log: str
    conference_guidelines: str

    # ── Outline Agent ───────────────────────────────────────────────
    outline: dict
    search_queries: list[str]

    # ── Literature Review Agent ─────────────────────────────────────
    search_results: list[dict]
    citations: list[dict]
    introduction: str
    related_work: str

    # ── Section Writing Agent ───────────────────────────────────────
    abstract: str
    methodology: str
    experiments: str
    conclusion: str

    # ── Assembly ────────────────────────────────────────────────────
    full_manuscript: str

    # ── Refinement Agent ────────────────────────────────────────────
    review_feedback: str
    refined_manuscript: str
    refinement_round: int
    max_refinement_rounds: int
    verdict: str

    # ── Final ───────────────────────────────────────────────────────
    final_manuscript: str
    status: str
