"""Shared fixtures for the paper-writing pipeline test suite."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

# ── Mock LLM ─────────────────────────────────────────────────────────


@dataclass
class MockMessage:
    content: str


class MockLLM:
    def __init__(self, response: str):
        self.response = response
        self.call_count = 0

    def invoke(self, messages):
        self.call_count += 1
        return MockMessage(content=self.response)


class SequentialMockLLM:
    def __init__(self, responses: list[str]):
        self.responses = responses
        self.call_count = 0

    def invoke(self, messages):
        idx = min(self.call_count, len(self.responses) - 1)
        self.call_count += 1
        return MockMessage(content=self.responses[idx])


# ── Sample state helper ──────────────────────────────────────────────

SAMPLE_IDEA = "We propose AdaSparse, an adaptive sparse attention mechanism."
SAMPLE_LOG = "Dense: 12.3 ppl at ctx=4096. AdaSparse: 12.5 ppl at ctx=4096."
SAMPLE_GUIDELINES = "Maximum 9 pages. Abstract under 250 words."


def make_base_state(**overrides) -> dict:
    state = {
        "idea_summary": SAMPLE_IDEA,
        "experimental_log": SAMPLE_LOG,
        "conference_guidelines": SAMPLE_GUIDELINES,
    }
    state.update(overrides)
    return state


# ── Mock agent responses ─────────────────────────────────────────────

OUTLINE_RESPONSE = json.dumps({
    "title": "AdaSparse: Adaptive Sparse Attention for Long-Context Transformers",
    "contributions": ["Adaptive sparse attention", "Theoretical bounds"],
    "search_queries": [
        "sparse attention transformers",
        "efficient transformers long context",
    ],
    "sections": [
        {"name": "Abstract", "key_points": ["Problem", "Method"], "approach": "Summarize"},
        {"name": "Introduction", "key_points": ["Motivation"], "approach": "Motivate"},
        {"name": "Related Work", "key_points": ["Sparse attention"], "approach": "Survey"},
        {"name": "Methodology", "key_points": ["TRP", "Top-k"], "approach": "Describe"},
        {"name": "Experiments", "key_points": ["Perplexity"], "approach": "Compare"},
        {"name": "Conclusion", "key_points": ["Summary"], "approach": "Wrap up"},
    ],
})

LIT_REVIEW_RESPONSE = json.dumps({
    "citations": [
        {
            "id": "cite_1",
            "title": "Attention Is All You Need",
            "authors": "Vaswani et al.",
            "year": "2017",
            "venue": "NeurIPS",
            "summary": "Introduced the Transformer.",
            "relevance": "Foundation architecture",
        },
    ],
    "introduction": "Transformers have become the backbone of modern NLP...",
    "related_work": "### Sparse Attention\nSeveral methods have been proposed...",
})

SECTION_WRITER_RESPONSE = json.dumps({
    "abstract": "We propose AdaSparse, an adaptive sparse attention mechanism...",
    "methodology": "Our approach consists of three components...",
    "experiments": "We evaluate AdaSparse on long-context benchmarks...",
    "conclusion": "We presented AdaSparse...",
})

REFINEMENT_PASS_RESPONSE = json.dumps({
    "verdict": "satisfactory",
    "issues": [],
    "refined_manuscript": "# AdaSparse\n\n## Abstract\nWe propose AdaSparse...",
})

REFINEMENT_NEEDS_WORK_RESPONSE = json.dumps({
    "verdict": "needs_refinement",
    "issues": [{"section": "Experiments", "severity": "moderate", "description": "Missing error bars", "suggestion": "Add std dev"}],
    "refined_manuscript": "# AdaSparse\n\n## Abstract\nWe propose AdaSparse (improved)...",
})
