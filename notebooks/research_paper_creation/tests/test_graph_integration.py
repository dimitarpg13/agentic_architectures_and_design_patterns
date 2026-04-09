"""Integration tests for the full LangGraph paper-writing pipeline."""

from unittest.mock import patch

import pytest

from tests.conftest import (
    LIT_REVIEW_RESPONSE,
    OUTLINE_RESPONSE,
    REFINEMENT_NEEDS_WORK_RESPONSE,
    REFINEMENT_PASS_RESPONSE,
    SECTION_WRITER_RESPONSE,
    SequentialMockLLM,
    make_base_state,
)
from config.settings import PipelineConfig
from workflow.graph import _strip_leading_header, build_paper_workflow


def _mock_config():
    return PipelineConfig.from_values(
        llm_provider="openai",
        llm_model="gpt-4o",
        search_provider="mock",
    )


class TestFullPipelinePassOnFirstReview:

    @patch("config.settings.create_llm")
    def test_completes_with_status(self, mock_create_llm):
        mock_create_llm.return_value = SequentialMockLLM([
            OUTLINE_RESPONSE,
            LIT_REVIEW_RESPONSE,
            SECTION_WRITER_RESPONSE,
            REFINEMENT_PASS_RESPONSE,
        ])

        graph = build_paper_workflow(_mock_config())
        result = graph.invoke(make_base_state())

        assert result["status"] == "completed"
        assert result["final_manuscript"] != ""

    @patch("config.settings.create_llm")
    def test_manuscript_contains_sections(self, mock_create_llm):
        mock_create_llm.return_value = SequentialMockLLM([
            OUTLINE_RESPONSE,
            LIT_REVIEW_RESPONSE,
            SECTION_WRITER_RESPONSE,
            REFINEMENT_PASS_RESPONSE,
        ])

        graph = build_paper_workflow(_mock_config())
        result = graph.invoke(make_base_state())

        manuscript = result["final_manuscript"]
        assert "Abstract" in manuscript or "AdaSparse" in manuscript


class TestRefinementLoop:

    @patch("config.settings.create_llm")
    def test_refines_then_completes(self, mock_create_llm):
        mock_create_llm.return_value = SequentialMockLLM([
            OUTLINE_RESPONSE,
            LIT_REVIEW_RESPONSE,
            SECTION_WRITER_RESPONSE,
            REFINEMENT_NEEDS_WORK_RESPONSE,
            REFINEMENT_PASS_RESPONSE,
        ])

        graph = build_paper_workflow(_mock_config())
        result = graph.invoke(make_base_state())

        assert result["status"] == "completed"
        assert result["refinement_round"] >= 1


class TestStripLeadingHeader:

    @pytest.mark.parametrize("raw,expected", [
        ("# Methodology\n\nBody text", "Body text"),
        ("## Experiments\n\nBody text", "Body text"),
        ("### Subsection\n\nBody text", "### Subsection\n\nBody text"),
        ("Body text without header", "Body text without header"),
        ("", ""),
        ("  # Indented Header\nBody", "Body"),
    ])
    def test_strips_h1_h2_preserves_h3(self, raw, expected):
        assert _strip_leading_header(raw) == expected


class TestMaxRefinementRounds:

    @patch("config.settings.create_llm")
    def test_stops_after_max_rounds(self, mock_create_llm):
        mock_create_llm.return_value = SequentialMockLLM([
            OUTLINE_RESPONSE,
            LIT_REVIEW_RESPONSE,
            SECTION_WRITER_RESPONSE,
            REFINEMENT_NEEDS_WORK_RESPONSE,
            REFINEMENT_NEEDS_WORK_RESPONSE,
            REFINEMENT_NEEDS_WORK_RESPONSE,
        ])

        cfg = _mock_config()
        cfg.max_refinement_rounds = 2
        graph = build_paper_workflow(cfg)
        result = graph.invoke(make_base_state())

        assert result["status"] == "max_rounds_reached"
        assert result["refinement_round"] <= 2
