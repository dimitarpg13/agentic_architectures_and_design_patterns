"""Unit tests for individual agents."""

import json

from tests.conftest import (
    LIT_REVIEW_RESPONSE,
    OUTLINE_RESPONSE,
    REFINEMENT_NEEDS_WORK_RESPONSE,
    REFINEMENT_PASS_RESPONSE,
    SECTION_WRITER_RESPONSE,
    MockLLM,
    make_base_state,
)
from agents.outline import OutlineAgent
from agents.literature_review import LiteratureReviewAgent
from agents.section_writer import SectionWriterAgent
from agents.refinement import RefinementAgent
from tools.web_search import MockSearchTool


class TestOutlineAgent:

    def test_produces_outline_and_queries(self):
        agent = OutlineAgent(MockLLM(OUTLINE_RESPONSE))
        result = agent(make_base_state())
        assert "outline" in result
        assert "search_queries" in result
        assert len(result["search_queries"]) > 0

    def test_outline_has_sections(self):
        agent = OutlineAgent(MockLLM(OUTLINE_RESPONSE))
        result = agent(make_base_state())
        assert len(result["outline"]["sections"]) == 6

    def test_handles_unparseable_json(self):
        agent = OutlineAgent(MockLLM("This is not JSON"))
        result = agent(make_base_state())
        assert result["outline"]["title"] == "Untitled Paper"

    def test_handles_fenced_json(self):
        fenced = f"```json\n{OUTLINE_RESPONSE}\n```"
        agent = OutlineAgent(MockLLM(fenced))
        result = agent(make_base_state())
        assert "AdaSparse" in result["outline"]["title"]


class TestLiteratureReviewAgent:

    def test_produces_citations_and_sections(self):
        agent = LiteratureReviewAgent(MockLLM(LIT_REVIEW_RESPONSE), MockSearchTool())
        state = make_base_state(
            outline=json.loads(OUTLINE_RESPONSE),
            search_queries=["sparse attention"],
        )
        result = agent(state)
        assert len(result["citations"]) > 0
        assert result["introduction"] != ""
        assert result["related_work"] != ""

    def test_deduplicates_search_results(self):
        agent = LiteratureReviewAgent(MockLLM(LIT_REVIEW_RESPONSE), MockSearchTool())
        state = make_base_state(
            outline=json.loads(OUTLINE_RESPONSE),
            search_queries=["query1", "query2", "query3"],
        )
        result = agent(state)
        urls = [r["url"] for r in result["search_results"]]
        assert len(urls) == len(set(urls))


class TestSectionWriterAgent:

    def test_produces_all_sections(self):
        agent = SectionWriterAgent(MockLLM(SECTION_WRITER_RESPONSE))
        state = make_base_state(
            outline=json.loads(OUTLINE_RESPONSE),
            citations=[],
            introduction="Intro text",
            related_work="Related work text",
        )
        result = agent(state)
        assert result["abstract"] != ""
        assert result["methodology"] != ""
        assert result["experiments"] != ""
        assert result["conclusion"] != ""


class TestRefinementAgent:

    def test_satisfactory_verdict(self):
        agent = RefinementAgent(MockLLM(REFINEMENT_PASS_RESPONSE))
        state = make_base_state(full_manuscript="# Paper\n\nContent...", refinement_round=0)
        result = agent(state)
        assert result["verdict"] == "satisfactory"
        assert result["refinement_round"] == 1

    def test_needs_refinement_verdict(self):
        agent = RefinementAgent(MockLLM(REFINEMENT_NEEDS_WORK_RESPONSE))
        state = make_base_state(full_manuscript="# Paper\n\nContent...", refinement_round=0)
        result = agent(state)
        assert result["verdict"] == "needs_refinement"
        assert result["review_feedback"] != ""

    def test_increments_round(self):
        agent = RefinementAgent(MockLLM(REFINEMENT_PASS_RESPONSE))
        state = make_base_state(full_manuscript="text", refinement_round=1)
        result = agent(state)
        assert result["refinement_round"] == 2
