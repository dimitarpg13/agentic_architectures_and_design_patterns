"""Tests for workflow.nodes.critic.CriticNode."""

import json

from tests.conftest import (
    CRITIC_NON_SALVAGEABLE_RESPONSE,
    CRITIC_PASS_RESPONSE,
    CRITIC_SALVAGEABLE_RESPONSE,
    MockLLM,
    make_base_state,
)
from workflow.nodes.critic import CriticNode


class TestParseResponse:

    def test_clean_json(self):
        result = CriticNode._parse_response(CRITIC_PASS_RESPONSE)
        assert result["verdict"] == "pass"
        assert result["issues"] == []

    def test_markdown_fenced_json(self):
        fenced = f"```json\n{CRITIC_SALVAGEABLE_RESPONSE}\n```"
        result = CriticNode._parse_response(fenced)
        assert result["verdict"] == "salvageable"
        assert len(result["issues"]) == 1

    def test_json_embedded_in_text(self):
        text = f"Here is my assessment:\n{CRITIC_PASS_RESPONSE}\nEnd."
        result = CriticNode._parse_response(text)
        assert result["verdict"] == "pass"

    def test_unparseable_response_returns_non_salvageable(self):
        result = CriticNode._parse_response("This is not JSON at all.")
        assert result["verdict"] == "non_salvageable"
        assert len(result["issues"]) == 1
        assert "not valid JSON" in result["issues"][0]["description"]

    def test_empty_string_returns_non_salvageable(self):
        result = CriticNode._parse_response("")
        assert result["verdict"] == "non_salvageable"

    def test_preserves_corrected_sql(self):
        result = CriticNode._parse_response(CRITIC_SALVAGEABLE_RESPONSE)
        assert "SELECT" in result["corrected_sql"]

    def test_fenced_without_language_tag(self):
        fenced = f"```\n{CRITIC_PASS_RESPONSE}\n```"
        result = CriticNode._parse_response(fenced)
        assert result["verdict"] == "pass"


class TestBuildPrompt:

    def test_includes_user_query(self):
        state = make_base_state(user_query="Revenue by nation?")
        prompt = CriticNode._build_prompt(state)
        assert "Revenue by nation?" in prompt

    def test_includes_generated_sql(self):
        state = make_base_state(generated_sql="SELECT n.n_name FROM nation n")
        prompt = CriticNode._build_prompt(state)
        assert "SELECT n.n_name" in prompt

    def test_includes_explanation(self):
        state = make_base_state(sql_explanation="Joins nation to lineitem")
        prompt = CriticNode._build_prompt(state)
        assert "Joins nation to lineitem" in prompt

    def test_default_explanation_when_missing(self):
        state = make_base_state()
        del state["sql_explanation"]
        prompt = CriticNode._build_prompt(state)
        assert "No explanation provided" in prompt


class TestCriticNodeCall:

    def test_pass_verdict(self):
        llm = MockLLM(CRITIC_PASS_RESPONSE)
        node = CriticNode(llm)
        result = node(make_base_state())

        assert result["critic_verdict"] == "pass"
        assert result["critic_issues"] == []
        assert result["critic_feedback"] == ""
        assert result["corrected_sql"] == ""

    def test_salvageable_verdict(self):
        llm = MockLLM(CRITIC_SALVAGEABLE_RESPONSE)
        node = CriticNode(llm)
        result = node(make_base_state())

        assert result["critic_verdict"] == "salvageable"
        assert len(result["critic_issues"]) == 1
        assert result["corrected_sql"] != ""

    def test_non_salvageable_verdict(self):
        llm = MockLLM(CRITIC_NON_SALVAGEABLE_RESPONSE)
        node = CriticNode(llm)
        result = node(make_base_state())

        assert result["critic_verdict"] == "non_salvageable"
        assert result["critic_feedback"] != ""
        assert result["corrected_sql"] == ""

    def test_invokes_llm_exactly_once(self):
        llm = MockLLM(CRITIC_PASS_RESPONSE)
        node = CriticNode(llm)
        node(make_base_state())
        assert llm.call_count == 1
