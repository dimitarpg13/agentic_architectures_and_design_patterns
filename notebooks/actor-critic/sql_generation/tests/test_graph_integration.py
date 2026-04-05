"""Integration tests for the full LangGraph workflow with mocked LLMs.

These tests exercise the real graph topology (edges, conditional routing,
state accumulation) while replacing the Vertex AI LLM calls with
deterministic mocks.
"""

from unittest.mock import patch

from tests.conftest import (
    ACTOR_SQL_RESPONSE,
    CRITIC_NON_SALVAGEABLE_RESPONSE,
    CRITIC_PASS_RESPONSE,
    CRITIC_SALVAGEABLE_RESPONSE,
    MockLLM,
    SequentialMockLLM,
)
from workflow.config import WorkflowConfig
from workflow.graph import build_sql_workflow


def _make_config(project_dir):
    cfg = WorkflowConfig.from_values("test-project")
    cfg.base_dir = project_dir
    return cfg


class TestPassOnFirstTry:
    """Actor generates correct SQL; Critic passes it immediately."""

    @patch("workflow.graph.ChatVertexAI")
    @patch("workflow.graph.ChatAnthropicVertex")
    def test_accepted_status(self, mock_claude_cls, mock_gemini_cls, project_dir):
        mock_claude_cls.return_value = MockLLM(ACTOR_SQL_RESPONSE)
        mock_gemini_cls.return_value = MockLLM(CRITIC_PASS_RESPONSE)

        graph = build_sql_workflow(_make_config(project_dir))
        result = graph.invoke({"user_query": "Top 10 nations by revenue"})

        assert result["status"] == "accepted"
        assert result["attempt"] == 1
        assert "SELECT" in result["final_sql"]

    @patch("workflow.graph.ChatVertexAI")
    @patch("workflow.graph.ChatAnthropicVertex")
    def test_correction_history_has_single_entry(
        self, mock_claude_cls, mock_gemini_cls, project_dir
    ):
        mock_claude_cls.return_value = MockLLM(ACTOR_SQL_RESPONSE)
        mock_gemini_cls.return_value = MockLLM(CRITIC_PASS_RESPONSE)

        graph = build_sql_workflow(_make_config(project_dir))
        result = graph.invoke({"user_query": "Top 10 nations by revenue"})

        assert len(result["correction_history"]) == 1
        assert result["correction_history"][0]["source"] == "actor"


class TestSalvageableCorrection:
    """Critic returns salvageable on first pass, then passes the correction."""

    @patch("workflow.graph.ChatVertexAI")
    @patch("workflow.graph.ChatAnthropicVertex")
    def test_accepted_after_correction(
        self, mock_claude_cls, mock_gemini_cls, project_dir
    ):
        mock_claude_cls.return_value = MockLLM(ACTOR_SQL_RESPONSE)
        mock_gemini_cls.return_value = SequentialMockLLM([
            CRITIC_SALVAGEABLE_RESPONSE,
            CRITIC_PASS_RESPONSE,
        ])

        graph = build_sql_workflow(_make_config(project_dir))
        result = graph.invoke({"user_query": "Revenue by nation"})

        assert result["status"] == "accepted"

    @patch("workflow.graph.ChatVertexAI")
    @patch("workflow.graph.ChatAnthropicVertex")
    def test_correction_history_has_two_entries(
        self, mock_claude_cls, mock_gemini_cls, project_dir
    ):
        mock_claude_cls.return_value = MockLLM(ACTOR_SQL_RESPONSE)
        mock_gemini_cls.return_value = SequentialMockLLM([
            CRITIC_SALVAGEABLE_RESPONSE,
            CRITIC_PASS_RESPONSE,
        ])

        graph = build_sql_workflow(_make_config(project_dir))
        result = graph.invoke({"user_query": "Revenue by nation"})

        assert len(result["correction_history"]) == 2
        sources = [e["source"] for e in result["correction_history"]]
        assert sources == ["actor", "critic_correction"]


class TestNonSalvageableRegeneration:
    """Critic returns non_salvageable; Actor regenerates; Critic passes."""

    @patch("workflow.graph.ChatVertexAI")
    @patch("workflow.graph.ChatAnthropicVertex")
    def test_accepted_after_regeneration(
        self, mock_claude_cls, mock_gemini_cls, project_dir
    ):
        mock_claude_cls.return_value = MockLLM(ACTOR_SQL_RESPONSE)
        mock_gemini_cls.return_value = SequentialMockLLM([
            CRITIC_NON_SALVAGEABLE_RESPONSE,
            CRITIC_PASS_RESPONSE,
        ])

        graph = build_sql_workflow(_make_config(project_dir))
        result = graph.invoke({"user_query": "Revenue by nation"})

        assert result["status"] == "accepted"
        assert result["attempt"] == 2

    @patch("workflow.graph.ChatVertexAI")
    @patch("workflow.graph.ChatAnthropicVertex")
    def test_actor_called_twice(
        self, mock_claude_cls, mock_gemini_cls, project_dir
    ):
        actor_llm = MockLLM(ACTOR_SQL_RESPONSE)
        mock_claude_cls.return_value = actor_llm
        mock_gemini_cls.return_value = SequentialMockLLM([
            CRITIC_NON_SALVAGEABLE_RESPONSE,
            CRITIC_PASS_RESPONSE,
        ])

        graph = build_sql_workflow(_make_config(project_dir))
        graph.invoke({"user_query": "Revenue by nation"})

        assert actor_llm.call_count == 2


class TestMaxAttemptsExhausted:
    """Critic never passes — workflow delivers best-effort after max attempts."""

    @patch("workflow.graph.ChatVertexAI")
    @patch("workflow.graph.ChatAnthropicVertex")
    def test_best_effort_status(
        self, mock_claude_cls, mock_gemini_cls, project_dir
    ):
        mock_claude_cls.return_value = MockLLM(ACTOR_SQL_RESPONSE)
        mock_gemini_cls.return_value = MockLLM(CRITIC_NON_SALVAGEABLE_RESPONSE)

        cfg = _make_config(project_dir)
        cfg.max_attempts = 2
        graph = build_sql_workflow(cfg)
        result = graph.invoke({"user_query": "Revenue by nation"})

        assert result["status"] == "best_effort"

    @patch("workflow.graph.ChatVertexAI")
    @patch("workflow.graph.ChatAnthropicVertex")
    def test_does_not_exceed_max_attempts(
        self, mock_claude_cls, mock_gemini_cls, project_dir
    ):
        actor_llm = MockLLM(ACTOR_SQL_RESPONSE)
        mock_claude_cls.return_value = actor_llm
        mock_gemini_cls.return_value = MockLLM(CRITIC_NON_SALVAGEABLE_RESPONSE)

        cfg = _make_config(project_dir)
        cfg.max_attempts = 2
        graph = build_sql_workflow(cfg)
        result = graph.invoke({"user_query": "Revenue by nation"})

        assert actor_llm.call_count <= cfg.max_attempts


class TestSalvageableLoopExhausted:
    """Critic keeps returning salvageable but never passes."""

    @patch("workflow.graph.ChatVertexAI")
    @patch("workflow.graph.ChatAnthropicVertex")
    def test_best_effort_after_repeated_corrections(
        self, mock_claude_cls, mock_gemini_cls, project_dir
    ):
        mock_claude_cls.return_value = MockLLM(ACTOR_SQL_RESPONSE)
        mock_gemini_cls.return_value = MockLLM(CRITIC_SALVAGEABLE_RESPONSE)

        cfg = _make_config(project_dir)
        cfg.max_attempts = 2
        graph = build_sql_workflow(cfg)
        result = graph.invoke({"user_query": "Revenue by nation"})

        assert result["status"] == "best_effort"
