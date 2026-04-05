"""Tests for workflow.nodes.actor.ActorNode."""

from tests.conftest import ACTOR_SQL_RESPONSE, MockLLM, make_base_state
from workflow.nodes.actor import ActorNode


class TestExtractSQL:

    def test_extracts_from_fenced_block(self):
        content = "Here is the SQL:\n```sql\nSELECT 1\n```\nDone."
        assert ActorNode._extract_sql(content) == "SELECT 1"

    def test_extracts_multiline_sql(self):
        content = "```sql\nSELECT\n  a,\n  b\nFROM t\n```"
        result = ActorNode._extract_sql(content)
        assert "SELECT" in result
        assert "FROM t" in result

    def test_case_insensitive_fence(self):
        content = "```SQL\nSELECT 1\n```"
        assert ActorNode._extract_sql(content) == "SELECT 1"

    def test_fallback_when_no_fence(self):
        content = "SELECT 1 FROM dual"
        assert ActorNode._extract_sql(content) == "SELECT 1 FROM dual"


class TestExtractExplanation:

    def test_extracts_explanation_section(self):
        content = (
            "```sql\nSELECT 1\n```\n\n"
            "## Explanation\nThis is a test query."
        )
        assert ActorNode._extract_explanation(content) == "This is a test query."

    def test_fallback_text_after_sql_block(self):
        content = "```sql\nSELECT 1\n```\n\nSome trailing explanation."
        assert ActorNode._extract_explanation(content) == "Some trailing explanation."

    def test_empty_when_no_explanation(self):
        content = "SELECT 1 FROM dual"
        assert ActorNode._extract_explanation(content) == ""

    def test_full_actor_response(self):
        result = ActorNode._extract_explanation(ACTOR_SQL_RESPONSE)
        assert "nation" in result.lower()


class TestBuildPrompt:

    def test_includes_user_query(self):
        node = ActorNode(llm=None)
        state = make_base_state(user_query="What are the top sellers?")
        prompt = node._build_prompt(state)
        assert "What are the top sellers?" in prompt

    def test_includes_feedback_when_present(self):
        node = ActorNode(llm=None)
        state = make_base_state(
            critic_feedback="Missing GROUP BY clause",
            generated_sql="SELECT bad",
        )
        prompt = node._build_prompt(state)
        assert "Missing GROUP BY clause" in prompt
        assert "Previous SQL (rejected)" in prompt

    def test_no_feedback_section_on_first_attempt(self):
        node = ActorNode(llm=None)
        state = make_base_state(critic_feedback="", generated_sql="")
        prompt = node._build_prompt(state)
        assert "Feedback" not in prompt
        assert "rejected" not in prompt


class TestActorNodeCall:

    def test_produces_correct_state_keys(self):
        llm = MockLLM(ACTOR_SQL_RESPONSE)
        node = ActorNode(llm)
        state = make_base_state(attempt=0)
        result = node(state)

        assert "generated_sql" in result
        assert "sql_explanation" in result
        assert "attempt" in result
        assert "correction_history" in result

    def test_increments_attempt(self):
        llm = MockLLM(ACTOR_SQL_RESPONSE)
        node = ActorNode(llm)

        result = node(make_base_state(attempt=0))
        assert result["attempt"] == 1

        result = node(make_base_state(attempt=2))
        assert result["attempt"] == 3

    def test_extracts_sql_from_response(self):
        llm = MockLLM(ACTOR_SQL_RESPONSE)
        node = ActorNode(llm)
        result = node(make_base_state())
        assert "SELECT" in result["generated_sql"]
        assert "nation" in result["generated_sql"].lower()

    def test_records_correction_history(self):
        llm = MockLLM(ACTOR_SQL_RESPONSE)
        node = ActorNode(llm)
        result = node(make_base_state(attempt=0))

        assert len(result["correction_history"]) == 1
        entry = result["correction_history"][0]
        assert entry["source"] == "actor"
        assert entry["attempt"] == 1

    def test_invokes_llm_exactly_once(self):
        llm = MockLLM(ACTOR_SQL_RESPONSE)
        node = ActorNode(llm)
        node(make_base_state())
        assert llm.call_count == 1
