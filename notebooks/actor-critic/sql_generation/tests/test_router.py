"""Tests for workflow.nodes.router — routing, correction, and finalization."""

from tests.conftest import make_base_state
from workflow.nodes.router import apply_correction, finalize, route_verdict


class TestRouteVerdict:

    def test_pass_routes_to_finalize(self):
        state = make_base_state(critic_verdict="pass", attempt=1, max_attempts=3)
        assert route_verdict(state) == "pass"

    def test_pass_takes_priority_even_at_max_attempts(self):
        state = make_base_state(critic_verdict="pass", attempt=3, max_attempts=3)
        assert route_verdict(state) == "pass"

    def test_salvageable_under_max(self):
        state = make_base_state(critic_verdict="salvageable", attempt=1, max_attempts=3)
        assert route_verdict(state) == "salvageable"

    def test_non_salvageable_under_max(self):
        state = make_base_state(critic_verdict="non_salvageable", attempt=1, max_attempts=3)
        assert route_verdict(state) == "non_salvageable"

    def test_salvageable_at_max_routes_to_max_attempts(self):
        state = make_base_state(critic_verdict="salvageable", attempt=3, max_attempts=3)
        assert route_verdict(state) == "max_attempts"

    def test_non_salvageable_at_max_routes_to_max_attempts(self):
        state = make_base_state(critic_verdict="non_salvageable", attempt=3, max_attempts=3)
        assert route_verdict(state) == "max_attempts"

    def test_unknown_verdict_treated_as_non_salvageable(self):
        state = make_base_state(critic_verdict="something_unexpected", attempt=1, max_attempts=3)
        assert route_verdict(state) == "non_salvageable"

    def test_missing_verdict_defaults_to_non_salvageable(self):
        state = make_base_state(attempt=1, max_attempts=3)
        del state["critic_verdict"]
        assert route_verdict(state) == "non_salvageable"

    def test_max_attempts_of_one(self):
        state = make_base_state(critic_verdict="salvageable", attempt=1, max_attempts=1)
        assert route_verdict(state) == "max_attempts"


class TestApplyCorrection:

    def test_replaces_generated_sql(self):
        state = make_base_state(
            corrected_sql="SELECT corrected FROM t",
            attempt=1,
        )
        result = apply_correction(state)
        assert result["generated_sql"] == "SELECT corrected FROM t"

    def test_increments_attempt(self):
        state = make_base_state(attempt=1)
        result = apply_correction(state)
        assert result["attempt"] == 2

    def test_records_correction_history(self):
        state = make_base_state(corrected_sql="SELECT fixed", attempt=1)
        result = apply_correction(state)

        assert len(result["correction_history"]) == 1
        entry = result["correction_history"][0]
        assert entry["source"] == "critic_correction"
        assert entry["sql"] == "SELECT fixed"
        assert entry["attempt"] == 2

    def test_updates_explanation(self):
        state = make_base_state(attempt=1)
        result = apply_correction(state)
        assert "Critic-corrected" in result["sql_explanation"]


class TestFinalize:

    def test_accepted_on_pass(self):
        state = make_base_state(
            critic_verdict="pass",
            generated_sql="SELECT good",
            sql_explanation="Good query.",
            attempt=1,
        )
        result = finalize(state)
        assert result["status"] == "accepted"
        assert result["final_sql"] == "SELECT good"
        assert result["final_explanation"] == "Good query."

    def test_best_effort_on_non_pass(self):
        state = make_base_state(
            critic_verdict="non_salvageable",
            generated_sql="SELECT best_effort",
            attempt=3,
        )
        result = finalize(state)
        assert result["status"] == "best_effort"
        assert result["final_sql"] == "SELECT best_effort"

    def test_best_effort_on_salvageable_at_max(self):
        state = make_base_state(
            critic_verdict="salvageable",
            generated_sql="SELECT almost",
            attempt=3,
        )
        result = finalize(state)
        assert result["status"] == "best_effort"
