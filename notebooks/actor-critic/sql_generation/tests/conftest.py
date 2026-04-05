"""Shared fixtures for the Actor-Critic SQL generation test suite."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

# ── Mock LLM infrastructure ─────────────────────────────────────────


@dataclass
class MockMessage:
    content: str


class MockLLM:
    """Returns the same response on every call."""

    def __init__(self, response: str):
        self.response = response
        self.call_count = 0
        self.last_messages = None

    def invoke(self, messages):
        self.call_count += 1
        self.last_messages = messages
        return MockMessage(content=self.response)


class SequentialMockLLM:
    """Returns a different response on each successive call.
    After exhausting the list, repeats the last response."""

    def __init__(self, responses: list[str]):
        self.responses = responses
        self.call_count = 0
        self.all_messages: list = []

    def invoke(self, messages):
        idx = min(self.call_count, len(self.responses) - 1)
        self.call_count += 1
        self.all_messages.append(messages)
        return MockMessage(content=self.responses[idx])


# ── File-system fixtures ─────────────────────────────────────────────

SAMPLE_ACTOR_PROMPT = "You are an SQL generation agent. Write correct SQL."
SAMPLE_CRITIC_PROMPT = "You are an SQL validation critic. Return JSON."
SAMPLE_DOMAIN_RULES = (
    "# TPC-H Domain Rules\n\n"
    "- Revenue = l_extendedprice * (1 - l_discount)\n"
    "- Default sort: descending by primary metric\n"
)
SAMPLE_DATA_DICT = (
    "# TPC-H Data Dictionary\n\n"
    "## lineitem\n"
    "| Column | Type | Description |\n"
    "|--------|------|-------------|\n"
    "| l_orderkey | INTEGER | FK to orders |\n"
    "| l_extendedprice | DECIMAL | Extended price |\n"
    "| l_discount | DECIMAL | Discount (0.00-0.10) |\n"
)


@pytest.fixture
def project_dir(tmp_path):
    """Create a minimal project directory with all prompt and use-case files."""
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "actor_system_prompt.md").write_text(SAMPLE_ACTOR_PROMPT)
    (prompts / "critic_system_prompt.md").write_text(SAMPLE_CRITIC_PROMPT)

    uc = tmp_path / "usecases" / "tpch"
    uc.mkdir(parents=True)
    (uc / "DOMAIN_RULES.md").write_text(SAMPLE_DOMAIN_RULES)
    (uc / "DATA_DICTIONARY.md").write_text(SAMPLE_DATA_DICT)

    return tmp_path


# ── Sample LLM responses ────────────────────────────────────────────

ACTOR_SQL_RESPONSE = """\
### SQL
```sql
SELECT n.n_name AS nation_name,
       SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_revenue
FROM nation n
JOIN customer c ON c.c_nationkey = n.n_nationkey
JOIN orders o ON o.o_custkey = c.c_custkey
JOIN lineitem l ON l.l_orderkey = o.o_orderkey
GROUP BY n.n_name
ORDER BY total_revenue DESC
LIMIT 10
```

## Explanation
Joins nation → customer → orders → lineitem, calculates revenue, \
groups by nation, returns top 10."""

CRITIC_PASS_RESPONSE = json.dumps({
    "verdict": "pass",
    "issues": [],
    "feedback": "",
    "corrected_sql": "",
})

CRITIC_SALVAGEABLE_RESPONSE = json.dumps({
    "verdict": "salvageable",
    "issues": [
        {
            "category": "logic",
            "severity": "medium",
            "description": "Missing GROUP BY column for n_name",
        }
    ],
    "feedback": "Add n.n_name to the GROUP BY clause.",
    "corrected_sql": (
        "SELECT n.n_name, SUM(l.l_extendedprice) AS rev "
        "FROM nation n JOIN lineitem l ON 1=1 "
        "GROUP BY n.n_name ORDER BY rev DESC LIMIT 10"
    ),
})

CRITIC_NON_SALVAGEABLE_RESPONSE = json.dumps({
    "verdict": "non_salvageable",
    "issues": [
        {
            "category": "schema",
            "severity": "high",
            "description": "Table 'revenue_summary' does not exist",
        }
    ],
    "feedback": (
        "The query references a non-existent table 'revenue_summary'. "
        "Use lineitem and nation tables instead."
    ),
    "corrected_sql": "",
})


# ── Reusable state helpers ───────────────────────────────────────────


def make_base_state(**overrides) -> dict:
    """Return a minimal valid state dict, with optional overrides."""
    state = {
        "user_query": "Show top 10 nations by revenue",
        "use_case": "tpch",
        "actor_system_message": "You are the Actor.",
        "critic_system_message": "You are the Critic.",
        "generated_sql": "SELECT 1",
        "sql_explanation": "Test query.",
        "critic_verdict": "",
        "critic_issues": [],
        "critic_feedback": "",
        "corrected_sql": "",
        "attempt": 0,
        "max_attempts": 3,
        "correction_history": [],
        "final_sql": "",
        "final_explanation": "",
        "status": "",
    }
    state.update(overrides)
    return state
