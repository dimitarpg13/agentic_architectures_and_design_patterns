"""Critic node — validates Actor-generated SQL using Gemini on Vertex AI."""

from __future__ import annotations

import json
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage

from workflow.state import SQLWorkflowState

logger = logging.getLogger(__name__)


class CriticNode:
    """Callable LangGraph node wrapping the Critic (Gemini) LLM."""

    def __init__(self, llm):
        self.llm = llm

    def __call__(self, state: SQLWorkflowState) -> dict:
        attempt = state.get("attempt", 0)
        logger.info("Critic validating SQL — attempt %d", attempt)

        prompt = self._build_prompt(state)

        messages = [
            SystemMessage(content=state["critic_system_message"]),
            HumanMessage(content=prompt),
        ]

        response = self.llm.invoke(messages)
        result = self._parse_response(response.content)

        verdict = result.get("verdict", "non_salvageable")
        issues = result.get("issues", [])
        feedback = result.get("feedback", "")
        corrected_sql = result.get("corrected_sql", "")

        logger.info(
            "Critic verdict: %s (%d issues)", verdict, len(issues)
        )

        return {
            "critic_verdict": verdict,
            "critic_issues": issues,
            "critic_feedback": feedback,
            "corrected_sql": corrected_sql,
        }

    @staticmethod
    def _build_prompt(state: SQLWorkflowState) -> str:
        return (
            f"## User's Original Question\n{state['user_query']}\n\n"
            f"## Generated SQL to Validate\n```sql\n{state['generated_sql']}\n```\n\n"
            f"## Actor's Explanation\n{state.get('sql_explanation', 'No explanation provided.')}\n\n"
            f"Evaluate this SQL against the Data Dictionary and Domain Rules "
            f"in your system message. Return your assessment as a JSON object."
        )

    @staticmethod
    def _parse_response(content: str) -> dict:
        """Extract JSON from the Critic's response, tolerating markdown fences."""
        text = content.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]  # remove opening fence
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass

        logger.warning("Could not parse Critic response as JSON — treating as non_salvageable")
        return {
            "verdict": "non_salvageable",
            "issues": [
                {
                    "category": "logic",
                    "severity": "high",
                    "description": "Critic response was not valid JSON.",
                }
            ],
            "feedback": f"Critic output could not be parsed. Raw response:\n{content[:500]}",
            "corrected_sql": "",
        }
