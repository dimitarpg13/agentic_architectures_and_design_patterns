"""Actor node — generates SQL from natural language using Claude on Vertex AI."""

from __future__ import annotations

import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage

from workflow.state import SQLWorkflowState

logger = logging.getLogger(__name__)


class ActorNode:
    """Callable LangGraph node wrapping the Actor (Claude) LLM."""

    def __init__(self, llm):
        self.llm = llm

    def __call__(self, state: SQLWorkflowState) -> dict:
        attempt = state.get("attempt", 0) + 1
        logger.info("Actor generating SQL — attempt %d", attempt)

        prompt = self._build_prompt(state)

        messages = [
            SystemMessage(content=state["actor_system_message"]),
            HumanMessage(content=prompt),
        ]

        response = self.llm.invoke(messages)
        content = response.content

        sql = self._extract_sql(content)
        explanation = self._extract_explanation(content)

        logger.info("Actor produced %d-char SQL", len(sql))

        return {
            "generated_sql": sql,
            "sql_explanation": explanation,
            "attempt": attempt,
            "correction_history": [
                {
                    "attempt": attempt,
                    "source": "actor",
                    "sql": sql,
                }
            ],
        }

    def _build_prompt(self, state: SQLWorkflowState) -> str:
        parts = [f"## User Question\n{state['user_query']}"]

        feedback = state.get("critic_feedback", "")
        if feedback:
            parts.append(
                f"## Feedback from Previous Validation\n"
                f"Your previous SQL was rejected. Address every issue below:\n\n"
                f"{feedback}"
            )

        previous = state.get("generated_sql", "")
        if previous and feedback:
            parts.append(
                f"## Your Previous SQL (rejected)\n```sql\n{previous}\n```"
            )

        return "\n\n".join(parts)

    @staticmethod
    def _extract_sql(content: str) -> str:
        match = re.search(
            r"```sql\s*\n(.*?)```", content, re.DOTALL | re.IGNORECASE
        )
        if match:
            return match.group(1).strip()
        # Fallback: treat the whole response as SQL if no fenced block found
        return content.strip()

    @staticmethod
    def _extract_explanation(content: str) -> str:
        match = re.search(
            r"##\s*Explanation\s*\n(.*?)(?:\n##|\Z)", content, re.DOTALL
        )
        if match:
            return match.group(1).strip()
        # Fallback: return everything after the SQL block
        parts = re.split(r"```sql.*?```", content, flags=re.DOTALL | re.IGNORECASE)
        if len(parts) > 1:
            return parts[-1].strip()
        return ""
