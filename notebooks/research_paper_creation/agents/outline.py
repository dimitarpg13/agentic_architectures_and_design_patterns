"""Outline Agent — produces a structured paper plan from raw materials."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "outline_agent.md"


class OutlineAgent:
    def __init__(self, llm):
        self.llm = llm
        self.system_prompt = PROMPT_PATH.read_text(encoding="utf-8")

    def __call__(self, state: dict) -> dict:
        logger.info("Outline Agent: generating structured outline")

        user_msg = (
            f"## Idea Summary\n{state['idea_summary']}\n\n"
            f"## Experimental Log\n{state['experimental_log']}"
        )
        guidelines = state.get("conference_guidelines", "")
        if guidelines:
            user_msg += f"\n\n## Conference Guidelines\n{guidelines}"

        response = self.llm.invoke([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_msg),
        ])

        outline = self._parse_json(response.content)
        search_queries = outline.get("search_queries", [])

        logger.info(
            "Outline Agent: produced outline with %d sections, %d search queries",
            len(outline.get("sections", [])),
            len(search_queries),
        )

        return {
            "outline": outline,
            "search_queries": search_queries,
        }

    @staticmethod
    def _parse_json(content: str) -> dict:
        text = content.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]
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
        logger.warning("Could not parse outline JSON, returning minimal structure")
        return {
            "title": "Untitled Paper",
            "contributions": [],
            "search_queries": [],
            "sections": [],
        }
