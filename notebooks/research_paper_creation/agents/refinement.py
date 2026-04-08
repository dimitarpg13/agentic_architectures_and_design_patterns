"""Refinement Agent — reviews and iteratively improves the manuscript."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "refinement_agent.md"


class RefinementAgent:
    def __init__(self, llm):
        self.llm = llm
        self.system_prompt = PROMPT_PATH.read_text(encoding="utf-8")

    def __call__(self, state: dict) -> dict:
        round_num = state.get("refinement_round", 0) + 1
        max_rounds = state.get("max_refinement_rounds", 2)
        logger.info("Refinement Agent: round %d/%d", round_num, max_rounds)

        manuscript = state.get("full_manuscript", "")
        guidelines = state.get("conference_guidelines", "")

        user_msg = f"## Full Manuscript\n{manuscript}"
        if guidelines:
            user_msg += f"\n\n## Conference Guidelines\n{guidelines}"
        user_msg += f"\n\n*This is refinement round {round_num} of {max_rounds}.*"

        response = self.llm.invoke([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_msg),
        ])

        parsed = self._parse_json(response.content)

        verdict = parsed.get("verdict", "satisfactory")
        issues = parsed.get("issues", [])
        refined = parsed.get("refined_manuscript", manuscript)

        logger.info(
            "Refinement Agent: verdict=%s, %d issues, refined %d chars",
            verdict, len(issues), len(refined),
        )

        return {
            "review_feedback": json.dumps(issues, indent=2) if issues else "",
            "refined_manuscript": refined,
            "full_manuscript": refined,
            "refinement_round": round_num,
            "verdict": verdict,
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
        logger.warning("Could not parse refinement JSON, treating as satisfactory")
        return {
            "verdict": "satisfactory",
            "issues": [],
            "refined_manuscript": content,
        }
