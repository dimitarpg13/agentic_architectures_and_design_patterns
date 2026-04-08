"""Section Writing Agent — drafts the core technical sections of the paper."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "section_writing_agent.md"


class SectionWriterAgent:
    def __init__(self, llm):
        self.llm = llm
        self.system_prompt = PROMPT_PATH.read_text(encoding="utf-8")

    def __call__(self, state: dict) -> dict:
        logger.info("Section Writer Agent: drafting core sections")

        outline_json = json.dumps(state.get("outline", {}), indent=2)
        citations_json = json.dumps(state.get("citations", []), indent=2)

        user_msg = (
            f"## Paper Outline\n```json\n{outline_json}\n```\n\n"
            f"## Idea Summary\n{state['idea_summary']}\n\n"
            f"## Experimental Log\n{state['experimental_log']}\n\n"
            f"## Introduction (already drafted)\n{state.get('introduction', '')}\n\n"
            f"## Related Work (already drafted)\n{state.get('related_work', '')}\n\n"
            f"## Citation Registry\n```json\n{citations_json}\n```"
        )

        response = self.llm.invoke([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_msg),
        ])

        parsed = self._parse_json(response.content)

        logger.info(
            "Section Writer Agent: abstract %d chars, methodology %d chars, "
            "experiments %d chars, conclusion %d chars",
            len(parsed.get("abstract", "")),
            len(parsed.get("methodology", "")),
            len(parsed.get("experiments", "")),
            len(parsed.get("conclusion", "")),
        )

        return {
            "abstract": parsed.get("abstract", ""),
            "methodology": parsed.get("methodology", ""),
            "experiments": parsed.get("experiments", ""),
            "conclusion": parsed.get("conclusion", ""),
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
        logger.warning("Could not parse section writer JSON, using raw content")
        return {
            "abstract": "",
            "methodology": content,
            "experiments": "",
            "conclusion": "",
        }
