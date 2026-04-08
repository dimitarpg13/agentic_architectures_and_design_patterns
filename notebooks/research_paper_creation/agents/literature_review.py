"""Literature Review Agent — searches for related work and drafts lit sections."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from tools.web_search import WebSearchTool

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "literature_review_agent.md"


class LiteratureReviewAgent:
    def __init__(self, llm, search_tool: WebSearchTool, max_results: int = 5):
        self.llm = llm
        self.search_tool = search_tool
        self.max_results = max_results
        self.system_prompt = PROMPT_PATH.read_text(encoding="utf-8")

    def __call__(self, state: dict) -> dict:
        queries = state.get("search_queries", [])
        logger.info("Literature Review Agent: searching with %d queries", len(queries))

        all_results = []
        for query in queries:
            results = self.search_tool.search(query, max_results=self.max_results)
            all_results.extend(results)
            logger.info("  Query '%s': %d results", query[:60], len(results))

        # Deduplicate by URL
        seen_urls: set[str] = set()
        unique_results = []
        for r in all_results:
            url = r.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(r)

        logger.info("Literature Review Agent: %d unique results after dedup", len(unique_results))

        outline_json = json.dumps(state.get("outline", {}), indent=2)
        results_text = "\n\n".join(
            f"### {r['title']}\n**URL:** {r['url']}\n{r['content']}"
            for r in unique_results
        )

        user_msg = (
            f"## Paper Outline\n```json\n{outline_json}\n```\n\n"
            f"## Search Results\n{results_text}"
        )

        response = self.llm.invoke([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_msg),
        ])

        parsed = self._parse_json(response.content)

        logger.info(
            "Literature Review Agent: %d citations, intro %d chars, related work %d chars",
            len(parsed.get("citations", [])),
            len(parsed.get("introduction", "")),
            len(parsed.get("related_work", "")),
        )

        return {
            "search_results": unique_results,
            "citations": parsed.get("citations", []),
            "introduction": parsed.get("introduction", ""),
            "related_work": parsed.get("related_work", ""),
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
        logger.warning("Could not parse literature review JSON")
        return {"citations": [], "introduction": content, "related_work": ""}
