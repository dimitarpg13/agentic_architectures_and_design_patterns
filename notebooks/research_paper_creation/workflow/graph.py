"""LangGraph graph for the multi-agent paper-writing pipeline.

Graph topology (mirrors PaperOrchestra's 5-step pipeline):

    START → generate_outline → search_literature → write_literature
          → write_sections → assemble → review → route
                                          ↑         │
                                          │    ┌────┴─────┐
                                          │  REFINE    DONE
                                          │    ↓         ↓
                                          └────┘     finalize → END
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from langgraph.graph import END, StateGraph

from agents.literature_review import LiteratureReviewAgent
from agents.outline import OutlineAgent
from agents.refinement import RefinementAgent
from agents.section_writer import SectionWriterAgent
from tools.web_search import create_search_tool
from workflow.state import PaperState

if TYPE_CHECKING:
    from config.settings import PipelineConfig
    from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)

_LEADING_HEADER = re.compile(r"^\s*#{1,2}\s+[^\n]*\n*")


def _strip_leading_header(text: str) -> str:
    """Remove a leading Markdown H1/H2 header line if present.

    Defence-in-depth: the prompts instruct agents not to include top-level
    headers, but LLMs sometimes ignore that.  Stripping here prevents
    duplicate / inconsistent headings in the assembled manuscript.
    """
    return _LEADING_HEADER.sub("", text, count=1).lstrip("\n") if text else ""


def build_paper_workflow(config: PipelineConfig) -> CompiledStateGraph:
    """Construct and compile the paper-writing LangGraph pipeline."""

    llm = config.build_llm()
    logger.info("LLM: %s / %s", config.llm_provider, config.llm_model)

    search_tool = create_search_tool(
        config.search_provider,
        api_key=config.tavily_api_key,
    )
    logger.info("Search: %s", config.search_provider)

    # ── Agent instances ─────────────────────────────────────────────
    outline_agent = OutlineAgent(llm)
    lit_agent = LiteratureReviewAgent(llm, search_tool, config.max_search_results)
    section_agent = SectionWriterAgent(llm)
    refinement_agent = RefinementAgent(llm)

    # ── Assembly & routing nodes ────────────────────────────────────

    def assemble_manuscript(state: PaperState) -> dict:
        """Stitch all drafted sections into a single Markdown manuscript."""
        title = state.get("outline", {}).get("title", "Untitled")
        logger.info("Assembling manuscript: %s", title)

        parts = [
            f"# {title}\n",
            "## Abstract\n",
            _strip_leading_header(state.get("abstract", "")),
            "\n## 1. Introduction\n",
            _strip_leading_header(state.get("introduction", "")),
            "\n## 2. Related Work\n",
            _strip_leading_header(state.get("related_work", "")),
            "\n## 3. Methodology\n",
            _strip_leading_header(state.get("methodology", "")),
            "\n## 4. Experiments\n",
            _strip_leading_header(state.get("experiments", "")),
            "\n## 5. Conclusion\n",
            _strip_leading_header(state.get("conclusion", "")),
        ]

        citations = state.get("citations", [])
        if citations:
            refs = "\n## References\n\n"
            for i, c in enumerate(citations, 1):
                refs += (
                    f"[{i}] {c.get('authors', 'Unknown')}. "
                    f"\"{c.get('title', '')}\" "
                    f"{c.get('venue', '')}, {c.get('year', '')}.\n\n"
                )
            parts.append(refs)

        manuscript = "\n".join(parts)
        logger.info("Assembled manuscript: %d characters", len(manuscript))

        return {
            "full_manuscript": manuscript,
            "refinement_round": 0,
            "max_refinement_rounds": config.max_refinement_rounds,
        }

    def route_after_review(state: PaperState) -> str:
        verdict = state.get("verdict", "satisfactory")
        round_num = state.get("refinement_round", 0)
        max_rounds = state.get("max_refinement_rounds", 2)

        if verdict == "satisfactory" or round_num >= max_rounds:
            logger.info("Routing → finalize (verdict=%s, round %d/%d)", verdict, round_num, max_rounds)
            return "done"
        logger.info("Routing → refine (round %d/%d)", round_num, max_rounds)
        return "refine"

    def finalize(state: PaperState) -> dict:
        manuscript = state.get("full_manuscript", "")
        round_num = state.get("refinement_round", 0)
        verdict = state.get("verdict", "")

        if verdict == "satisfactory":
            status = "completed"
        else:
            status = "max_rounds_reached"

        logger.info("Finalized: %s after %d refinement round(s)", status, round_num)
        return {"final_manuscript": manuscript, "status": status}

    # ── Build graph ─────────────────────────────────────────────────
    graph = StateGraph(PaperState)

    graph.add_node("generate_outline", outline_agent)
    graph.add_node("search_literature", lit_agent)
    graph.add_node("write_sections", section_agent)
    graph.add_node("assemble", assemble_manuscript)
    graph.add_node("review", refinement_agent)
    graph.add_node("finalize", finalize)

    graph.set_entry_point("generate_outline")
    graph.add_edge("generate_outline", "search_literature")
    graph.add_edge("search_literature", "write_sections")
    graph.add_edge("write_sections", "assemble")
    graph.add_edge("assemble", "review")
    graph.add_conditional_edges("review", route_after_review, {
        "refine": "review",
        "done": "finalize",
    })
    graph.add_edge("finalize", END)

    compiled = graph.compile()
    logger.info("Paper workflow compiled successfully")
    return compiled
