from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from agents import SpecializedAgent
from config import Settings
from llm import build_llm_client
from orchestrator import Orchestrator


async def run_workflow(objective: str) -> str:
    settings = Settings.from_env()
    llm = build_llm_client(settings)
    base_dir = Path(__file__).resolve().parent

    agents = [
        SpecializedAgent(
            name="research_agent",
            role="Finds and structures knowledge for the objective.",
            server_script=base_dir / "research_tools_server.py",
            llm=llm,
        ),
        SpecializedAgent(
            name="analytics_agent",
            role="Produces quantitative metrics and KPI summaries.",
            server_script=base_dir / "analytics_tools_server.py",
            llm=llm,
        ),
    ]

    orchestrator = Orchestrator(llm=llm, agents=agents)
    await orchestrator.initialize()
    try:
        return await orchestrator.run(objective)
    finally:
        await orchestrator.shutdown()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MCP multi-agent workflow example with orchestrator + specialized agents."
    )
    parser.add_argument(
        "--objective",
        default=(
            "Prepare a short executive brief on MCP-based multi-agent workflows and "
            "include a KPI status example."
        ),
        help="Goal for the orchestrator.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = asyncio.run(run_workflow(args.objective))
    print("\n=== Final Orchestrated Output ===")
    print(result)


if __name__ == "__main__":
    main()


