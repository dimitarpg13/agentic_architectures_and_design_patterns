from __future__ import annotations

from pathlib import Path

from agents import SpecializedAgent
from llm_clients import build_llm_client
from orchestrator_langgraph import LangGraphOrchestrator
from settings import Settings


async def run_multi_agent_workflow(
    objective: str,
    *,
    provider: str | None = None,
    openai_api_key: str | None = None,
    anthropic_api_key: str | None = None,
    openai_model: str | None = None,
    anthropic_model: str | None = None,
) -> str:
    settings = Settings.from_env()

    if provider:
        settings = Settings(
            llm_provider=provider,
            openai_api_key=openai_api_key or settings.openai_api_key,
            anthropic_api_key=anthropic_api_key or settings.anthropic_api_key,
            openai_model=openai_model or settings.openai_model,
            anthropic_model=anthropic_model or settings.anthropic_model,
        )
    else:
        settings = Settings(
            llm_provider=settings.llm_provider,
            openai_api_key=openai_api_key or settings.openai_api_key,
            anthropic_api_key=anthropic_api_key or settings.anthropic_api_key,
            openai_model=openai_model or settings.openai_model,
            anthropic_model=anthropic_model or settings.anthropic_model,
        )

    llm = build_llm_client(settings)
    base_dir = Path(__file__).resolve().parent

    agents = [
        SpecializedAgent(
            name="research_agent",
            role="Researches and structures domain knowledge.",
            server_script=base_dir / "research_tools_server.py",
            llm=llm,
        ),
        SpecializedAgent(
            name="analytics_agent",
            role="Performs numerical and KPI-oriented analysis.",
            server_script=base_dir / "analytics_tools_server.py",
            llm=llm,
        ),
    ]

    orchestrator = LangGraphOrchestrator(llm=llm, agents=agents)
    await orchestrator.initialize()
    try:
        return await orchestrator.run(objective)
    finally:
        await orchestrator.shutdown()


