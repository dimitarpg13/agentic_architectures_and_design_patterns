from __future__ import annotations

from pathlib import Path
from typing import Any

from a2a_channel import A2AChannel
from llm_clients import build_llm_client
from orchestrator import ADKA2AOrchestrator
from settings import Settings
from specialized_agents import ADKSpecializedAgent


def _provider_to_adk_model(provider: str, settings: Settings) -> str:
    if provider == "anthropic":
        return settings.anthropic_model
    return settings.openai_model


async def run_workflow(
    objective: str,
    *,
    provider: str | None = None,
    openai_api_key: str | None = None,
    anthropic_api_key: str | None = None,
    openai_model: str | None = None,
    anthropic_model: str | None = None,
) -> dict[str, Any]:
    base = Settings.from_env()
    settings = Settings(
        llm_provider=(provider or base.llm_provider).strip().lower(),
        openai_api_key=openai_api_key or base.openai_api_key,
        anthropic_api_key=anthropic_api_key or base.anthropic_api_key,
        openai_model=openai_model or base.openai_model,
        anthropic_model=anthropic_model or base.anthropic_model,
    )
    llm = build_llm_client(settings)

    folder = Path(__file__).resolve().parent
    channel = A2AChannel()
    adk_model = _provider_to_adk_model(settings.llm_provider, settings)

    agents = [
        ADKSpecializedAgent(
            name="research_agent",
            role="Finds and structures conceptual insights for the objective.",
            model_name=adk_model,
            server_script=folder / "research_tools_server.py",
            llm=llm,
            a2a_channel=channel,
        ),
        ADKSpecializedAgent(
            name="analytics_agent",
            role="Computes KPI-style quantitative evidence for the objective.",
            model_name=adk_model,
            server_script=folder / "analytics_tools_server.py",
            llm=llm,
            a2a_channel=channel,
        ),
    ]

    orchestrator = ADKA2AOrchestrator(
        llm=llm,
        agents=agents,
        a2a_channel=channel,
        adk_model_name=adk_model,
    )
    await orchestrator.initialize()
    try:
        return await orchestrator.run(objective)
    finally:
        await orchestrator.shutdown()


