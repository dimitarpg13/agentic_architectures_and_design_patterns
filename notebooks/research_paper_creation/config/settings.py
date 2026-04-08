"""Pipeline configuration with configurable LLM backend and search provider."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent


def create_llm(provider: str, model: str, **kwargs) -> BaseChatModel:
    """Factory for creating LLM instances from any supported provider."""
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, **kwargs)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, **kwargs)

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, **kwargs)

    if provider == "google_vertex":
        from langchain_google_vertexai import ChatVertexAI
        return ChatVertexAI(model_name=model, **kwargs)

    raise ValueError(f"Unknown LLM provider: {provider}")


@dataclass
class PipelineConfig:
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o"
    llm_kwargs: dict = field(default_factory=dict)
    search_provider: str = "tavily"
    tavily_api_key: str = ""
    max_refinement_rounds: int = 2
    max_search_results: int = 5
    base_dir: Path = field(default=BASE_DIR)
    langsmith_api_key: str = ""
    langsmith_project: str = "paper-orchestra"

    # ── Factories ───────────────────────────────────────────────────

    @classmethod
    def from_env(cls, env_path: str | Path | None = None) -> PipelineConfig:
        if env_path is None:
            env_path = BASE_DIR / ".env"
        load_dotenv(env_path)

        kwargs: dict = {}
        provider = os.getenv("LLM_PROVIDER", "openai")
        if provider == "google_vertex":
            kwargs["project"] = os.getenv("GCP_PROJECT_ID", "")
            kwargs["location"] = os.getenv("GCP_LOCATION", "us-central1")

        cfg = cls(
            llm_provider=provider,
            llm_model=os.getenv("LLM_MODEL", "gpt-4o"),
            llm_kwargs=kwargs,
            search_provider=os.getenv("SEARCH_PROVIDER", "tavily"),
            tavily_api_key=os.getenv("TAVILY_API_KEY", ""),
            max_refinement_rounds=int(os.getenv("MAX_REFINEMENT_ROUNDS", "2")),
            max_search_results=int(os.getenv("MAX_SEARCH_RESULTS_PER_QUERY", "5")),
            langsmith_api_key=os.getenv("LANGSMITH_API_KEY", ""),
            langsmith_project=os.getenv("LANGSMITH_PROJECT", "paper-orchestra"),
        )
        cfg._apply_langsmith_env()
        return cfg

    @classmethod
    def from_values(
        cls,
        llm_provider: str = "openai",
        llm_model: str = "gpt-4o",
        *,
        search_provider: str = "tavily",
        tavily_api_key: str = "",
        max_refinement_rounds: int = 2,
        max_search_results: int = 5,
        langsmith_api_key: str = "",
        langsmith_project: str = "paper-orchestra",
        **llm_kwargs,
    ) -> PipelineConfig:
        cfg = cls(
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_kwargs=llm_kwargs,
            search_provider=search_provider,
            tavily_api_key=tavily_api_key,
            max_refinement_rounds=max_refinement_rounds,
            max_search_results=max_search_results,
            langsmith_api_key=langsmith_api_key,
            langsmith_project=langsmith_project,
        )
        cfg._apply_langsmith_env()
        return cfg

    # ── Helpers ─────────────────────────────────────────────────────

    def build_llm(self) -> BaseChatModel:
        return create_llm(self.llm_provider, self.llm_model, **self.llm_kwargs)

    def _apply_langsmith_env(self) -> None:
        if self.langsmith_api_key:
            os.environ["LANGSMITH_TRACING"] = "true"
            os.environ["LANGSMITH_API_KEY"] = self.langsmith_api_key
            os.environ["LANGSMITH_PROJECT"] = self.langsmith_project
            logger.info("LangSmith tracing enabled — project: %s", self.langsmith_project)
