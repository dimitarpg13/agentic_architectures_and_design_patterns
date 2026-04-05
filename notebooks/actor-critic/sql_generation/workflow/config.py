"""
Workflow configuration with three secret-management strategies:
  1. .env file (local development)
  2. Hardcoded values (notebook experimentation)
  3. Google Cloud Secret Manager (production)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass
class WorkflowConfig:
    gcp_project_id: str
    gcp_location_gemini: str = "us-central1"
    gcp_location_claude: str = "us-east5"
    actor_model: str = "claude-sonnet-4-20250514"
    critic_model: str = "gemini-2.5-flash"
    max_attempts: int = 3
    use_case: str = "tpch"
    langsmith_api_key: str = ""
    langsmith_project: str = "sql-generation-actor-critic"
    base_dir: Path = field(default=BASE_DIR)

    # ── Factory: from .env file ─────────────────────────────────────

    @classmethod
    def from_env(cls, env_path: str | Path | None = None) -> WorkflowConfig:
        if env_path is None:
            env_path = BASE_DIR / ".env"
        load_dotenv(env_path)

        cfg = cls(
            gcp_project_id=os.environ["GCP_PROJECT_ID"],
            gcp_location_gemini=os.getenv("GCP_LOCATION_GEMINI", "us-central1"),
            gcp_location_claude=os.getenv("GCP_LOCATION_CLAUDE", "us-east5"),
            actor_model=os.getenv("ACTOR_MODEL", "claude-sonnet-4-20250514"),
            critic_model=os.getenv("CRITIC_MODEL", "gemini-2.5-flash"),
            max_attempts=int(os.getenv("MAX_VALIDATION_ATTEMPTS", "3")),
            use_case=os.getenv("USE_CASE", "tpch"),
            langsmith_api_key=os.getenv("LANGSMITH_API_KEY", ""),
            langsmith_project=os.getenv(
                "LANGSMITH_PROJECT", "sql-generation-actor-critic"
            ),
        )
        cfg._apply_langsmith_env()
        return cfg

    # ── Factory: from explicit values (notebook) ────────────────────

    @classmethod
    def from_values(
        cls,
        gcp_project_id: str,
        *,
        gcp_location_gemini: str = "us-central1",
        gcp_location_claude: str = "us-east5",
        actor_model: str = "claude-sonnet-4-20250514",
        critic_model: str = "gemini-2.5-flash",
        max_attempts: int = 3,
        use_case: str = "tpch",
        langsmith_api_key: str = "",
        langsmith_project: str = "sql-generation-actor-critic",
    ) -> WorkflowConfig:
        cfg = cls(
            gcp_project_id=gcp_project_id,
            gcp_location_gemini=gcp_location_gemini,
            gcp_location_claude=gcp_location_claude,
            actor_model=actor_model,
            critic_model=critic_model,
            max_attempts=max_attempts,
            use_case=use_case,
            langsmith_api_key=langsmith_api_key,
            langsmith_project=langsmith_project,
        )
        cfg._apply_langsmith_env()
        return cfg

    # ── Factory: from Google Cloud Secret Manager ───────────────────

    @classmethod
    def from_secret_manager(
        cls,
        gcp_project_id: str,
        secret_prefix: str = "sql-gen",
    ) -> WorkflowConfig:
        from google.cloud import secretmanager

        client = secretmanager.SecretManagerServiceClient()

        def _get_secret(name: str, default: str = "") -> str:
            secret_id = f"{secret_prefix}-{name}"
            resource = f"projects/{gcp_project_id}/secrets/{secret_id}/versions/latest"
            try:
                response = client.access_secret_version(request={"name": resource})
                return response.payload.data.decode("UTF-8")
            except Exception:
                logger.warning("Secret %s not found, using default", secret_id)
                return default

        cfg = cls(
            gcp_project_id=gcp_project_id,
            gcp_location_gemini=_get_secret("gcp-location-gemini", "us-central1"),
            gcp_location_claude=_get_secret("gcp-location-claude", "us-east5"),
            actor_model=_get_secret("actor-model", "claude-sonnet-4-20250514"),
            critic_model=_get_secret("critic-model", "gemini-2.5-flash"),
            max_attempts=int(_get_secret("max-attempts", "3")),
            use_case=_get_secret("use-case", "tpch"),
            langsmith_api_key=_get_secret("langsmith-api-key", ""),
            langsmith_project=_get_secret(
                "langsmith-project", "sql-generation-actor-critic"
            ),
        )
        cfg._apply_langsmith_env()
        return cfg

    # ── Internals ───────────────────────────────────────────────────

    def _apply_langsmith_env(self) -> None:
        """Propagate LangSmith config to environment variables so
        LangGraph's automatic tracing picks them up."""
        if self.langsmith_api_key:
            os.environ["LANGSMITH_TRACING"] = "true"
            os.environ["LANGSMITH_API_KEY"] = self.langsmith_api_key
            os.environ["LANGSMITH_PROJECT"] = self.langsmith_project
            logger.info(
                "LangSmith tracing enabled — project: %s", self.langsmith_project
            )
        else:
            logger.info("LangSmith API key not set — tracing disabled")
