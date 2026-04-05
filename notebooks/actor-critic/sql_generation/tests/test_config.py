"""Tests for workflow.config.WorkflowConfig."""

import os
from pathlib import Path

import pytest

from workflow.config import WorkflowConfig


class TestFromValues:

    def test_creates_config_with_defaults(self):
        cfg = WorkflowConfig.from_values("my-project")
        assert cfg.gcp_project_id == "my-project"
        assert cfg.gcp_location_gemini == "us-central1"
        assert cfg.gcp_location_claude == "us-east5"
        assert cfg.actor_model == "claude-sonnet-4-20250514"
        assert cfg.critic_model == "gemini-2.5-flash"
        assert cfg.max_attempts == 3
        assert cfg.use_case == "tpch"

    def test_overrides_defaults(self):
        cfg = WorkflowConfig.from_values(
            "my-project",
            max_attempts=5,
            use_case="custom",
            critic_model="gemini-2.5-pro",
        )
        assert cfg.max_attempts == 5
        assert cfg.use_case == "custom"
        assert cfg.critic_model == "gemini-2.5-pro"


class TestFromEnv:

    def test_reads_env_file(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text(
            "GCP_PROJECT_ID=test-project-123\n"
            "GCP_LOCATION_GEMINI=europe-west4\n"
            "MAX_VALIDATION_ATTEMPTS=5\n"
            "USE_CASE=custom_ds\n"
        )
        cfg = WorkflowConfig.from_env(env_path=env_file)
        assert cfg.gcp_project_id == "test-project-123"
        assert cfg.gcp_location_gemini == "europe-west4"
        assert cfg.max_attempts == 5
        assert cfg.use_case == "custom_ds"

    def test_missing_project_id_raises(self, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        env_file.write_text("USE_CASE=tpch\n")
        monkeypatch.delenv("GCP_PROJECT_ID", raising=False)
        with pytest.raises(KeyError):
            WorkflowConfig.from_env(env_path=env_file)


class TestLangSmithEnv:

    def test_sets_env_vars_when_key_provided(self, monkeypatch):
        monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
        monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
        monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)

        WorkflowConfig.from_values(
            "proj",
            langsmith_api_key="lsv2_test_key",
            langsmith_project="my-project",
        )
        assert os.environ["LANGSMITH_TRACING"] == "true"
        assert os.environ["LANGSMITH_API_KEY"] == "lsv2_test_key"
        assert os.environ["LANGSMITH_PROJECT"] == "my-project"

        # Clean up so LangSmith doesn't attempt real tracing in later tests
        monkeypatch.delenv("LANGSMITH_TRACING")
        monkeypatch.delenv("LANGSMITH_API_KEY")
        monkeypatch.delenv("LANGSMITH_PROJECT")

    def test_does_not_set_env_vars_when_key_empty(self, monkeypatch):
        monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
        monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
        monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)

        WorkflowConfig.from_values("proj", langsmith_api_key="")
        assert "LANGSMITH_API_KEY" not in os.environ


class TestBaseDir:

    def test_default_base_dir_is_a_directory(self):
        cfg = WorkflowConfig.from_values("proj")
        assert isinstance(cfg.base_dir, Path)

    def test_base_dir_can_be_overridden(self, tmp_path):
        cfg = WorkflowConfig.from_values("proj")
        cfg.base_dir = tmp_path
        assert cfg.base_dir == tmp_path
