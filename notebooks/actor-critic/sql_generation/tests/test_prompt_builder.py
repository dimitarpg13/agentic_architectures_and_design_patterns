"""Tests for utils.prompt_builder.PromptBuilder."""

from utils.prompt_builder import PromptBuilder

from tests.conftest import (
    SAMPLE_ACTOR_PROMPT,
    SAMPLE_CRITIC_PROMPT,
    SAMPLE_DATA_DICT,
    SAMPLE_DOMAIN_RULES,
)


class TestBuildActorSystemPrompt:

    def test_contains_base_prompt(self, project_dir):
        pb = PromptBuilder(base_dir=project_dir, use_case="tpch")
        result = pb.build_actor_system_prompt()
        assert SAMPLE_ACTOR_PROMPT in result

    def test_contains_domain_rules(self, project_dir):
        pb = PromptBuilder(base_dir=project_dir, use_case="tpch")
        result = pb.build_actor_system_prompt()
        assert "Revenue = l_extendedprice" in result

    def test_contains_data_dictionary(self, project_dir):
        pb = PromptBuilder(base_dir=project_dir, use_case="tpch")
        result = pb.build_actor_system_prompt()
        assert "l_extendedprice" in result
        assert "DECIMAL" in result

    def test_includes_metadata_when_provided(self, project_dir):
        pb = PromptBuilder(base_dir=project_dir, use_case="tpch")
        result = pb.build_actor_system_prompt(dataset_metadata="schema.lineitem")
        assert "schema.lineitem" in result
        assert "Metadata" in result

    def test_excludes_metadata_section_when_none(self, project_dir):
        pb = PromptBuilder(base_dir=project_dir, use_case="tpch")
        result = pb.build_actor_system_prompt()
        assert "Metadata" not in result

    def test_includes_refresh_date(self, project_dir):
        pb = PromptBuilder(base_dir=project_dir, use_case="tpch")
        result = pb.build_actor_system_prompt(data_refresh_date="2026-02-20")
        assert "2026-02-20" in result

    def test_refresh_date_defaults_to_unknown(self, project_dir):
        pb = PromptBuilder(base_dir=project_dir, use_case="tpch")
        result = pb.build_actor_system_prompt()
        assert "unknown" in result

    def test_includes_current_datetime(self, project_dir):
        pb = PromptBuilder(base_dir=project_dir, use_case="tpch")
        result = pb.build_actor_system_prompt()
        assert "UTC" in result


class TestBuildCriticSystemPrompt:

    def test_contains_base_prompt(self, project_dir):
        pb = PromptBuilder(base_dir=project_dir, use_case="tpch")
        result = pb.build_critic_system_prompt()
        assert SAMPLE_CRITIC_PROMPT in result

    def test_contains_data_dictionary(self, project_dir):
        pb = PromptBuilder(base_dir=project_dir, use_case="tpch")
        result = pb.build_critic_system_prompt()
        assert "l_extendedprice" in result

    def test_contains_domain_rules(self, project_dir):
        pb = PromptBuilder(base_dir=project_dir, use_case="tpch")
        result = pb.build_critic_system_prompt()
        assert "Revenue = l_extendedprice" in result


class TestMissingFiles:

    def test_missing_domain_rules_returns_placeholder(self, tmp_path):
        prompts = tmp_path / "prompts"
        prompts.mkdir()
        (prompts / "actor_system_prompt.md").write_text("base prompt")
        (prompts / "critic_system_prompt.md").write_text("critic prompt")
        uc = tmp_path / "usecases" / "tpch"
        uc.mkdir(parents=True)
        (uc / "DATA_DICTIONARY.md").write_text("data dict")

        pb = PromptBuilder(base_dir=tmp_path, use_case="tpch")
        result = pb.build_actor_system_prompt()
        assert "[File not found:" in result

    def test_missing_usecase_dir_returns_placeholder(self, tmp_path):
        prompts = tmp_path / "prompts"
        prompts.mkdir()
        (prompts / "actor_system_prompt.md").write_text("base prompt")

        pb = PromptBuilder(base_dir=tmp_path, use_case="nonexistent")
        result = pb.build_actor_system_prompt()
        assert "[File not found:" in result
