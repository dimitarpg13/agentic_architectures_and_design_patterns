"""Assembles system prompts from on-disk markdown files.

Mirrors the PromptBuilder.construct_system_message pattern from
actor-critic/user_input.md — combining a base behavioral prompt with
domain rules, data dictionary, and dataset metadata.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


class PromptBuilder:
    """Reads prompt fragments from disk and assembles them into
    complete system messages for the Actor and the Critic."""

    def __init__(self, use_case: str, base_dir: Path | None = None):
        self.use_case = use_case
        self.base_dir = base_dir or Path(__file__).resolve().parent.parent
        self._prompts_dir = self.base_dir / "prompts"
        self._usecase_dir = self.base_dir / "usecases" / use_case

    def _read(self, path: Path) -> str:
        text = path.read_text(encoding="utf-8")
        logger.debug("Loaded %d chars from %s", len(text), path.name)
        return text

    @property
    def domain_rules(self) -> str:
        return self._read(self._usecase_dir / "DOMAIN_RULES.md")

    @property
    def data_dictionary(self) -> str:
        return self._read(self._usecase_dir / "DATA_DICTIONARY.md")

    def build_actor_system_message(self) -> str:
        base_prompt = self._read(self._prompts_dir / "actor_system_prompt.md")
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        return (
            f"# Instructions\n{base_prompt}\n\n"
            f"## Domain Rules\n{self.domain_rules}\n\n"
            f"## Data Dictionary\n{self.data_dictionary}\n\n"
            f"## Current Date & Time\n{now}\n"
        )

    def build_critic_system_message(self) -> str:
        base_prompt = self._read(self._prompts_dir / "critic_system_prompt.md")

        return (
            f"# Instructions\n{base_prompt}\n\n"
            f"## Data Dictionary Reference\n{self.data_dictionary}\n\n"
            f"## Domain Rules Reference\n{self.domain_rules}\n"
        )
