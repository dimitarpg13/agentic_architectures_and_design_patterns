"""System-prompt assembly following the PromptBuilder pattern.

The builder reads domain documents from the ``usecases/<slug>/`` and
``prompts/`` subdirectories, then stitches them into a single system
message that grounds the LLM.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class PromptBuilder:
    """Construct system messages for the Actor and Critic agents."""

    def __init__(self, base_dir: Path, use_case: str = "tpch"):
        self.base_dir = Path(base_dir)
        self.prompts_dir = self.base_dir / "prompts"
        self.use_case_dir = self.base_dir / "usecases" / use_case

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_actor_system_prompt(
        self,
        dataset_metadata: Optional[str] = None,
        data_refresh_date: Optional[str] = None,
    ) -> str:
        """Assemble the full Actor system message.

        Mirrors the ``PromptBuilder.construct_system_message`` layout from
        the actor-critic reference design (see ``user_input.md``).
        """
        base_prompt = self._read("prompts/actor_system_prompt.md")
        domain_rules = self._read_usecase("DOMAIN_RULES.md")
        data_dictionary = self._read_usecase("DATA_DICTIONARY.md")

        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        refresh = data_refresh_date or "unknown"

        sections = [
            "# Instructions\n" + base_prompt,
            "## **Business Context**\n" + domain_rules,
            "## **Data Dictionary**\n" + data_dictionary,
        ]

        if dataset_metadata:
            sections.append("## **Metadata**\n" + dataset_metadata)

        sections.append(f"## **Data Refresh Date**\nThe data was last refreshed on {refresh}.")
        sections.append(f"## **Current Date & Time**\n{now_utc}")

        return "\n\n".join(sections)

    def build_critic_system_prompt(self) -> str:
        """Assemble the full Critic system message.

        The Critic needs the data dictionary and domain rules in context
        to validate SQL against the actual schema and business logic.
        """
        base_prompt = self._read("prompts/critic_system_prompt.md")
        data_dictionary = self._read_usecase("DATA_DICTIONARY.md")
        domain_rules = self._read_usecase("DOMAIN_RULES.md")

        sections = [
            "# Instructions\n" + base_prompt,
            "## **Data Dictionary Reference**\n" + data_dictionary,
            "## **Domain Rules Reference**\n" + domain_rules,
        ]

        return "\n\n".join(sections)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _read(self, relative: str) -> str:
        path = self.base_dir / relative
        if path.exists():
            text = path.read_text(encoding="utf-8").strip()
            logger.debug("Loaded %d chars from %s", len(text), path.name)
            return text
        logger.warning("File not found: %s", relative)
        return f"[File not found: {relative}]"

    def _read_usecase(self, filename: str) -> str:
        path = self.use_case_dir / filename
        if path.exists():
            text = path.read_text(encoding="utf-8").strip()
            logger.debug("Loaded %d chars from %s", len(text), path.name)
            return text
        logger.warning("Use-case file not found: %s", path)
        return f"[File not found: usecases/{self.use_case_dir.name}/{filename} — create it before running the workflow]"

