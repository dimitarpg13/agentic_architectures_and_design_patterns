from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass
class A2AMessage:
    sender: str
    recipient: str
    text: str
    timestamp_utc: str
    raw_payload: Any


class A2AChannel:
    """
    Lightweight local A2A communication channel.
    Uses a2a-sdk message helpers when available, otherwise falls back to dict payloads.
    """

    def __init__(self) -> None:
        self._log: list[A2AMessage] = []
        self._new_text_message = None
        try:
            from a2a.utils import new_agent_text_message

            self._new_text_message = new_agent_text_message
        except Exception:
            self._new_text_message = None

    def send(self, *, sender: str, recipient: str, text: str) -> A2AMessage:
        payload: Any
        if self._new_text_message is not None:
            try:
                payload = self._new_text_message(text=text)
            except TypeError:
                payload = self._new_text_message(text)
        else:
            payload = {"from": sender, "to": recipient, "text": text}

        message = A2AMessage(
            sender=sender,
            recipient=recipient,
            text=text,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            raw_payload=payload,
        )
        self._log.append(message)
        return message

    def history(self) -> list[A2AMessage]:
        return list(self._log)


