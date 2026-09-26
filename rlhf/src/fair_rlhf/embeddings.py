"""Small text helpers shared by reward-model preparation and training."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


def normalize_chat_context(
    value: object,
    *,
    fallback_prompt: str | None = None,
) -> tuple[ChatMessage, ...]:
    """Convert serialized chat messages to a simple immutable representation."""

    if value is None:
        return (ChatMessage("user", str(fallback_prompt)),)

    messages = tuple(
        ChatMessage(str(message["role"]), str(message["content"]))
        for message in value
        if isinstance(message, Mapping)
    )
    if not messages:
        raise ValueError("context must contain at least one message")
    return messages


def prompt_from_context(context: Sequence[ChatMessage]) -> str:
    """Return the canonical prompt text used for IDs and split grouping."""

    if len(context) == 1 and context[0].role == "user":
        return context[0].content
    payload = [message.__dict__ for message in context]
    return "chat_context_v1:" + json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def stable_response_id(prompt: str, response: str) -> str:
    """Hash one response together with the prompt that gives it meaning."""

    return sha256(f"{prompt}\0{response}".encode()).hexdigest()
