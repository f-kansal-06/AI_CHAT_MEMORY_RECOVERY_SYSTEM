"""Conversation import helpers for local text, JSON, and JSONL exports."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any


def _messages_from_object(data: Any) -> list[dict[str, str]]:
    if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict) and "mapping" in data[0]:
        data = data[0]
    if isinstance(data, dict):
        if isinstance(data.get("conversations"), list):
            conversations = data["conversations"]
            if len(conversations) != 1:
                raise ValueError("This file contains multiple conversations; import one conversation at a time")
            data = conversations[0]
        if isinstance(data.get("mapping"), dict):
            mapped = []
            for node in data["mapping"].values():
                message = node.get("message") if isinstance(node, dict) else None
                if not isinstance(message, dict):
                    continue
                author = message.get("author", {})
                role = author.get("role", "user") if isinstance(author, dict) else "user"
                body = message.get("content", {})
                if isinstance(body, dict):
                    body = body.get("parts", body.get("text", ""))
                mapped.append({"id": message.get("id"), "role": role, "content": body, "create_time": message.get("create_time", node.get("create_time", 0))})
            if mapped and any(item.get("create_time") for item in mapped):
                mapped.sort(key=lambda item: item.get("create_time") or 0)
            data = mapped
        elif "messages" in data:
            data = data["messages"]
        elif "conversation" in data and isinstance(data["conversation"], list):
            data = data["conversation"]
        elif "contents" in data:
            data = data["contents"]
        elif "turns" in data:
            data = data["turns"]
        else:
            raise ValueError("JSON object must contain messages, conversation, or turns")
    if not isinstance(data, list):
        raise ValueError("Conversation data must be a list of messages")
    messages = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Message {i + 1} must be a JSON object")
        role = str(item.get("role", item.get("speaker", "user"))).lower()
        role = {"human": "user", "ai": "assistant", "model": "assistant"}.get(role, role)
        if role not in {"user", "assistant", "system"}:
            role = "user"
        content = item.get("content", item.get("text", item.get("message", "")))
        if isinstance(content, list):
            content = "\n".join(str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in content)
        content = str(content).strip()
        if content:
            messages.append({"message_id": str(item.get("message_id") or item.get("id") or uuid.uuid4()), "role": role, "content": content})
    if not messages:
        raise ValueError("No non-empty messages found")
    return messages


def parse_conversation_text(text: str) -> list[dict[str, str]]:
    """Parse simple `User:` / `Assistant:` transcripts, keeping unmarked lines."""
    messages: list[dict[str, str]] = []
    active_role: str | None = None
    active: list[str] = []

    def flush() -> None:
        nonlocal active, active_role
        body = "\n".join(active).strip()
        if body:
            messages.append({"message_id": str(uuid.uuid4()), "role": active_role or "user", "content": body})
        active, active_role = [], None

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        prefix, sep, rest = stripped.partition(":")
        role = {"user": "user", "human": "user", "assistant": "assistant", "ai": "assistant", "system": "system"}.get(prefix.strip().lower()) if sep else None
        if role:
            flush()
            active_role = role
            if rest.strip():
                active.append(rest.strip())
        else:
            active.append(stripped)
    flush()
    if not messages and text.strip():
        messages.append({"message_id": str(uuid.uuid4()), "role": "user", "content": text.strip()})
    return messages


def load_conversation(path: str | Path) -> tuple[str, list[dict[str, str]]]:
    path = Path(path)
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8-sig")
    if suffix == ".txt":
        title = path.stem
        messages = parse_conversation_text(text)
    elif suffix == ".json":
        data = json.loads(text)
        title = str(data.get("title") or path.stem) if isinstance(data, dict) else path.stem
        messages = _messages_from_object(data)
    elif suffix == ".jsonl":
        title = path.stem
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        messages = _messages_from_object(rows)
    else:
        raise ValueError("Supported conversation formats are TXT, JSON, and JSONL")
    return title, messages
