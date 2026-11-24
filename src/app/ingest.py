"""Conversation import helpers for local text, JSON, and JSONL exports."""

from __future__ import annotations

import json
import re
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


def parse_loading_older_paste(text: str) -> tuple[str, list[dict[str, str]]] | None:
    """Split the ChatGPT page-copy layout shown by the user into prompt and reply."""
    # Some ChatGPT page selections omit role labels and begin with this page
    # status line. In that layout, the title/question sits above the first
    # blank line and the assistant's answer starts below it.
    lines = text.splitlines()
    first_content = next((i for i, line in enumerate(lines) if line.strip()), None)
    if first_content is not None and lines[first_content].strip().lower().startswith("loading older messages"):
        page_lines = lines[first_content + 1:]
        separator = next((i for i, line in enumerate(page_lines) if not line.strip()), None)
        if separator is not None:
            header = [line.strip() for line in page_lines[:separator] if line.strip()]
            answer = "\n".join(page_lines[separator + 1:]).strip()
            if header and answer:
                question_line = next((line for line in reversed(header) if "?" in line), "")
                question_match = re.search(r"([^.!?]*\?)\s*$", question_line)
                question = ""
                title_parts = list(header)
                if question_match:
                    decorative = re.search(r"[⭐🌟]{2,}", question_line)
                    if decorative:
                        question = question_line[decorative.end():].strip()
                        title_parts[-1] = question_line[:decorative.start()].strip()
                    else:
                        question = question_match.group(1).strip()
                        title_parts[-1] = question_line[:question_match.start()].strip()
                if question:
                    title_parts = [re.sub(r"[⭐🌟]+", "", part).strip(" :-") for part in title_parts]
                    title = " / ".join(part for part in title_parts if part) or "Pasted chat"
                    messages = [
                        {"message_id": str(uuid.uuid4()), "role": "user", "content": question},
                        {"message_id": str(uuid.uuid4()), "role": "assistant", "content": answer},
                    ]
                    return title, messages
    return None


def parse_conversation_text(text: str) -> list[dict[str, str]]:
    """Parse pasted chat headings such as User, You said, ChatGPT, or Gemini."""
    page_paste = parse_loading_older_paste(text)
    if page_paste:
        return page_paste[1]

    messages: list[dict[str, str]] = []
    active_role: str | None = None
    active: list[str] = []

    def flush() -> None:
        nonlocal active, active_role
        body = "\n".join(active).strip()
        if body:
            messages.append({"message_id": str(uuid.uuid4()), "role": active_role or "user", "content": body})
        active, active_role = [], None

    role_names = {
        "user": "user", "human": "user", "you": "user", "you said": "user", "user said": "user",
        "assistant": "assistant", "ai": "assistant", "i asked": "assistant", "chatgpt": "assistant", "chatgpt said": "assistant",
        "gemini": "assistant", "gemini said": "assistant", "bard": "assistant", "bard said": "assistant",
        "model": "assistant", "system": "system",
    }
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith("[uploaded image"):
            continue
        prefix, sep, rest = stripped.partition(":")
        role = role_names.get(prefix.strip().lower()) if sep else role_names.get(stripped.lower())
        if role:
            flush()
            active_role = role
            if rest.strip():
                active.append(rest.strip())
        else:
            # Ignore wrapper metadata from exported/pasted transcript files.
            # It describes the attachment, not a message in the conversation.
            if not messages and active_role is None and (
                stripped.lower() == "conversation transcript"
                or stripped.lower().startswith("scope:")
            ):
                continue
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
