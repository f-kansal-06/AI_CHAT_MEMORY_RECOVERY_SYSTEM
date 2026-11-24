"""Create readable summaries and recovery notes from stored memory entries."""

from __future__ import annotations

from typing import Any


def render_summary(
    title: str,
    memories: list[dict[str, Any]],
    messages: list[dict[str, Any]] | None = None,
    stats: dict[str, Any] | None = None,
) -> str:
    """Render a concise, role-aware summary with source-turn references."""
    included = [row for row in memories if not row.get("excluded", 0)]
    source_rows = {
        row["message_id"]: {**row, "sequence": int(row.get("sequence", index))}
        for index, row in enumerate(messages or [])
    }

    def speaker(row: dict[str, Any]) -> str:
        for source in row.get("source_message_ids", []):
            if source in source_rows:
                return str(source_rows[source].get("role", "unknown"))
        return "unknown"

    def recency(row: dict[str, Any]) -> int:
        return max(
            (
                int(source_rows[source].get("sequence", 0))
                for source in row.get("source_message_ids", [])
                if source in source_rows
            ),
            default=0,
        )

    def row_line(row: dict[str, Any]) -> str:
        turns = sorted({
            int(source_rows[source].get("sequence", 0)) + 1
            for source in row.get("source_message_ids", [])
            if source in source_rows
        })
        reference = f" (turn {', '.join(map(str, turns))})" if turns else ""
        text = str(row.get("text", "")).replace("*", "").strip()
        return f"- {text}{reference}"

    lines = [
        f"# Chat summary: {title}",
        "",
        f"Based on {len(messages or [])} messages. Extracted locally with transparent text rules; check the cited turns for nuance.",
    ]
    if stats and stats.get("original_tokens"):
        reduction = float(stats.get("compression_ratio", 0)) * 100
        lines.append(
            f"Estimated text reduction: {reduction:.1f}% by a rough word/character count (not a model tokenizer)."
        )

    def add_section(
        heading: str,
        rows: list[dict[str, Any]],
        limit: int,
        empty_note: str | None = None,
    ) -> None:
        if not rows and not empty_note:
            return
        lines.extend(["", f"## {heading}"])
        if not rows:
            lines.append(f"- {empty_note}")
            return
        ranked = sorted(
            rows,
            key=lambda row: (float(row.get("importance", 0)), recency(row)),
            reverse=True,
        )
        lines.extend(row_line(row) for row in ranked[:limit])
        if len(ranked) > limit:
            lines.append(f"- {len(ranked) - limit} lower-priority points left out to keep this summary short.")

    user_context = [
        row for row in included
        if speaker(row) == "user"
        and row.get("kind") in {"fact", "requirement", "preference"}
    ]
    decisions = [
        row for row in included
        if speaker(row) == "user"
        and row.get("kind") in {"decision", "agreement"}
        and row.get("status") == "confirmed"
    ]
    assistant_guidance = [
        row for row in included
        if speaker(row) == "assistant"
        and row.get("kind") in {"suggestion", "conclusion", "next_action"}
        and row.get("status") != "acknowledged"
    ]
    disputed = [
        row for row in included
        if row.get("kind") in {"claim", "rejected_suggestion"}
    ]
    open_items = [
        row for row in included
        if row.get("kind") in {"uncertainty", "unresolved_question", "next_action"}
        and not (speaker(row) == "assistant" and row.get("kind") == "next_action")
    ]

    add_section("Your goals and requirements", user_context, 10)
    add_section(
        "Decisions you confirmed",
        decisions,
        8,
        "No explicit user decision was detected. Suggestions below are not confirmed choices.",
    )
    add_section("Assistant guidance and ideas (not confirmed decisions)", assistant_guidance, 8)
    add_section("Disagreements or rejected ideas", disputed, 6)
    add_section("Open questions and next steps", open_items, 6)

    if not included:
        lines.extend(["", "No useful memory points were detected. Try pasting the chat with speaker labels."])
    return "\n".join(lines) + "\n"


def render_recovery(title: str, memories: list[dict[str, Any]]) -> str:
    """Render a compact continuation note for callers that do not have turns."""
    included = [row for row in memories if not row.get("excluded", 0)]
    lines = [
        f"# Conversation recovery: {title}",
        "",
        "Use confirmed decisions as the current direction. Suggestions and uncertain points still need review.",
    ]
    sections = [
        ("Confirmed decisions and requirements", {"decision", "requirement", "preference", "agreement"}),
        ("Important context", {"fact", "conclusion"}),
        ("Suggestions (not confirmed)", {"suggestion"}),
        ("Disagreements and rejected ideas", {"claim", "rejected_suggestion"}),
        ("Open questions and next steps", {"uncertainty", "unresolved_question", "next_action"}),
    ]
    for heading, kinds in sections:
        rows = [row for row in included if row.get("kind") in kinds]
        if not rows:
            continue
        lines.extend(["", f"## {heading}"])
        for row in sorted(rows, key=lambda item: float(item.get("importance", 0)), reverse=True):
            refs = row.get("source_message_ids", [])
            ref_text = f" (source IDs: {', '.join(refs)})" if refs else ""
            lines.append(f"- {row.get('text', '')}{ref_text}")
    if not included:
        lines.extend(["", "No included memory points are available."])
    return "\n".join(lines) + "\n"
