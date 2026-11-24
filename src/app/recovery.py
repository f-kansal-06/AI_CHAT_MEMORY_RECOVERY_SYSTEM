"""Render recovered continuation context from stored memory entries only."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def render_recovery(title: str, memories: list[dict[str, Any]]) -> str:
    excluded_count = sum(bool(row.get("excluded", 0)) for row in memories)
    memories = [row for row in memories if not row.get("excluded", 0)]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in memories:
        groups[row["kind"]].append(row)
    labels = {
        "fact": "Important facts", "decision": "Decisions and conclusions", "conclusion": "Decisions and conclusions",
        "requirement": "User requirements", "preference": "User requirements", "agreement": "Agreements",
        "claim": "Disagreements and disputed claims", "rejected_suggestion": "Disagreements and rejected suggestions",
        "suggestion": "Suggestions (not confirmed)", "uncertainty": "Uncertainties", "unresolved_question": "Unresolved questions", "next_action": "Next actions",
    }
    ordered = ["fact", "decision", "conclusion", "requirement", "preference", "agreement", "claim", "rejected_suggestion", "suggestion", "uncertainty", "unresolved_question", "next_action"]
    lines = [f"# Conversation recovery: {title}", "", "## Overview", f"Included {len(memories)} memory items; excluded {excluded_count}. This recovery uses the saved memory and source references; it does not reread the original transcript."]
    for kind in ordered:
        rows = groups.get(kind, [])
        if not rows:
            continue
        lines.extend(["", f"## {labels[kind]}"])
        for row in rows:
            refs = ", ".join(row["source_message_ids"])
            status = row["status"]
            suffix = f" [{status}; confidence {row['confidence']:.2f}; importance {row['importance']:.2f}"
            if row.get("model_importance") is not None:
                suffix += f"; model score {row['model_importance']:.2f}"
            suffix += f"; source: {refs}]"
            if row.get("conflict_group"):
                suffix += f" [conflict group: {row['conflict_group']}]"
            lines.append(f"- {row['text']}{suffix}")
    lines.extend(["", "## Suggested next steps", "- Review unresolved items and disputed claims against their source messages.", "- Continue from the open next actions above.", "", "## Compact continuation context", f"Continue the conversation titled '{title}'. Treat items marked proposed, uncertain, disputed, rejected, or unresolved according to their status; do not treat suggestions as agreed facts. Source message IDs are included for traceability."])
    for kind in ordered:
        for row in groups.get(kind, []):
            lines.append(f"- ({row['kind']}; {row['status']}) {row['text']}")
    return "\n".join(lines)
