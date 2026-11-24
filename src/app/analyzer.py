"""Conservative, fully local memory extraction with message-level provenance.

This module intentionally uses transparent rules rather than pretending that a
turn salience classifier can perform semantic fact extraction.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from src.data.cleaning import count_tokens

_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
_DISAGREEMENT = re.compile(r"\b(i disagree|we disagree|that is wrong|that's wrong|not true|incorrect|false claim|i reject|we reject)\b", re.I)
_BARE_ACKNOWLEDGEMENT = re.compile(r"^(?:agreed|sounds good|that'?s right|that is right|correct|yes)[.! ]*$", re.I)
_UNCERTAIN = re.compile(r"\b(maybe|perhaps|possibly|might|may be|could be|not sure|i am uncertain|i'm uncertain|uncertain whether|uncertain if|unclear whether|unknown whether|unverified|i think|i believe|seems)\b", re.I)
_SUGGESTION = re.compile(r"\b(i suggest|suggestion|could try|you could|consider|recommend|option is|one option|we could)\b", re.I)
_AGREEMENT = re.compile(r"\b(agreed|we agree|sounds good|that's right|that is right|correct|let's do|we'll use|we will use|decided|confirmed|yes,? (?:that|this|it))\b", re.I)
_REQUIREMENT = re.compile(r"\b(requirement|must|need to|needs to|should|please|ensure|make sure|do not|don't|avoid|only|always|never)\b", re.I)
_DECISION = re.compile(r"\b(decided|decision|we will use|we'll use|going with|chosen|selected|settled on|finalized)\b", re.I)
_ACTION = re.compile(r"\b(next|todo|to do|action item|follow up|will (?:investigate|build|review|send|check|implement)|need to (?:investigate|build|review|send|check|implement))\b", re.I)
_CONCLUSION = re.compile(r"\b(conclusion|therefore|in summary|we found|result is|it follows)\b", re.I)
_PREFERENCE = re.compile(r"\b(i prefer|i like|i dislike|i hate|my preference|prefer(?:s)?|rather than)\b", re.I)
_QUESTION = re.compile(r"\?\s*$")
_UNRESOLVED = re.compile(r"\b(still need to decide|need to determine|not yet decided|open question|remains unclear|need to find out)\b", re.I)


def _kind_and_status(text: str, role: str) -> tuple[str, str, float]:
    if _UNCERTAIN.search(text):
        return "uncertainty", "uncertain", 0.55
    if role == "user" and re.match(r"\s*(?:i reject|we reject)\b", text, re.I):
        return "rejected_suggestion", "rejected", 0.78
    if _DISAGREEMENT.search(text):
        return "claim", "disputed", 0.62
    if _UNRESOLVED.search(text):
        return "unresolved_question", "unresolved", 0.72
    if _SUGGESTION.search(text):
        return "suggestion", "proposed", 0.68
    if _ACTION.search(text):
        return "next_action", "open", 0.75
    if _REQUIREMENT.search(text) and role == "user":
        return "requirement", "stated", 0.82
    if _QUESTION.search(text):
        return "unresolved_question", "unresolved", 0.68
    if _DECISION.search(text):
        return "decision", "confirmed", 0.88 if role == "user" else 0.75
    if _AGREEMENT.search(text):
        return "agreement", "confirmed", 0.86
    if _ACTION.search(text):
        return "next_action", "open", 0.75
    if _CONCLUSION.search(text):
        return "conclusion", "stated", 0.76
    if _PREFERENCE.search(text):
        return "preference", "stated", 0.82
    if len(text.split()) >= 5:
        return "fact", "stated", 0.60 if role == "assistant" else 0.76
    return "", "", 0.0


def analyze_conversation(messages: list[dict[str, Any]], importance_scores: dict[str, float] | None = None, importance_model: str | None = None) -> dict[str, Any]:
    """Create compact entries; duplicates link all source message IDs."""
    entries: list[dict[str, Any]] = []
    seen: dict[tuple[str, str], dict[str, Any]] = {}
    ordered = sorted(messages, key=lambda m: int(m.get("sequence", 0)))
    for message in ordered:
        text = str(message.get("content", ""))
        role = str(message.get("role", "user"))
        mid = str(message["message_id"])
        for sentence in _SENTENCE.split(text):
            sentence = sentence.strip(" \t-*•")
            # A bare acknowledgement has no subject and cannot be recovered
            # meaningfully without guessing what it refers to.
            if _BARE_ACKNOWLEDGEMENT.fullmatch(sentence):
                continue
            if len(sentence.split()) < 3 and not _AGREEMENT.search(sentence):
                continue
            kind, status, confidence = _kind_and_status(sentence, role)
            if not kind:
                continue
            norm = " ".join(re.findall(r"\w+", sentence.lower()))
            key = (kind, norm)
            if key in seen:
                if mid not in seen[key]["source_message_ids"]:
                    seen[key]["source_message_ids"].append(mid)
                continue
            entry = {
                "kind": kind,
                "status": status,
                "text": sentence,
                "confidence": confidence,
                "source_message_ids": [mid],
                "conflict_group": None,
                "importance": _combined_importance(sentence, kind, mid, importance_scores),
                "model_importance": (float(importance_scores[mid]) if importance_scores and mid in importance_scores else None),
            }
            seen[key] = entry
            entries.append(entry)

    # Disagreement is marked only when a message uses an explicit cue.
    # Pairing sentences by word overlap and negation produced false conflicts.

    entries.sort(key=lambda e: (-e["importance"], e["source_message_ids"][0]))
    stats = {
        "message_count": len(messages),
        "original_tokens": sum(count_tokens(str(m.get("content", ""))) for m in messages),
        "memory_tokens": sum(count_tokens(e["text"]) for e in entries),
        "entry_count": len(entries),
    }
    stats["compression_ratio"] = 1 - stats["memory_tokens"] / max(stats["original_tokens"], 1)
    return {
        "entries": entries, "stats": stats, "overview": _overview(entries),
        "extractor": f"local_rules_v1+{importance_model}" if importance_model else "local_rules_v1",
        "metadata": {"extractor_version": "local_rules_v1", "importance_model": importance_model, "importance_model_used": bool(importance_scores)},
    }


def _importance(text: str, kind: str) -> float:
    base = {"decision": .96, "requirement": .95, "preference": .88, "agreement": .88, "conclusion": .85, "fact": .80, "uncertainty": .78, "rejected_suggestion": .78, "suggestion": .56, "unresolved_question": .68, "next_action": .76, "claim": .65}.get(kind, .5)
    if len(text.split()) > 40:
        base -= .05
    return round(base, 3)


def _combined_importance(text: str, kind: str, message_id: str, scores: dict[str, float] | None) -> float:
    rule_score = _importance(text, kind)
    if not scores or message_id not in scores:
        return rule_score
    # The classifier predicts benchmark turn salience; rule extraction provides
    # the distinct structural signal. Keep both signals visible on each entry.
    return round(0.75 * rule_score + 0.25 * float(scores[message_id]), 3)


def _overview(entries: list[dict[str, Any]]) -> str:
    counts: dict[str, int] = defaultdict(int)
    for e in entries:
        counts[e["kind"]] += 1
    kinds = ", ".join(f"{n} {k.replace('_', ' ')}" for k, n in sorted(counts.items()))
    highlights = [e["text"] for e in entries if e["kind"] in {"decision", "requirement", "conclusion", "fact"}][:3]
    if highlights:
        return f"{len(entries)} memory items ({kinds}). Key context: " + " ".join(highlights)
    return f"Extracted {len(entries)} distinct memory items: {kinds}."
