"""Conservative, fully local memory extraction with message-level provenance.

This module intentionally uses transparent rules rather than pretending that a
turn salience classifier can perform semantic fact extraction.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any



def _count_tokens(text: str) -> int:
    """Return a lightweight, dependency-free token estimate."""
    if not text:
        return 0
    return max(len(text.split()), int(len(text) / 4))

_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
_DISAGREEMENT = re.compile(r"\b(i disagree|we disagree|that is wrong|that's wrong|not true|incorrect|false claim|i reject|we reject)\b", re.I)
_BARE_ACKNOWLEDGEMENT = re.compile(r"^(?:agreed|sounds good|that'?s right|that is right|correct|yes)[.! ]*$", re.I)
_UNCERTAIN = re.compile(r"\b(maybe|perhaps|possibly|might|may be|could be|not sure|i am uncertain|i'm uncertain|uncertain whether|uncertain if|unclear whether|unknown whether|unverified)\b", re.I)
_SUGGESTION = re.compile(r"\b(i suggest(?:ed)?|i'd suggest|i would suggest|the advice (?:is|was) to|i added|strongest version would|suggested plan|keep applying|apply broadly|apply if eligible|seek referrals|improve your resume|do not pause|don't pause|do not wait|don't wait|focus on|suggestion|could try|you could|consider|recommend|option is|one option|we could)\b", re.I)
_RECOMMENDATION = re.compile(r"\b(i recommend|i'd recommend|i would recommend|i'd choose|i would choose|i'd pick|i would pick|my recommendation|best choice is)\b", re.I)
_AGREEMENT = re.compile(r"\b(we agree|agreed(?: on| to| that)|sounds good|that's right|that is right|let's do|yes,? (?:that|this|it))\b", re.I)
_REQUIREMENT = re.compile(r"\b(requirement|must|need to|needs to|should|please|ensure|make sure|do not|don't|avoid|only|always|never)\b", re.I)
_DECISION = re.compile(r"\b(decided to|we decided|i decided|decision to|we will use|we'll use|going with|let's go with|let us go with|let's choose|let us choose|i choose|we choose|we chose|i chose|we have chosen|i have chosen|we settled on|i settled on|we finalized|i finalized)\b", re.I)
_ACTION = re.compile(r"\b(next\s+(?:step|action)|next,|todo|to do|action item|follow up|will (?:investigate|build|review|send|check|implement)|need to (?:investigate|build|review|send|check|implement))\b", re.I)
_CONCLUSION = re.compile(r"\b(i actually think|my favorite|that shift|the key is|the main gaps?|the important distinction|therefore|in summary|we found|result is|it follows|very few build|can be more|more impressive|not automatically|depends on|the reason|the issue|the problem|that means|this means|as a result|connects .{0,50} to)\b", re.I)
_PREFERENCE = re.compile(r"\b(i prefer|i like|i dislike|i hate|i don't want|i dont want|i do not want|i want to avoid|my concern is|my preference|prefer(?:s)?|rather than)\b", re.I)
_QUESTION = re.compile(r"(?:\?\s*$|^\s*(?:why|what|who|when|where|which|how|can|could|would|shouldn['’]?t|should|is|are|am i|so am i|do|does|did|will|have|has|so what|so do you have|resume impact)\b)", re.I)
_UNRESOLVED = re.compile(r"\b(still need to decide|need to determine|not yet decided|open question|remains unclear|need to find out)\b", re.I)
_METRIC_ROW = re.compile(r"^\s*(?:metric\s+score|technical depth|business value|recruiter understanding|mba value|swe value|industry relevance|overall)\s+[\d.]+\s*$", re.I)
_LIST_HEADER = re.compile(r"\b(suggested technical components|technical components|potential features|possible concepts|examples|main gaps|gaps to validate|weekly allocation|aim for|scope includes|key features|suggested features)\b", re.I)


def _split_sentences(text: str) -> list[str]:
    protected_dot = "\u2024"
    protected = re.sub(
        r"\b(?:e\.g|i\.e)\.",
        lambda match: match.group(0).replace(".", protected_dot),
        text,
        flags=re.I,
    )
    protected = re.sub(
        r"(?m)^(\s*\d+)\.(?=\s)",
        lambda match: match.group(1) + protected_dot,
        protected,
    )
    return [part.replace(protected_dot, ".") for part in _SENTENCE.split(protected)]


def _kind_and_status(text: str, role: str) -> tuple[str, str, float]:
    if _RECOMMENDATION.search(text):
        return "suggestion", "proposed", 0.78
    if _LIST_HEADER.search(text) and role == "assistant":
        return "suggestion", "proposed", 0.68
    if role == "user" and re.match(r"\s*(?:no[, ]+)?(?:i reject|we reject)\b", text, re.I):
        return "rejected_suggestion", "rejected", 0.78
    if _DISAGREEMENT.search(text):
        return "claim", "disputed", 0.62
    if _UNRESOLVED.search(text):
        return "unresolved_question", "unresolved", 0.72
    # A question can already have an answer later in the chat. Keep explicit
    # open-question phrases above; do not label every question unresolved.
    if _QUESTION.search(text):
        if role == "user":
            return "question", "asked", 0.64
        return "", "", 0.0
    if _SUGGESTION.search(text):
        return "suggestion", "proposed", 0.68
    if _UNCERTAIN.search(text):
        return "uncertainty", "uncertain", 0.55
    if _ACTION.search(text):
        return "next_action", "open", 0.75
    if _REQUIREMENT.search(text) and role == "user":
        return "requirement", "stated", 0.82
    if _DECISION.search(text):
        if role == "user":
            return "decision", "confirmed", 0.88
        return "suggestion", "proposed", 0.72
    if _AGREEMENT.search(text):
        if role == "user":
            return "agreement", "confirmed", 0.86
        return "agreement", "acknowledged", 0.55
    if _ACTION.search(text):
        return "next_action", "open", 0.75
    if _CONCLUSION.search(text):
        return "conclusion", "stated", 0.76
    if _PREFERENCE.search(text):
        if role == "user":
            return "preference", "stated", 0.82
        return "suggestion", "proposed", 0.65
    # User statements provide personal context. Assistant prose is not treated
    # as a fact by default: doing so filled summaries with generic explanations
    # and made guidance look verified. Keep assistant content only when a rule
    # above identifies its role (advice, conclusion, action, etc.).
    if role == "user" and len(text.split()) >= 5:
        return "fact", "stated", 0.76
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
        raw_sentences = _split_sentences(text)
        sentences: list[str] = []
        position = 0
        while position < len(raw_sentences):
            sentence = raw_sentences[position].strip(" \t-*•")
            if sentence.endswith(":"):
                bullet_items: list[str] = []
                next_position = position + 1
                while next_position < len(raw_sentences):
                    candidate = raw_sentences[next_position].strip()
                    if not re.match(r"^(?:[-*•]|\d+[.)])\s+", candidate):
                        break
                    bullet_items.append(re.sub(r"^(?:[-*•]|\d+[.)])\s+", "", candidate).strip())
                    next_position += 1
                if bullet_items:
                    header = sentence[:-1].strip()
                    contents = "; ".join(bullet_items)
                    if role == "assistant" and not _LIST_HEADER.search(sentence):
                        sentence = f"Suggested plan ({header}): {contents}"
                    else:
                        sentence = f"{sentence} {contents}"
                    position = next_position - 1
            elif (
                _RECOMMENDATION.search(sentence)
                and sentence.endswith(":")
                and position + 1 < len(raw_sentences)
                and len(raw_sentences[position + 1].split()) <= 8
            ):
                sentence = f"{sentence} {raw_sentences[position + 1].strip()}"
                position += 1
            sentences.append(sentence)
            position += 1
        for sentence in sentences:
            decorative = re.search(r"[⭐🌟]{2,}", sentence)
            if decorative and "?" in sentence[decorative.end():]:
                sentence = sentence[decorative.end():]
            sentence = sentence.replace("⭐", "").replace("🌟", "").replace("*", "").strip()
            # A bare acknowledgement has no subject and cannot be recovered
            # meaningfully without guessing what it refers to.
            if _BARE_ACKNOWLEDGEMENT.fullmatch(sentence):
                continue
            if "|" in sentence or _METRIC_ROW.fullmatch(sentence):
                continue
            # Questions in assistant answers are often examples of questions a
            # user might ask, not unresolved questions in the current chat.
            if role != "user" and _QUESTION.search(sentence):
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
        "original_tokens": sum(_count_tokens(str(m.get("content", ""))) for m in messages),
        "memory_tokens": sum(_count_tokens(e["text"]) for e in entries),
        "entry_count": len(entries),
    }
    stats["compression_ratio"] = 1 - stats["memory_tokens"] / max(stats["original_tokens"], 1)
    return {
        "entries": entries, "stats": stats, "overview": _overview(entries),
        "extractor": f"local_rules_v1+{importance_model}" if importance_model else "local_rules_v1",
        "metadata": {"extractor_version": "local_rules_v1", "importance_model": importance_model, "importance_model_used": bool(importance_scores)},
    }


def _importance(text: str, kind: str) -> float:
    base = {"decision": .96, "requirement": .95, "preference": .88, "agreement": .88, "conclusion": .85, "fact": .80, "uncertainty": .78, "rejected_suggestion": .78, "suggestion": .56, "unresolved_question": .68, "question": .64, "next_action": .76, "claim": .65}.get(kind, .5)
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
