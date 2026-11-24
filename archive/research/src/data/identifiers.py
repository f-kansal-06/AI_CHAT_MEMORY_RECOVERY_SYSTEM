"""Exact aliases for historical benchmark evidence identifier conventions."""

from typing import Any


def turn_id_aliases(turn: dict[str, Any], conversation_id: str) -> set[str]:
    session = str(turn.get("session_id", ""))
    index = int(turn.get("turn_idx", 0))
    aliases = {
        f"{conversation_id}_{session}_{index}",
        f"{conversation_id}_{turn.get('global_turn_idx')}",
        f"{conversation_id}_{session}:{index}",
    }
    # LoCoMo evidence references Dn:k are 1-based while TurnRecord indices
    # are zero-based. This alias is namespaced and exact.
    if session.upper().startswith("D"):
        aliases.add(f"{conversation_id}_{session}:{index + 1}")
    return aliases


def questions_grounded_in_turns(questions: list[Any], turns: list[dict[str, Any]]) -> list[Any]:
    """Keep only questions whose complete evidence set exists in these turns."""
    aliases_by_conversation: dict[str, set[str]] = {}
    for turn in turns:
        cid = str(turn.get("conv_id", "default"))
        aliases_by_conversation.setdefault(cid, set()).update(turn_id_aliases(turn, cid))
    valid = []
    for question in questions:
        evidence = list(getattr(question, "evidence_turn_ids", []))
        available = aliases_by_conversation.get(str(getattr(question, "conv_id", "")), set())
        if evidence and set(evidence).issubset(available):
            valid.append(question)
    return valid
