"""
Unified schema definitions for conversation memory importance benchmark.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
import json


@dataclass
class TurnRecord:
    conv_id: str
    session_id: str
    turn_idx: int
    global_turn_idx: int
    speaker: str
    text: str
    n_tokens: int
    label: int  # 0 or 1
    linked_question_ids: List[str] = field(default_factory=list)
    linked_category: str = "none"
    source: str = "longmemeval"  # "longmemeval" | "locomo" | "custom" | "wildchat"
    lang: str = "en"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TurnRecord":
        return cls(
            conv_id=str(data["conv_id"]),
            session_id=str(data["session_id"]),
            turn_idx=int(data["turn_idx"]),
            global_turn_idx=int(data["global_turn_idx"]),
            speaker=str(data["speaker"]),
            text=str(data["text"]),
            n_tokens=int(data["n_tokens"]),
            label=int(data["label"]),
            linked_question_ids=list(data.get("linked_question_ids", [])),
            linked_category=str(data.get("linked_category", "none")),
            source=str(data.get("source", "longmemeval")),
            lang=str(data.get("lang", "en")),
        )


@dataclass
class QuestionRecord:
    question_id: str
    conv_id: str
    category: str
    evidence_turn_ids: List[str] = field(default_factory=list)
    question_text: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QuestionRecord":
        return cls(
            question_id=str(data["question_id"]),
            conv_id=str(data["conv_id"]),
            category=str(data["category"]),
            evidence_turn_ids=list(data.get("evidence_turn_ids", [])),
            question_text=data.get("question_text"),
        )


def validate_turn_record(record: Dict[str, Any]) -> bool:
    """Validate that a dictionary strictly conforms to the TurnRecord unified schema."""
    required_fields = {
        "conv_id": (str, int),
        "session_id": (str, int),
        "turn_idx": int,
        "global_turn_idx": int,
        "speaker": str,
        "text": str,
        "n_tokens": int,
        "label": int,
        "linked_question_ids": list,
        "linked_category": str,
        "source": str,
        "lang": str,
    }
    for field_name, expected_type in required_fields.items():
        if field_name not in record:
            return False
        if not isinstance(record[field_name], expected_type):
            return False
    if record["label"] not in (0, 1):
        return False
    if record["source"] not in ("longmemeval", "locomo", "custom", "wildchat"):
        return False
    return True

