"""
Dataset loading and input formatting for DistilBERT and baselines.
Format: [speaker] current_turn [SEP] previous_turn_text (context window: 0, 1, 2).
"""

from typing import List, Dict, Any, Optional
import json
from pathlib import Path


def format_turn_input(
    current_turn: Dict[str, Any],
    previous_turns: List[Dict[str, Any]],
    context_window: int = 1,
    max_prev_chars: int = 400,
) -> str:
    """
    Construct model input text:
    [speaker] current_turn [SEP] previous_turn_text
    context_window can be 0, 1, or 2.
    """
    speaker = current_turn.get("speaker", "user")
    text = current_turn.get("text", "").strip()
    prefix = f"[{speaker}] {text}"

    if context_window <= 0 or not previous_turns:
        return prefix

    selected_prev = previous_turns[-context_window:]
    prev_snippets = []
    for prev in selected_prev:
        p_speaker = prev.get("speaker", "assistant")
        p_text = prev.get("text", "").strip()
        if len(p_text) > max_prev_chars:
            p_text = p_text[:max_prev_chars] + "..."
        prev_snippets.append(f"[{p_speaker}] {p_text}")

    context_str = " [SEP] ".join(prev_snippets)
    return f"{prefix} [SEP] {context_str}"


def load_turns_from_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_formatted_samples(
    turns: List[Dict[str, Any]],
    context_window: int = 1,
) -> List[Dict[str, Any]]:
    """
    Group turns by conversation and construct input texts with previous turns.
    Returns list of dicts with:
    - input_text: formatted string
    - label: int (0 or 1)
    - category: str
    - n_tokens: int
    - conv_id, session_id, turn_idx, global_turn_idx
    """
    # Group turns by conv_id maintaining order
    convs: Dict[str, List[Dict[str, Any]]] = {}
    for t in turns:
        cid = t["conv_id"]
        if cid not in convs:
            convs[cid] = []
        convs[cid].append(t)

    formatted_samples = []
    for cid, conv_turns in convs.items():
        # Sort by global_turn_idx or session + turn_idx
        conv_turns = sorted(conv_turns, key=lambda x: x.get("global_turn_idx", 0))
        for i, turn in enumerate(conv_turns):
            prev_turns = conv_turns[:i]
            input_text = format_turn_input(turn, prev_turns, context_window=context_window)
            sample = dict(turn)
            sample["input_text"] = input_text
            formatted_samples.append(sample)

    return formatted_samples

