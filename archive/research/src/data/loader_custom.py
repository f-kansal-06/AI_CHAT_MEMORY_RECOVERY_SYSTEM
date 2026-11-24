"""
Custom dataset loader.
Loads ~4,221-turn custom dataset from JSONL or JSON format.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from src.data.schema import TurnRecord, QuestionRecord
from src.data.cleaning import (
    normalize_whitespace,
    count_tokens,
    head_tail_truncate,
    is_english,
    CategoryMapper,
)

logger = logging.getLogger(__name__)


def load_custom_dataset(
    file_path: str,
    category_mapper: Optional[CategoryMapper] = None,
    filter_english: bool = True,
    max_turn_words: int = 250,
) -> Tuple[List[TurnRecord], List[QuestionRecord]]:
    """
    Load custom dataset turns and questions.
    Supports either JSONL (one record per turn) or standard JSON conversation lists.
    """
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Custom dataset file not found: {file_path}")

    if category_mapper is None:
        category_mapper = CategoryMapper()

    all_turns: List[TurnRecord] = []
    all_questions: List[QuestionRecord] = []
    seen_texts_by_session: Dict[str, set] = {}

    # Check if JSONL or JSON
    with open(p, "r", encoding="utf-8") as f:
        first_char = ""
        for line in f:
            stripped = line.strip()
            if stripped:
                first_char = stripped[0]
                break

    if first_char == "[":
        # Full JSON array
        with open(p, "r", encoding="utf-8") as f:
            raw_items = json.load(f)
    else:
        # JSONL format
        raw_items = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    raw_items.append(json.loads(line))

    # Parse records
    for i, item in enumerate(raw_items):
        cid = str(item.get("conv_id", item.get("conversation_id", f"custom_c_{i // 20}")))
        sid = str(item.get("session_id", f"s_{i // 10}"))
        t_idx = int(item.get("turn_idx", i % 10))
        g_idx = int(item.get("global_turn_idx", i))
        speaker = str(item.get("speaker", item.get("role", "user"))).lower()
        if speaker not in ("user", "assistant", "system"):
            speaker = "user" if speaker in ("human", "client") else "assistant"

        raw_text = normalize_whitespace(item.get("text", item.get("content", "")))
        if not raw_text:
            continue

        sess_key = f"{cid}_{sid}"
        if sess_key not in seen_texts_by_session:
            seen_texts_by_session[sess_key] = set()

        norm_key = raw_text.lower()
        if norm_key in seen_texts_by_session[sess_key]:
            continue
        seen_texts_by_session[sess_key].add(norm_key)

        if filter_english and not is_english(raw_text):
            continue

        raw_n_tokens = count_tokens(raw_text)
        truncated_text = head_tail_truncate(raw_text, max_words=max_turn_words)

        label = int(item.get("label", item.get("importance", 0)))
        raw_cat = str(item.get("category", item.get("linked_category", "none")))
        mapped_cat = category_mapper.map_category("custom", raw_cat) if label == 1 else "none"

        qids = item.get("linked_question_ids", [])
        if label == 1 and not qids:
            qids = [f"q_{cid}_{sid}_{t_idx}"]

        rec = TurnRecord(
            conv_id=cid,
            session_id=sid,
            turn_idx=t_idx,
            global_turn_idx=g_idx,
            speaker=speaker,
            text=truncated_text,
            n_tokens=raw_n_tokens,
            label=label,
            linked_question_ids=qids,
            linked_category=mapped_cat,
            source="custom",
            lang=str(item.get("lang", "en")),
        )
        all_turns.append(rec)

        if label == 1:
            for q in qids:
                all_questions.append(
                    QuestionRecord(
                        question_id=q,
                        conv_id=cid,
                        category=mapped_cat,
                        evidence_turn_ids=[f"{cid}_{sid}_{t_idx}"],
                    )
                )

    logger.info(f"Loaded Custom Dataset: {len(all_turns)} turns, {len(all_questions)} question links.")
    return all_turns, all_questions

