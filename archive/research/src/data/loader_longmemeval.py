"""
LongMemEval dataset loader.
Parses official LongMemEval_S format with question haystacks and turn-level has_answer evidence flags.
Handles official longmemeval_s_cleaned.json schema with haystack_sessions and haystack_session_ids.
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


def load_longmemeval(
    file_path: str,
    category_mapper: Optional[CategoryMapper] = None,
    filter_english: bool = False,
    max_turn_words: int = 250,
    max_instances: Optional[int] = None,
) -> Tuple[List[TurnRecord], List[QuestionRecord]]:
    """
    Load and parse LongMemEval JSON into unified TurnRecord and QuestionRecord.
    Supports official haystack_sessions and haystack_session_ids.
    """
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"LongMemEval file not found: {file_path}")

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        instances = data.get("instances", data.get("data", [data]))
    elif isinstance(data, list):
        instances = data
    else:
        raise ValueError(f"Unexpected JSON format in {file_path}")

    if max_instances is not None and len(instances) > max_instances:
        instances = instances[:max_instances]

    if category_mapper is None:
        category_mapper = CategoryMapper()

    all_turns: List[TurnRecord] = []
    all_questions: List[QuestionRecord] = []
    dropped_non_english = 0

    for item in instances:
        qid = str(item.get("question_id", item.get("id", f"q_{len(all_questions)}")))
        q_text = item.get("question", "")
        raw_cat = item.get("question_type", item.get("category", "single_fact"))
        mapped_cat = category_mapper.map_category("longmemeval", raw_cat)
        cid = str(item.get("conv_id", qid))

        raw_sessions = item.get("haystack_sessions", item.get("sessions", []))
        raw_session_ids = item.get("haystack_session_ids", [])

        evidence_turn_ids = []
        global_turn_idx = 0

        for s_idx, session_turns in enumerate(raw_sessions):
            if s_idx < len(raw_session_ids):
                sid = str(raw_session_ids[s_idx])
            else:
                sid = f"s_{s_idx}"

            if isinstance(session_turns, dict) and "turns" in session_turns:
                turns_list = session_turns["turns"]
            elif isinstance(session_turns, list):
                turns_list = session_turns
            else:
                continue

            seen_texts_in_session = set()

            for t_idx, turn in enumerate(turns_list):
                speaker = str(turn.get("role", turn.get("speaker", "user"))).lower()
                if speaker not in ("user", "assistant", "system"):
                    speaker = "user" if speaker in ("human", "client") else "assistant"

                text = normalize_whitespace(turn.get("content", turn.get("text", "")))
                if not text:
                    continue

                norm_key = text.lower()
                if norm_key in seen_texts_in_session:
                    continue
                seen_texts_in_session.add(norm_key)

                if filter_english and not is_english(text):
                    dropped_non_english += 1
                    continue

                raw_n_tokens = count_tokens(text)
                truncated_text = head_tail_truncate(text, max_words=max_turn_words)

                has_answer = turn.get("has_answer", turn.get("evidence", False))
                label = 1 if bool(has_answer) else 0

                turn_id = f"{cid}_{sid}_{t_idx}"
                linked_qids = [qid] if label == 1 else []
                linked_cat = mapped_cat if label == 1 else "none"

                if label == 1:
                    evidence_turn_ids.append(turn_id)

                rec = TurnRecord(
                    conv_id=cid,
                    session_id=sid,
                    turn_idx=t_idx,
                    global_turn_idx=global_turn_idx,
                    speaker=speaker,
                    text=truncated_text,
                    n_tokens=raw_n_tokens,
                    label=label,
                    linked_question_ids=linked_qids,
                    linked_category=linked_cat,
                    source="longmemeval",
                    lang="en",
                )
                all_turns.append(rec)
                global_turn_idx += 1

        all_questions.append(
            QuestionRecord(
                question_id=qid,
                conv_id=cid,
                category=mapped_cat,
                evidence_turn_ids=evidence_turn_ids,
                question_text=q_text,
            )
        )

    logger.info(
        f"Loaded LongMemEval: {len(all_turns)} turns, {len(all_questions)} questions."
    )
    return all_turns, all_questions
