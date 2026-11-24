"""
LoCoMo dataset loader.
Parses multi-session conversations and question QA items with evidence references (e.g. "D1:3").
Handles official locomo10.json schema with dia_id per turn and dict-of-sessions conversation.
"""

import json
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Set
from src.data.schema import TurnRecord, QuestionRecord
from src.data.cleaning import (
    normalize_whitespace,
    count_tokens,
    head_tail_truncate,
    is_english,
    CategoryMapper,
)

logger = logging.getLogger(__name__)


def parse_locomo_evidence_id(ev_str: str) -> List[str]:
    """
    Extract normalized dia_id references like "D1:3" from evidence strings.
    """
    results = []
    # Split by comma or semicolon
    parts = re.split(r"[,;]+", str(ev_str))
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Standardize D1:3
        m = re.match(r"(?:session[_\s]*)?([dD]?\d+|s\d+)[:\-\s]+(\d+)", part, re.IGNORECASE)
        if m:
            sess_num = m.group(1).upper()
            if not sess_num.startswith("D"):
                sess_num = f"D{sess_num.lstrip('S')}"
            turn_idx = int(m.group(2))
            results.append(f"{sess_num}:{turn_idx}")
        else:
            m2 = re.match(r"(\d+)", part)
            if m2:
                results.append(f"D1:{m2.group(1)}")
    return results


def load_locomo(
    file_path: str,
    category_mapper: Optional[CategoryMapper] = None,
    filter_english: bool = False,
    max_turn_words: int = 250,
) -> Tuple[List[TurnRecord], List[QuestionRecord]]:
    """
    Load and parse LoCoMo conversations and QA pairs into unified TurnRecord and QuestionRecord.
    """
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"LoCoMo file not found: {file_path}")

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        conv_items = data.get("conversations", data.get("data", [data]))
    elif isinstance(data, list):
        conv_items = data
    else:
        raise ValueError(f"Unexpected JSON format in {file_path}")

    if category_mapper is None:
        category_mapper = CategoryMapper()

    all_turns: List[TurnRecord] = []
    all_questions: List[QuestionRecord] = []

    for c_idx, conv in enumerate(conv_items):
        cid = str(conv.get("sample_id", conv.get("conv_id", f"locomo_{c_idx}")))

        # In locomo10.json, 'conversation' is a dict mapping 'session_1', 'session_2', ... to list of turn dicts
        raw_conv = conv.get("conversation", {})
        session_turn_map: Dict[str, List[Dict[str, Any]]] = {}

        if isinstance(raw_conv, dict):
            # Extract session keys sorted numerically
            sess_keys = [k for k in raw_conv.keys() if "session_" in k and not k.endswith("_date_time")]
            # Sort by session number
            def extract_snum(s):
                m = re.search(r"\d+", s)
                return int(m.group(0)) if m else 0
            sess_keys.sort(key=extract_snum)

            for sk in sess_keys:
                s_num = extract_snum(sk)
                d_id = f"D{s_num}"
                session_turn_map[d_id] = raw_conv[sk]
        elif isinstance(raw_conv, list):
            for s_idx, sess in enumerate(raw_conv):
                d_id = f"D{s_idx + 1}"
                session_turn_map[d_id] = sess.get("turns", []) if isinstance(sess, dict) else sess

        # Questions QA pairs
        qa_list = conv.get("qa", conv.get("questions", []))
        evidence_turn_lookup: Dict[str, List[Tuple[str, str]]] = {}
        # dia_id (e.g. "D1:3") -> list of (question_id, mapped_category)

        for q_idx, qa in enumerate(qa_list):
            qid = str(qa.get("question_id", f"{cid}_q{q_idx}"))
            q_text = qa.get("question", "")
            raw_cat = str(qa.get("category", "1"))
            mapped_cat = category_mapper.map_category("locomo", raw_cat)

            raw_ev = qa.get("evidence", [])
            if isinstance(raw_ev, str):
                raw_ev = [raw_ev]

            q_ev_ids = []
            for ev_item in raw_ev:
                parsed_evs = parse_locomo_evidence_id(str(ev_item))
                for ev_key in parsed_evs:
                    if ev_key not in evidence_turn_lookup:
                        evidence_turn_lookup[ev_key] = []
                    evidence_turn_lookup[ev_key].append((qid, mapped_cat))
                    q_ev_ids.append(f"{cid}_{ev_key}")

            all_questions.append(
                QuestionRecord(
                    question_id=qid,
                    conv_id=cid,
                    category=mapped_cat,
                    evidence_turn_ids=q_ev_ids,
                    question_text=q_text,
                )
            )

        # Build turn records
        global_turn_idx = 0
        for sid, raw_turns in session_turn_map.items():
            seen_texts = set()
            for t_idx, turn in enumerate(raw_turns):
                dia_id = turn.get("dia_id", f"{sid}:{t_idx + 1}")
                speaker = str(turn.get("speaker", "user")).lower()
                if speaker not in ("user", "assistant", "system"):
                    speaker = "user"

                text = normalize_whitespace(turn.get("text", turn.get("content", "")))
                if not text:
                    continue

                norm_key = text.lower()
                if norm_key in seen_texts:
                    continue
                seen_texts.add(norm_key)

                raw_n_tokens = count_tokens(text)
                truncated_text = head_tail_truncate(text, max_words=max_turn_words)

                # Check evidence matches
                ev_matches = evidence_turn_lookup.get(dia_id, [])
                label = 1 if len(ev_matches) > 0 else 0
                linked_qids = [m[0] for m in ev_matches]
                linked_cat = ev_matches[0][1] if ev_matches else "none"

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
                    source="locomo",
                    lang="en",
                )
                all_turns.append(rec)
                global_turn_idx += 1

    logger.info(f"Loaded LoCoMo: {len(all_turns)} turns, {len(all_questions)} questions.")
    return all_turns, all_questions
