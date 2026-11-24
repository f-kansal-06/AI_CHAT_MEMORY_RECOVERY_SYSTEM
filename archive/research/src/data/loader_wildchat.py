"""
WildChat dataset streaming and loading module.
Streams English conversations from allenai/WildChat-1M or local sample fallback for analysis (E6).
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Generator, Optional
from src.data.schema import TurnRecord
from src.data.cleaning import (
    normalize_whitespace,
    count_tokens,
    head_tail_truncate,
    is_english,
)

logger = logging.getLogger(__name__)


def stream_wildchat_conversations(
    max_conversations: int = 5000,
    min_turns: int = 6,
    local_fallback_path: Optional[str] = "data/raw/wildchat_sample.jsonl",
) -> Generator[List[TurnRecord], None, None]:
    """
    Stream conversations from allenai/WildChat-1M, filtering for English and >= min_turns.
    Falls back to local file if remote streaming is unavailable.
    """
    streamed_count = 0
    loaded_from_remote = False

    try:
        from datasets import load_dataset
        logger.info("Attempting to stream allenai/WildChat-1M from Hugging Face...")
        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        for item in ds:
            # Check language
            lang = str(item.get("language", "")).lower()
            if "en" not in lang and lang != "english":
                continue

            raw_conv = item.get("conversation", item.get("turns", []))
            if len(raw_conv) < min_turns:
                continue

            turns = _parse_wildchat_item(item, f"wc_{streamed_count}")
            if len(turns) >= min_turns:
                yield turns
                streamed_count += 1
                loaded_from_remote = True
                if streamed_count >= max_conversations:
                    break
    except Exception as e:
        logger.warning(f"Could not stream WildChat-1M from Hugging Face ({e}). Checking local fallback.")

    if not loaded_from_remote and local_fallback_path:
        p = Path(local_fallback_path)
        if p.exists():
            logger.info(f"Loading WildChat conversations from local fallback: {local_fallback_path}")
            current_conv_id = None
            current_turns: List[TurnRecord] = []
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    turn_data = json.loads(line)
                    cid = turn_data.get("conv_id")
                    if current_conv_id is not None and cid != current_conv_id:
                        if len(current_turns) >= min_turns:
                            yield current_turns
                            streamed_count += 1
                            if streamed_count >= max_conversations:
                                break
                        current_turns = []
                    current_conv_id = cid
                    current_turns.append(TurnRecord.from_dict(turn_data))

            if current_turns and len(current_turns) >= min_turns and streamed_count < max_conversations:
                yield current_turns


def _parse_wildchat_item(item: Dict[str, Any], conv_id: str) -> List[TurnRecord]:
    raw_conv = item.get("conversation", item.get("turns", []))
    turns: List[TurnRecord] = []
    seen_texts = set()

    for t_idx, turn in enumerate(raw_conv):
        speaker = str(turn.get("role", "user")).lower()
        if speaker not in ("user", "assistant", "system"):
            speaker = "user"
        content = normalize_whitespace(turn.get("content", ""))
        if not content:
            continue

        norm_key = content.lower()
        if norm_key in seen_texts:
            continue
        seen_texts.add(norm_key)

        rec = TurnRecord(
            conv_id=conv_id,
            session_id="s0",
            turn_idx=t_idx,
            global_turn_idx=t_idx,
            speaker=speaker,
            text=head_tail_truncate(content, max_words=250),
            n_tokens=count_tokens(content),
            label=0,
            linked_question_ids=[],
            linked_category="none",
            source="wildchat",
            lang="en",
        )
        turns.append(rec)

    return turns

