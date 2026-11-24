"""
Data cleaning, text normalization, language filtering, truncation, and category mapping.
"""

import re
import yaml
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


def normalize_whitespace(text: str) -> str:
    """Normalize excess whitespace and line breaks."""
    if not text:
        return ""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def count_tokens(text: str) -> int:
    """
    Approximate token count for the raw turn before model truncation.
    Roughly 1.3 tokens per whitespace-delimited word or len/4 characters.
    """
    if not text:
        return 0
    words = text.split()
    return max(len(words), int(len(text) / 4))


def head_tail_truncate(text: str, max_words: int = 250) -> str:
    """
    Truncate very long turns using head + tail.
    """
    words = text.split()
    if len(words) <= max_words:
        return text
    half = max_words // 2
    head = " ".join(words[:half])
    tail = " ".join(words[-half:])
    return f"{head} ... [TRUNCATED] ... {tail}"


def is_english(text: str) -> bool:
    """
    Fast filter for English text. Checks ASCII ratio first before slow langdetect.
    """
    if not text or len(text.strip()) < 3:
        return True

    # Fast ASCII printable character ratio check
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    ratio = ascii_chars / max(len(text), 1)
    if ratio > 0.85:
        return True
    if ratio < 0.50:
        return False

    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 42
        return detect(text) == "en"
    except Exception:
        return ratio > 0.70


class CategoryMapper:
    """Maps dataset-specific categories to unified taxonomy defined in configs/label_map.yaml."""

    def __init__(self, config_path: str = "configs/label_map.yaml"):
        p = Path(config_path)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {
                "shared_categories": [
                    "single_fact", "multi_hop", "temporal",
                    "knowledge_update", "preference", "other", "none"
                ]
            }

    def map_category(self, source: str, raw_category: Optional[str]) -> str:
        if not raw_category or str(raw_category).strip().lower() in ("none", "0", "null", ""):
            return "none"
        raw_cat = str(raw_category).strip().lower()
        source_maps = self.config.get(source, {})
        if raw_cat in source_maps:
            return source_maps[raw_cat]
        # General substring match
        for key, mapped in source_maps.items():
            if str(key).lower() in raw_cat:
                return mapped
        # Check standard targets
        targets = self.config.get("shared_categories", [])
        for t in targets:
            if t in raw_cat:
                return t
        return "other"
