"""
Heuristic baselines: Random, Recency (truncation), and Length.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from src.baselines.base import MemoryBaseline


class RandomBaseline(MemoryBaseline):
    """Assigns independent uniform random scores in [0, 1]."""

    def __init__(self, seed: int = 42):
        super().__init__("random")
        self.seed = seed

    def predict_proba(
        self,
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> np.ndarray:
        rng = np.random.RandomState(self.seed)
        return rng.uniform(0.0, 1.0, size=len(texts))


class RecencyBaseline(MemoryBaseline):
    """
    Recency / Truncation baseline: Later turns score higher.
    Reflects standard conversational FIFO / sliding context buffer behavior.
    """

    def __init__(self):
        super().__init__("recency")

    def predict_proba(
        self,
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> np.ndarray:
        n = len(texts)
        if n == 0:
            return np.array([])
        if metadata is None:
            # Fallback: relative index within the provided list
            return np.linspace(0.0, 1.0, n)

        # Compute max global_turn_idx per conv_id
        conv_max_idx = {}
        for meta in metadata:
            cid = meta.get("conv_id", "default")
            g_idx = meta.get("global_turn_idx", meta.get("turn_idx", 0))
            conv_max_idx[cid] = max(conv_max_idx.get(cid, 0), g_idx)

        scores = np.zeros(n, dtype=np.float32)
        for i, meta in enumerate(metadata):
            cid = meta.get("conv_id", "default")
            g_idx = meta.get("global_turn_idx", meta.get("turn_idx", 0))
            max_idx = conv_max_idx.get(cid, 1)
            scores[i] = (g_idx + 1.0) / (max_idx + 1.0) if max_idx > 0 else 1.0

        return np.clip(scores, 0.0, 1.0)


class LengthBaseline(MemoryBaseline):
    """
    Turn length baseline: Turns with more tokens receive higher scores.
    """

    def __init__(self):
        super().__init__("length")

    def predict_proba(
        self,
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> np.ndarray:
        n = len(texts)
        if n == 0:
            return np.array([])

        scores = np.zeros(n, dtype=np.float32)
        for i in range(n):
            if metadata and i < len(metadata):
                tokens = metadata[i].get("n_tokens", len(texts[i].split()))
            else:
                tokens = len(texts[i].split())
            scores[i] = float(tokens)

        max_tok = float(np.max(scores)) if n > 0 and np.max(scores) > 0 else 1.0
        return np.clip(scores / max_tok, 0.0, 1.0)

