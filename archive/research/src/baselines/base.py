"""
Base class for memory importance baselines.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np


class MemoryBaseline(ABC):
    """Abstract base class for all memory importance score models."""

    def __init__(self, name: str):
        self.name = name

    def fit(
        self,
        texts: List[str],
        labels: List[int],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> "MemoryBaseline":
        """Fit baseline on training data if applicable."""
        return self

    @abstractmethod
    def predict_proba(
        self,
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> np.ndarray:
        """
        Predict probability/score in [0, 1] for each turn.
        Returns 1D numpy array of shape (N,).
        """
        pass

