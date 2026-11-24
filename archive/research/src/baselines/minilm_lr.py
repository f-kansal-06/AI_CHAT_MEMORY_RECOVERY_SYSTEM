"""
Frozen sentence-transformers/all-MiniLM-L6-v2 embeddings + Logistic Regression baseline.
"""

from typing import List, Dict, Any, Optional
import logging
import numpy as np
from sklearn.linear_model import LogisticRegression
from src.baselines.base import MemoryBaseline

logger = logging.getLogger(__name__)


class MiniLMLogisticRegression(MemoryBaseline):
    """Frozen all-MiniLM-L6-v2 encoder + balanced Logistic Regression."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        random_state: int = 42,
    ):
        super().__init__("minilm_lr")
        self.model_name = model_name
        self.random_state = random_state
        self.encoder = None
        self.clf = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
        )
        self.is_fitted = False
        self._init_encoder()

    def _init_encoder(self):
        try:
            import os
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(self.model_name, local_files_only=True)
        except Exception as e:
            logger.warning(
                f"Could not load SentenceTransformer({self.model_name}): {e}. "
                "Will use hashing feature encoder fallback."
            )
            self.encoder = None

    def _encode(self, texts: List[str]) -> np.ndarray:
        if self.encoder is not None:
            try:
                # Truncate texts to 150 chars for fast CPU encoding
                fast_texts = [t[:150] for t in texts]
                embeddings = self.encoder.encode(
                    fast_texts,
                    batch_size=128,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
                return embeddings
            except Exception as e:
                logger.warning(f"SentenceTransformer encoding failed: {e}. Falling back.")

        # Fallback deterministic pseudo-embedding (384-d feature vector)
        from sklearn.feature_extraction.text import HashingVectorizer
        hasher = HashingVectorizer(n_features=384, alternate_sign=False)
        return hasher.fit_transform(texts).toarray()

    def fit(
        self,
        texts: List[str],
        labels: List[int],
        metadata: Optional[List[Dict[str, Any]]] = None,
        max_train_turns: int = 1500,
    ) -> "MiniLMLogisticRegression":
        if len(texts) == 0:
            return self

        unique_labels = set(labels)
        if len(unique_labels) < 2:
            self.is_fitted = False
            self.constant_label = list(unique_labels)[0]
            return self

        # Subset for fast CPU laptop runs if large
        if len(texts) > max_train_turns:
            texts = texts[:max_train_turns]
            labels = labels[:max_train_turns]

        X = self._encode(texts)
        y = np.array(labels)
        self.clf.fit(X, y)
        self.is_fitted = True
        return self

    def predict_proba(
        self,
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> np.ndarray:
        n = len(texts)
        if n == 0:
            return np.array([])
        if not self.is_fitted:
            const_val = getattr(self, "constant_label", 0.5)
            return np.full(n, float(const_val), dtype=np.float32)

        X = self._encode(texts)
        probs = self.clf.predict_proba(X)
        if probs.shape[1] == 2:
            return probs[:, 1]
        elif probs.shape[1] == 1:
            return probs[:, 0]
        return np.full(n, 0.5, dtype=np.float32)
