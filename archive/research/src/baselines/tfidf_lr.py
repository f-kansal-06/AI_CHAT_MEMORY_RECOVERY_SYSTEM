"""
TF-IDF + Logistic Regression baseline.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from src.baselines.base import MemoryBaseline


class TfidfLogisticRegression(MemoryBaseline):
    """TF-IDF vectorizer + balanced Logistic Regression classifier."""

    def __init__(self, max_features: int = 5000, random_state: int = 42):
        super().__init__("tfidf_lr")
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            ngram_range=(1, 2),
            lowercase=True,
        )
        self.clf = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
        )
        self.is_fitted = False

    def fit(
        self,
        texts: List[str],
        labels: List[int],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> "TfidfLogisticRegression":
        if len(texts) == 0:
            return self

        # Handle edge cases where all labels are 0 or 1
        unique_labels = set(labels)
        if len(unique_labels) < 2:
            self.is_fitted = False
            self.constant_label = list(unique_labels)[0]
            return self

        X = self.vectorizer.fit_transform(texts)
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

        X = self.vectorizer.transform(texts)
        probs = self.clf.predict_proba(X)
        if probs.shape[1] == 2:
            return probs[:, 1]
        elif probs.shape[1] == 1:
            return probs[:, 0]
        return np.full(n, 0.5, dtype=np.float32)

