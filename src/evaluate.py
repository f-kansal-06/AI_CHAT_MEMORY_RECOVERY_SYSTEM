"""
Comprehensive evaluation CLI and runner for baselines and trained DistilBERT models.
Generates turn-level, ranking, budgeted retention, slices, and latency metrics.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import load_turns_from_jsonl, build_formatted_samples
from src.data.schema import QuestionRecord
from src.train import (
    TurnImportanceDataset,
    get_tokenizer,
    DistilBertForMemoryImportance,
    CATEGORY_MAP,
)
from src.eval.metrics import (
    compute_turn_level_metrics,
    find_best_f1_threshold,
    evaluate_ranking,
    compute_budgeted_retention,
    compute_slice_metrics,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def score_with_distilbert(
    model: DistilBertForMemoryImportance,
    samples: List[Dict[str, Any]],
    tokenizer,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    ds = TurnImportanceDataset(samples, tokenizer, max_length=128)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    all_scores = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            scores = outputs["importance_scores"].cpu().numpy()
            all_scores.extend(scores.tolist())
    return np.array(all_scores, dtype=np.float32)


def measure_inference_latency(
    model: DistilBertForMemoryImportance,
    sample_text: str,
    tokenizer,
    n_runs: int = 10,
) -> float:
    """Measure mean per-turn inference latency on CPU (batch size 1) in milliseconds."""
    device = torch.device("cpu")
    model.to(device)
    model.eval()

    sample = [{"input_text": sample_text, "label": 0, "linked_category": "none"}]
    ds = TurnImportanceDataset(sample, tokenizer, max_length=128)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    batch = next(iter(loader))
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask)

    start_t = time.perf_counter()
    for _ in range(n_runs):
        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
    total_time = time.perf_counter() - start_t
    return float((total_time / n_runs) * 1000.0)  # in ms


def run_full_evaluation(
    model_or_scores,
    test_turns: List[Dict[str, Any]],
    val_turns: Optional[List[Dict[str, Any]]] = None,
    questions: Optional[List[QuestionRecord]] = None,
    context_window: int = 1,
    model_name: str = "distilbert",
    checkpoint_path: Optional[str] = None,
    val_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Run comprehensive evaluation on test set.
    """
    test_samples = build_formatted_samples(test_turns, context_window=context_window)
    y_test = np.array([s.get("label", 0) for s in test_samples])

    if questions is None:
        questions = []

    # Get predictions
    if isinstance(model_or_scores, np.ndarray):
        scores = model_or_scores
        model_obj = None
    elif hasattr(model_or_scores, "predict_proba"):
        # Baseline model
        texts = [s["input_text"] for s in test_samples]
        scores = model_or_scores.predict_proba(texts, metadata=test_samples)
        model_obj = None
    elif isinstance(model_or_scores, DistilBertForMemoryImportance):
        tokenizer = get_tokenizer()
        scores = score_with_distilbert(model_or_scores, test_samples, tokenizer)
        model_obj = model_or_scores
    else:
        raise ValueError(f"Unsupported model type: {type(model_or_scores)}")

    # Threshold selection: strictly on validation set if available
    threshold = 0.5
    if val_threshold is not None:
        threshold = val_threshold
    elif val_turns is not None:
        val_samples = build_formatted_samples(val_turns, context_window=context_window)
        val_y = np.array([s.get("label", 0) for s in val_samples])
        if model_obj is not None:
            val_scores = score_with_distilbert(model_obj, val_samples, tokenizer)
        elif hasattr(model_or_scores, "predict_proba"):
            val_texts = [s["input_text"] for s in val_samples]
            val_scores = model_or_scores.predict_proba(val_texts, metadata=val_samples)
        else:
            val_scores = np.full(len(val_y), 0.5)
        threshold = find_best_f1_threshold(val_y, val_scores)

    # 1. Turn-level metrics
    turn_metrics = compute_turn_level_metrics(y_test, scores, threshold=threshold)

    # 2. Ranking metrics
    ranking_metrics = evaluate_ranking(test_samples, scores)

    # 3. Budgeted retention
    budget_metrics = compute_budgeted_retention(test_samples, scores, questions)

    # 4. Slices
    slice_metrics = compute_slice_metrics(test_samples, scores, threshold=threshold)

    # 5. Efficiency metrics (for PyTorch models)
    efficiency = {}
    if model_obj is not None:
        num_params = sum(p.numel() for p in model_obj.parameters())
        efficiency["num_parameters"] = num_params
        if checkpoint_path and Path(checkpoint_path).exists():
            efficiency["size_on_disk_mb"] = float(Path(checkpoint_path).stat().st_size / (1024 * 1024))
        sample_txt = test_samples[0]["input_text"] if test_samples else "Hello world"
        efficiency["cpu_latency_ms"] = measure_inference_latency(model_obj, sample_txt, tokenizer)

    results = {
        "model_name": model_name,
        "n_test_turns": len(test_samples),
        "prevalence": float(np.mean(y_test)) if len(y_test) > 0 else 0.0,
        "turn_metrics": turn_metrics,
        "ranking_metrics": ranking_metrics,
        "budget_retention": budget_metrics,
        "slices": slice_metrics,
        "efficiency": efficiency,
    }

    return results
