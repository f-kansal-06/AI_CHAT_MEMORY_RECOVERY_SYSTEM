"""
Conventional Evaluation Script for Conversation Memory System.

Computes and displays standard, conventional evaluation metrics:
- Accuracy
- Precision
- Recall (Sensitivity)
- Specificity (True Negative Rate)
- F1-Score
- ROC-AUC
- PR-AUC
- Confusion Matrix (True Positives, False Positives, True Negatives, False Negatives)
- Token Reduction Rate
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import load_turns_from_jsonl, build_formatted_samples
from src.baselines.heuristics import RandomBaseline, RecencyBaseline, LengthBaseline
from src.baselines.tfidf_lr import TfidfLogisticRegression
from src.eval.metrics import compute_conventional_metrics, format_conventional_metrics_table, find_best_f1_threshold
from src.eval.plots import generate_evaluation_dashboard


def evaluate_dataset_conventional(
    dataset_name: str = "custom",
    baseline_type: str = "tfidf",
    threshold: float = None,
):
    proc_dir = Path("data/processed")
    train_file = proc_dir / f"{dataset_name}_train.jsonl"
    val_file = proc_dir / f"{dataset_name}_val.jsonl"
    test_file = proc_dir / f"{dataset_name}_test.jsonl"

    if not test_file.exists():
        print(f"Error: {test_file} not found. Please ensure processed data exists.")
        return

    print(f"\n[Data] Loading {dataset_name} test split from {test_file}...")
    test_turns = load_turns_from_jsonl(str(test_file))
    test_samples = build_formatted_samples(test_turns, context_window=1)
    y_test = np.array([s.get("label", 0) for s in test_samples])
    texts_test = [s["input_text"] for s in test_samples]

    # Baseline selection
    if baseline_type == "random":
        model = RandomBaseline()
    elif baseline_type == "recency":
        model = RecencyBaseline()
    elif baseline_type == "length":
        model = LengthBaseline()
    else:
        # Default TF-IDF + Logistic Regression
        print(f"[Model] Fitting TF-IDF + Logistic Regression on {train_file.name}...")
        train_turns = load_turns_from_jsonl(str(train_file))
        train_samples = build_formatted_samples(train_turns, context_window=1)
        train_texts = [s["input_text"] for s in train_samples]
        train_y = np.array([s.get("label", 0) for s in train_samples])

        model = TfidfLogisticRegression(max_features=5000)
        model.fit(train_texts, train_y)

    # Score test set
    scores = model.predict_proba(texts_test, metadata=test_samples)

    # Threshold selection
    if threshold is None:
        if val_file.exists():
            val_turns = load_turns_from_jsonl(str(val_file))
            val_samples = build_formatted_samples(val_turns, context_window=1)
            val_y = np.array([s.get("label", 0) for s in val_samples])
            val_scores = model.predict_proba([s["input_text"] for s in val_samples], metadata=val_samples)
            threshold = find_best_f1_threshold(val_y, val_scores)
        else:
            threshold = 0.5

    # Compute conventional metrics
    metrics = compute_conventional_metrics(y_test, scores, threshold=threshold)

    # Print clean formatted table
    title = f"{model.name.upper()} ON {dataset_name.upper()} TEST SET"
    table = format_conventional_metrics_table(metrics, title=title)
    print("\n" + table)

    # Calculate token reduction
    y_pred = (scores >= threshold).astype(int)
    total_tokens = sum(len(s["text"].split()) for s in test_samples)
    retained_tokens = sum(len(s["text"].split()) for s, p in zip(test_samples, y_pred) if p == 1)
    reduction_pct = (1.0 - (retained_tokens / max(1, total_tokens))) * 100

    print(f"\n[Context Compression & Token Reduction]")
    print(f"Total Test Tokens      : {total_tokens:,}")
    print(f"Retained Memory Tokens : {retained_tokens:,}")
    print(f"Token Reduction Rate   : {reduction_pct:.2f}% context pruned")
    print("=" * 65 + "\n")

    # Generate plots and visual representations
    plot_dir = f"reports/plots/{dataset_name}_{baseline_type}"
    print(f"[Visuals] Generating plots and dashboards in {plot_dir}...")
    saved_plots = generate_evaluation_dashboard(
        y_true=y_test,
        y_prob=scores,
        metrics=metrics,
        raw_tokens=total_tokens,
        retained_tokens=retained_tokens,
        output_dir=plot_dir,
        model_name=f"{model.name.upper()} ({dataset_name.upper()})"
    )

    print("[Visuals] Plots successfully generated:")
    for name, p in saved_plots.items():
        print(f"  • {name:<18} -> {p}")
    print("=" * 65 + "\n")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate memory models using conventional metrics.")
    parser.add_argument("--dataset", type=str, default="custom", choices=["custom", "locomo", "longmemeval"])
    parser.add_argument("--model", type=str, default="tfidf", choices=["tfidf", "recency", "length", "random"])
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    evaluate_dataset_conventional(
        dataset_name=args.dataset,
        baseline_type=args.model,
        threshold=args.threshold
    )
