"""
Visual Plotting Suite for Conventional Machine Learning Evaluation Metrics.

Generates:
1. Confusion Matrix Heatmap (TP, FP, TN, FN)
2. ROC Curve (Receiver Operating Characteristic) with AUC
3. Precision-Recall Curve with PR-AUC
4. Performance Metrics Bar Chart (Accuracy, Precision, Recall, Specificity, F1)
5. Token Reduction & Context Compression Chart
6. Unified 2x2 Evaluation Dashboard
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc


def plot_confusion_matrix(
    cm: np.ndarray,
    output_path: str = "reports/plots/confusion_matrix.png",
    title: str = "Confusion Matrix",
):
    """Plot an annotated confusion matrix heatmap."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(6, 5))
    
    labels = [["True Neg (TN)", "False Pos (FP)"], ["False Neg (FN)", "True Pos (TP)"]]
    annot = np.array([[f"{labels[i][j]}\n{cm[i, j]:,}" for j in range(2)] for i in range(2)])
    
    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar=True,
        xticklabels=["Discard (0)", "Keep (1)"],
        yticklabels=["Discard (0)", "Keep (1)"],
        linewidths=1.5,
        annot_kws={"fontsize": 11, "fontweight": "bold"},
    )
    plt.title(title, fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Predicted Label", fontsize=12, fontweight="bold")
    plt.ylabel("True Label", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    roc_auc: float,
    output_path: str = "reports/plots/roc_curve.png",
    title: str = "ROC Curve",
):
    """Plot ROC curve with AUC and reference line."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#2980b9", lw=2.5, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="#7f8c8d", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.50)")
    
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    plt.ylabel("True Positive Rate (Recall)", fontsize=11, fontweight="bold")
    plt.title(title, fontsize=14, fontweight="bold", pad=12)
    plt.legend(loc="lower right", fontsize=10, frameon=True)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    pr_auc: float,
    output_path: str = "reports/plots/pr_curve.png",
    title: str = "Precision-Recall Curve",
):
    """Plot Precision-Recall curve with PR-AUC and prevalence baseline."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    prevalence = float(np.mean(y_true)) if len(y_true) > 0 else 0.0
    
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color="#27ae60", lw=2.5, label=f"PR Curve (AUC = {pr_auc:.4f})")
    plt.axhline(y=prevalence, color="#e74c3c", lw=1.5, linestyle="--", label=f"Baseline Prevalence ({prevalence:.1%})")
    
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("Recall", fontsize=11, fontweight="bold")
    plt.ylabel("Precision", fontsize=11, fontweight="bold")
    plt.title(title, fontsize=14, fontweight="bold", pad=12)
    plt.legend(loc="lower left", fontsize=10, frameon=True)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_metrics_bar(
    metrics: Dict[str, Any],
    output_path: str = "reports/plots/metrics_bar.png",
    title: str = "Evaluation Metrics Summary",
):
    """Plot bar chart comparing conventional classification metrics."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    metric_names = ["Accuracy", "Precision", "Recall", "Specificity", "F1-Score"]
    values = [
        metrics.get("accuracy", 0.0) * 100,
        metrics.get("precision", 0.0) * 100,
        metrics.get("recall", 0.0) * 100,
        metrics.get("specificity", 0.0) * 100,
        metrics.get("f1", 0.0) * 100,
    ]
    colors = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6", "#e74c3c"]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(metric_names, values, color=colors, width=0.55, edgecolor="black", linewidth=1.2)
    plt.ylim([0, 110])
    plt.ylabel("Score (%)", fontsize=12, fontweight="bold")
    plt.title(title, fontsize=14, fontweight="bold", pad=12)
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, h + 2, f"{h:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_token_reduction_chart(
    raw_tokens: int,
    retained_tokens: int,
    reduction_pct: float,
    output_path: str = "reports/plots/token_reduction.png",
    title: str = "Context Compression & Token Reduction",
):
    """Plot token reduction comparison before and after memory filtering."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    categories = ["Raw Full Context", "Compressed Memory"]
    tokens = [raw_tokens, retained_tokens]
    colors = ["#e74c3c", "#27ae60"]

    plt.figure(figsize=(7, 5))
    bars = plt.bar(categories, tokens, color=colors, width=0.45, edgecolor="black", linewidth=1.2)
    plt.ylabel("Token Count (words)", fontsize=12, fontweight="bold")
    plt.title(f"{title}\n({reduction_pct:.1f}% Context Pruned)", fontsize=14, fontweight="bold", pad=12)
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, h + max(tokens) * 0.02, f"{h:,} tokens", ha="center", va="bottom", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def generate_evaluation_dashboard(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metrics: Dict[str, Any],
    raw_tokens: int,
    retained_tokens: int,
    output_dir: str = "reports/plots",
    model_name: str = "Memory Importance Classifier",
) -> Dict[str, str]:
    """Generate individual plots and an all-in-one 2x2 evaluation dashboard."""
    os.makedirs(output_dir, exist_ok=True)
    cm = np.array(metrics.get("confusion_matrix", [[0, 0], [0, 0]]))
    roc_auc = metrics.get("roc_auc", 0.5)
    pr_auc = metrics.get("pr_auc", 0.0)
    reduction_pct = (1.0 - (retained_tokens / max(1, raw_tokens))) * 100

    # 1. Individual plots
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    roc_path = os.path.join(output_dir, "roc_curve.png")
    pr_path = os.path.join(output_dir, "pr_curve.png")
    bar_path = os.path.join(output_dir, "metrics_bar.png")
    token_path = os.path.join(output_dir, "token_reduction.png")
    dash_path = os.path.join(output_dir, "conventional_evaluation_dashboard.png")

    plot_confusion_matrix(cm, output_path=cm_path, title=f"Confusion Matrix ({model_name})")
    plot_roc_curve(y_true, y_prob, roc_auc, output_path=roc_path, title=f"ROC Curve ({model_name})")
    plot_precision_recall_curve(y_true, y_prob, pr_auc, output_path=pr_path, title=f"PR Curve ({model_name})")
    plot_metrics_bar(metrics, output_path=bar_path, title=f"Performance Metrics ({model_name})")
    plot_token_reduction_chart(raw_tokens, retained_tokens, reduction_pct, output_path=token_path)

    # 2. Unified 2x2 Dashboard Figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(f"Conventional Evaluation Dashboard: {model_name}", fontsize=16, fontweight="bold", y=0.98)

    # Subplot 1: Confusion Matrix
    labels = [["TN", "FP"], ["FN", "TP"]]
    annot = np.array([[f"{labels[i][j]}\n{cm[i, j]:,}" for j in range(2)] for i in range(2)])
    sns.heatmap(
        cm, annot=annot, fmt="", cmap="Blues", cbar=False, ax=axes[0, 0],
        xticklabels=["Discard (0)", "Keep (1)"], yticklabels=["Discard (0)", "Keep (1)"],
        linewidths=1.2, annot_kws={"fontsize": 11, "fontweight": "bold"}
    )
    axes[0, 0].set_title(f"Confusion Matrix (Acc: {metrics.get('accuracy', 0):.1%})", fontweight="bold")
    axes[0, 0].set_xlabel("Predicted")
    axes[0, 0].set_ylabel("True Label")

    # Subplot 2: Metrics Bar Chart
    metric_names = ["Accuracy", "Precision", "Recall", "Specificity", "F1"]
    values = [
        metrics.get("accuracy", 0.0) * 100,
        metrics.get("precision", 0.0) * 100,
        metrics.get("recall", 0.0) * 100,
        metrics.get("specificity", 0.0) * 100,
        metrics.get("f1", 0.0) * 100,
    ]
    colors = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6", "#e74c3c"]
    bars = axes[0, 1].bar(metric_names, values, color=colors, width=0.55, edgecolor="black")
    axes[0, 1].set_ylim([0, 110])
    axes[0, 1].set_ylabel("Score (%)", fontweight="bold")
    axes[0, 1].set_title("Classification Scores", fontweight="bold")
    axes[0, 1].grid(axis="y", linestyle="--", alpha=0.5)
    for b in bars:
        h = b.get_height()
        axes[0, 1].text(b.get_x() + b.get_width() / 2.0, h + 2, f"{h:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=10)

    # Subplot 3: ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    axes[1, 0].plot(fpr, tpr, color="#2980b9", lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    axes[1, 0].plot([0, 1], [0, 1], color="#7f8c8d", lw=1.5, linestyle="--", label="Random")
    axes[1, 0].set_xlabel("False Positive Rate", fontweight="bold")
    axes[1, 0].set_ylabel("True Positive Rate", fontweight="bold")
    axes[1, 0].set_title("ROC Curve", fontweight="bold")
    axes[1, 0].legend(loc="lower right")
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)

    # Subplot 4: Token Reduction
    cat_tokens = ["Full Context", "Retained Memory"]
    tok_vals = [raw_tokens, retained_tokens]
    tok_bars = axes[1, 1].bar(cat_tokens, tok_vals, color=["#e74c3c", "#27ae60"], width=0.45, edgecolor="black")
    axes[1, 1].set_ylabel("Tokens (words)", fontweight="bold")
    axes[1, 1].set_title(f"Context Reduction: {reduction_pct:.1f}% Pruned", fontweight="bold")
    axes[1, 1].grid(axis="y", linestyle="--", alpha=0.5)
    for b in tok_bars:
        h = b.get_height()
        axes[1, 1].text(b.get_x() + b.get_width() / 2.0, h + max(tok_vals) * 0.02, f"{h:,}", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    plt.savefig(dash_path, dpi=300)
    plt.close()

    return {
        "confusion_matrix": cm_path,
        "roc_curve": roc_path,
        "pr_curve": pr_path,
        "metrics_bar": bar_path,
        "token_reduction": token_path,
        "dashboard": dash_path,
    }

