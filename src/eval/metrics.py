"""
Comprehensive evaluation metrics for conversation memory importance.
Implements:
1. Turn-level: PR-AUC, ROC-AUC, Precision, Recall, F1 (val-tuned threshold), ECE.
2. Ranking (per conversation): Recall@K (5%, 10%, 20%), nDCG@K, MRR.
3. Budgeted retention: Token budget (5%, 10%, 20%, 30%, 50%), evidence recall,
   question-level all-evidence-retained rate, actual token reduction.
4. Slices: Category, length buckets, speaker.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)
from src.data.schema import QuestionRecord


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE) across n equal-width probability bins."""
    if len(y_true) == 0:
        return 0.0

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total_samples = len(y_true)

    for i in range(n_bins):
        bin_lower = bins[i]
        bin_upper = bins[i + 1]
        # Include upper bound on last bin
        if i == n_bins - 1:
            mask = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        else:
            mask = (y_prob >= bin_lower) & (y_prob < bin_upper)

        bin_count = np.sum(mask)
        if bin_count > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            ece += (bin_count / total_samples) * abs(bin_acc - bin_conf)

    return float(ece)


def find_best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find the classification threshold that maximizes F1 score on validation set."""
    if len(y_true) == 0 or np.sum(y_true) == 0 or np.sum(y_true) == len(y_true):
        return 0.5

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # Avoid division by zero
    f1s = np.zeros_like(thresholds)
    for i, t in enumerate(thresholds):
        p = precisions[i]
        r = recalls[i]
        if (p + r) > 0:
            f1s[i] = 2 * (p * r) / (p + r)
        else:
            f1s[i] = 0.0

    best_idx = np.argmax(f1s) if len(f1s) > 0 else 0
    return float(thresholds[best_idx]) if len(thresholds) > 0 else 0.5


def compute_turn_level_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute turn-level classification and ranking metrics."""
    if len(y_true) == 0:
        return {"pr_auc": 0.0, "roc_auc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "ece": 0.0}

    # Handle single-class edge cases
    if len(np.unique(y_true)) < 2:
        pr_auc = float(np.mean(y_true))
        roc_auc = 0.5
    else:
        try:
            pr_auc = float(average_precision_score(y_true, y_prob))
        except Exception:
            pr_auc = 0.0
        try:
            roc_auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            roc_auc = 0.5

    y_pred = (y_prob >= threshold).astype(int)
    p = float(precision_score(y_true, y_pred, zero_division=0))
    r = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    ece = compute_expected_calibration_error(y_true, y_prob)

    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "precision": p,
        "recall": r,
        "f1": f1,
        "ece": ece,
        "threshold": threshold,
    }


def compute_dcg(relevance: np.ndarray, k: int) -> float:
    """Compute Discounted Cumulative Gain at rank K."""
    rel = relevance[:k]
    if len(rel) == 0:
        return 0.0
    discounts = np.log2(np.arange(len(rel)) + 2.0)
    return float(np.sum(rel / discounts))


def compute_ndcg_at_k(relevance: np.ndarray, k: int) -> float:
    """Compute normalized DCG at rank K for binary relevance."""
    dcg = compute_dcg(relevance, k)
    ideal_rel = np.sort(relevance)[::-1]
    idcg = compute_dcg(ideal_rel, k)
    if idcg <= 0.0:
        return 0.0
    return float(dcg / idcg)


def compute_ranking_metrics_per_conversation(
    conv_turns: List[Dict[str, Any]],
    scores: np.ndarray,
    k_fractions: Tuple[float, ...] = (0.05, 0.10, 0.20),
) -> Dict[str, float]:
    """
    Compute Recall@K, nDCG@K for K in {5%, 10%, 20%} and MRR for a single conversation.
    """
    n = len(conv_turns)
    if n == 0:
        return {}

    labels = np.array([t.get("label", 0) for t in conv_turns])
    total_positives = int(np.sum(labels))
    if total_positives == 0:
        # Conversation has no evidence turns
        return None

    # Rank turns descending by predicted score
    ranked_indices = np.argsort(scores)[::-1]
    ranked_labels = labels[ranked_indices]

    results = {}
    for frac in k_fractions:
        k = max(1, int(np.ceil(n * frac)))
        k = min(k, n)
        retained_pos = np.sum(ranked_labels[:k])
        recall_at_k = float(retained_pos / total_positives)
        ndcg_at_k = compute_ndcg_at_k(ranked_labels, k)
        pct_key = int(frac * 100)
        results[f"recall@{pct_key}%"] = recall_at_k
        results[f"ndcg@{pct_key}%"] = ndcg_at_k

    # MRR of first evidence turn
    pos_positions = np.where(ranked_labels == 1)[0]
    if len(pos_positions) > 0:
        first_pos_rank = pos_positions[0] + 1  # 1-based rank
        mrr = float(1.0 / first_pos_rank)
    else:
        mrr = 0.0
    results["mrr"] = mrr

    return results


def evaluate_ranking(
    all_turns: List[Dict[str, Any]],
    all_scores: np.ndarray,
    k_fractions: Tuple[float, ...] = (0.05, 0.10, 0.20),
) -> Dict[str, float]:
    """Aggregate ranking metrics across all conversations containing evidence turns."""
    conv_indices: Dict[str, List[int]] = {}
    for idx, t in enumerate(all_turns):
        cid = t.get("conv_id", "default")
        if cid not in conv_indices:
            conv_indices[cid] = []
        conv_indices[cid].append(idx)

    per_conv_metrics: List[Dict[str, float]] = []
    for cid, indices in conv_indices.items():
        c_turns = [all_turns[i] for i in indices]
        c_scores = all_scores[indices]
        m = compute_ranking_metrics_per_conversation(c_turns, c_scores, k_fractions)
        if m is not None:
            per_conv_metrics.append(m)

    if not per_conv_metrics:
        keys = [f"recall@{int(f*100)}%" for f in k_fractions] + [f"ndcg@{int(f*100)}%" for f in k_fractions] + ["mrr"]
        return {k: 0.0 for k in keys}

    avg_metrics = {}
    for key in per_conv_metrics[0].keys():
        avg_metrics[key] = float(np.mean([m[key] for m in per_conv_metrics]))
    return avg_metrics


def compute_budgeted_retention(
    all_turns: List[Dict[str, Any]],
    all_scores: np.ndarray,
    questions: List[QuestionRecord],
    budget_fractions: Tuple[float, ...] = (0.05, 0.10, 0.20, 0.30, 0.50),
) -> Dict[str, Any]:
    """
    Budgeted retention evaluation:
    For token budgets of B% of total conversation tokens, keep highest-scored turns
    without exceeding the budget limit.
    Computes:
    - evidence_recall: fraction of all evidence turns kept
    - all_evidence_retained_rate: fraction of questions whose every evidence turn is kept
    - actual_token_reduction: 1.0 - (retained_tokens / total_tokens)
    """
    # Group turns and scores by conv_id
    conv_data: Dict[str, List[Tuple[Dict[str, Any], float]]] = {}
    for t, s in zip(all_turns, all_scores):
        cid = t.get("conv_id", "default")
        if cid not in conv_data:
            conv_data[cid] = []
        conv_data[cid].append((t, float(s)))

    # Map questions by conv_id
    questions_by_conv: Dict[str, List[QuestionRecord]] = {}
    for q in questions:
        cid = q.conv_id
        if cid not in questions_by_conv:
            questions_by_conv[cid] = []
        questions_by_conv[cid].append(q)

    budget_results: Dict[str, Dict[str, float]] = {}

    for b_frac in budget_fractions:
        pct_key = f"{int(b_frac * 100)}%"
        total_evidence_turns = 0
        retained_evidence_turns = 0

        total_questions = 0
        fully_retained_questions = 0

        total_tokens_all = 0
        total_retained_tokens_all = 0

        for cid, items in conv_data.items():
            conv_total_tokens = sum(t.get("n_tokens", 1) for t, _ in items)
            budget_limit = int(conv_total_tokens * b_frac)
            total_tokens_all += conv_total_tokens

            # Sort turns strictly descending by score
            sorted_items = sorted(items, key=lambda x: x[1], reverse=True)

            retained_turn_ids = set()
            current_tokens = 0

            # Strict budget constraint: never exceed budget_limit
            for t, s in sorted_items:
                t_tokens = t.get("n_tokens", 1)
                if current_tokens + t_tokens <= budget_limit:
                    current_tokens += t_tokens
                    t_id = f"{t.get('conv_id')}_{t.get('session_id')}_{t.get('turn_idx')}"
                    retained_turn_ids.add(t_id)
                    # Also record by global turn idx or tuple
                    retained_turn_ids.add(f"{cid}_{t.get('global_turn_idx')}")

            total_retained_tokens_all += current_tokens

            # Check evidence turns in this conversation
            for t, _ in items:
                if t.get("label", 0) == 1:
                    total_evidence_turns += 1
                    t_id = f"{t.get('conv_id')}_{t.get('session_id')}_{t.get('turn_idx')}"
                    t_gid = f"{cid}_{t.get('global_turn_idx')}"
                    if t_id in retained_turn_ids or t_gid in retained_turn_ids:
                        retained_evidence_turns += 1

            # Check questions in this conversation
            conv_questions = questions_by_conv.get(cid, [])
            for q in conv_questions:
                if not q.evidence_turn_ids:
                    continue
                total_questions += 1
                # Check if all evidence turns are retained
                all_kept = True
                for ev_id in q.evidence_turn_ids:
                    # Match against retained IDs
                    if ev_id not in retained_turn_ids:
                        # Try suffix match or alternative index format
                        matched = any(ev_id in ret or ret in ev_id for ret in retained_turn_ids)
                        if not matched:
                            all_kept = False
                            break
                if all_kept:
                    fully_retained_questions += 1

        ev_recall = (retained_evidence_turns / total_evidence_turns) if total_evidence_turns > 0 else 0.0
        all_ev_rate = (fully_retained_questions / total_questions) if total_questions > 0 else 0.0
        token_reduction = 1.0 - (total_retained_tokens_all / max(total_tokens_all, 1))

        budget_results[pct_key] = {
            "evidence_recall": float(ev_recall),
            "all_evidence_retained_rate": float(all_ev_rate),
            "token_reduction": float(token_reduction),
            "retained_tokens": int(total_retained_tokens_all),
            "total_tokens": int(total_tokens_all),
        }

    return budget_results


def compute_slice_metrics(
    all_turns: List[Dict[str, Any]],
    all_scores: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Dict[str, float]]:
    """Compute sliced metrics by speaker, conversation length bucket, and category."""
    slices: Dict[str, Dict[str, float]] = {}

    # 1. By Speaker
    for spk in ("user", "assistant"):
        indices = [i for i, t in enumerate(all_turns) if t.get("speaker") == spk]
        if indices:
            y_t = np.array([all_turns[i].get("label", 0) for i in indices])
            y_p = all_scores[indices]
            slices[f"speaker_{spk}"] = compute_turn_level_metrics(y_t, y_p, threshold)

    # 2. By Conversation Length Bucket
    conv_lengths: Dict[str, int] = {}
    for t in all_turns:
        cid = t.get("conv_id", "default")
        conv_lengths[cid] = conv_lengths.get(cid, 0) + 1

    buckets = {
        "len_short_<20": lambda l: l < 20,
        "len_medium_20-50": lambda l: 20 <= l <= 50,
        "len_long_>50": lambda l: l > 50,
    }

    for b_name, cond in buckets.items():
        indices = [i for i, t in enumerate(all_turns) if cond(conv_lengths.get(t.get("conv_id", "default"), 0))]
        if indices:
            y_t = np.array([all_turns[i].get("label", 0) for i in indices])
            y_p = all_scores[indices]
            slices[b_name] = compute_turn_level_metrics(y_t, y_p, threshold)

    # 3. By Category (for positive evidence turns)
    categories = set(t.get("linked_category", "none") for t in all_turns if t.get("label", 0) == 1)
    for cat in categories:
        if cat == "none":
            continue
        # Compare positive turns of this category against all negative turns
        pos_indices = [i for i, t in enumerate(all_turns) if t.get("label") == 1 and t.get("linked_category") == cat]
        neg_indices = [i for i, t in enumerate(all_turns) if t.get("label", 0) == 0]
        comb_indices = pos_indices + neg_indices
        if pos_indices and neg_indices:
            y_t = np.array([all_turns[i].get("label", 0) for i in comb_indices])
            y_p = all_scores[comb_indices]
            slices[f"category_{cat}"] = compute_turn_level_metrics(y_t, y_p, threshold)

    return slices

