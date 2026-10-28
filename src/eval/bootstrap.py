"""
Bootstrap Confidence Interval computation for conversation memory importance evaluation.
Performs 1,000 resamples by CONVERSATION (not turns) to compute 95% CIs.
"""

from typing import List, Dict, Any, Tuple, Callable
import numpy as np


def compute_bootstrap_ci(
    conv_ids: List[str],
    eval_fn: Callable[[List[str]], float],
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Compute bootstrap confidence interval by resampling conversation IDs with replacement.
    """
    rng = np.random.RandomState(seed)
    unique_convs = np.array(list(set(conv_ids)))
    n_convs = len(unique_convs)

    if n_convs == 0:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0, "std": 0.0}

    boot_estimates = np.zeros(n_resamples, dtype=np.float64)

    for b in range(n_resamples):
        resampled_cids = rng.choice(unique_convs, size=n_convs, replace=True).tolist()
        boot_estimates[b] = eval_fn(resampled_cids)

    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_estimates, alpha * 100))
    upper = float(np.percentile(boot_estimates, (1.0 - alpha) * 100))
    mean_val = float(np.mean(boot_estimates))
    std_val = float(np.std(boot_estimates))

    return {
        "mean": mean_val,
        "lower": lower,
        "upper": upper,
        "std": std_val,
    }


def compute_paired_bootstrap_comparison(
    conv_ids: List[str],
    eval_fn_a: Callable[[List[str]], float],
    eval_fn_b: Callable[[List[str]], float],
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Compute paired bootstrap comparison (A vs B) by resampling conversation IDs.
    Returns CIs for A, B, and the difference (A - B).
    """
    rng = np.random.RandomState(seed)
    unique_convs = np.array(list(set(conv_ids)))
    n_convs = len(unique_convs)

    diffs = np.zeros(n_resamples, dtype=np.float64)
    scores_a = np.zeros(n_resamples, dtype=np.float64)
    scores_b = np.zeros(n_resamples, dtype=np.float64)

    for b in range(n_resamples):
        resampled_cids = rng.choice(unique_convs, size=n_convs, replace=True).tolist()
        s_a = eval_fn_a(resampled_cids)
        s_b = eval_fn_b(resampled_cids)
        scores_a[b] = s_a
        scores_b[b] = s_b
        diffs[b] = s_a - s_b

    alpha = (1.0 - ci) / 2.0
    diff_lower = float(np.percentile(diffs, alpha * 100))
    diff_upper = float(np.percentile(diffs, (1.0 - alpha) * 100))

    return {
        "mean_a": float(np.mean(scores_a)),
        "ci_a": [float(np.percentile(scores_a, alpha * 100)), float(np.percentile(scores_a, (1.0 - alpha) * 100))],
        "mean_b": float(np.mean(scores_b)),
        "ci_b": [float(np.percentile(scores_b, alpha * 100)), float(np.percentile(scores_b, (1.0 - alpha) * 100))],
        "mean_diff": float(np.mean(diffs)),
        "ci_diff_95": [diff_lower, diff_upper],
        "p_value_heuristic": float(np.mean(diffs <= 0.0) if np.mean(diffs) > 0 else np.mean(diffs >= 0.0)),
    }

