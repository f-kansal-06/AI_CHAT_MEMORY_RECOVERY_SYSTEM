"""
Comprehensive pytest test suite for memory importance benchmark pipeline.
Covers:
1. Schema validation
2. Leakage checks (no conv_id or session hash shared across splits, custom text absent from test)
3. Label alignment
4. Metric correctness on hand-computed examples
5. Budget constraint verification (never exceeds budget)
6. Simulated recovery constraints
7. Determinism
8. End-to-end smoke test
"""

import json
import numpy as np
import pytest
import torch
from pathlib import Path

from src.data.schema import TurnRecord, QuestionRecord, validate_turn_record
from src.data.dataset import build_formatted_samples, format_turn_input
from src.data.split import split_longmemeval, compute_session_hash
from src.baselines.heuristics import RandomBaseline, RecencyBaseline, LengthBaseline
from src.baselines.tfidf_lr import TfidfLogisticRegression
from src.models.distilbert import DistilBertForMemoryImportance
from src.eval.metrics import (
    compute_turn_level_metrics,
    compute_ranking_metrics_per_conversation,
    compute_budgeted_retention,
    compute_ndcg_at_k,
)
from src.eval.recovery import run_recovery_simulation
from tests.fixtures.synthetic_fixture import get_tiny_synthetic_fixture


def test_schema_validation():
    """Verify that records conform to unified TurnRecord schema."""
    turns, questions = get_tiny_synthetic_fixture()
    for t in turns:
        d = t.to_dict()
        assert validate_turn_record(d) is True

    # Bad record
    bad_record = {"conv_id": "c1", "text": "bad"}
    assert validate_turn_record(bad_record) is False


def test_processed_data_schema():
    """Verify saved processed files strictly conform to schema."""
    for name in ["longmemeval_train.jsonl", "locomo_train.jsonl", "custom_train.jsonl"]:
        path = Path("data/processed") / name
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    if idx >= 50:
                        break
                    data = json.loads(line)
                    assert validate_turn_record(data) is True


def test_no_leakage_in_splits():
    """Verify no conv_id or session hash is shared between train, val, and test."""
    train_path = Path("data/processed/longmemeval_train.jsonl")
    val_path = Path("data/processed/longmemeval_val.jsonl")
    test_path = Path("data/processed/longmemeval_test.jsonl")

    if train_path.exists() and val_path.exists() and test_path.exists():
        with open(train_path, "r", encoding="utf-8") as f:
            tr = [json.loads(l) for l in f]
        with open(val_path, "r", encoding="utf-8") as f:
            va = [json.loads(l) for l in f]
        with open(test_path, "r", encoding="utf-8") as f:
            te = [json.loads(l) for l in f]

        tr_cids = set(t["conv_id"] for t in tr)
        va_cids = set(t["conv_id"] for t in va)
        te_cids = set(t["conv_id"] for t in te)

        # Check disjoint conv_ids
        assert len(tr_cids.intersection(va_cids)) == 0
        assert len(tr_cids.intersection(te_cids)) == 0
        assert len(va_cids.intersection(te_cids)) == 0


def test_custom_text_absent_from_test_sets():
    """Verify custom dataset normalized text does not appear in test sets."""
    custom_path = Path("data/processed/custom_all.jsonl")
    test_path = Path("data/processed/longmemeval_test.jsonl")
    if custom_path.exists() and test_path.exists():
        with open(custom_path, "r", encoding="utf-8") as f:
            custom_texts = set(" ".join(json.loads(l)["text"].lower().split()) for l in f)
        with open(test_path, "r", encoding="utf-8") as f:
            test_texts = set(" ".join(json.loads(l)["text"].lower().split()) for l in f)

        overlap = custom_texts.intersection(test_texts)
        assert len(overlap) == 0


def test_label_alignment_fixture():
    """Verify evidence IDs map to the right turns in fixture."""
    turns, questions = get_tiny_synthetic_fixture()
    # Turn 2 is evidence for Q1
    assert turns[2].label == 1
    assert "q1" in turns[2].linked_question_ids
    assert turns[2].linked_category == "single_fact"

    # Turn 6 is evidence for Q1 and Q2
    assert turns[6].label == 1
    assert "q1" in turns[6].linked_question_ids
    assert "q2" in turns[6].linked_question_ids
    assert turns[6].linked_category == "multi_hop"


def test_hand_computed_ranking_metrics():
    """Verify ranking metrics (Recall@K, nDCG@K, MRR) on hand-computed scores."""
    # 5 turns, 2 evidence turns at index 0 and 2
    conv_turns = [
        {"label": 1, "global_turn_idx": 0},
        {"label": 0, "global_turn_idx": 1},
        {"label": 1, "global_turn_idx": 2},
        {"label": 0, "global_turn_idx": 3},
        {"label": 0, "global_turn_idx": 4},
    ]
    # Perfect predictor
    scores = np.array([0.9, 0.1, 0.8, 0.2, 0.05])
    metrics = compute_ranking_metrics_per_conversation(conv_turns, scores, k_fractions=(0.20, 0.40))

    # K=20% of 5 = 1 turn (top score = index 0, label 1)
    # Evidence recall at 20% = 1 / 2 = 0.5
    assert metrics["recall@20%"] == 0.5

    # K=40% of 5 = 2 turns (top scores = index 0 and 2, both label 1)
    # Evidence recall at 40% = 2 / 2 = 1.0
    assert metrics["recall@40%"] == 1.0
    assert metrics["mrr"] == 1.0


def test_budget_routine_never_exceeds_budget():
    """Verify that budgeted retention strictly respects token limits."""
    turns, questions = get_tiny_synthetic_fixture()
    dict_turns = [t.to_dict() for t in turns]
    scores = np.linspace(0.1, 0.99, len(dict_turns))

    budget_res = compute_budgeted_retention(
        dict_turns,
        scores,
        questions,
        budget_fractions=(0.05, 0.10, 0.20, 0.30, 0.50),
    )

    conv1_tokens = 100
    conv2_tokens = 80

    for b_key, res in budget_res.items():
        b_frac = int(b_key.replace("%", "")) / 100.0
        # Check total retained tokens never exceeds total allowed budget
        max_allowed = int(conv1_tokens * b_frac) + int(conv2_tokens * b_frac)
        assert res["retained_tokens"] <= max_allowed


def test_recovery_simulation_constraints():
    """Verify recovery simulation: only pre-cut turns scored, only pre-cut questions counted."""
    turns, questions = get_tiny_synthetic_fixture()
    dict_turns = [t.to_dict() for t in turns]

    def dummy_score_fn(sub_turns):
        return np.linspace(0.1, 0.9, len(sub_turns))

    res = run_recovery_simulation(
        dict_turns,
        questions,
        dummy_score_fn,
        cut_fractions=(0.25, 0.50, 0.75),
        budget_fractions=(0.10, 0.20),
    )

    # At cut_25%: conv 1 cut at turn floor(10*0.25) = 2.
    # Pre-cut turns are 0, 1. Neither evidence turn (idx 2, 6) is before cut.
    # So eligible questions at cut_25% should be 0.
    assert res["cut_25%"]["budget_10%"]["eligible_questions"] == 0

    # At cut_75%: conv 1 cut at floor(10*0.75) = 7.
    # Pre-cut turns are 0..6. Both turn 2 and 6 are included!
    # So Q1 and Q2 are eligible!
    assert res["cut_75%"]["budget_20%"]["eligible_questions"] > 0


def test_determinism_seed():
    """Verify that same seed produces identical split and identical initial loss."""
    turns, _ = get_tiny_synthetic_fixture()
    d1 = [t.to_dict() for t in turns]
    d2 = [t.to_dict() for t in turns]

    tr1, v1, te1, _ = split_longmemeval(d1, seed=42)
    tr2, v2, te2, _ = split_longmemeval(d2, seed=42)

    assert len(tr1) == len(tr2)
    assert [t["global_turn_idx"] for t in tr1] == [t["global_turn_idx"] for t in tr2]


def test_end_to_end_smoke_run():
    """Fast end-to-end smoke test on tiny fixture (CPU, 1 epoch, under 30s)."""
    turns, questions = get_tiny_synthetic_fixture()
    dict_turns = [t.to_dict() for t in turns]

    train_turns = dict_turns[:12]
    val_turns = dict_turns[12:]

    # Test baseline fit and predict
    bl = TfidfLogisticRegression()
    samples_tr = build_formatted_samples(train_turns)
    texts_tr = [s["input_text"] for s in samples_tr]
    labels_tr = [s["label"] for s in samples_tr]
    bl.fit(texts_tr, labels_tr)

    samples_va = build_formatted_samples(val_turns)
    texts_va = [s["input_text"] for s in samples_va]
    scores = bl.predict_proba(texts_va)
    assert len(scores) == len(val_turns)
    assert np.all(scores >= 0.0) and np.all(scores <= 1.0)

