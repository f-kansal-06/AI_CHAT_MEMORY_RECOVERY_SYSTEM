"""Tune DistilBERT on the existing leakage-safe LongMemEval split.

Uses a seeded randomized search, early stopping, and validation PR-AUC. The
held-out test set is evaluated only after the best validation checkpoint wins.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baselines.tfidf_lr import TfidfLogisticRegression
from src.data.dataset import build_formatted_samples, load_turns_from_jsonl
from src.data.schema import QuestionRecord
from src.data.identifiers import questions_grounded_in_turns
from src.eval.metrics import compute_turn_level_metrics, find_best_f1_threshold
from src.evaluate import run_full_evaluation, score_with_distilbert
from src.train import get_tokenizer, run_training
from src.models.distilbert import DistilBertForMemoryImportance

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("memory_tuning")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("longmemeval", "custom"), default="custom")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--max-train-turns", type=int, default=2000)
    parser.add_argument("--output", default="results/tuning")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    data_prefix = f"data/processed/{args.dataset}"
    train = load_turns_from_jsonl(f"{data_prefix}_train.jsonl")
    val = load_turns_from_jsonl(f"{data_prefix}_val.jsonl")
    test = load_turns_from_jsonl(f"{data_prefix}_test.jsonl")
    questions = [QuestionRecord.from_dict(json.loads(line)) for line in Path(f"{data_prefix}_questions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    questions = questions_grounded_in_turns(questions, test)
    data_fingerprint = hashlib.sha256("".join(Path(f"{data_prefix}_{split}.jsonl").read_text(encoding="utf-8") for split in ("train", "val", "test")).encode("utf-8")).hexdigest()

    grid = list(itertools.product(
        [1e-5, 2e-5], [8, 16], [0.1, 0.2], [128, 256], [0.01, 0.05], [3, 5]
    ))
    random.Random(args.seed).shuffle(grid)
    trials = []
    best = None
    for idx, (lr, batch_size, dropout, max_length, weight_decay, epochs) in enumerate(grid[:args.trials], 1):
        config = {
            "model_name_or_path": "distilbert-base-uncased", "context_window": 1,
            "max_length": max_length, "batch_size": batch_size, "epochs": epochs,
            "lr": lr, "weight_decay": weight_decay, "dropout": dropout,
            "warmup_ratio": 0.1, "grad_clip": 1.0, "patience": 2,
            "aux_weight": 0.0, "max_train_turns": args.max_train_turns,
            "max_val_turns": 1500,
        }
        trial_dir = out / args.dataset / f"trial_{idx:02d}"
        trial_meta_path = trial_dir / "trial_metadata.json"
        logger.info("Trial %d/%d: %s", idx, min(args.trials, len(grid)), config)
        checkpoint_file = trial_dir / "best_model.pt"
        metadata_matches = False
        if trial_meta_path.exists():
            trial_meta = json.loads(trial_meta_path.read_text(encoding="utf-8"))
            metadata_matches = trial_meta.get("data_fingerprint") == data_fingerprint and trial_meta.get("config") == config
        if checkpoint_file.exists() and (metadata_matches or not trial_meta_path.exists()):
            saved = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
            if saved.get("config") != config:
                raise RuntimeError(f"Existing trial checkpoint config mismatch: {checkpoint_file}")
            result = {"best_val_pr_auc": saved["val_pr_auc"], "best_threshold": saved["best_threshold"], "checkpoint_path": str(checkpoint_file)}
            logger.info("Reusing completed trial checkpoint %s", checkpoint_file)
        else:
            result = run_training(train, val, config, str(trial_dir), seed=args.seed)
        trial_dir.mkdir(parents=True, exist_ok=True)
        trial_meta_path.write_text(json.dumps({"data_fingerprint": data_fingerprint, "config": config, "seed": args.seed}, indent=2), encoding="utf-8")
        record = {"trial": idx, "config": config, "best_val_pr_auc": result["best_val_pr_auc"], "best_threshold": result["best_threshold"], "checkpoint_path": Path(result["checkpoint_path"]).as_posix()}
        trials.append(record)
        (out / f"{args.dataset}_trials.json").write_text(json.dumps(trials, indent=2), encoding="utf-8")
        if best is None or record["best_val_pr_auc"] > best["best_val_pr_auc"]:
            best = record
        del result
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if best is None:
        raise RuntimeError("No tuning trials completed")
    ckpt = torch.load(best["checkpoint_path"], map_location="cpu", weights_only=False)
    model = DistilBertForMemoryImportance(
        model_name_or_path=best["config"]["model_name_or_path"],
        pos_weight=ckpt["pos_weight"], aux_weight=best["config"]["aux_weight"],
        dropout_rate=best["config"]["dropout"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    tokenizer = get_tokenizer(best["config"]["model_name_or_path"])
    samples = build_formatted_samples(test, context_window=best["config"]["context_window"])
    scores = score_with_distilbert(model, samples, tokenizer, max_length=best["config"]["max_length"])
    labels = np.asarray([s["label"] for s in samples])
    metrics = compute_turn_level_metrics(labels, scores, threshold=best["best_threshold"])
    test_eval = run_full_evaluation(scores, test, questions=questions, model_name="best_tuned_distilbert", val_threshold=best["best_threshold"])

    # A light lexical baseline is measured on the same test split.
    baseline_train = train
    if len(train) > args.max_train_turns:
        selected_indices = set(np.random.RandomState(args.seed).choice(len(train), size=args.max_train_turns, replace=False).tolist())
        baseline_train = [turn for i, turn in enumerate(train) if i in selected_indices]
    tr_samples = build_formatted_samples(baseline_train)
    tfidf = TfidfLogisticRegression().fit([s["input_text"] for s in tr_samples], [s["label"] for s in tr_samples])
    tfidf_scores = tfidf.predict_proba([s["input_text"] for s in samples])
    val_samples = build_formatted_samples(val)
    val_scores = tfidf.predict_proba([s["input_text"] for s in val_samples])
    val_labels = np.asarray([s["label"] for s in val_samples])
    baseline_threshold = find_best_f1_threshold(val_labels, val_scores)
    baseline_metrics = compute_turn_level_metrics(labels, tfidf_scores, threshold=baseline_threshold)
    report = {
        "dataset": args.dataset,
        "data_fingerprint_sha256": data_fingerprint,
        "label_provenance": "weak heuristic labels; exploratory only" if args.dataset == "custom" else "benchmark evidence labels",
        "train_turns": len(train), "validation_turns": len(val), "test_turns": len(test),
        "positive_counts": {"train": sum(t.get("label", 0) == 1 for t in train), "validation": sum(t.get("label", 0) == 1 for t in val), "test": sum(t.get("label", 0) == 1 for t in test)},
        "selection_metric": "validation PR-AUC",
        "test_used_for_selection": False,
        "trials": trials,
        "best_trial": best,
        "best_model_test_metrics": metrics,
        "best_model_test_evaluation": test_eval,
        "tfidf_baseline_test_metrics": baseline_metrics,
        "tfidf_baseline_validation_threshold": baseline_threshold,
    }
    (out / "tuning_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Best validation PR-AUC %.4f; test PR-AUC %.4f, F1 %.4f", best["best_val_pr_auc"], metrics["pr_auc"], metrics["f1"])
    logger.info("Best checkpoint: %s", best["checkpoint_path"])


if __name__ == "__main__":
    main()
