"""
Data Preparation and Leakage-Safe Splitting Pipeline.
Loads raw data, performs schema validation, generates leakage-safe splits,
and writes processed JSONL and split ID files to data/processed and data/splits.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List

from src.data.schema import TurnRecord, QuestionRecord, validate_turn_record
from src.data.loader_longmemeval import load_longmemeval
from src.data.loader_locomo import load_locomo
from src.data.loader_custom import load_custom_dataset
from src.data.cleaning import CategoryMapper
from src.data.split import split_longmemeval, split_locomo_5fold, split_custom_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def save_turns_to_jsonl(turns: List[TurnRecord], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for t in turns:
            f.write(json.dumps(t.to_dict()) + "\n")


def save_questions_to_jsonl(questions: List[QuestionRecord], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q.to_dict()) + "\n")


def prepare_all_data(
    longmemeval_raw: str = "data/raw/longmemeval_s.json",
    locomo_raw: str = "data/raw/locomo.json",
    custom_raw: str = "data/raw/custom_4221.jsonl",
    output_processed_dir: str = "data/processed",
    output_splits_dir: str = "data/splits",
    seed: int = 42,
    max_longmemeval_instances: int = 25,  # 25 full haystacks gives ~13,000 turns, fast & rich for lab
):
    proc_dir = Path(output_processed_dir)
    splits_dir = Path(output_splits_dir)
    proc_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    cat_mapper = CategoryMapper()

    # 1. LongMemEval
    logger.info(f"Processing LongMemEval from {longmemeval_raw}...")
    lme_turns, lme_questions = load_longmemeval(
        longmemeval_raw,
        category_mapper=cat_mapper,
        max_instances=max_longmemeval_instances,
    )
    # Validate schema
    for t in lme_turns:
        assert validate_turn_record(t.to_dict()), f"Invalid turn record: {t}"

    save_turns_to_jsonl(lme_turns, proc_dir / "longmemeval_all.jsonl")
    save_questions_to_jsonl(lme_questions, proc_dir / "longmemeval_questions.jsonl")

    # Leakage-safe split for LongMemEval
    lme_dict_turns = [t.to_dict() for t in lme_turns]
    tr_turns, val_turns, te_turns, lme_split_stats = split_longmemeval(lme_dict_turns, seed=seed)
    logger.info(f"LongMemEval split stats: {lme_split_stats}")

    with open(proc_dir / "longmemeval_train.jsonl", "w", encoding="utf-8") as f:
        for t in tr_turns:
            f.write(json.dumps(t) + "\n")
    with open(proc_dir / "longmemeval_val.jsonl", "w", encoding="utf-8") as f:
        for t in val_turns:
            f.write(json.dumps(t) + "\n")
    with open(proc_dir / "longmemeval_test.jsonl", "w", encoding="utf-8") as f:
        for t in te_turns:
            f.write(json.dumps(t) + "\n")

    with open(splits_dir / "longmemeval_splits.json", "w", encoding="utf-8") as f:
        json.dump(lme_split_stats, f, indent=2)

    # 2. LoCoMo
    logger.info(f"Processing LoCoMo from {locomo_raw}...")
    loc_turns, loc_questions = load_locomo(locomo_raw, category_mapper=cat_mapper)
    for t in loc_turns:
        assert validate_turn_record(t.to_dict()), f"Invalid turn record: {t}"

    save_turns_to_jsonl(loc_turns, proc_dir / "locomo_all.jsonl")
    save_questions_to_jsonl(loc_questions, proc_dir / "locomo_questions.jsonl")

    loc_dict_turns = [t.to_dict() for t in loc_turns]
    loc_folds = split_locomo_5fold(loc_dict_turns, seed=seed)
    logger.info(f"LoCoMo 5-fold CV prepared with {len(loc_folds)} folds.")

    with open(splits_dir / "locomo_folds.json", "w", encoding="utf-8") as f:
        json.dump(loc_folds, f, indent=2)

    # Also prepare standard fold 0 train/val/test splits for direct evaluations like E4
    f0 = loc_folds[0]
    loc_tr = [t for t in loc_dict_turns if t["conv_id"] in f0["train_conv_ids"]]
    loc_va = [t for t in loc_dict_turns if t["conv_id"] in f0["val_conv_ids"]]
    loc_te = [t for t in loc_dict_turns if t["conv_id"] in f0["test_conv_ids"]]

    with open(proc_dir / "locomo_train.jsonl", "w", encoding="utf-8") as f:
        for t in loc_tr:
            f.write(json.dumps(t) + "\n")
    with open(proc_dir / "locomo_val.jsonl", "w", encoding="utf-8") as f:
        for t in loc_va:
            f.write(json.dumps(t) + "\n")
    with open(proc_dir / "locomo_test.jsonl", "w", encoding="utf-8") as f:
        for t in loc_te:
            f.write(json.dumps(t) + "\n")

    # 3. Custom Dataset (~4,221 turns)
    logger.info(f"Processing Custom Dataset from {custom_raw}...")
    cust_turns, cust_questions = load_custom_dataset(custom_raw, category_mapper=cat_mapper)
    for t in cust_turns:
        assert validate_turn_record(t.to_dict()), f"Invalid turn record: {t}"

    save_turns_to_jsonl(cust_turns, proc_dir / "custom_all.jsonl")
    save_questions_to_jsonl(cust_questions, proc_dir / "custom_questions.jsonl")

    cust_dict_turns = [t.to_dict() for t in cust_turns]
    c_tr, c_va, c_te = split_custom_dataset(cust_dict_turns, seed=seed)
    logger.info(f"Custom dataset split: train={len(c_tr)}, val={len(c_va)}, test={len(c_te)}")

    with open(proc_dir / "custom_train.jsonl", "w", encoding="utf-8") as f:
        for t in c_tr:
            f.write(json.dumps(t) + "\n")
    with open(proc_dir / "custom_val.jsonl", "w", encoding="utf-8") as f:
        for t in c_va:
            f.write(json.dumps(t) + "\n")
    with open(proc_dir / "custom_test.jsonl", "w", encoding="utf-8") as f:
        for t in c_te:
            f.write(json.dumps(t) + "\n")

    cust_split_stats = {
        "total_turns": len(cust_dict_turns),
        "train_turns": len(c_tr),
        "val_turns": len(c_va),
        "test_turns": len(c_te),
        "train_convs": len(set(t["conv_id"] for t in c_tr)),
        "val_convs": len(set(t["conv_id"] for t in c_va)),
        "test_convs": len(set(t["conv_id"] for t in c_te)),
    }
    with open(splits_dir / "custom_splits.json", "w", encoding="utf-8") as f:
        json.dump(cust_split_stats, f, indent=2)

    logger.info("Data preparation completed successfully for all three datasets!")


if __name__ == "__main__":
    prepare_all_data()

