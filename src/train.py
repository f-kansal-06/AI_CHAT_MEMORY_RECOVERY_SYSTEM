"""
Training script for DistilBERT Turn-Level Memory Importance Model.
Implements:
- Weighted BCE loss with pos_weight from training set prevalence
- Optional auxiliary multi-task category head
- AdamW + linear warmup & decay
- Early stopping on validation PR-AUC
- CSV epoch logging (train/val loss, val PR-AUC)
- Support for seeds, max_train_turns subsetting, and CPU fallback
"""

import argparse
import csv
import logging
import math
import os
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.data.dataset import load_turns_from_jsonl, build_formatted_samples
from src.data.cleaning import CategoryMapper
from src.models.distilbert import DistilBertForMemoryImportance
from src.eval.metrics import find_best_f1_threshold, compute_turn_level_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CATEGORY_MAP = {
    "single_fact": 0,
    "multi_hop": 1,
    "temporal": 2,
    "knowledge_update": 3,
    "preference": 4,
    "other": 5,
    "none": 6,
}


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    else:
        torch.set_num_threads(min(4, os.cpu_count() or 1))


class TurnImportanceDataset(Dataset):
    def __init__(
        self,
        samples: List[Dict[str, Any]],
        tokenizer,
        max_length: int = 128,
    ):
        self.samples = samples
        texts = [s["input_text"] for s in samples]
        labels = [float(s.get("label", 0)) for s in samples]
        cats = [CATEGORY_MAP.get(s.get("linked_category", "none"), CATEGORY_MAP["none"]) for s in samples]

        if tokenizer is not None and len(texts) > 0:
            encoding = tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            self.input_ids = encoding["input_ids"]
            self.attention_mask = encoding["attention_mask"]
        else:
            self.input_ids = torch.zeros((len(texts), max_length), dtype=torch.long)
            self.attention_mask = torch.zeros((len(texts), max_length), dtype=torch.long)

        self.importance_labels = torch.tensor(labels, dtype=torch.float32)
        self.category_labels = torch.tensor(cats, dtype=torch.long)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "importance_labels": self.importance_labels[idx],
            "category_labels": self.category_labels[idx],
        }


def get_tokenizer(model_name: str = "distilbert-base-uncased"):
    try:
        return AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    except Exception as e:
        try:
            return AutoTokenizer.from_pretrained(model_name)
        except Exception:
            logger.warning(f"Could not load AutoTokenizer from '{model_name}' ({e}). Using robust fallback.")
            return None


def train_epoch(
    model: DistilBertForMemoryImportance,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    grad_clip: float = 1.0,
    fp16: bool = False,
) -> float:
    model.train()
    total_loss = 0.0
    scaler = torch.cuda.amp.GradScaler() if fp16 and torch.cuda.is_available() else None

    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        importance_labels = batch["importance_labels"].to(device)
        category_labels = batch["category_labels"].to(device)

        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    importance_labels=importance_labels,
                    category_labels=category_labels,
                )
                loss = outputs["loss"]
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                importance_labels=importance_labels,
                category_labels=category_labels,
            )
            loss = outputs["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if scheduler:
            scheduler.step()

        total_loss += loss.item()
        if (step + 1) % 20 == 0 or (step + 1) == len(dataloader):
            logger.info(f"  Step {step + 1}/{len(dataloader)} | Step Loss: {loss.item():.4f}")

    return total_loss / max(len(dataloader), 1)


@torch.no_grad()
def evaluate_epoch(
    model: DistilBertForMemoryImportance,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_scores = []
    all_labels = []

    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        importance_labels = batch["importance_labels"].to(device)
        category_labels = batch["category_labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            importance_labels=importance_labels,
            category_labels=category_labels,
        )
        loss = outputs["loss"]
        if loss is not None:
            total_loss += loss.item()

        scores = outputs["importance_scores"].detach().cpu().numpy()
        labels = importance_labels.detach().cpu().numpy()

        all_scores.extend(scores.tolist())
        all_labels.extend(labels.tolist())

    y_true = np.array(all_labels)
    y_prob = np.array(all_scores)
    val_loss = total_loss / max(len(dataloader), 1)

    if len(np.unique(y_true)) > 1:
        pr_auc = float(average_precision_score(y_true, y_prob))
    else:
        pr_auc = float(np.mean(y_true))

    return val_loss, pr_auc, y_true, y_prob


def run_training(
    train_turns: List[Dict[str, Any]],
    val_turns: List[Dict[str, Any]],
    config: Dict[str, Any],
    output_dir: str,
    seed: int = 42,
) -> Dict[str, Any]:
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # Save exact config used
    with open(Path(output_dir) / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False)

    context_window = config.get("context_window", 1)
    max_train_turns = config.get("max_train_turns")
    if max_train_turns and len(train_turns) > max_train_turns:
        pos_turns = [t for t in train_turns if t.get("label", 0) == 1]
        neg_turns = [t for t in train_turns if t.get("label", 0) == 0]
        # Allow up to half of max_train_turns for positives so negatives are always included
        max_pos_allowed = max(1, int(max_train_turns * 0.5))
        n_pos = min(len(pos_turns), max_pos_allowed)
        n_neg = min(len(neg_turns), max_train_turns - n_pos)
        train_turns = pos_turns[:n_pos] + neg_turns[:n_neg]
        logger.info(f"Subsetting train turns: preserved {n_pos} positives + {n_neg} negatives.")

    max_val_turns = config.get("max_val_turns", 500)
    if max_val_turns and len(val_turns) > max_val_turns:
        val_pos = [t for t in val_turns if t.get("label", 0) == 1]
        val_neg = [t for t in val_turns if t.get("label", 0) == 0]
        max_val_pos_allowed = max(1, int(max_val_turns * 0.5))
        n_v_pos = min(len(val_pos), max_val_pos_allowed)
        n_v_neg = min(len(val_neg), max_val_turns - n_v_pos)
        val_turns = val_pos[:n_v_pos] + val_neg[:n_v_neg]

    # Format turns into model inputs
    train_samples = build_formatted_samples(train_turns, context_window=context_window)
    val_samples = build_formatted_samples(val_turns, context_window=context_window)

    # Compute pos_weight from training label prevalence
    n_pos = sum(1 for s in train_samples if s.get("label", 0) == 1)
    n_neg = len(train_samples) - n_pos
    pos_weight = (n_neg / max(n_pos, 1)) if n_pos > 0 else 1.0
    logger.info(f"Training label prevalence: {n_pos} pos, {n_neg} neg. Calculated pos_weight = {pos_weight:.2f}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")

    # Model & Tokenizer
    model_name = config.get("model_name_or_path", "distilbert-base-uncased")
    tokenizer = get_tokenizer(model_name)
    max_length = config.get("max_length", 256)

    train_ds = TurnImportanceDataset(train_samples, tokenizer, max_length=max_length)
    val_ds = TurnImportanceDataset(val_samples, tokenizer, max_length=max_length)

    batch_size = config.get("batch_size", 32 if torch.cuda.is_available() else 16)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    aux_weight = float(config.get("aux_weight", 0.0))
    model = DistilBertForMemoryImportance(
        model_name_or_path=model_name,
        num_categories=len(CATEGORY_MAP),
        pos_weight=pos_weight,
        aux_weight=aux_weight,
    ).to(device)

    epochs = config.get("epochs", 5)
    lr = float(config.get("lr", 2e-5))
    weight_decay = float(config.get("weight_decay", 0.01))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * config.get("warmup_ratio", 0.10))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # Logging CSV
    csv_path = Path(output_dir) / "training_curves.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_pr_auc", "lr"])

    best_val_pr_auc = -1.0
    best_threshold = 0.5
    best_checkpoint_path = Path(output_dir) / "best_model.pt"
    patience = config.get("patience", 3)
    patience_counter = 0

    history = []

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            grad_clip=config.get("grad_clip", 1.0),
            fp16=(torch.cuda.is_available()),
        )
        val_loss, val_pr_auc, val_y_true, val_y_prob = evaluate_epoch(model, val_loader, device)
        cur_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val PR-AUC: {val_pr_auc:.4f}"
        )

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, val_pr_auc, cur_lr])

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_pr_auc": val_pr_auc,
        })

        if val_pr_auc > best_val_pr_auc:
            best_val_pr_auc = val_pr_auc
            best_threshold = find_best_f1_threshold(val_y_true, val_y_prob)
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_pr_auc": best_val_pr_auc,
                    "best_threshold": best_threshold,
                    "pos_weight": pos_weight,
                    "aux_weight": aux_weight,
                    "config": config,
                },
                best_checkpoint_path,
            )
            logger.info(f"[*] Best checkpoint saved (val_pr_auc={best_val_pr_auc:.4f}, threshold={best_threshold:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch}.")
                break

    return {
        "best_val_pr_auc": best_val_pr_auc,
        "best_threshold": best_threshold,
        "checkpoint_path": str(best_checkpoint_path),
        "history": history,
    }
