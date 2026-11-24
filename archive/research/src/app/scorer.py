"""Optional local DistilBERT importance scorer for imported conversations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.data.dataset import format_turn_input
from src.models.distilbert import DistilBertForMemoryImportance
from src.train import TurnImportanceDataset, get_tokenizer
from torch.utils.data import DataLoader


class LocalImportanceScorer:
    def __init__(self, report_path: str | Path = "results/tuning/tuning_results.json"):
        report_path = Path(report_path)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        best = report["best_trial"]
        config = best["config"]
        checkpoint = torch.load(best["checkpoint_path"], map_location="cpu", weights_only=False)
        self.model = DistilBertForMemoryImportance(
            model_name_or_path=config["model_name_or_path"], pos_weight=checkpoint["pos_weight"],
            aux_weight=config.get("aux_weight", 0), dropout_rate=config.get("dropout", .2), local_files_only=True,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.tokenizer = get_tokenizer(config["model_name_or_path"], local_files_only=True)
        self.max_length = int(config["max_length"])
        self.context_window = int(config.get("context_window", 1))
        self.name = f"DistilBERT/{report['dataset']}"

    def score(self, messages: list[dict[str, Any]]) -> dict[str, float]:
        ordered = sorted(messages, key=lambda m: int(m.get("sequence", 0)))
        samples = []
        for i, msg in enumerate(ordered):
            current = {"speaker": msg.get("role", "user"), "text": msg.get("content", "")}
            prev = [{"speaker": x.get("role", "user"), "text": x.get("content", "")} for x in ordered[:i]]
            samples.append({"input_text": format_turn_input(current, prev, self.context_window), "label": 0, "linked_category": "none", "message_id": msg["message_id"]})
        dataset = TurnImportanceDataset(samples, self.tokenizer, max_length=self.max_length)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        scores: list[float] = []
        with torch.inference_mode():
            for batch in loader:
                output = self.model(input_ids=batch["input_ids"].to(self.device), attention_mask=batch["attention_mask"].to(self.device))
                scores.extend(output["importance_scores"].detach().cpu().tolist())
        return {sample["message_id"]: float(score) for sample, score in zip(samples, scores)}


def load_local_scorer(report_path: str | Path = "results/tuning/tuning_results.json") -> LocalImportanceScorer | None:
    if not Path(report_path).exists():
        return None
    try:
        return LocalImportanceScorer(report_path)
    except Exception as exc:
        # The local rule extractor still works if an optional checkpoint is
        # missing or incompatible; callers display that fallback to the user.
        print(f"Local importance model unavailable ({exc}); using rule-based importance only.")
        return None
