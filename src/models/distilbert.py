"""
DistilBERT model with binary importance head and optional auxiliary multi-task category head.
"""

import logging
from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertConfig, PreTrainedModel

logger = logging.getLogger(__name__)


class DistilBertForMemoryImportance(nn.Module):
    """
    DistilBERT encoder with:
    1. Turn-level binary importance head (1 logit, weighted BCE)
    2. Optional auxiliary category classification head (num_categories logits, CrossEntropy)
    """

    def __init__(
        self,
        model_name_or_path: str = "distilbert-base-uncased",
        num_categories: int = 7,  # single_fact, multi_hop, temporal, knowledge_update, preference, other, none
        dropout_rate: float = 0.2,
        pos_weight: Optional[float] = None,
        aux_weight: float = 0.0,
    ):
        super().__init__()
        self.num_categories = num_categories
        self.aux_weight = aux_weight

        try:
            self.distilbert = DistilBertModel.from_pretrained(model_name_or_path, local_files_only=True)
            hidden_size = self.distilbert.config.dim
        except Exception:
            try:
                self.distilbert = DistilBertModel.from_pretrained(model_name_or_path)
                hidden_size = self.distilbert.config.dim
            except Exception as e:
                logger.warning(
                    f"Could not load pretrained weights for '{model_name_or_path}' ({e}). "
                    "Initializing randomly initialized DistilBertConfig."
                )
                config = DistilBertConfig(
                    vocab_size=30522,
                    max_position_embeddings=512,
                    sinusoidal_pos_embds=False,
                    n_layers=4,
                    n_heads=8,
                    dim=256,
                    hidden_dim=1024,
                    dropout=0.1,
                    attention_dropout=0.1,
                )
                self.distilbert = DistilBertModel(config)
                hidden_size = 256

        self.pre_classifier = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

        # Primary importance head (1 logit)
        self.importance_head = nn.Linear(hidden_size, 1)

        # Auxiliary category head
        if num_categories > 1:
            self.aux_head = nn.Linear(hidden_size, num_categories)
        else:
            self.aux_head = None

        # Pos weight for imbalanced BCE
        if pos_weight is not None and pos_weight > 0:
            self.register_buffer("pos_weight_tensor", torch.tensor([pos_weight], dtype=torch.float32))
        else:
            self.pos_weight_tensor = None

        self.aux_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def get_importance_loss_fn(self, device: torch.device) -> nn.BCEWithLogitsLoss:
        if self.pos_weight_tensor is not None:
            return nn.BCEWithLogitsLoss(pos_weight=self.pos_weight_tensor.to(device))
        return nn.BCEWithLogitsLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        importance_labels: Optional[torch.Tensor] = None,
        category_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass.
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            importance_labels: [batch_size] float in {0.0, 1.0}
            category_labels: [batch_size] int category indices or -100
        """
        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_state = distilbert_output[0]  # [batch_size, seq_len, hidden_size]
        # Pool at [CLS] token (index 0)
        cls_token = hidden_state[:, 0]

        pooled = self.pre_classifier(cls_token)
        pooled = self.relu(pooled)
        pooled = self.dropout(pooled)

        importance_logits = self.importance_head(pooled).squeeze(-1)  # [batch_size]
        importance_scores = torch.sigmoid(importance_logits)

        aux_logits = None
        if self.aux_head is not None:
            aux_logits = self.aux_head(pooled)  # [batch_size, num_categories]

        loss = None
        imp_loss_val = None
        aux_loss_val = None

        if importance_labels is not None:
            imp_loss_fn = self.get_importance_loss_fn(input_ids.device)
            imp_loss_val = imp_loss_fn(importance_logits, importance_labels.float())
            loss = imp_loss_val

            if self.aux_head is not None and category_labels is not None and self.aux_weight > 0:
                aux_loss_val = self.aux_loss_fn(aux_logits, category_labels.long())
                loss = loss + (self.aux_weight * aux_loss_val)

        return {
            "loss": loss,
            "importance_loss": imp_loss_val,
            "aux_loss": aux_loss_val,
            "importance_logits": importance_logits,
            "importance_scores": importance_scores,
            "aux_logits": aux_logits,
        }
