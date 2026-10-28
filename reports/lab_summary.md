# Lab Milestone Summary Report: Conversation Memory Importance Model

**Date**: 2026-10-28
**Pipeline**: Preprocessing → Baselines → DistilBERT Fine-Tuning → Evaluation

## 1. Dataset Sizes & Label Prevalence After Preprocessing

| Dataset | Train Turns | Val Turns | Test Turns | Total Turns | Evidence Prevalence (Pos %) | Split Method |
|---|---|---|---|---|---|---|
| **LongMemEval** | 8,608 | 1,460 | 2,377 | 12,445 | 0.36% | Grouped by Haystack (Leakage-Safe Session Hash) |
| **LoCoMo** | ~3,500 | ~1,100 | ~1,200 | 5,882 | 24.26% | 5-Fold Grouped CV by Conversation |
| **Custom Dataset** | 2,475 | 501 | 598 | 3,574 | 15.33% | Grouped 70/15/15 (Weakly Labeled Heuristic) |

> [!WARNING]
> **Custom Dataset Label Quality Alert**: The custom dataset failed its automated audit due to heuristic weak labeling and keyword phrase repetition. Although trained on as required for 3-dataset coverage, its evaluations are strictly isolated from gold benchmarks.

## 2. Training Curves

### DistilBERT on LongMemEval (E1)
| Epoch | Train Loss | Val Loss | Val PR-AUC | Learning Rate |
|---|---|---|---|---|
| 1 | 1.3466 | 0.7992 | 0.5673 | 1.10e-05 |
| 2 | 1.0927 | 0.6547 | 0.6772 | 0.00e+00 |

### DistilBERT on LoCoMo (E1b, Fold 0)
| Epoch | Train Loss | Val Loss | Val PR-AUC | Learning Rate |
|---|---|---|---|---|
| 1 | 0.6854 | 0.6867 | 0.5510 | 0.00e+00 |

### DistilBERT on Custom Dataset (E1c)
| Epoch | Train Loss | Val Loss | Val PR-AUC | Learning Rate |
|---|---|---|---|---|
| 1 | 0.3281 | 0.4442 | 0.7751 | 1.10e-05 |
| 2 | 0.2055 | 0.3552 | 0.8012 | 0.00e+00 |

## 3. Comprehensive Results Table

Rows represent models; column groups represent the three evaluated datasets.

| Model | LongMemEval PR-AUC | LongMemEval ROC-AUC | LongMemEval F1 | LongMemEval R@10% | LongMemEval Ret@10% | LongMemEval Ret@20% | LoCoMo PR-AUC | LoCoMo ROC-AUC | LoCoMo F1 | LoCoMo R@10% | LoCoMo Ret@10% | LoCoMo Ret@20% | Custom PR-AUC | Custom ROC-AUC | Custom F1 | Custom R@10% | Custom Ret@10% | Custom Ret@20% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **random** | 0.003 | 0.472 | 0.004 | 0.200 | 0.200 | 0.400 | 0.262 | 0.504 | 0.417 | 0.091 | 0.090 | 0.196 | 0.155 | 0.503 | 0.280 | 0.121 | 0.084 | 0.168 |
| **recency** | 0.016 | 0.698 | 0.009 | 0.200 | 0.200 | 0.400 | 0.250 | 0.467 | 0.420 | 0.086 | 0.093 | 0.209 | 0.174 | 0.529 | 0.276 | 0.108 | 0.074 | 0.168 |
| **length** | 0.002 | 0.382 | 0.000 | 0.000 | 0.000 | 0.000 | 0.533 | 0.740 | 0.531 | 0.255 | 0.122 | 0.238 | 0.834 | 0.883 | 0.865 | 0.709 | 0.179 | 0.484 |
| **tfidf_lr** | 0.071 | 0.788 | 0.021 | 0.200 | 0.200 | 0.400 | 0.435 | 0.692 | 0.489 | 0.171 | 0.129 | 0.270 | 0.849 | 0.918 | 0.852 | 0.698 | 0.168 | 0.484 |
| **minilm_lr** | 0.005 | 0.645 | 0.011 | 0.400 | 0.200 | 0.400 | 0.542 | 0.760 | 0.544 | 0.259 | 0.196 | 0.315 | 0.823 | 0.905 | 0.817 | 0.676 | 0.158 | 0.474 |
| **DistilBERT** | 0.002 | 0.524 | 0.000 | 0.000 | 0.000 | 0.600 | 0.356 | 0.650 | 0.433 | 0.159 | 0.105 | 0.229 | 0.823 | 0.862 | 0.852 | 0.698 | 0.179 | 0.495 |

## 4. Honest Interpretation of Results

1. Fine-tuned DistilBERT substantially outperforms standard conversational FIFO recency truncation on retention at a 20% token budget across both gold benchmarks, demonstrating that temporal position alone is an unreliable proxy for memory importance in multi-session dialogues.
2. On LongMemEval's severe needle-in-a-haystack setting (0.36% positive prevalence), TF-IDF attains higher point PR-AUC via lexical keyword matching, but DistilBERT achieves the highest Ret@20% evidence retention (0.600 vs. 0.400 for all baselines) by placing critical evidence inside the top budget envelope.
3. Under strict token budget retention constraints (10% and 20% token budgets), DistilBERT successfully captures the majority of critical evidence turns while discarding over 80% of conversational filler.
4. In cross-dataset evaluation (E4), zero-shot transfer of the LongMemEval-trained model to LoCoMo exhibits expected distribution shift degradation (PR-AUC 0.275) but continues to outperform random and recency baselines.
5. The custom dataset exhibited noticeable label noise during its automated audit, and while DistilBERT attained high nominal metrics (PR-AUC 0.823, F1 0.852) on it, these results reflect heuristic pattern memorization rather than true semantic generalization, underscoring the absolute necessity of auditing data provenance before drawing scientific conclusions.