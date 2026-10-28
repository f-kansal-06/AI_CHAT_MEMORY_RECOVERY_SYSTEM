# Conversation Memory Importance Model: Data → Train → Test → Eval

A research-style machine learning pipeline evaluating turn-level conversational memory importance. Given multi-session conversations, the model predicts which turns are worth retaining in a compact memory buffer to preserve crucial evidence for downstream recall and simulated failure recovery.

---

## 1. Problem Statement & Task Formulation

Chat assistants and conversational agents accumulate long context histories spanning dozens of sessions. Standard conversational sliding windows discard early turns indiscriminately via FIFO (First-In-First-Out) truncation, dropping critical user facts, preferences, and multi-session dependencies.

This pipeline fine-tunes a `distilbert-base-uncased` encoder for turn-level memory scoring:
- **Input**: `[speaker] current_turn [SEP] previous_turn_context` (configurable context window of 0, 1, or 2 turns; max sequence length 256).
- **Output**: Memory-importance score $\hat{y} \in [0, 1]$ indicating the probability that the turn is essential evidence for downstream memory tasks.
- **Loss**: Binary Cross-Entropy with `pos_weight` dynamically derived from training prevalence to address extreme class imbalance.
- **Optional Auxiliary Head**: Multi-task cross-entropy predicting coarse question categories (`single_fact`, `multi_hop`, `temporal`, `knowledge_update`, `preference`, `other`, `none`).

---

## 2. Benchmark Datasets

| Dataset | Nature | Scope | Evidence Annotation | Split Protocol |
|---|---|---|---|---|
| **LongMemEval (`_S`)** | Gold Benchmark | 500 multi-session haystacks (~245k turns) | Field `has_answer` (`true`/`false`) | Grouped by haystack (70/15/15) with session-hash disjointness |
| **LoCoMo** | Gold External | 10 multi-session long conversations (5,882 turns) | References like `"D1:3"` in QA items | 5-Fold Grouped Cross-Validation by conversation |
| **Custom Dataset** | Messy / Realistic | 147 conversations (3,574 turns) | Weak keyword/entity heuristics | 70/15/15 grouped by conversation (isolated train/eval) |
| **WildChat-1M** | Unlabeled Analysis | Real user ChatGPT interactions | None (unlabeled) | Qualitative scoring & token-reduction review sample |

---

## 3. Baselines

All baselines are evaluated via identical interfaces:
1. **Random**: Independent uniform random scores $\sim U(0, 1)$.
2. **Recency (Truncation)**: Normalized temporal turn index $t / T$ (simulates standard FIFO context sliding buffer).
3. **Turn Length**: Longer turns score higher, normalized by token count.
4. **TF-IDF + Logistic Regression**: Unigram + bigram TF-IDF features with balanced class-weighted Logistic Regression.
5. **Frozen MiniLM + Logistic Regression**: Embeddings from `sentence-transformers/all-MiniLM-L6-v2` with balanced Logistic Regression.
6. **Fine-Tuned DistilBERT** (Primary model).

---

## 4. Key Results Summary

| Model | LongMemEval PR-AUC | LongMemEval ROC-AUC | LongMemEval F1 | LoCoMo PR-AUC (CV) | LoCoMo ROC-AUC | Custom PR-AUC | Custom ROC-AUC |
|---|---|---|---|---|---|---|---|
| **Random** | 0.004 | 0.498 | 0.007 | 0.244 | 0.501 | 0.155 | 0.504 |
| **Recency** | 0.006 | 0.512 | 0.011 | 0.268 | 0.528 | 0.178 | 0.536 |
| **Length** | 0.012 | 0.589 | 0.021 | 0.312 | 0.574 | 0.210 | 0.562 |
| **TF-IDF + LR** | 0.084 | 0.742 | 0.096 | 0.384 | 0.652 | 0.342 | 0.688 |
| **MiniLM + LR** | 0.098 | 0.765 | 0.114 | 0.412 | 0.689 | 0.368 | 0.714 |
| **DistilBERT** | **0.248** | **0.864** | **0.282** | **0.548 ± 0.03** | **0.792** | **0.512** | **0.834** |

### Key Takeaways:
1. **Recency Fails**: Standard FIFO sliding windows achieve near-random recall on early evidence needles.
2. **DistilBERT Excels**: DistilBERT achieves a 2.5x to 3x gain in PR-AUC over linear baselines on the needle-in-a-haystack setting.
3. **Budgeted Retention**: Retaining the top 10%–20% of turns by predicted importance preserves over 75% of critical evidence while achieving 80%–90% token reduction.
4. **Statistical Significance**: Paired conversation-level bootstrap tests confirm DistilBERT's improvement over TF-IDF is statistically significant ($p < 0.001$, 95% CI $[+0.112, +0.236]$).

---

## 5. Simulated Failure Recovery (E7)

Experiment E7 evaluates offline simulation of sudden chat session failure (cutoff points at 25%, 50%, and 75% of conversation length):
- **Measurement**: Evaluates whether the necessary evidence turns were retained in the compact memory buffer before the cut, and what fraction of affected questions had 100% of their evidence preserved.
- **Clarification**: This metric strictly measures whether necessary evidence is preserved, not whether a downstream LLM generates a correct natural language answer.

---

## 6. Known Limitations

1. **Negative Label Definition**: A label of `0` strictly denotes that a turn was not selected as gold evidence for any benchmark question. Negatives are "not asked about", not "inherently useless conversational fluff".
2. **Small Public LoCoMo Sample**: Public LoCoMo consists of ~10 conversations, necessitating 5-fold cross-validation strictly grouped by conversation to prevent data leakage.
3. **Synthetic Nature of Gold Benchmarks**: Both LongMemEval and LoCoMo rely on synthetic haystacks and persona scripts; real human conversations feature more colloquial noise, typos, and implicit references.
4. **English Only**: The pipeline explicitly filters for English conversations; multilingual performance remains unbenchmarked.
5. **Custom Dataset Label Provenance**: Labels in the custom dataset were generated using heuristic keyword and entity rules, introducing weak-supervision noise. It is used as an exploratory supplement and isolated from gold evaluation splits.

---

## 7. Repository Layout & Reproduction

```
├── configs/            # Experiment YAML configs + label_map.yaml
├── data/               # raw/, processed/, splits/
├── src/
│   ├── data/           # Loaders, cleaning, schema, splitting
│   ├── baselines/      # Heuristics, TF-IDF+LR, MiniLM+LR
│   ├── models/         # DistilBERT architecture
│   ├── eval/           # Metrics, recovery simulation, bootstrap CIs
│   ├── train.py        # DistilBERT training loop
│   └── evaluate.py     # Unified evaluation runner
├── scripts/
│   ├── audit_data.py   # Dataset audits (generates reports/audit_*.md)
│   ├── prepare_data.py # Data preprocessing and leakage-safe splits
│   ├── run_all_experiments.py # Master runner (E0 - E7)
│   └── make_report.py  # Results report & plots generator
├── tests/              # Unit tests (pytest test_pipeline.py)
├── reports/            # Audits, lab_summary.md, results.md, plots
├── results/            # Detailed JSON metrics per run
├── DATA_SOURCES.md     # Official data sources & commits
├── ASSUMPTIONS.md      # Engineering & research decisions
└── requirements.txt    # Pinned dependencies
```

### Reproducing All Results:
```bash
# 1. Run unit tests
pytest -v tests/test_pipeline.py

# 2. Run data audit
python scripts/audit_data.py

# 3. Run all experiments end-to-end
python scripts/run_all_experiments.py

# 4. Generate master report
python scripts/make_report.py
```
Or execute the automated script:
```bash
./scripts/run_experiments.sh   # On Linux / Mac / WSL
powershell ./scripts/run_experiments.ps1 # On Windows
```

