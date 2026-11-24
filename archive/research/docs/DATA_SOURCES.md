# Data Sources Documentation

This document records the exact sources, repositories, commits/versions, schemas, and acquisition notes for all datasets used in this benchmark.

## 1. LongMemEval (`_S` variant)
- **Repository**: [xiaowu0162/LongMemEval](https://github.com/xiaowu0162/LongMemEval)
- **Paper**: *LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory* (Wu et al., 2024)
- **Variant**: `LongMemEval_S` (single-hop and multi-hop questions with haystacks of synthetic/distractor and evidence sessions).
- **Format**: JSON containing an array of instances. Each instance represents a question haystack:
  - `question_id`: Unique identifier for the question.
  - `question`: Downstream question text.
  - `question_type` / `category`: Task category (e.g., single_session_fact, multi_session_fact, temporal_reasoning, preference, update).
  - `sessions`: List of sessions in the conversation haystack.
    - Each session contains `session_id`, `date`/`timestamp`, and `turns`.
    - Turns inside evidence sessions carry answer-evidence flags: field `has_answer` (`true`/`false` or `1`/`0`).
- **Acquisition**: Loaded from raw JSON or cloned repository. If remote network is restricted, a faithful synthetic fixture generated matching the exact official schema in `tests/fixtures/longmemeval_fixture.json` and `data/raw/longmemeval_s.json` is used for reproducible offline testing.

## 2. LoCoMo
- **Repository**: [snap-research/locomo](https://github.com/snap-research/locomo)
- **Paper**: *LoCoMo: Long-Context Conversational Memory Benchmark* (Snap Research, 2024)
- **Format**: Multi-session long conversations with associated QA pairs:
  - Dialogues: Structured into sessions/dialogues (e.g., D1, D2, ...).
  - QA items: Contain question, gold answer, question category (1: single-fact, 2: multi-hop, 3: temporal, 4: causal/reasoning, 5: user preference), and evidence references (e.g. `evidence: ["D1:3", "D2:5"]`).
- **Parsing**: References like `"D1:3"` are resolved to Session `D1`, turn index `3` (0- or 1-based, aligned with dialogue index).
- **Evaluation protocol**: Grouped 5-fold cross-validation by conversation (`conv_id`) to avoid data leakage across folds. Also used as the unseen external test set for LongMemEval-trained models (E4).

## 3. Custom Dataset (~4,221 turns)
- **Path**: Configurable via `custom_path` (default `data/raw/custom_4221.jsonl`).
- **Nature**: Realistic multi-turn conversation logs with weak/heuristic importance annotations simulating a messy enterprise or production conversational log.
- **Audit**: Subjected to complete statistical audit (`scripts/audit_data.py`), testing label noise, token length distributions, duplicate rates, and hash-overlap with LongMemEval and LoCoMo test sets.
- **Usage**: Used in E1c (isolated training/eval) and E3 (combined training if audit criteria allow), never placed in the primary validation/test splits of LongMemEval.

## 4. WildChat-1M
- **Hugging Face**: [`allenai/WildChat-1M`](https://huggingface.co/datasets/allenai/WildChat-1M)
- **Nature**: Real-world user interactions with ChatGPT (unlabeled for downstream memory evidence).
- **Usage**: Analysis only (E6). Streamed to extract >= 5,000 English conversations with at least 6 turns. Used to score turn distributions, simulate token reduction, and export top-100 and bottom-100 turns for manual inspection. No labeled evaluation metric is computed on it.

