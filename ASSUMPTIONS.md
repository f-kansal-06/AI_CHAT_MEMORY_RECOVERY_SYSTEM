# Project Assumptions & Design Decisions

This document records key engineering and research assumptions made across the data, model, training, and evaluation pipelines in accordance with Rule 3 ("Make reasonable assumptions instead of stopping to ask; record each one in ASSUMPTIONS.md").

## 1. Task Definition & Label Semantics
- **Negative Label Definition**: A label of `0` denotes that the turn was not annotated as necessary evidence for any known downstream question in the benchmark. As noted in the prompt, this does not imply the turn is useless conversational fluff; it only reflects task-specific non-evidence status.
- **Evidence Label Union**: If multiple questions reference the same conversation turn, the turn's binary label is positive (`1`), and its linked question IDs are aggregated into `linked_question_ids`. Its category is mapped via priority or set to the dominant category.
- **Taxonomy Normalization**: All dataset question categories are mapped into seven standard coarse classes: `single_fact`, `multi_hop`, `temporal`, `knowledge_update`, `preference`, `other`, `none` (for unlinked/negative turns).

## 2. Token Budgeting & Retention
- **Strict Budget Ceiling**: In the budgeted retention routine, turns are selected greedily in descending order of predicted importance score. A candidate turn is added if and only if `current_tokens + turn.n_tokens <= budget_limit`. If a turn would exceed the remaining token budget, it is skipped (and smaller subsequent turns may still fit if within the remaining allowance, or strict truncation stops at the boundary). The budget is never exceeded.
- **True vs. Truncated Tokens**: Turns may be truncated to `max_length = 256` for DistilBERT encoder input, but their true pre-truncation `n_tokens` (measured by whitespace/standard BPE count) is preserved and used for all budget retention calculations.
- **All-Evidence-Retained Rate**: A question is considered successfully preserved if and only if **100%** of its annotated evidence turns are present in the retained memory subset.

## 3. LoCoMo Evidence Mapping
- LoCoMo dialogue items list evidence in formats like `"D1:3"` or `"session_1:3"`. We assume `D<session_idx>:<turn_idx>` where session index identifies the sub-dialogue within the conversation and turn index is 0- or 1-indexed relative to the start of that sub-dialogue. Our parser handles both 0-indexed and 1-indexed conventions gracefully.
- Public LoCoMo contains ~10 long multi-session conversations. Because of the small sample size, 5-fold cross-validation is strictly grouped by `conv_id` (folds contain disjoint conversations) to prevent intra-conversation leakage.

## 4. Leakage-Safe Splitting
- In LongMemEval, filler sessions are frequently reused across multiple question haystacks. To prevent train-to-test contamination, we compute a SHA-256 hash of normalized text for every session. Any session hash appearing in train cannot appear in val or test. If a conversation shares sessions across split boundaries, it is either reassigned to the same split or offending duplicate sessions are pruned, logging counts before and after.

## 5. Offline Fallbacks & Fixtures
- In restricted execution environments or when external API rate-limits prevent downloading raw multi-gigabyte corpora, synthetic fixtures conforming exactly to the official schema are generated in `tests/fixtures/` and `data/raw/` so the full pipeline, audits, baseline training, DistilBERT fine-tuning, and evaluation metrics run reproducibly end-to-end.
- For WildChat-1M (unlabeled analysis), if Hugging Face streaming is inaccessible due to firewall/offline constraints, an offline sample of multi-turn conversational sessions is analyzed for score distribution and review turn exports.

## 6. Simulated Recovery (E7)
- Cut points are placed deterministically at 25%, 50%, and 75% of total conversation turns.
- Only turns prior to the cut point are visible to scoring and retention.
- Only questions whose evidence turns strictly occurred *before* the cut point are evaluated for recovery coverage.

