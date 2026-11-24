# Data Audit Report: LONGMEMEVAL

**Audit Timestamp**: 2026-10-28
**Source**: `longmemeval`
**Turn Count**: 245,925
**Conversation Count**: 500
**Question Count**: 500

## 1. Summary Statistics

| Metric | Value |
|---|---|
| Total Turns | 245,925 |
| Total Conversations | 500 |
| Positive Evidence Turns | 896 (0.36%) |
| Negative (Non-Evidence) Turns | 245,029 (99.64%) |
| Mean Turns / Conversation | 491.9 ± 30.3 |
| Median Turns / Conversation | 489.5 |
| Token Length (Mean ± Std) | 248.0 ± 264.2 tokens |
| Token Length Median | 108.0 tokens |
| Token Length 95th Percentile | 721.0 tokens |
| Max Turn Length | 19,140 tokens |
| Internal Duplicate Rate | 22.96% |

## 2. Speaker & Category Distribution

### Speaker Distribution
- `user`: 121,854 turns (49.5%)
- `assistant`: 124,071 turns (50.5%)

### Question Category Distribution (Positive Turns)
- `other`: 449 (50.1%)
- `preference`: 44 (4.9%)
- `multi_hop`: 259 (28.9%)
- `knowledge_update`: 144 (16.1%)

## 3. Cross-Dataset Leakage & Overlap Audit
- No cross-dataset reference hashes checked.