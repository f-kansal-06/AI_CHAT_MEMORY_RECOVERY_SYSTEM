# Data Audit Report: LOCOMO

**Audit Timestamp**: 2026-10-28
**Source**: `locomo`
**Turn Count**: 5,882
**Conversation Count**: 10
**Question Count**: 1,986

## 1. Summary Statistics

| Metric | Value |
|---|---|
| Total Turns | 5,882 |
| Total Conversations | 10 |
| Positive Evidence Turns | 1,427 (24.26%) |
| Negative (Non-Evidence) Turns | 4,455 (75.74%) |
| Mean Turns / Conversation | 588.2 ± 112.0 |
| Median Turns / Conversation | 646.0 |
| Token Length (Mean ± Std) | 30.5 ± 17.1 tokens |
| Token Length Median | 27.0 tokens |
| Token Length 95th Percentile | 64.0 tokens |
| Max Turn Length | 113 tokens |
| Internal Duplicate Rate | 0.17% |

## 2. Speaker & Category Distribution

### Speaker Distribution
- `user`: 5,882 turns (100.0%)

### Question Category Distribution (Positive Turns)
- `multi_hop`: 246 (17.2%)
- `single_fact`: 605 (42.4%)
- `temporal`: 137 (9.6%)
- `other`: 437 (30.6%)
- `preference`: 2 (0.1%)

## 3. Cross-Dataset Leakage & Overlap Audit
- **Overlap with longmemeval**: 2 turns (0.03%)