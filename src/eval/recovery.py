"""
Simulated Failure Recovery Evaluation (Experiment E7).
Offline simulation measuring evidence retention and question recovery coverage
when a conversation is cut off at 25%, 50%, and 75% of its length.
"""

from typing import List, Dict, Any, Tuple, Optional, Callable
import numpy as np
from src.data.schema import QuestionRecord


def run_recovery_simulation(
    turns: List[Dict[str, Any]],
    questions: List[QuestionRecord],
    score_fn: Callable[[List[Dict[str, Any]]], np.ndarray],
    cut_fractions: Tuple[float, ...] = (0.25, 0.50, 0.75),
    budget_fractions: Tuple[float, ...] = (0.05, 0.10, 0.20),
) -> Dict[str, Any]:
    """
    Run simulated failure recovery for a given scoring method.
    Strict constraints:
    1. Cut points at 25%, 50%, 75% of turns.
    2. Only turns before the cut point are scored or restored.
    3. Only questions whose evidence turns ALL lie before the cut point are counted.
    4. Budget constraint: restored memory never exceeds budget_limit = pre_cut_tokens * budget_fraction.
    """
    # Group turns by conversation
    conv_turns_map: Dict[str, List[Dict[str, Any]]] = {}
    for t in turns:
        cid = t.get("conv_id", "default")
        if cid not in conv_turns_map:
            conv_turns_map[cid] = []
        conv_turns_map[cid].append(t)

    # Sort turns by global_turn_idx
    for cid in conv_turns_map:
        conv_turns_map[cid] = sorted(
            conv_turns_map[cid],
            key=lambda x: x.get("global_turn_idx", x.get("turn_idx", 0)),
        )

    # Group questions by conversation
    conv_questions_map: Dict[str, List[QuestionRecord]] = {}
    for q in questions:
        cid = q.conv_id
        if cid not in conv_questions_map:
            conv_questions_map[cid] = []
        conv_questions_map[cid].append(q)

    results: Dict[str, Any] = {}

    for cut_frac in cut_fractions:
        cut_key = f"cut_{int(cut_frac * 100)}%"
        results[cut_key] = {}

        # Prepare and score pre-cut turns for all conversations once per cut fraction
        conv_pre_cut_info = {}
        for cid, c_turns in conv_turns_map.items():
            n_c = len(c_turns)
            if n_c == 0:
                continue
            cut_idx = max(1, int(np.floor(n_c * cut_frac)))
            pre_cut_turns = c_turns[:cut_idx]
            conv_pre_cut_tokens = sum(t.get("n_tokens", 1) for t in pre_cut_turns)
            pre_cut_scores = score_fn(pre_cut_turns)
            if len(pre_cut_scores) != len(pre_cut_turns):
                raise ValueError(f"Score count {len(pre_cut_scores)} does not match pre-cut turns {len(pre_cut_turns)}")
            ranked_pairs = sorted(
                zip(pre_cut_turns, pre_cut_scores),
                key=lambda x: x[1],
                reverse=True,
            )
            conv_pre_cut_info[cid] = (pre_cut_turns, ranked_pairs, conv_pre_cut_tokens)

        for b_frac in budget_fractions:
            budget_key = f"budget_{int(b_frac * 100)}%"

            total_eligible_questions = 0
            recovered_questions = 0

            total_pre_cut_evidence = 0
            retained_pre_cut_evidence = 0

            total_pre_cut_tokens = 0
            total_restored_tokens = 0

            for cid, (pre_cut_turns, ranked_pairs, conv_pre_cut_tokens) in conv_pre_cut_info.items():
                total_pre_cut_tokens += conv_pre_cut_tokens

                budget_limit = int(conv_pre_cut_tokens * b_frac)
                current_budget = 0
                restored_turn_ids = set()

                for t, s in ranked_pairs:
                    tok = t.get("n_tokens", 1)
                    if current_budget + tok <= budget_limit:
                        current_budget += tok
                        t_id = f"{t.get('conv_id')}_{t.get('session_id')}_{t.get('turn_idx')}"
                        restored_turn_ids.add(t_id)
                        restored_turn_ids.add(f"{cid}_{t.get('global_turn_idx')}")

                total_restored_tokens += current_budget

                # Pre-cut evidence turns
                for t in pre_cut_turns:
                    if t.get("label", 0) == 1:
                        total_pre_cut_evidence += 1
                        t_id = f"{t.get('conv_id')}_{t.get('session_id')}_{t.get('turn_idx')}"
                        t_gid = f"{cid}_{t.get('global_turn_idx')}"
                        if t_id in restored_turn_ids or t_gid in restored_turn_ids:
                            retained_pre_cut_evidence += 1

                # Check eligible questions: ALL evidence turns must lie strictly before cut_idx
                pre_cut_turn_ids_all = set(
                    f"{t.get('conv_id')}_{t.get('session_id')}_{t.get('turn_idx')}"
                    for t in pre_cut_turns
                )
                pre_cut_gids_all = set(
                    f"{cid}_{t.get('global_turn_idx')}" for t in pre_cut_turns
                )

                for q in conv_questions_map.get(cid, []):
                    if not q.evidence_turn_ids:
                        continue

                    # Verify that ALL evidence turns are before the cut
                    all_evidence_before_cut = True
                    for ev_id in q.evidence_turn_ids:
                        in_pre_cut = (ev_id in pre_cut_turn_ids_all) or (ev_id in pre_cut_gids_all) or any(ev_id in x or x in ev_id for x in pre_cut_turn_ids_all)
                        if not in_pre_cut:
                            all_evidence_before_cut = False
                            break

                    if not all_evidence_before_cut:
                        continue  # Question not eligible at this cut point

                    total_eligible_questions += 1

                    # Check if all evidence is inside restored memory
                    all_evidence_restored = True
                    for ev_id in q.evidence_turn_ids:
                        in_restored = (ev_id in restored_turn_ids) or any(ev_id in x or x in ev_id for x in restored_turn_ids)
                        if not in_restored:
                            all_evidence_restored = False
                            break

                    if all_evidence_restored:
                        recovered_questions += 1

            recovery_coverage = (recovered_questions / total_eligible_questions) if total_eligible_questions > 0 else 0.0
            evidence_recall = (retained_pre_cut_evidence / total_pre_cut_evidence) if total_pre_cut_evidence > 0 else 0.0
            compression_ratio = (total_restored_tokens / max(total_pre_cut_tokens, 1))

            results[cut_key][budget_key] = {
                "recovery_coverage": float(recovery_coverage),
                "evidence_recall": float(evidence_recall),
                "compression_ratio": float(compression_ratio),
                "eligible_questions": int(total_eligible_questions),
                "recovered_questions": int(recovered_questions),
                "restored_tokens": int(total_restored_tokens),
                "pre_cut_tokens": int(total_pre_cut_tokens),
            }

    return results

