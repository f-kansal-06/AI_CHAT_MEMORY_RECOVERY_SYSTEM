"""
Leakage-safe splitting strategies for LongMemEval, LoCoMo, and Custom datasets.
Ensures no session content hash or conversation ID leaks between train/val/test splits.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set
import numpy as np

logger = logging.getLogger(__name__)


def compute_session_hash(session_turns: List[Dict[str, Any]]) -> str:
    """Compute SHA-256 hash of normalized text for a session to prevent leakage."""
    normalized_texts = [
        " ".join(t.get("text", "").strip().lower().split())
        for t in session_turns
    ]
    payload = "|".join(normalized_texts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def split_longmemeval(
    turns: List[Dict[str, Any]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Split LongMemEval grouped by conversation/question with strict session hash de-duplication.
    Filler sessions are shared across haystacks, so compute a content hash per session
    and guarantee no session hash appears in more than one split (drop offending shared sessions).
    """
    rng = np.random.RandomState(seed)

    # Group turns by conversation and by session
    conv_to_turns: Dict[str, List[Dict[str, Any]]] = {}
    conv_to_sessions: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    for t in turns:
        cid = str(t["conv_id"])
        sid = str(t["session_id"])
        if cid not in conv_to_turns:
            conv_to_turns[cid] = []
            conv_to_sessions[cid] = {}
        conv_to_turns[cid].append(t)
        if sid not in conv_to_sessions[cid]:
            conv_to_sessions[cid][sid] = []
        conv_to_sessions[cid][sid].append(t)

    # Compute hashes for all sessions
    session_hashes: Dict[Tuple[str, str], str] = {}
    for cid, sessions in conv_to_sessions.items():
        for sid, s_turns in sessions.items():
            shash = compute_session_hash(s_turns)
            session_hashes[(cid, sid)] = shash

    # Shuffle conversations
    all_cids = sorted(list(conv_to_turns.keys()))
    rng.shuffle(all_cids)

    n_total = len(all_cids)
    n_train = max(1, int(n_total * train_ratio))
    n_val = max(1, int(n_total * val_ratio))

    train_cids = set(all_cids[:n_train])
    val_cids = set(all_cids[n_train : n_train + n_val])
    test_cids = set(all_cids[n_train + n_val :])

    # Collect train session hashes
    train_hashes: Set[str] = set()
    for cid in train_cids:
        for sid in conv_to_sessions[cid]:
            train_hashes.add(session_hashes[(cid, sid)])

    # For val conversations: keep sessions whose hash is NOT in train
    val_hashes: Set[str] = set()
    val_turns: List[Dict[str, Any]] = []
    dropped_val_sessions = 0

    for cid in val_cids:
        for sid, s_turns in conv_to_sessions[cid].items():
            shash = session_hashes[(cid, sid)]
            if shash in train_hashes:
                # Offending shared filler session: drop it from val
                dropped_val_sessions += 1
            else:
                val_hashes.add(shash)
                val_turns.extend(s_turns)

    # For test conversations: keep sessions whose hash is NOT in train and NOT in val
    test_turns: List[Dict[str, Any]] = []
    dropped_test_sessions = 0

    for cid in test_cids:
        for sid, s_turns in conv_to_sessions[cid].items():
            shash = session_hashes[(cid, sid)]
            if shash in train_hashes or shash in val_hashes:
                dropped_test_sessions += 1
            else:
                test_turns.extend(s_turns)

    train_turns = [t for t in turns if str(t["conv_id"]) in train_cids]

    stats = {
        "initial_convs": n_total,
        "train_convs": len(train_cids),
        "val_convs": len(val_cids),
        "test_convs": len(test_cids),
        "dropped_val_sessions_due_to_leakage": dropped_val_sessions,
        "dropped_test_sessions_due_to_leakage": dropped_test_sessions,
        "train_turns": len(train_turns),
        "val_turns": len(val_turns),
        "test_turns": len(test_turns),
        "train_pos": sum(1 for t in train_turns if t.get("label") == 1),
        "val_pos": sum(1 for t in val_turns if t.get("label") == 1),
        "test_pos": sum(1 for t in test_turns if t.get("label") == 1),
    }

    return train_turns, val_turns, test_turns, stats


def split_locomo_5fold(
    turns: List[Dict[str, Any]],
    seed: int = 42,
) -> List[Dict[str, Any]]:
    rng = np.random.RandomState(seed)
    conv_ids = sorted(list(set(str(t["conv_id"]) for t in turns)))
    rng.shuffle(conv_ids)

    n_folds = 5
    folds_convs: List[List[str]] = [[] for _ in range(n_folds)]
    for i, cid in enumerate(conv_ids):
        folds_convs[i % n_folds].append(cid)

    fold_assignments = []
    for test_idx in range(n_folds):
        val_idx = (test_idx + 1) % n_folds
        train_indices = [i for i in range(n_folds) if i != test_idx and i != val_idx]

        test_cids = folds_convs[test_idx]
        val_cids = folds_convs[val_idx]
        train_cids = [cid for i in train_indices for cid in folds_convs[i]]

        fold_assignments.append({
            "fold": test_idx,
            "train_conv_ids": train_cids,
            "val_conv_ids": val_cids,
            "test_conv_ids": test_cids,
        })

    return fold_assignments


def split_custom_dataset(
    turns: List[Dict[str, Any]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    conv_ids = sorted(list(set(str(t["conv_id"]) for t in turns)))
    if len(conv_ids) > 1:
        rng = np.random.RandomState(seed)
        rng.shuffle(conv_ids)
        n_total = len(conv_ids)
        n_train = max(1, int(n_total * train_ratio))
        n_val = max(1, int(n_total * val_ratio))
        train_cids = set(conv_ids[:n_train])
        val_cids = set(conv_ids[n_train : n_train + n_val])
        test_cids = set(conv_ids[n_train + n_val :])

        train_turns = [t for t in turns if str(t["conv_id"]) in train_cids]
        val_turns = [t for t in turns if str(t["conv_id"]) in val_cids]
        test_turns = [t for t in turns if str(t["conv_id"]) in test_cids]
        return train_turns, val_turns, test_turns
    else:
        n = len(turns)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_turns = turns[:n_train]
        val_turns = turns[n_train : n_train + n_val]
        test_turns = turns[n_train + n_val :]
        return train_turns, val_turns, test_turns
