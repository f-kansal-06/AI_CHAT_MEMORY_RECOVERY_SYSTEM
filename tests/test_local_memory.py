import json

from src.app.analyzer import analyze_conversation
from src.app.ingest import load_conversation, parse_conversation_text
from src.app.recovery import render_recovery
from src.app.store import MemoryStore
from src.data.schema import QuestionRecord
from src.data.identifiers import questions_grounded_in_turns
from src.eval.recovery import run_recovery_simulation
import numpy as np


def test_txt_json_jsonl_ingestion(tmp_path):
    transcript = tmp_path / "chat.txt"
    transcript.write_text("User: Save the project under C:/demo.\nAssistant: Understood.\n", encoding="utf-8")
    title, msgs = load_conversation(transcript)
    assert title == "chat"
    assert [m["role"] for m in msgs] == ["user", "assistant"]

    json_path = tmp_path / "chat.json"
    json_path.write_text(json.dumps({"title": "Named chat", "messages": [{"role": "human", "content": "Hello"}]}), encoding="utf-8")
    assert load_conversation(json_path)[0] == "Named chat"
    assert load_conversation(json_path)[1][0]["role"] == "user"

    jsonl_path = tmp_path / "chat.jsonl"
    jsonl_path.write_text('{"speaker":"assistant","text":"Hi"}\n', encoding="utf-8")
    assert load_conversation(jsonl_path)[1][0]["content"] == "Hi"

    export_path = tmp_path / "export.json"
    export_path.write_text(json.dumps({"title": "Export", "mapping": {
        "node2": {"message": {"id": "a2", "author": {"role": "assistant"}, "content": {"parts": ["Second message"]}, "create_time": 2}},
        "node1": {"message": {"id": "u1", "author": {"role": "user"}, "content": {"parts": ["First message"]}, "create_time": 1}},
    }}), encoding="utf-8")
    title, exported = load_conversation(export_path)
    assert title == "Export"
    assert [m["message_id"] for m in exported] == ["u1", "a2"]


def test_extraction_preserves_stance_sources_and_deduplicates():
    messages = [
        {"message_id": "u1", "sequence": 0, "role": "user", "content": "Please use SQLite for persistent local storage. The project requires local execution."},
        {"message_id": "a1", "sequence": 1, "role": "assistant", "content": "I suggest using cloud storage instead."},
        {"message_id": "u2", "sequence": 2, "role": "user", "content": "No, do not use cloud storage. We will use SQLite."},
        {"message_id": "u3", "sequence": 3, "role": "user", "content": "Please use SQLite for persistent local storage."},
    ]
    analysis = analyze_conversation(messages)
    assert any(e["kind"] == "requirement" and "SQLite" in e["text"] for e in analysis["entries"])
    assert any(e["kind"] == "rejected_suggestion" for e in analysis["entries"])
    assert any(e["kind"] == "requirement" and "Please use SQLite" in e["text"] for e in analysis["entries"])
    duplicate = next(e for e in analysis["entries"] if e["text"] == "Please use SQLite for persistent local storage.")
    assert duplicate["source_message_ids"] == ["u1", "u3"]
    assert analysis["stats"]["memory_tokens"] <= analysis["stats"]["original_tokens"]


def test_unresolved_and_uncertain_statements_keep_distinct_status():
    analysis = analyze_conversation([
        {"message_id": "m1", "sequence": 0, "role": "user", "content": "We still need to decide which local model will be used."},
        {"message_id": "m2", "sequence": 1, "role": "assistant", "content": "Maybe the current cache is incomplete."},
        {"message_id": "m3", "sequence": 2, "role": "user", "content": "Please preserve uncertain claims separately from confirmed decisions."},
    ])
    assert any(e["kind"] == "unresolved_question" and e["status"] == "unresolved" for e in analysis["entries"])
    assert any(e["kind"] == "uncertainty" and e["status"] == "uncertain" for e in analysis["entries"])
    assert any(e["kind"] == "requirement" and e["status"] == "stated" for e in analysis["entries"])


def test_opposed_assistant_suggestion_is_kept_as_rejected_with_link():
    analysis = analyze_conversation([
        {"message_id": "a", "sequence": 0, "role": "assistant", "content": "One option is to send every conversation to a hosted LLM for summarization."},
        {"message_id": "u", "sequence": 1, "role": "user", "content": "Do not use hosted APIs."},
    ])
    rejected = next(e for e in analysis["entries"] if e["kind"] == "rejected_suggestion")
    requirement = next(e for e in analysis["entries"] if e["kind"] == "requirement")
    assert rejected["status"] == "rejected"
    assert rejected["conflict_group"] == requirement["conflict_group"]
    assert requirement["status"] == "stated"


def test_sqlite_reimport_analyze_recover_search_delete(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    messages = parse_conversation_text("User: We decided to use SQLite for the local project.\nAssistant: I suggest cloud storage instead.\nUser: No, do not use cloud storage.")
    cid = store.import_conversation("Demo", messages)
    assert store.import_conversation("Duplicate", messages) == cid
    analysis = analyze_conversation(store.messages(cid))
    store.save_analysis(cid, analysis)
    assert store.save_analysis(cid, analysis)
    assert store.latest_run(cid)["status"] == "complete"
    assert store.search("SQLite")
    recovered = render_recovery("Demo", store.memories(cid))
    assert "Compact continuation context" in recovered
    assert "source:" in recovered
    assert store.delete_conversation(cid)
    assert store.list_conversations() == []


def test_update_replaces_source_and_invalidates_old_memory(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    cid = store.import_conversation("Old", [{"message_id": "old", "role": "user", "content": "We decided to use Python for the prototype."}])
    store.save_analysis(cid, analyze_conversation(store.messages(cid)))
    store.update_conversation(cid, "New", [{"message_id": "new", "role": "user", "content": "We decided to use Rust for this prototype."}])
    assert store.messages(cid)[0]["content"].endswith("Rust for this prototype.")
    assert store.memories(cid) == []
    assert store.latest_run(cid)["status"] == "stale"


def test_interrupted_analysis_keeps_previous_memory(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    cid = store.import_conversation("Atomic", [{"message_id": "m", "role": "user", "content": "We decided to use Python for the prototype."}])
    good = analyze_conversation(store.messages(cid))
    store.save_analysis(cid, good)
    before = store.memories(cid)
    try:
        store.save_analysis(cid, {"entries": [{"text": "incomplete"}], "stats": {}})
    except KeyError:
        pass
    assert store.memories(cid) == before
    assert store.latest_run(cid)["status"] == "failed"


def test_memory_recovery_does_not_need_transcript(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    cid = store.import_conversation("Recovery", [{"message_id": "m1", "role": "user", "content": "We decided to use Python for the prototype."}])
    store.save_analysis(cid, analyze_conversation(store.messages(cid)))
    # Deleting source messages demonstrates the renderer consumes stored memory only.
    with store._connect() as db:
        db.execute("DELETE FROM messages WHERE conversation_id=?", (cid,))
    assert "use Python for the prototype" in render_recovery("Recovery", store.memories(cid))


def test_recovery_uses_exact_evidence_ids_not_substrings():
    turns = [
        {"conv_id": "c", "session_id": "s_1", "turn_idx": 1, "global_turn_idx": 1, "label": 0, "n_tokens": 1},
        {"conv_id": "c", "session_id": "s_1", "turn_idx": 10, "global_turn_idx": 10, "label": 0, "n_tokens": 1},
    ]
    question = QuestionRecord("q", "c", "single_fact", ["c_s_1_10"])
    result = run_recovery_simulation(turns, [question], lambda sub: np.asarray([.9, .1]), cut_fractions=(1.0,), budget_fractions=(.5,))
    # The 50% budget retains ID 1. It must not falsely satisfy evidence ID 10.
    assert result["cut_100%"]["budget_50%"]["recovery_coverage"] == 0.0


def test_recovery_matches_locomo_one_based_evidence_reference():
    turns = [
        {"conv_id": "c", "session_id": "D1", "turn_idx": i, "global_turn_idx": i, "label": int(i == 2), "n_tokens": 1}
        for i in range(3)
    ]
    question = QuestionRecord("q", "c", "single_fact", ["c_D1:3"])
    result = run_recovery_simulation(turns, [question], lambda sub: np.asarray([.1, .2, .9]), cut_fractions=(1.0,), budget_fractions=(1.0,))
    assert result["cut_100%"]["budget_100%"]["recovered_questions"] == 1


def test_questions_with_pruned_evidence_are_excluded_from_split_metrics():
    turns = [{"conv_id": "c", "session_id": "s1", "turn_idx": 0, "global_turn_idx": 0}]
    grounded = QuestionRecord("q1", "c", "single_fact", ["c_s1_0"])
    pruned = QuestionRecord("q2", "c", "multi_hop", ["c_s1_0", "c_s2_0"])
    assert questions_grounded_in_turns([grounded, pruned], turns) == [grounded]
