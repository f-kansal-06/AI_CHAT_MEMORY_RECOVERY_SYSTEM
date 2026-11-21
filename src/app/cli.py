"""Interactive terminal interface for the local conversation memory system."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.app.analyzer import analyze_conversation
from src.app.ingest import load_conversation
from src.app.recovery import render_recovery
from src.app.store import MemoryStore
from src.app.scorer import load_local_scorer
from src.data.cleaning import count_tokens


def _choose(rows: list[dict], prompt: str = "Conversation ID") -> str | None:
    if not rows:
        print("No conversations stored.")
        return None
    for i, row in enumerate(rows, 1):
        print(f"{i}. {row['title']} | {row['conversation_id']} | {row['message_count']} messages | {row['memory_count']} memory items")
    value = input(f"{prompt} or list number (blank cancels): ").strip()
    if not value:
        return None
    if value.isdigit() and 1 <= int(value) <= len(rows):
        return rows[int(value) - 1]["conversation_id"]
    return value


def import_file(store: MemoryStore) -> str | None:
    path = input("Conversation file path (TXT/JSON/JSONL): ").strip().strip('"')
    if path.lower() == "manual":
        return manual_input(store)
    if not path:
        return None
    title, messages = load_conversation(path)
    existing = store.list_conversations()
    if existing:
        update_id = input("Existing conversation ID to replace (blank imports/deduplicates): ").strip()
        if update_id:
            store.update_conversation(update_id, title, messages, source_path=str(Path(path).resolve()))
            print(f"Updated conversation {update_id}; analyze it again to refresh its memory.")
            return update_id
    cid = store.import_conversation(title, messages, source_path=str(Path(path).resolve()))
    print(f"Imported {len(messages)} messages. Conversation ID: {cid}")
    return cid


def manual_input(store: MemoryStore) -> str | None:
    print("Enter messages as User: ... or Assistant: ... . Enter /done on its own line to finish.")
    transcript = []
    while True:
        line = input()
        if line.strip() == "/done":
            break
        transcript.append(line)
    title = input("Conversation title: ").strip() or "Manual conversation"
    from src.app.ingest import parse_conversation_text
    messages = parse_conversation_text("\n".join(transcript))
    if not messages:
        print("No messages entered.")
        return None
    cid = store.import_conversation(title, messages)
    print(f"Imported {len(messages)} messages. Conversation ID: {cid}")
    return cid


def analyze_one(store: MemoryStore, cid: str | None = None) -> None:
    if not cid:
        cid = _choose(store.list_conversations())
    if not cid:
        return
    messages = store.messages(cid)
    if not messages:
        print("Conversation not found or contains no messages.")
        return
    print(f"Analyzing {len(messages)} messages locally...")
    scorer = load_local_scorer()
    scores = scorer.score(messages) if scorer else None
    analysis = analyze_conversation(messages, scores, scorer.name if scorer else None)
    store.save_analysis(cid, analysis)
    stats = analysis["stats"]
    print(analysis["overview"])
    print(f"Importance scorer: {analysis['extractor']}")
    print(f"Tokens: {stats['original_tokens']} -> {stats['memory_tokens']} | compression: {stats['compression_ratio']:.1%}")
    print(f"Stored {stats['entry_count']} memory items.")


def show_conversations(store: MemoryStore) -> None:
    rows = store.list_conversations()
    if not rows:
        print("No conversations stored.")
    for row in rows:
        print(f"{row['conversation_id']}  {row['title']} | {row['message_count']} messages | {row['memory_count']} memories")
    if rows:
        cid = _choose(rows)
        if cid:
            selected = next((item for item in rows if item["conversation_id"] == cid), None)
            print(f"\n{selected['title'] if selected else cid}")
            for memory in store.memories(cid):
                print(f"- [{memory['kind']} / {memory['status']}] {memory['text']} (sources: {', '.join(memory['source_message_ids'])})")


def recover(store: MemoryStore) -> None:
    rows = store.list_conversations()
    cid = _choose(rows)
    if not cid:
        return
    row = next((r for r in rows if r["conversation_id"] == cid), None)
    if not row:
        print("Conversation not found.")
        return
    memories = store.memories(cid)
    if not memories:
        print("No stored memory yet. Analyze this conversation first.")
        return
    output = render_recovery(row["title"], memories)
    print(output)
    dest = input("Save recovery context to file? Enter path or leave blank: ").strip().strip('"')
    if dest:
        Path(dest).write_text(output, encoding="utf-8")
        print(f"Recovery written to {dest}")


def compare_size(store: MemoryStore) -> None:
    rows = store.list_conversations()
    cid = _choose(rows)
    if not cid:
        return
    row = next((r for r in rows if r["conversation_id"] == cid), None)
    messages, memories = store.messages(cid), store.memories(cid)
    original = sum(count_tokens(m["content"]) for m in messages)
    compact = sum(count_tokens(m["text"]) for m in memories)
    print(f"{row['title'] if row else cid}: {original} transcript tokens -> {compact} memory tokens ({1 - compact / max(original, 1):.1%} reduction)")


def evaluate_model() -> None:
    result_path = Path("results/tuning/tuning_results.json")
    if result_path.exists():
        print("Latest local tuned-model report:")
        print(json.dumps(json.loads(result_path.read_text(encoding="utf-8")), indent=2)[:6000])
        return
    result_path = Path("results/e0_baselines/e0_results.json")
    if not result_path.exists():
        print("No saved evaluation results found. Run the research evaluation pipeline first.")
        return
    results = json.loads(result_path.read_text(encoding="utf-8"))
    print("Saved benchmark evaluation (historical artifact; not a live evaluation):")
    print(json.dumps(results, indent=2)[:6000])


def main() -> None:
    parser = argparse.ArgumentParser(description="Local conversation memory and recovery")
    parser.add_argument("--db", default="data/chat_memory.sqlite3", help="SQLite database file")
    parser.add_argument("--import-file", help="Import a TXT, JSON, or JSONL conversation, then exit")
    args = parser.parse_args()
    store = MemoryStore(args.db)
    if args.import_file:
        cid = store.import_conversation(*load_conversation(args.import_file), source_path=str(Path(args.import_file).resolve()))
        print(f"Imported. Conversation ID: {cid}")
        return

    actions = {
        "1": ("Import conversation file", lambda: import_file(store)),
        "2": ("Analyze conversation and generate memory", lambda: analyze_one(store)),
        "3": ("View stored conversations and memories", lambda: show_conversations(store)),
        "4": ("Recover conversation", lambda: recover(store)),
        "5": ("Search stored memory", lambda: print(json.dumps(store.search(input("Search: ")), indent=2, ensure_ascii=False))),
        "6": ("Compare transcript and memory size", lambda: compare_size(store)),
        "7": ("Evaluate model performance", evaluate_model),
        "8": ("Delete conversation and associated memory", lambda: delete_conversation(store)),
        "9": ("Exit", None),
    }
    while True:
        print("\n=== Local Conversation Memory & Recovery ===")
        for key, (label, _) in actions.items():
            print(f"{key}. {label}")
        choice = input("Select: ").strip()
        if choice == "9":
            break
        action = actions.get(choice)
        if not action:
            print("Choose one of the listed options.")
            continue
        try:
            action[1]()
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            print(f"Could not complete that action: {exc}")


def delete_conversation(store: MemoryStore) -> None:
    cid = _choose(store.list_conversations())
    if cid:
        print("Deleted." if store.delete_conversation(cid) else "Conversation not found.")


if __name__ == "__main__":
    main()
