"""Simple terminal workflow for local conversation memory and recovery."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from src.app.analyzer import analyze_conversation
from src.app.ingest import load_conversation, parse_conversation_text
from src.app.recovery import render_recovery
from src.app.store import MemoryStore
from src.data.cleaning import count_tokens

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMO_TRANSCRIPT = PROJECT_ROOT / "examples" / "realistic_conversation.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "recovery_context.md"


def _choose(rows: list[dict], prompt: str = "Conversation ID") -> str | None:
    if not rows:
        print("No conversations stored.")
        return None
    for i, row in enumerate(rows, 1):
        print(f"{i}. {row['title']} | {row['conversation_id']} | {row['message_count']} messages | {row['memory_count']} memories")
    value = input(f"{prompt} or list number (blank cancels): ").strip()
    if not value:
        return None
    if value.isdigit() and 1 <= int(value) <= len(rows):
        return rows[int(value) - 1]["conversation_id"]
    return value


def import_file(store: MemoryStore) -> str | None:
    path = input("Conversation file path (TXT/JSON/JSONL), or 'manual': ").strip().strip('"')
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
    messages = parse_conversation_text("\n".join(transcript))
    if not messages:
        print("No messages entered.")
        return None
    cid = store.import_conversation(title, messages)
    print(f"Imported {len(messages)} messages. Conversation ID: {cid}")
    return cid


def analyze_one(store: MemoryStore, cid: str | None = None, use_model: bool = False) -> str | None:
    if not cid:
        cid = _choose(store.list_conversations())
    if not cid:
        return None
    messages = store.messages(cid)
    if not messages:
        print("Conversation not found or contains no messages.")
        return None
    print(f"Analyzing {len(messages)} messages locally...")
    scorer = None
    if use_model:
        # Keep PyTorch/Transformers startup cost out of the normal terminal path.
        from src.app.scorer import load_local_scorer
        scorer = load_local_scorer()
    scores = scorer.score(messages) if scorer else None
    analysis = analyze_conversation(messages, scores, scorer.name if scorer else None)
    store.save_analysis(cid, analysis)
    stats = analysis["stats"]
    print(analysis["overview"])
    if scorer:
        print(f"Importance scoring: {scorer.name} (optional; it ranks messages, it does not extract facts).")
    else:
        print("Analysis: local rules; no model download or hosted API used.")
    print(f"Transcript size: {stats['original_tokens']} tokens; memory: {stats['memory_tokens']} tokens ({stats['compression_ratio']:.1%} reduction).")
    print(f"Saved {stats['entry_count']} memory items.")
    return cid


def review_memories(store: MemoryStore, cid: str) -> None:
    memories = store.memories(cid)
    if not memories:
        print("No memories to review. Analyze the conversation first.")
        return
    source_by_id = {m["message_id"]: m for m in store.messages(cid)}
    print("\nReview each extracted item against its source. Enter=keep, e=exclude, c=correct, r=restore, q=finish.")
    for number, memory in enumerate(memories, 1):
        excluded = bool(memory.get("excluded", 0))
        marker = "EXCLUDED" if excluded else "included"
        print(f"\n{number}. [{memory['kind']} / {memory['status']} / {marker}] {memory['text']}")
        print(f"   Confidence: {memory['confidence']:.2f} | Importance: {memory['importance']:.2f}")
        for source_id in memory["source_message_ids"]:
            source = source_by_id.get(source_id)
            if source:
                content = source["content"].replace("\n", " ")
                if len(content) > 220:
                    content = content[:217] + "..."
                print(f"   Source {source_id} ({source['role']}): {content}")
        command = input("   Action [Enter/e/c/r/q]: ").strip().lower()
        if command == "q":
            break
        if command == "e":
            store.set_memory_excluded(memory["memory_id"], True)
            print("   Excluded from recovery output.")
        elif command == "r":
            store.set_memory_excluded(memory["memory_id"], False)
            print("   Restored to recovery output.")
        elif command == "c":
            corrected = input("   Corrected memory text: ").strip()
            try:
                if corrected:
                    store.update_memory_text(memory["memory_id"], corrected)
                    print("   Correction saved; source reference preserved.")
                else:
                    print("   No correction entered; original text kept.")
            except ValueError as exc:
                print(f"   Correction not saved: {exc}")


def _open_in_notepad(path: Path) -> None:
    if os.name != "nt":
        print(f"Open this file in a text editor: {path}")
        return
    try:
        subprocess.Popen(["notepad.exe", str(path.resolve())])
        print("Opened the recovery document in Notepad.")
    except OSError as exc:
        print(f"Saved the recovery document, but could not launch Notepad: {exc}")


def write_recovery_file(store: MemoryStore, cid: str, output_path: Path = DEFAULT_OUTPUT) -> Path | None:
    row = next((item for item in store.list_conversations() if item["conversation_id"] == cid), None)
    if not row:
        print("Conversation not found.")
        return None
    memories = store.memories(cid)
    if not any(not item.get("excluded", 0) for item in memories):
        print("No included memories. Review or analyze this conversation before recovery.")
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_recovery(row["title"], memories), encoding="utf-8")
    print(f"Recovery document updated: {output_path.resolve()}")
    _open_in_notepad(output_path)
    return output_path


def run_guided_demo(store: MemoryStore, output_path: Path = DEFAULT_OUTPUT) -> None:
    print("\n=== Guided local memory demo ===")
    print("This sample run uses local rules and SQLite. No chatbot or hosted API is called.")
    title, messages = load_conversation(DEMO_TRANSCRIPT)
    cid = store.import_conversation(title, messages, source_path=str(DEMO_TRANSCRIPT))
    print(f"1. Imported sample: {title} ({len(store.messages(cid))} messages).")
    cid = analyze_one(store, cid)
    if not cid:
        return
    print("2. Review the extracted memories below. You can keep, exclude, correct, or restore each item.")
    review_memories(store, cid)
    print("3. Writing the recovery document from the saved memories only.")
    write_recovery_file(store, cid, output_path)


def show_conversations(store: MemoryStore) -> None:
    rows = store.list_conversations()
    if not rows:
        print("No conversations stored.")
        return
    cid = _choose(rows)
    if not cid:
        return
    selected = next((item for item in rows if item["conversation_id"] == cid), None)
    if not selected:
        print("Conversation not found.")
        return
    print(f"\n{selected['title']}")
    for memory in store.memories(cid):
        marker = "excluded" if memory.get("excluded", 0) else "included"
        print(f"- [{memory['kind']} / {memory['status']} / {marker}] {memory['text']} (sources: {', '.join(memory['source_message_ids'])})")


def recover(store: MemoryStore, output_path: Path = DEFAULT_OUTPUT) -> None:
    cid = _choose(store.list_conversations())
    if cid:
        write_recovery_file(store, cid, output_path)


def compare_size(store: MemoryStore) -> None:
    cid = _choose(store.list_conversations())
    if not cid:
        return
    row = next((r for r in store.list_conversations() if r["conversation_id"] == cid), None)
    messages = store.messages(cid)
    memories = [m for m in store.memories(cid) if not m.get("excluded", 0)]
    original = sum(count_tokens(m["content"]) for m in messages)
    compact = sum(count_tokens(m["text"]) for m in memories)
    print(f"{row['title'] if row else cid}: {original} transcript tokens -> {compact} included memory tokens ({1 - compact / max(original, 1):.1%} reduction)")


def evaluate_model() -> None:
    result_path = PROJECT_ROOT / "results" / "tuning" / "tuning_results.json"
    if result_path.exists():
        print("Latest local tuned-model report (turn importance experiment, not a live memory-quality score):")
        print(json.dumps(json.loads(result_path.read_text(encoding="utf-8")), indent=2)[:6000])
        return
    result_path = PROJECT_ROOT / "results" / "e0_baselines" / "e0_results.json"
    if not result_path.exists():
        print("No saved evaluation results found. Run the research evaluation pipeline first.")
        return
    results = json.loads(result_path.read_text(encoding="utf-8"))
    print("Saved benchmark evaluation (historical artifact; not a live evaluation):")
    print(json.dumps(results, indent=2)[:6000])


def more_tools(store: MemoryStore) -> None:
    options = {"1": ("Compare transcript and memory size", lambda: compare_size(store)),
               "2": ("View saved model experiment report", evaluate_model),
               "3": ("Delete a conversation and its memories", lambda: delete_conversation(store)),
               "4": ("Back", None)}
    while True:
        print("\n--- More tools ---")
        for key, (label, _) in options.items():
            print(f"{key}. {label}")
        choice = input("Select: ").strip()
        if choice == "4":
            return
        action = options.get(choice)
        if action:
            action[1]()
        else:
            print("Choose one of the listed options.")


def delete_conversation(store: MemoryStore) -> None:
    cid = _choose(store.list_conversations())
    if cid:
        print("Deleted." if store.delete_conversation(cid) else "Conversation not found.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Local conversation memory and recovery")
    parser.add_argument("--db", default=str(PROJECT_ROOT / "data" / "chat_memory.sqlite3"), help="SQLite database file")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Recovery document written and opened in Notepad")
    parser.add_argument("--import-file", help="Import a TXT, JSON, or JSONL conversation, then exit")
    parser.add_argument("--use-importance-model", action="store_true", help="Load the optional local turn-importance model during analysis")
    args = parser.parse_args()
    store = MemoryStore(args.db)
    output_path = Path(args.output)
    if args.import_file:
        cid = store.import_conversation(*load_conversation(args.import_file), source_path=str(Path(args.import_file).resolve()))
        print(f"Imported. Conversation ID: {cid}")
        return

    actions = {
        "1": ("Run guided demo (sample -> review -> Notepad)", lambda: run_guided_demo(store, output_path)),
        "2": ("Import a conversation", lambda: import_file(store)),
        "3": ("Analyze a conversation", lambda: analyze_one(store, use_model=args.use_importance_model)),
        "4": ("Review, correct, or exclude memories", lambda: review_selected(store)),
        "5": ("Write recovery document and open in Notepad", lambda: recover(store, output_path)),
        "6": ("View conversations and memories", lambda: show_conversations(store)),
        "7": ("Search included memories", lambda: print(json.dumps(store.search(input("Search: ")), indent=2, ensure_ascii=False))),
        "8": ("More tools", lambda: more_tools(store)),
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


def review_selected(store: MemoryStore) -> None:
    cid = _choose(store.list_conversations())
    if cid:
        review_memories(store, cid)


if __name__ == "__main__":
    main()
