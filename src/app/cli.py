"""Small terminal app: summarize a sample chat or paste a chat for Notepad."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Any

from src.app.analyzer import analyze_conversation
from src.app.ingest import load_conversation, parse_conversation_text, parse_loading_older_paste
from src.app.recovery import render_summary
from src.app.store import MemoryStore

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMO_TRANSCRIPT = PROJECT_ROOT / "examples" / "sample_chat.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "chat_summary.md"


def _open_in_notepad(path: Path) -> None:
    if os.name != "nt":
        print(f"Summary saved. Open it in a text editor: {path.resolve()}")
        return
    try:
        subprocess.Popen(["notepad.exe", str(path.resolve())])
        print("Opened the summary in Notepad.")
    except OSError as exc:
        print(f"Summary saved, but Notepad could not be opened: {exc}")


def _summarize_and_save(
    store: MemoryStore,
    title: str,
    messages: list[dict[str, Any]],
    output_path: Path,
    source_path: str | None = None,
) -> Path:
    conversation_id = store.import_conversation(title, messages, source_path=source_path)
    # Analyze the rows actually stored. An identical chat may already exist,
    # in which case its original message IDs must remain the evidence links.
    stored_messages = store.messages(conversation_id)
    analysis = analyze_conversation(stored_messages)
    store.save_analysis(conversation_id, analysis)
    conversation = next(
        row for row in store.list_conversations()
        if row["conversation_id"] == conversation_id
    )
    summary = render_summary(
        conversation["title"],
        store.memories(conversation_id),
        stored_messages,
        stats=analysis["stats"],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(summary, encoding="utf-8")
    print(f"Processed {len(stored_messages)} messages and extracted {analysis['stats']['entry_count']} points locally.")
    print(f"Summary saved to: {output_path.resolve()}")
    _open_in_notepad(output_path)
    return output_path


def run_guided_demo(store: MemoryStore, output_path: Path = DEFAULT_OUTPUT) -> None:
    print("\nSummarizing the prepared sample chat...")
    title, messages = load_conversation(DEMO_TRANSCRIPT)
    _summarize_and_save(
        store, title, messages, output_path, source_path=str(DEMO_TRANSCRIPT)
    )


def paste_chat(store: MemoryStore, output_path: Path = DEFAULT_OUTPUT) -> None:
    print("\nCopy your conversation from ChatGPT, then paste it here.")
    print("Keep speaker headings such as 'You said:' and 'ChatGPT said:' if they are present.")
    print("When the full chat is pasted, type /done on a new line and press Enter.")
    pasted_lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "/done":
            break
        pasted_lines.append(line)
    pasted_text = "\n".join(pasted_lines).strip()
    if not pasted_text:
        print("No chat was pasted.")
        return

    page_paste = parse_loading_older_paste(pasted_text)
    if page_paste:
        suggested_title, messages = page_paste
    else:
        suggested_title = "Pasted chat"
        messages = parse_conversation_text(pasted_text)
    if not messages:
        print("No messages could be read from that paste.")
        return
    roles = {message["role"] for message in messages}
    if not {"user", "assistant"}.issubset(roles):
        print("I could not identify both chat speakers. I will summarize the pasted text, but decisions may be harder to classify.")
        print("For the clearest result, include headings such as 'You said:' and 'ChatGPT said:'.")

    title = input(f"Chat title (Enter keeps '{suggested_title}'): ").strip()
    _summarize_and_save(store, title or suggested_title, messages, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize chats locally and open the result in Notepad")
    parser.add_argument("--db", default=str(PROJECT_ROOT / "data" / "chat_memory.sqlite3"), help=argparse.SUPPRESS)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Where to write the summary file")
    parser.add_argument("--import-file", help=argparse.SUPPRESS)
    args = parser.parse_args()

    store = MemoryStore(args.db)
    output_path = Path(args.output)
    if args.import_file:
        title, messages = load_conversation(args.import_file)
        _summarize_and_save(store, title, messages, output_path, source_path=str(Path(args.import_file).resolve()))
        return

    actions = {
        "1": ("Summarize the sample chat in Notepad", lambda: run_guided_demo(store, output_path)),
        "2": ("Paste my ChatGPT chat and summarize it", lambda: paste_chat(store, output_path)),
        "3": ("Exit", None),
    }
    while True:
        print("\n=== Chat Summary to Notepad ===")
        for key, (label, _) in actions.items():
            print(f"{key}. {label}")
        choice = input("Select: ").strip()
        if choice == "3":
            return
        action = actions.get(choice)
        if not action:
            print("Choose 1, 2, or 3.")
            continue
        try:
            action[1]()
        except (OSError, ValueError, KeyError) as exc:
            print(f"Could not create the summary: {exc}")


if __name__ == "__main__":
    main()
