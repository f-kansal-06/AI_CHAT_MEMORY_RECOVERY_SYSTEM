"""Run a realistic, self-contained import → analyze → persist → recover demo."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.app.analyzer import analyze_conversation
from src.app.ingest import load_conversation
from src.app.recovery import render_recovery
from src.app.scorer import load_local_scorer
from src.app.store import MemoryStore


def main() -> None:
    demo = Path(__file__).resolve().parents[1] / "examples" / "realistic_conversation.json"
    title, messages = load_conversation(demo)
    with tempfile.TemporaryDirectory(prefix="chat-memory-demo-") as temp:
        store = MemoryStore(Path(temp) / "demo.sqlite3")
        conversation_id = store.import_conversation(title, messages, str(demo))
        scorer = load_local_scorer()
        analysis = analyze_conversation(store.messages(conversation_id), scorer.score(messages) if scorer else None, scorer.name if scorer else None)
        store.save_analysis(conversation_id, analysis)
        print(f"Imported {len(messages)} messages; saved {analysis['stats']['entry_count']} memory entries.")
        print(f"Importance: {analysis['extractor']} | {analysis['stats']['original_tokens']} -> {analysis['stats']['memory_tokens']} tokens ({analysis['stats']['compression_ratio']:.1%} reduction)")
        print("\n" + render_recovery(title, store.memories(conversation_id)))


if __name__ == "__main__":
    main()
