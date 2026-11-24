"""Start the guided terminal-to-Notepad lab demonstration."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.app.cli import DEFAULT_OUTPUT, run_guided_demo
from src.app.store import MemoryStore


def main() -> None:
    store = MemoryStore(PROJECT_ROOT / "data" / "chat_memory.sqlite3")
    run_guided_demo(store, DEFAULT_OUTPUT)


if __name__ == "__main__":
    main()
