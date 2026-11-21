"""SQLite persistence for conversations, messages, memories, and processing runs."""

from __future__ import annotations

import json
import hashlib
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class MemoryStore:
    def __init__(self, path: str | Path = "data/chat_memory.sqlite3"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.path, timeout=30)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _initialize(self) -> None:
        with self._connect() as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                conversation_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                source_path TEXT,
                content_hash TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                message_id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
                sequence INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                UNIQUE(conversation_id, sequence)
            );
            CREATE TABLE IF NOT EXISTS memories (
                memory_id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
                kind TEXT NOT NULL,
                status TEXT NOT NULL,
                text TEXT NOT NULL,
                confidence REAL NOT NULL,
                importance REAL NOT NULL,
                model_importance REAL,
                conflict_group TEXT,
                source_message_ids TEXT NOT NULL,
                normalized_text TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(conversation_id, kind, normalized_text)
            );
            CREATE TABLE IF NOT EXISTS processing_runs (
                run_id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
                status TEXT NOT NULL,
                extractor TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_memories_text ON memories(text);
            CREATE INDEX IF NOT EXISTS idx_memories_conversation ON memories(conversation_id, importance DESC);
            """)
            columns = {row[1] for row in db.execute("PRAGMA table_info(conversations)")}
            if "content_hash" not in columns:
                db.execute("ALTER TABLE conversations ADD COLUMN content_hash TEXT")
            db.execute("CREATE INDEX IF NOT EXISTS idx_conversation_hash ON conversations(content_hash)")
            memory_columns = {row[1] for row in db.execute("PRAGMA table_info(memories)")}
            if "model_importance" not in memory_columns:
                db.execute("ALTER TABLE memories ADD COLUMN model_importance REAL")

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def import_conversation(self, title: str, messages: list[dict[str, Any]], source_path: str | None = None, conversation_id: str | None = None) -> str:
        cid = conversation_id or str(uuid.uuid4())
        now = self._now()
        digest = hashlib.sha256("\n".join(f"{m.get('role', 'user')}:{m['content']}" for m in messages).encode("utf-8")).hexdigest()
        with self._connect() as db:
            duplicate = db.execute("SELECT conversation_id FROM conversations WHERE content_hash=?", (digest,)).fetchone()
            if duplicate:
                return str(duplicate["conversation_id"])
            existing = db.execute("SELECT conversation_id FROM conversations WHERE conversation_id=?", (cid,)).fetchone()
            if existing:
                return cid
            db.execute("INSERT INTO conversations (conversation_id,title,source_path,content_hash,created_at,updated_at) VALUES (?, ?, ?, ?, ?, ?)", (cid, title, source_path, digest, now, now))
            for seq, msg in enumerate(messages):
                db.execute("INSERT INTO messages VALUES (?, ?, ?, ?, ?)", (msg.get("message_id") or str(uuid.uuid4()), cid, seq, msg.get("role", "user"), msg["content"]))
        return cid

    def messages(self, conversation_id: str) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("SELECT message_id, sequence, role, content FROM messages WHERE conversation_id=? ORDER BY sequence", (conversation_id,)).fetchall()
        return [dict(row) for row in rows]

    def update_conversation(self, conversation_id: str, title: str, messages: list[dict[str, Any]], source_path: str | None = None) -> None:
        now = self._now()
        digest = hashlib.sha256("\n".join(f"{m.get('role', 'user')}:{m['content']}" for m in messages).encode("utf-8")).hexdigest()
        with self._connect() as db:
            if not db.execute("SELECT 1 FROM conversations WHERE conversation_id=?", (conversation_id,)).fetchone():
                raise KeyError(f"Unknown conversation: {conversation_id}")
            db.execute("DELETE FROM memories WHERE conversation_id=?", (conversation_id,))
            db.execute("DELETE FROM messages WHERE conversation_id=?", (conversation_id,))
            db.execute("UPDATE processing_runs SET status='stale',finished_at=? WHERE conversation_id=? AND status='complete'", (now, conversation_id))
            db.execute("UPDATE conversations SET title=?,source_path=?,content_hash=?,updated_at=? WHERE conversation_id=?", (title, source_path, digest, now, conversation_id))
            for seq, msg in enumerate(messages):
                db.execute("INSERT INTO messages VALUES (?, ?, ?, ?, ?)", (msg.get("message_id") or str(uuid.uuid4()), conversation_id, seq, msg.get("role", "user"), msg["content"]))

    def save_analysis(self, conversation_id: str, analysis: dict[str, Any]) -> str:
        run_id, now = str(uuid.uuid4()), self._now()
        with self._connect() as db:
            if not db.execute("SELECT 1 FROM conversations WHERE conversation_id=?", (conversation_id,)).fetchone():
                raise KeyError(f"Unknown conversation: {conversation_id}")
            db.execute("INSERT INTO processing_runs VALUES (?, ?, ?, ?, ?, ?, ?)", (run_id, conversation_id, "running", analysis.get("extractor", "local_rules_v1"), "{}", now, None))
        try:
            with self._connect() as db:
                db.execute("DELETE FROM memories WHERE conversation_id=?", (conversation_id,))
                for entry in analysis["entries"]:
                    normalized = " ".join(entry["text"].lower().split())
                    db.execute("INSERT OR IGNORE INTO memories (memory_id,conversation_id,kind,status,text,confidence,importance,model_importance,conflict_group,source_message_ids,normalized_text,created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (
                        str(uuid.uuid4()), conversation_id, entry["kind"], entry["status"], entry["text"], float(entry["confidence"]), float(entry["importance"]), entry.get("model_importance"), entry.get("conflict_group"), json.dumps(entry["source_message_ids"]), normalized, now,
                    ))
                run_metadata = {"stats": analysis.get("stats", {}), **analysis.get("metadata", {})}
                db.execute("UPDATE processing_runs SET status=?, metadata_json=?, finished_at=? WHERE run_id=?", ("complete", json.dumps(run_metadata), self._now(), run_id))
                db.execute("UPDATE conversations SET updated_at=? WHERE conversation_id=?", (self._now(), conversation_id))
        except Exception:
            with self._connect() as db:
                db.execute("UPDATE processing_runs SET status=?, finished_at=? WHERE run_id=?", ("failed", self._now(), run_id))
            raise
        return run_id

    def list_conversations(self) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("""SELECT c.conversation_id, c.title, c.created_at, c.updated_at,
                    COUNT(DISTINCT m.message_id) AS message_count, COUNT(DISTINCT x.memory_id) AS memory_count
                    FROM conversations c LEFT JOIN messages m USING(conversation_id)
                    LEFT JOIN memories x USING(conversation_id) GROUP BY c.conversation_id ORDER BY c.updated_at DESC""").fetchall()
        return [dict(row) for row in rows]

    def memories(self, conversation_id: str) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("SELECT * FROM memories WHERE conversation_id=? ORDER BY importance DESC, created_at", (conversation_id,)).fetchall()
        result = [dict(row) for row in rows]
        for row in result:
            row["source_message_ids"] = json.loads(row["source_message_ids"])
        return result

    def search(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        terms = [term for term in query.split() if term]
        if not terms:
            return []
        where = " AND ".join("(m.text LIKE ? OR m.kind LIKE ?)" for _ in terms)
        params: list[Any] = []
        for term in terms:
            params.extend((f"%{term}%", f"%{term}%"))
        params.append(limit)
        with self._connect() as db:
            rows = db.execute(f"SELECT m.*, c.title FROM memories m JOIN conversations c USING(conversation_id) WHERE {where} ORDER BY m.importance DESC LIMIT ?", params).fetchall()
        result = [dict(row) for row in rows]
        for row in result:
            row["source_message_ids"] = json.loads(row["source_message_ids"])
        return result

    def delete_conversation(self, conversation_id: str) -> bool:
        with self._connect() as db:
            cur = db.execute("DELETE FROM conversations WHERE conversation_id=?", (conversation_id,))
        return cur.rowcount > 0

    def latest_run(self, conversation_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM processing_runs WHERE conversation_id=? ORDER BY started_at DESC LIMIT 1", (conversation_id,)).fetchone()
        return dict(row) if row else None
