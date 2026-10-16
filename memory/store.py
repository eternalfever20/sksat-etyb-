"""SQLite facts scoped to a workspace and optionally to a dataset fingerprint."""

from __future__ import annotations

import re
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class MemoryInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: str = Field(min_length=1, max_length=40)
    key: str = Field(min_length=1, max_length=120)
    value: str = Field(min_length=1, max_length=2000)
    source: str = Field(default="user", max_length=300)


def parse_plain_definition(sentence: str) -> dict[str, str]:
    """Split a short user-written definition while preserving exclusion rules."""
    sentence = sentence.strip().rstrip(".")
    match = re.match(
        r"^(.*?)\s+(means|is|=|excludes|includes?|defined as)\s+(.*?)$",
        sentence,
        flags=re.IGNORECASE,
    )
    if match and match.group(1).strip() and match.group(3).strip():
        key, relation, definition = match.groups()
        relation = relation.casefold()
        if relation in {"excludes", "include", "includes"}:
            definition = f"{relation} {definition.strip()}"
        return {
            "kind": "metric_definition",
            "key": key.strip(),
            "value": definition.strip(),
        }
    if re.match(r"^always use\b", sentence, re.IGNORECASE):
        return {
            "kind": "preference",
            "key": "Analysis preference",
            "value": sentence,
        }
    raise ValueError(
        "Use a sentence like ‘active customers means accounts that are not cancelled’."
    )


class MemoryStore:
    def __init__(self, path: str, workspace_id: str = "local-default"):
        self.path, self.workspace_id = path, workspace_id
        if path != ":memory:":
            Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        return db

    def _init(self):
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS memories(
                id TEXT PRIMARY KEY, workspace_id TEXT NOT NULL, dataset_fingerprint TEXT,
                kind TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL, source TEXT NOT NULL,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL, last_used_at TEXT,
                UNIQUE(workspace_id,dataset_fingerprint,kind,key))""")

    def upsert(self, data: dict, fingerprint: str | None = None) -> dict:
        item = MemoryInput.model_validate(data)
        now = datetime.now(UTC).isoformat()
        with self._connect() as db:
            existing = db.execute(
                "SELECT id,created_at FROM memories WHERE workspace_id=? AND dataset_fingerprint IS ? AND kind=? AND key=?",
                (self.workspace_id, fingerprint, item.kind, item.key),
            ).fetchone()
            mid = existing["id"] if existing else uuid.uuid4().hex
            created = existing["created_at"] if existing else now
            if existing:
                db.execute(
                    "UPDATE memories SET value=?,source=?,updated_at=? WHERE id=?",
                    (item.value, item.source, now, mid),
                )
            else:
                db.execute(
                    """INSERT INTO memories(id,workspace_id,dataset_fingerprint,kind,key,value,source,created_at,updated_at,last_used_at)
                    VALUES(?,?,?,?,?,?,?,?,?,NULL)""",
                    (
                        mid,
                        self.workspace_id,
                        fingerprint,
                        item.kind,
                        item.key,
                        item.value,
                        item.source,
                        created,
                        now,
                    ),
                )
        return {
            "id": mid,
            "kind": item.kind,
            "key": item.key,
            "value": item.value,
            "dataset_fingerprint": fingerprint,
            "updated_at": now,
            "updated": existing is not None,
        }

    def list(self, fingerprint: str | None = None) -> list[dict]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM memories WHERE workspace_id=? AND (dataset_fingerprint IS NULL OR dataset_fingerprint IS ?) ORDER BY updated_at DESC",
                (self.workspace_id, fingerprint),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_all(self) -> list[dict]:
        """List every memory in this workspace for the management UI."""
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM memories WHERE workspace_id=? ORDER BY updated_at DESC",
                (self.workspace_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def relevant(self, question: str, fingerprint: str | None = None) -> list[dict]:
        terms = set(re.findall(r"[a-z0-9_]+", question.lower()))
        terms -= {
            "what",
            "which",
            "how",
            "does",
            "did",
            "are",
            "was",
            "were",
            "the",
            "and",
            "for",
            "with",
            "from",
            "that",
            "this",
            "show",
            "tell",
            "give",
            "calculate",
            "compute",
            "please",
            "data",
            "table",
            "is",
            "it",
            "about",
            "my",
            "of",
            "in",
            "on",
            "to",
            "by",
        }
        preferred: dict[tuple[str, str], dict] = {}
        for m in self.list(fingerprint):
            if (
                m["dataset_fingerprint"] is not None
                and m["dataset_fingerprint"] != fingerprint
            ):
                continue
            hay = set(
                re.findall(
                    r"[a-z0-9_]+",
                    (m["key"] + " " + m["value"] + " " + m["kind"]).lower(),
                )
            )
            if terms and not terms.intersection(hay):
                continue
            identity = (m["kind"].casefold(), m["key"].casefold())
            previous = preferred.get(identity)
            is_dataset_fact = (
                m["dataset_fingerprint"] == fingerprint and fingerprint is not None
            )
            previous_is_dataset_fact = (
                previous is not None
                and previous["dataset_fingerprint"] == fingerprint
                and fingerprint is not None
            )
            if previous is None or (is_dataset_fact and not previous_is_dataset_fact):
                preferred[identity] = m
        selected = sorted(
            preferred.values(), key=lambda item: item["updated_at"], reverse=True
        )
        if selected:
            with self._connect() as db:
                ids = [m["id"] for m in selected]
                db.executemany(
                    "UPDATE memories SET last_used_at=? WHERE id=?",
                    [(datetime.now(UTC).isoformat(), i) for i in ids],
                )
        return selected[:30]

    def delete(self, memory_id: str) -> bool:
        with self._connect() as db:
            cur = db.execute(
                "DELETE FROM memories WHERE id=? AND workspace_id=?",
                (memory_id, self.workspace_id),
            )
            return cur.rowcount > 0

    def clear(self) -> int:
        with self._connect() as db:
            cur = db.execute(
                "DELETE FROM memories WHERE workspace_id=?", (self.workspace_id,)
            )
            return cur.rowcount
