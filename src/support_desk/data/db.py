"""SQLite lifecycle for the mock account database.

Phase 2 builds parameterised query tools on top of this. The model never sees raw SQL.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from support_desk import config


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or config.DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    # Phase 4 runs account and knowledge in parallel threads; sqlite3.threadsafety == 3
    # (serialized) on CPython, so sharing one connection across them is safe.
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def build(db_path: Path | None = None) -> sqlite3.Connection:
    """Drop and recreate the database from seed.sql. Safe to run any time."""
    conn = connect(db_path)
    conn.executescript(config.SEED_SQL.read_text(encoding="utf-8"))
    conn.commit()
    return conn


def table_counts(conn: sqlite3.Connection) -> dict[str, int]:
    tables = ("customers", "orders", "payments", "shipments")
    return {
        t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]  # noqa: S608 - fixed literals
        for t in tables
    }
