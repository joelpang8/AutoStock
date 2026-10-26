"""SQLite storage: every event, decision, order and alert, plus the bot state.

One database file (DB_PATH, default autostock.db). Tables:
  events     every engine event as JSON, the full record
  decisions  each AI reply: action, ticker, usd, reason, cost, and what came of it
  orders     each submitted order: status, fill price, slippage, the AI's reason
  alerts     every alert pushed to the owner
  state      one row: the BotState that must survive a restart (halt, pause, ...)

Writes come from the engine thread and reads from the check-in bot's thread, so
one connection is shared behind a lock. WAL mode and synchronous=FULL: a write
that returned is on disk. Logging never raises into the engine; state does,
because trading on a state we failed to save or read would be unsafe.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from dataclasses import asdict
from datetime import datetime, timezone

from state import BotState, StateError

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY, ts TEXT NOT NULL, kind TEXT NOT NULL, data TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS events_ts ON events(ts);
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY, ts TEXT NOT NULL, outcome TEXT NOT NULL,
    action TEXT, ticker TEXT, usd REAL, reason TEXT, why TEXT,
    cost_usd REAL, input_tokens INTEGER, output_tokens INTEGER, request_id TEXT, raw TEXT);
CREATE INDEX IF NOT EXISTS decisions_ts ON decisions(ts);
CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY, ts TEXT NOT NULL, client_order_id TEXT, order_id TEXT,
    action TEXT, ticker TEXT, usd REAL, status TEXT, fill_price REAL, filled_qty REAL,
    slippage_pct REAL, reason TEXT, error TEXT);
CREATE INDEX IF NOT EXISTS orders_ts ON orders(ts);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY, ts TEXT NOT NULL, kind TEXT NOT NULL, msg TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS state (
    id INTEGER PRIMARY KEY CHECK (id = 1), data TEXT NOT NULL, updated TEXT NOT NULL);
"""

DECISION_KINDS = {"decision", "rejected", "order", "order_failed", "invalid"}
ORDER_KINDS = {"order", "order_failed"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, path: str):
        self.path = path
        self.lock = threading.Lock()
        try:
            self.conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=FULL")
            self.conn.executescript(SCHEMA)
        except sqlite3.DatabaseError as e:
            raise StateError(f"can't open database {path}: {e}. "
                             "Refusing to start: fix or move it deliberately.") from e

    def _write(self, sql: str, args=()):
        with self.lock:
            self.conn.execute(sql, args)

    def _read(self, sql: str, args=()) -> list:
        with self.lock:
            return self.conn.execute(sql, args).fetchall()

    # --- event log (the engine's record callback) ---------------------------
    def record(self, event: dict):
        try:
            self._record(event)
        except Exception:
            log.exception("could not store event %s", event)

    def _record(self, e: dict):
        ts, kind = e.get("ts") or _now(), e.get("kind", "?")
        with self.lock:
            self.conn.execute("BEGIN")
            try:
                self.conn.execute("INSERT INTO events (ts, kind, data) VALUES (?, ?, ?)",
                                  (ts, kind, json.dumps(e, default=str)))
                d = e.get("decision") or {}
                if kind in DECISION_KINDS:
                    usage = e.get("usage") or {}
                    self.conn.execute(
                        "INSERT INTO decisions (ts, outcome, action, ticker, usd, reason, why, "
                        "cost_usd, input_tokens, output_tokens, request_id, raw) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (ts, kind, d.get("action"), d.get("ticker"), d.get("usd"), d.get("reason"),
                         e.get("why") or e.get("error"), e.get("cost_usd"),
                         usage.get("input"), usage.get("output"), e.get("request_id"), e.get("raw")))
                if kind in ORDER_KINDS:
                    self.conn.execute(
                        "INSERT INTO orders (ts, client_order_id, order_id, action, ticker, usd, "
                        "status, fill_price, filled_qty, slippage_pct, reason, error) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (ts, e.get("client_order_id"), e.get("order_id"), d.get("action"),
                         d.get("ticker"), d.get("usd"), e.get("status", "failed"),
                         e.get("fill_price"), e.get("filled_qty"), e.get("slippage_pct"),
                         d.get("reason"), e.get("error")))
                self.conn.execute("COMMIT")
            except BaseException:
                self.conn.execute("ROLLBACK")
                raise

    def record_alert(self, kind: str, msg: str):
        try:
            self._write("INSERT INTO alerts (ts, kind, msg) VALUES (?, ?, ?)", (_now(), kind, msg))
        except Exception:
            log.exception("could not store alert")

    def events_since(self, ts: str) -> list[dict]:
        """Events at or after an ISO UTC timestamp, oldest first."""
        return [json.loads(r[0]) for r in
                self._read("SELECT data FROM events WHERE ts >= ? ORDER BY id", (ts,))]


class SqliteStateStore:
    """Same interface as the old JSON StateStore: load() / save(state)."""

    def __init__(self, db: Database):
        self.db = db

    def load(self) -> BotState:
        try:
            rows = self.db._read("SELECT data FROM state WHERE id = 1")
            return BotState(**json.loads(rows[0][0])) if rows else BotState()
        except Exception as e:
            raise StateError(f"can't read bot state from {self.db.path}: {e}. "
                             "Refusing to start: fix it deliberately.") from e

    def save(self, state: BotState) -> None:
        self.db._write("INSERT INTO state (id, data, updated) VALUES (1, ?, ?) "
                       "ON CONFLICT(id) DO UPDATE SET data = excluded.data, updated = excluded.updated",
                       (json.dumps(asdict(state), sort_keys=True), _now()))

    def import_json(self, path: str) -> bool:
        """One-time move of an old state.json (keeps a halt across the upgrade)."""
        if not os.path.exists(path) or self.db._read("SELECT 1 FROM state WHERE id = 1"):
            return False
        try:
            with open(path) as f:
                state = BotState(**json.load(f))
        except Exception as e:
            raise StateError(f"can't import {path}: {e}. Fix or remove it deliberately.") from e
        self.save(state)
        os.replace(path, path + ".imported")
        return True
