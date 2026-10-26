"""SQLite storage: event/decision/order/alert logging and the bot state."""
import json
import sqlite3
import threading

import pytest

from state import BotState, StateError
from storage import Database, SqliteStateStore
from test_engine import make, parts  # noqa: F401  (parts is a fixture)


@pytest.fixture
def db(tmp_path):
    return Database(str(tmp_path / "bot.db"))


def rows(db, sql):
    return db._read(sql)


BUY = {"action": "buy", "ticker": "QQQ", "usd": 15.0, "reason": "momentum"}


def test_order_event_lands_in_events_decisions_and_orders(db):
    db.record({"ts": "2026-10-22T15:07:30+00:00", "kind": "order", "decision": BUY,
               "order_id": "o1", "client_order_id": "ab-20261022T1505", "status": "filled",
               "fill_price": 100.05, "filled_qty": 0.15, "slippage_pct": 0.05,
               "cost_usd": 0.0004, "usage": {"input": 900, "output": 120},
               "request_id": "req_1", "raw": "{}"})
    assert rows(db, "SELECT kind FROM events") == [("order",)]
    assert rows(db, "SELECT outcome, action, ticker, usd, reason, cost_usd, input_tokens, "
                    "output_tokens FROM decisions") == [
        ("order", "buy", "QQQ", 15.0, "momentum", 0.0004, 900, 120)]
    assert rows(db, "SELECT client_order_id, status, fill_price, slippage_pct, reason "
                    "FROM orders") == [("ab-20261022T1505", "filled", 100.05, 0.05, "momentum")]


def test_rejected_and_invalid_are_decisions_not_orders(db):
    db.record({"ts": "t1", "kind": "rejected", "decision": BUY, "why": "over cap"})
    db.record({"ts": "t2", "kind": "invalid", "error": "bad json", "raw": "nope"})
    db.record({"ts": "t3", "kind": "idle", "why": "market closed"})
    assert rows(db, "SELECT outcome, why FROM decisions ORDER BY id") == [
        ("rejected", "over cap"), ("invalid", "bad json")]
    assert rows(db, "SELECT COUNT(*) FROM orders") == [(0,)]
    assert rows(db, "SELECT COUNT(*) FROM events") == [(3,)]


def test_failed_order_is_logged_with_error(db):
    db.record({"ts": "t", "kind": "order_failed", "decision": BUY, "error": "timeout"})
    assert rows(db, "SELECT status, error FROM orders") == [("failed", "timeout")]


def test_bad_event_never_raises_and_writes_nothing_partial(db):
    db.record({"ts": "t", "kind": "order", "decision": "not a dict"})
    assert rows(db, "SELECT COUNT(*) FROM events") == [(0,)]


def test_alerts_are_stored(db):
    db.record_alert("halt", "HALTED: floor")
    assert rows(db, "SELECT kind, msg FROM alerts") == [("halt", "HALTED: floor")]


def test_events_since(db):
    for ts in ("2026-10-21T20:00:00+00:00", "2026-10-22T14:00:00+00:00"):
        db.record({"ts": ts, "kind": "idle"})
    assert [e["ts"] for e in db.events_since("2026-10-22T04:00:00+00:00")] == [
        "2026-10-22T14:00:00+00:00"]


def test_logging_from_two_threads(db):
    def burst():
        for _ in range(50):
            db.record({"ts": "t", "kind": "idle"})
    ts = [threading.Thread(target=burst) for _ in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert rows(db, "SELECT COUNT(*) FROM events") == [(200,)]


# --- state ----------------------------------------------------------------------
def test_state_roundtrip_and_survives_reopen(tmp_path):
    path = str(tmp_path / "bot.db")
    SqliteStateStore(Database(path)).save(BotState(halted=True, halt_reason="floor", paused=True))
    st = SqliteStateStore(Database(path)).load()
    assert st.halted and st.halt_reason == "floor" and st.paused


def test_missing_state_is_blank(db):
    assert SqliteStateStore(db).load() == BotState()


def test_unreadable_state_raises(db):
    db._write("INSERT INTO state (id, data, updated) VALUES (1, '{\"bogus\": 1}', '')")
    with pytest.raises(StateError):
        SqliteStateStore(db).load()


def test_corrupt_database_file_refuses_to_open(tmp_path):
    p = tmp_path / "bot.db"
    p.write_bytes(b"this is not a sqlite file" * 100)
    with pytest.raises(StateError):
        Database(str(p))


def test_old_state_json_is_imported_once(db, tmp_path):
    old = tmp_path / "state.json"
    old.write_text(json.dumps({"halted": True, "halt_reason": "floor", "paused": False,
                               "order_times": [], "ai_spend": {}}))
    store = SqliteStateStore(db)
    assert store.import_json(str(old))
    assert store.load().halted
    assert not old.exists() and (tmp_path / "state.json.imported").exists()
    assert not store.import_json(str(old))


def test_import_never_overwrites_existing_state(db, tmp_path):
    store = SqliteStateStore(db)
    store.save(BotState(halted=True, halt_reason="floor"))
    old = tmp_path / "state.json"
    old.write_text(json.dumps({"halted": False}))
    assert not store.import_json(str(old))
    assert store.load().halted


# --- engine end to end ---------------------------------------------------------
def test_engine_cycle_is_logged_and_halt_survives_restart(parts):
    db = parts.store.db
    e = make(parts)
    e.record = db.record
    e.run_cycle()
    assert rows(db, "SELECT action, ticker, status FROM orders") == [("buy", "QQQ", "filled")]
    e.halt("test")
    assert make(parts).g.halted
