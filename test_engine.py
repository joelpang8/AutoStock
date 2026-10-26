"""Engine tests with fake broker, data and decider. No network."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS

import pytest

from broker import BrokerError
from data import MarketClosed, Snapshot, StaleDataError, TickerView
from decision import (BudgetExhausted, DailyBudget, Decision, DecisionResult)
from engine import Engine
from guardrails import Guardrails
from state import BotState, StateError
from storage import Database, SqliteStateStore

NOW = datetime(2026, 10, 22, 15, 7, 30, tzinfo=timezone.utc)   # slot 15:05


class FakeBroker:
    def __init__(self, equity=300.0):
        self._equity, self.cash = equity, 250.0
        self.held = {}               # {ticker: {"usd", "pl_pct", "qty"}}
        self.orders = {}             # client_order_id -> order
        self.open = []
        self.submitted, self.cancelled_ids = [], []
        self.closed_all = 0
        self.submit_error = None     # raise this from submit_market
        self.lands_anyway = False    # ...but the order still reached Alpaca
        self.fill_status = "filled"

    # guardrails interface
    def equity(self):
        if self._equity is None:
            raise BrokerError("equity unreadable")
        return self._equity
    def settled_cash(self): return self.cash
    def position_value(self, t): return self.held.get(t, {}).get("usd", 0.0)
    def cancel_all_orders(self): self.open = []
    def close_all_positions(self):
        self.closed_all += 1
        self.held = {}

    # engine extras
    def positions(self): return dict(self.held)
    def open_orders(self, ticker=None): return list(self.open)
    def order_by_client_id(self, cid): return self.orders.get(cid)
    def order(self, oid): return next(o for o in self.orders.values() if o.id == oid)
    def cancel_order(self, oid): self.cancelled_ids.append(oid)

    def submit_market(self, ticker, side, cid, usd=None, qty=None):
        self.submitted.append(dict(ticker=ticker, side=side, cid=cid, usd=usd, qty=qty))
        o = NS(id=f"id-{cid}", client_order_id=cid, status=self.fill_status,
               filled_avg_price="100.05", filled_qty="0.15", symbol=ticker)
        if self.submit_error:
            if self.lands_anyway:
                self.orders[cid] = o
            raise self.submit_error
        self.orders[cid] = o
        return o


def snapshot(positions=None):
    tv = lambda s: TickerView(s, 100.0, 0.02, 0.1, 0.0, 1.0, 3.0)
    return Snapshot(NOW, NOW.replace(hour=20), 300.0, 250.0, positions or {},
                    {"SPY": tv("SPY"), "QQQ": tv("QQQ")}, {})


class FakeData:
    def __init__(self):
        self.exc, self.snap, self.calls = None, None, 0
    def snapshot(self, broker, tickers):
        self.calls += 1
        if self.exc:
            raise self.exc
        return self.snap or snapshot(broker.positions())


class FakeDecider:
    def __init__(self):
        self.budget = DailyBudget(1.0)
        self.calls = 0
        self.next = Decision("buy", "QQQ", 15.0, "test buy")
        self.error = ""
        self.exc = None
    def decide(self, snap):
        self.calls += 1
        if self.exc:
            raise self.exc
        self.budget.add(0.0004)
        d = None if self.error else self.next
        return DecisionResult(d, 0.0004, {"input": 900, "output": 120}, "req_1", "{}", self.error)


@pytest.fixture
def parts(tmp_path):
    b = FakeBroker()
    return NS(broker=b, data=FakeData(), decider=FakeDecider(), alerts=[], events=[],
              store=SqliteStateStore(Database(str(tmp_path / "bot.db"))))


def make(p):
    return Engine(p.broker, p.data, p.decider, Guardrails(p.broker), p.store, ["SPY", "QQQ"],
                  notify=lambda k, m: p.alerts.append((k, m)), record=p.events.append,
                  now=lambda: NOW, sleep=lambda s: None)


# --- happy path --------------------------------------------------------------
def test_buy_is_submitted_by_dollars_with_slot_id(parts):
    ev = make(parts).run_cycle()
    assert ev["kind"] == "order" and ev["status"] == "filled"
    assert parts.broker.submitted == [dict(ticker="QQQ", side="buy", cid="ab-20261022T1505",
                                           usd=15.0, qty=None)]
    assert ev["slippage_pct"] == pytest.approx(0.05)
    assert ev["decision"]["reason"] == "test buy"
    assert any(k == "trade" and "BUY QQQ $15.00" in m for k, m in parts.alerts)


def test_hold_places_nothing(parts):
    parts.decider.next = Decision("hold", "NONE", 0.0, "quiet market")
    ev = make(parts).run_cycle()
    assert ev["kind"] == "decision" and not parts.broker.submitted


def test_sell_of_whole_position_uses_qty(parts):
    parts.broker.held = {"SPY": {"usd": 18.0, "pl_pct": 1.0, "qty": 0.18}}
    parts.decider.next = Decision("sell", "SPY", 17.8, "take profit")
    make(parts).run_cycle()
    assert parts.broker.submitted[0]["qty"] == 0.18 and parts.broker.submitted[0]["usd"] is None


def test_sell_over_per_trade_cap_is_rejected(parts):
    # guardrails cap sells at $20 too, so a $40 position is sold in two orders
    parts.broker.held = {"SPY": {"usd": 40.0, "pl_pct": 1.0, "qty": 0.4}}
    parts.decider.next = Decision("sell", "SPY", 40.0, "exit")
    assert make(parts).run_cycle()["kind"] == "rejected"


def test_partial_sell_uses_dollars(parts):
    parts.broker.held = {"SPY": {"usd": 40.0, "pl_pct": 1.0, "qty": 0.4}}
    parts.decider.next = Decision("sell", "SPY", 20.0, "trim")
    make(parts).run_cycle()
    assert parts.broker.submitted[0]["usd"] == 20.0


# --- holds on bad inputs -------------------------------------------------------
def test_market_closed_is_idle_and_makes_no_ai_call(parts):
    parts.data.exc = MarketClosed("closed")
    assert make(parts).run_cycle()["kind"] == "idle"
    assert parts.decider.calls == 0 and not parts.alerts


@pytest.mark.parametrize("exc", [StaleDataError("old quotes"), BrokerError("api down")])
def test_bad_data_holds_and_alerts(parts, exc):
    parts.data.exc = exc
    assert make(parts).run_cycle()["kind"] == "hold"
    assert parts.decider.calls == 0 and parts.alerts[0][0] == "error"


def test_unreadable_equity_holds(parts):
    parts.broker._equity = None
    assert make(parts).run_cycle()["kind"] == "hold"
    assert parts.data.calls == 0 and not parts.broker.submitted


def test_invalid_ai_reply_holds_and_alerts(parts):
    parts.decider.error = "not JSON"
    ev = make(parts).run_cycle()
    assert ev["kind"] == "invalid" and not parts.broker.submitted
    assert "invalid AI reply" in parts.alerts[0][1]


def test_budget_exhausted_holds(parts):
    parts.decider.exc = BudgetExhausted("daily AI budget $1.00 reached")
    assert make(parts).run_cycle()["kind"] == "hold"
    assert not parts.broker.submitted


def test_guardrail_rejection_places_nothing(parts):
    parts.decider.next = Decision("buy", "QQQ", 25.0, "too big")
    ev = make(parts).run_cycle()
    assert ev["kind"] == "rejected" and "per-trade cap" in ev["why"]
    assert not parts.broker.submitted


def test_repeated_alerts_are_deduplicated(parts):
    parts.data.exc = StaleDataError("old quotes")
    e = make(parts)
    e.run_cycle(); e.run_cycle(); e.run_cycle()
    assert len(parts.alerts) == 1


# --- floor ------------------------------------------------------------------------
def test_floor_breach_liquidates_halts_and_persists(parts):
    parts.broker.held = {"SPY": {"usd": 40.0, "pl_pct": -50.0, "qty": 0.4}}
    parts.broker._equity = 199.0
    e = make(parts)
    assert e.run_cycle()["kind"] == "halt"
    assert parts.broker.closed_all == 1 and parts.decider.calls == 0
    assert any(k == "halt" for k, _ in parts.alerts)
    assert parts.store.load().halted

    # restart: still halted even though equity looks fine now
    parts.broker._equity = 300.0
    e2 = make(parts)
    assert e2.run_cycle()["kind"] == "halted"
    assert parts.decider.calls == 0 and not parts.broker.submitted
    e2.resume()
    assert not parts.store.load().halted
    assert e2.run_cycle()["kind"] == "order"


def test_halted_retries_liquidation_until_flat(parts):
    e = make(parts)
    e.halt("owner halt")
    parts.broker.held = {"SPY": {"usd": 40.0, "pl_pct": 0.0, "qty": 0.4}}  # sell didn't stick
    e.run_cycle()
    assert parts.broker.closed_all == 2   # once on halt, once retried


def test_pause_and_resume(parts):
    e = make(parts)
    e.pause()
    assert make(parts).run_cycle()["kind"] == "paused"   # survives restart
    e2 = make(parts); e2.resume()
    assert e2.run_cycle()["kind"] == "order"


# --- no double orders ----------------------------------------------------------------
def test_restart_in_same_slot_does_not_resubmit(parts):
    make(parts).run_cycle()
    ev = make(parts).run_cycle()       # crash + restart, same 5-minute slot
    assert ev["kind"] == "skip" and len(parts.broker.submitted) == 1
    assert parts.decider.calls == 1    # no second AI call either


def test_submit_error_but_order_landed_is_not_resubmitted(parts):
    parts.broker.submit_error = TimeoutError("read timeout")
    parts.broker.lands_anyway = True
    ev = make(parts).run_cycle()
    assert ev["kind"] == "order" and ev["status"] == "filled"
    assert len(parts.broker.submitted) == 1


def test_submit_error_and_no_order_reports_failure(parts):
    parts.broker.submit_error = TimeoutError("read timeout")
    ev = make(parts).run_cycle()
    assert ev["kind"] == "order_failed"
    assert any("failed" in m for _, m in parts.alerts)


def test_open_order_blocks_new_decisions(parts):
    parts.broker.open = [NS(id="o9", symbol="SPY", client_order_id="ab-x", side="buy",
                            submitted_at=NOW - timedelta(minutes=1))]
    assert make(parts).run_cycle()["kind"] == "skip"
    assert parts.decider.calls == 0 and not parts.broker.cancelled_ids


def test_stuck_open_order_is_cancelled(parts):
    parts.broker.open = [NS(id="o9", symbol="SPY", client_order_id="ab-x", side="buy",
                            submitted_at=NOW - timedelta(minutes=30))]
    make(parts).run_cycle()
    assert parts.broker.cancelled_ids == ["o9"]


def test_unfilled_order_is_reported_not_retried(parts):
    parts.broker.fill_status = "new"
    ev = make(parts).run_cycle()
    assert ev["status"] == "new" and len(parts.broker.submitted) == 1


# --- hourly cap and AI spend survive restarts -------------------------------------------
def test_hourly_cap_survives_restart(parts):
    import time as _t
    parts.store.save(BotState(order_times=[_t.time() - 60] * 12))
    ev = make(parts).run_cycle()
    assert ev["kind"] == "rejected" and "hourly" in ev["why"]


def test_ai_spend_survives_restart(parts):
    e = make(parts)
    e.run_cycle()
    assert parts.store.load().ai_spend
    e2 = make(parts)
    assert e2.decider.budget.spent


# --- stored state ----------------------------------------------------------------------------
def test_corrupt_state_refuses_to_start(parts):
    parts.store.db._write("INSERT INTO state (id, data, updated) VALUES (1, '{not json', '')")
    with pytest.raises(StateError):
        make(parts)


def test_unexpected_exception_does_not_crash_loop(parts):
    parts.decider.exc = RuntimeError("boom")
    ev = make(parts).run_cycle()
    assert ev["kind"] == "error" and "boom" in parts.alerts[0][1]
