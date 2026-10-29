"""QLD trend strategy: the signal, order sizing, the watchdog, and the engine
running it through the real Guardrails. No network.

The engine tests use the limits the strategy needs (QLD allowed, $300 per
ticker), which are also guardrails.py's defaults since 2026-10-28."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS

import pytest

from data import ET, Snapshot, TickerView
from decision import DailyBudget, Decision, DecisionResult, DecisionUnavailable
from engine import Engine
from guardrails import Guardrails, Limits
from test_engine import NOW, FakeBroker, FakeData
from trend import CASH_BUFFER, HOLD, SIGNAL, TrendStrategy

LIMITS = Limits(allowed_tickers={"QQQ", "QLD", "SPY"}, max_ticker_usd=300.0)


class FakeBars:
    """Daily QQQ closes: 199 at 100, then `last` (yesterday). Today's bar is
    also returned, to check it's ignored."""
    def __init__(self, last=110.0, n=200, exc=None):
        self.last, self.n, self.exc, self.requests = last, n, exc, []
    def get_stock_bars(self, req):
        self.requests.append(req)
        if self.exc:
            raise self.exc
        day0 = NOW.astimezone(ET).replace(hour=0, minute=0) - timedelta(days=self.n)
        closes = [100.0] * (self.n - 1) + [self.last]
        bars = [NS(timestamp=day0 + timedelta(days=i), close=c) for i, c in enumerate(closes)]
        bars.append(NS(timestamp=NOW, close=1.0))   # today's partial bar
        return NS(data={SIGNAL: bars})


class FakeWatchdog:
    def __init__(self, reply="approve"):
        self.budget, self.reply, self.prompts = DailyBudget(1.0), reply, []
    def decide(self, snap):
        self.prompts.append(snap.to_prompt())
        usd = float(snap.context.split("$")[-1].split(" ")[0])
        d = {"approve": Decision("buy", HOLD, usd, "data looks normal"),
             "veto": Decision("hold", "NONE", 0.0, "QQQ down 4% today"),
             "change": Decision("buy", HOLD, 5.0, "smaller"),
             "invalid": None}[self.reply]
        return DecisionResult(d, 0.0004, {"input": 900, "output": 40}, "req", "{}",
                              "" if d else "bad json")


def snap(cash=300.0, positions=None, tickers=("QQQ", "QLD")):
    tv = lambda s: TickerView(s, 100.0, 0.02, 0.1, 0.0, 1.0, 3.0)
    return Snapshot(NOW, NOW.replace(hour=20), cash, cash, positions or {},
                    {s: tv(s) for s in tickers}, {}, "context")


def strat(last=110.0, reply="approve", **kw):
    return TrendStrategy(FakeBars(last, **kw), FakeWatchdog(reply), LIMITS, now=lambda: NOW)


# --- signal ------------------------------------------------------------------------
def test_signal_above_average_is_risk_on_and_ignores_todays_bar():
    s = strat(last=110.0)
    on, close, avg = s.signal()
    assert on and close == 110.0 and avg == pytest.approx(100.05)
    req = s.bars.requests[0]
    assert req.feed.value == "sip" and req.adjustment.value == "all"


def test_signal_below_average_is_risk_off():
    assert not strat(last=90.0).signal()[0]


def test_signal_fetched_once_a_day():
    s = strat()
    s.signal(); s.signal()
    assert len(s.bars.requests) == 1


def test_too_little_history_is_unavailable():
    with pytest.raises(DecisionUnavailable):
        strat(n=150).signal()


def test_data_error_is_unavailable():
    with pytest.raises(DecisionUnavailable):
        strat(exc=RuntimeError("down")).signal()


# --- sizing ----------------------------------------------------------------------------
def test_risk_on_buys_up_to_the_per_trade_cap():
    r = strat().decide(snap(cash=300.0))
    assert (r.decision.action, r.decision.ticker, r.decision.usd) == ("buy", HOLD, 20.0)
    assert "watchdog: data looks normal" in r.decision.reason
    assert r.cost_usd == 0.0004


def test_buy_leaves_a_cash_buffer():
    r = strat().decide(snap(cash=12.345))
    assert r.decision.usd == round(12.345 - CASH_BUFFER - 0.005, 2)   # 11.34, rounded down


def test_fully_invested_holds_without_asking_claude():
    s = strat()
    r = s.decide(snap(cash=0.80, positions={HOLD: {"usd": 299.0, "qty": 2.99, "pl_pct": 0}}))
    assert r.decision.action == "hold" and "fully invested" in r.decision.reason
    assert not s.watchdog.prompts


def test_per_ticker_cap_limits_the_last_buy():
    r = strat().decide(snap(cash=100.0, positions={HOLD: {"usd": 290.0, "qty": 2.9, "pl_pct": 0}}))
    assert r.decision.usd == 10.0


def test_risk_off_sells_in_capped_steps_without_asking_claude():
    s = strat(last=90.0)
    r = s.decide(snap(cash=0.0, positions={HOLD: {"usd": 75.5, "qty": 0.75, "pl_pct": 0}}))
    assert (r.decision.action, r.decision.usd) == ("sell", 20.0)
    r = s.decide(snap(cash=0.0, positions={HOLD: {"usd": 7.25, "qty": 0.07, "pl_pct": 0}}))
    assert r.decision.usd == 7.25
    assert not s.watchdog.prompts


def test_risk_off_in_cash_holds():
    r = strat(last=90.0).decide(snap())
    assert r.decision.action == "hold" and "risk off, in cash" in r.decision.reason


def test_no_qld_quote_holds_and_says_why():
    r = strat().decide(snap(tickers=("QQQ",)))
    assert r.decision.action == "hold" and "no usable quote" in r.decision.reason
    no_qld = Limits(allowed_tickers={"QQQ", "SPY"})
    s = TrendStrategy(FakeBars(), FakeWatchdog(), no_qld, now=lambda: NOW)
    assert "not in the guardrails' allowed tickers" in s.decide(snap(tickers=("QQQ",))).decision.reason


# --- watchdog ------------------------------------------------------------------------
def test_watchdog_sees_the_proposed_order():
    s = strat()
    s.decide(snap())
    assert "PROPOSED ORDER from the rule: buy $20.00 of QLD" in s.watchdog.prompts[0]


def test_watchdog_veto_becomes_a_hold():
    r = strat(reply="veto").decide(snap())
    assert r.decision.action == "hold" and r.decision.reason == "watchdog veto: QQQ down 4% today"


def test_watchdog_changing_the_order_is_rejected():
    r = strat(reply="change").decide(snap())
    assert r.decision is None and "instead of approving" in r.error


def test_watchdog_invalid_reply_passes_through_as_invalid():
    r = strat(reply="invalid").decide(snap())
    assert r.decision is None and r.error == "bad json"


# --- through the real engine and guardrails ---------------------------------------------
@pytest.fixture
def run(tmp_path):
    from storage import Database, SqliteStateStore

    def make(last=110.0, reply="approve", cash=300.0, held=None):
        b = FakeBroker()
        b.cash = cash
        b.held = held or {}
        data = FakeData()
        data.snap = None
        tv = lambda s: TickerView(s, 100.0, 0.02, 0.1, 0.0, 1.0, 3.0)
        data.snapshot = lambda broker, tickers: Snapshot(
            NOW, NOW.replace(hour=20), broker.equity(), broker.settled_cash(), broker.positions(),
            {s: tv(s) for s in ("QQQ", "QLD")}, {}, "context")
        alerts = []
        g = Guardrails(b, LIMITS)
        e = Engine(b, data, strat(last, reply), g,
                   SqliteStateStore(Database(str(tmp_path / f"{last}{reply}{cash}.db"))),
                   ["QQQ", "QLD"], notify=lambda k, m: alerts.append((k, m)),
                   now=lambda: NOW, sleep=lambda s: None)
        return NS(engine=e, broker=b, alerts=alerts)
    return make


def test_engine_buys_qld_through_the_guardrails(run):
    r = run()
    ev = r.engine.run_cycle()
    assert ev["kind"] == "order" and ev["status"] == "filled"
    assert r.broker.submitted[0] == dict(ticker="QLD", side="buy", cid="ab-20261022T1505",
                                         usd=20.0, qty=None)


def test_engine_sells_whole_small_position_by_qty(run):
    r = run(last=90.0, held={"QLD": {"usd": 15.0, "pl_pct": 0.0, "qty": 0.15}})
    ev = r.engine.run_cycle()
    assert ev["kind"] == "order" and r.broker.submitted[0]["qty"] == 0.15


def test_engine_veto_places_nothing(run):
    r = run(reply="veto")
    ev = r.engine.run_cycle()
    assert ev["kind"] == "decision" and not r.broker.submitted


def test_engine_floor_still_liquidates(run):
    r = run(held={"QLD": {"usd": 150.0, "pl_pct": -40.0, "qty": 1.5}})
    r.broker._equity = 199.0
    ev = r.engine.run_cycle()
    assert ev["kind"] == "halt" and r.broker.closed_all == 1 and not r.broker.submitted


def test_default_limits_fit_the_strategy():
    """guardrails.py as the owner set it on 2026-10-28: QLD allowed, $300 per ticker,
    everything else unchanged."""
    lim = Limits()
    assert HOLD in lim.allowed_tickers and lim.max_ticker_usd == 300.0
    assert (lim.equity_floor, lim.max_trade_usd, lim.max_orders_per_hour) == (200.0, 20.0, 12)
