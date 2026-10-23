"""Unit tests for the data layer with fake Alpaca clients. No network."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS

import pytest

from broker import BrokerDataError
from data import ET, MarketClosed, MarketData, StaleDataError

NOW = datetime(2026, 10, 22, 15, 0, tzinfo=timezone.utc)  # 11:00 ET, Thursday


def quote(bid=100.0, ask=100.02, age_s=5):
    return NS(bid_price=bid, ask_price=ask, timestamp=NOW - timedelta(seconds=age_s))


def snap(q=None, prev_close=99.0):
    return NS(latest_quote=q or quote(), previous_daily_bar=NS(close=prev_close))


def day_bar(days_ago, close):
    d = (NOW.astimezone(ET) - timedelta(days=days_ago)).replace(hour=0, minute=0)
    return NS(timestamp=d, close=close, open=close)


class FakeData:
    def __init__(self, snaps, intraday=None, daily=None):
        self.snaps, self.intraday, self.daily = snaps, intraday or {}, daily or {}
        self.requests = []

    def get_stock_snapshot(self, req):
        self.requests.append(req)
        return self.snaps

    def get_stock_bars(self, req):
        self.requests.append(req)
        return NS(data=self.daily if req.timeframe.unit_value.value == "Day" else self.intraday)


class FakeBroker:
    def __init__(self, is_open=True, equity=300.0):
        self.is_open, self._equity = is_open, equity
    def clock(self):
        return NS(is_open=self.is_open, next_open="tomorrow",
                  next_close=NOW.replace(hour=20), timestamp=NOW)
    def equity(self):
        if self._equity is None:
            raise BrokerDataError("equity missing")
        return self._equity
    def settled_cash(self): return 250.0
    def positions(self): return {"SPY": {"usd": 40.0, "pl_pct": 1.5}}


def md(data):
    return MarketData(data, now=lambda: NOW)


def test_snapshot_happy_path():
    data = FakeData(
        {"SPY": snap(), "QQQ": snap(quote(200, 200.04), prev_close=202)},
        intraday={"SPY": [NS(open=99.5)]},
        daily={"SPY": [day_bar(d, 95 + d) for d in (8, 7, 6, 5, 2, 1, 0)]})
    s = md(data).snapshot(FakeBroker(), ["SPY", "QQQ"])
    spy = s.tickers["SPY"]
    assert spy.price == pytest.approx(100.01)
    assert spy.chg_today_pct == pytest.approx((100.01 / 99 - 1) * 100)
    assert spy.chg_1h_pct == pytest.approx((100.01 / 99.5 - 1) * 100)
    # completed sessions are days 8,7,6,5,2,1; five sessions back is day 7 (close 102)
    assert spy.chg_5d_pct == pytest.approx((100.01 / 102 - 1) * 100)
    assert s.tickers["QQQ"].chg_1h_pct is None and s.tickers["QQQ"].chg_5d_pct is None
    assert s.equity == 300 and s.spendable_cash == 250 and not s.excluded
    assert all(r.feed.value == "iex" for r in data.requests)


def test_prompt_text_is_compact_and_complete():
    data = FakeData({"SPY": snap(), "QQQ": snap(quote(age_s=999))})
    text = md(data).snapshot(FakeBroker(), ["SPY", "QQQ"]).to_prompt()
    assert "11:00 ET" in text and "closes 16:00 ET" in text
    assert "equity $300.00, spendable cash $250.00" in text
    assert "SPY $40.00 (+1.5%)" in text
    assert "\nSPY 100.01 " in text
    assert "not tradable now: QQQ" in text
    assert len(text) < 1000


def test_market_closed():
    with pytest.raises(MarketClosed):
        md(FakeData({})).snapshot(FakeBroker(is_open=False), ["SPY"])


def test_unreadable_account_propagates():
    with pytest.raises(BrokerDataError):
        md(FakeData({"SPY": snap()})).snapshot(FakeBroker(equity=None), ["SPY"])


@pytest.mark.parametrize("bad,why", [
    (None, "no quote"),
    (quote(age_s=300), "old"),
    (quote(bid=0), "bad quote"),
    (quote(bid=float("nan")), "bad quote"),
    (quote(bid=101, ask=100), "bad quote"),
    (quote(bid=100, ask=101), "too wide"),
])
def test_bad_ticker_is_excluded(bad, why):
    snaps = {"SPY": snap(), "QQQ": NS(latest_quote=bad, previous_daily_bar=None)}
    s = md(FakeData(snaps)).snapshot(FakeBroker(), ["SPY", "QQQ"])
    assert "QQQ" not in s.tickers and why in s.excluded["QQQ"]
    assert "SPY" in s.tickers


def test_missing_ticker_is_excluded():
    s = md(FakeData({"SPY": snap()})).snapshot(FakeBroker(), ["SPY", "QQQ"])
    assert "QQQ" in s.excluded


def test_all_stale_raises():
    with pytest.raises(StaleDataError):
        md(FakeData({"SPY": snap(quote(age_s=600))})).snapshot(FakeBroker(), ["SPY"])
