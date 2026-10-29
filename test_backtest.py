"""Replay test plumbing: no look-ahead, the simulated cash account, earnings."""
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace as NS

import pytest
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from backtest import (ACTIVE_PARAGRAPH, CAUTIOUS_PARAGRAPH, SPREAD_PCT, VARIANTS, NasdaqEarnings,
                      ReplayBars, SimAccount, SimClock, max_drawdown, ticker_views)
from data import ET

DAY = date(2026, 10, 27)


def at(h, m, d=DAY):
    return datetime(d.year, d.month, d.day, h, m, tzinfo=ET)


def five_bars(d=DAY, n=78, start=100.0):
    return [NS(timestamp=at(9, 30, d) + timedelta(minutes=5 * i), open=start + i, high=start + i + 1,
               low=start + i - 1, close=start + i + 0.5) for i in range(n)]


def daily_bars(days=10, end=DAY):
    return [NS(timestamp=datetime(end.year, end.month, end.day, tzinfo=ET) - timedelta(days=days - i),
               close=90.0 + i) for i in range(days + 1)]   # the last one is `end` itself


# --- no look-ahead -------------------------------------------------------------------
def test_five_minute_bars_are_clipped_to_the_clock():
    clock = SimClock()
    clock.t = at(10, 0)
    rb = ReplayBars({}, {"SPY": five_bars()}, clock)
    got = rb.get_stock_bars(StockBarsRequest(symbol_or_symbols=["SPY"],
                                             timeframe=TimeFrame(5, TimeFrameUnit.Minute)))
    bars = got.data["SPY"]
    assert bars[-1].timestamp == at(9, 55)   # the 10:00 bar isn't complete at 10:00
    assert len(bars) == 6


def test_todays_daily_bar_is_never_served():
    clock = SimClock()
    clock.t = at(15, 59)
    rb = ReplayBars({"SPY": daily_bars()}, {}, clock)
    got = rb.get_stock_bars(StockBarsRequest(symbol_or_symbols=["SPY"], timeframe=TimeFrame.Day))
    assert all(b.timestamp.astimezone(ET).date() < DAY for b in got.data["SPY"])


def test_request_end_is_respected():
    clock = SimClock()
    clock.t = at(15, 0)
    rb = ReplayBars({}, {"SPY": five_bars()}, clock)
    got = rb.get_stock_bars(StockBarsRequest(symbol_or_symbols=["SPY"],
                                             timeframe=TimeFrame(5, TimeFrameUnit.Minute),
                                             start=at(9, 30), end=at(9, 45)))
    assert len(got.data["SPY"]) == 3


def test_ticker_views_use_last_completed_bar():
    views = ticker_views({"SPY": five_bars()}, {"SPY": daily_bars()}, at(10, 0), ["SPY"])
    v = views["SPY"]
    assert v.price == 105.5                       # the 9:55 bar's close
    assert v.chg_today_pct == pytest.approx((105.5 / 99.0 - 1) * 100)   # vs yesterday's close


# --- simulated cash account --------------------------------------------------------
def account(prices, today):
    return SimAccount(prices=lambda: prices, today=lambda: today["d"],
                      next_session=lambda d: d + timedelta(days=1))


def test_buy_pays_half_the_spread_and_uses_cash():
    p, today = {"SPY": 100.0}, {"d": DAY}
    a = account(p, today)
    a.buy("SPY", 20.0)
    assert a.settled_cash() == pytest.approx(280.0)
    assert a.qty["SPY"] == pytest.approx(20.0 / (100 * (1 + SPREAD_PCT / 200)))
    assert a.equity() == pytest.approx(300.0 - a.spread_paid)


def test_sale_proceeds_settle_next_session():
    p, today = {"SPY": 100.0}, {"d": DAY}
    a = account(p, today)
    a.buy("SPY", 20.0)
    a.sell("SPY", qty=a.qty["SPY"])
    assert a.settled_cash() == pytest.approx(280.0)        # not yet settled
    assert a.equity() == pytest.approx(300.0 - a.spread_paid)
    today["d"] = DAY + timedelta(days=1)
    assert a.settled_cash() == pytest.approx(300.0 - a.spread_paid)
    assert not a.qty


def test_partial_sell_keeps_cost_basis_proportional():
    p, today = {"SPY": 100.0}, {"d": DAY}
    a = account(p, today)
    a.buy("SPY", 40.0)
    a.sell("SPY", usd=20.0)
    assert a.cost["SPY"] == pytest.approx(20.0, rel=1e-3)
    p["SPY"] = 110.0
    assert a.positions()["SPY"]["pl_pct"] == pytest.approx(10.0 - SPREAD_PCT / 2, abs=0.01)


def test_close_all_positions_sells_everything():
    p, today = {"SPY": 100.0, "QQQ": 50.0}, {"d": DAY}
    a = account(p, today)
    a.buy("SPY", 20.0)
    a.buy("QQQ", 10.0)
    a.close_all_positions()
    assert not a.qty and a.trades == 4


# --- earnings, prompts, report math ----------------------------------------------------
def test_nasdaq_earnings_next_date_as_of_the_clock(tmp_path):
    clock = SimClock()
    cal = {date(2026, 10, 28): [("MSFT", "time-after-hours"), ("IBM", "time-after-hours")],
           date(2026, 10, 30): [("AAPL", "time-pre-market")]}
    e = NasdaqEarnings({"MSFT", "AAPL"}, date(2026, 10, 26), date(2026, 10, 31),
                       tmp_path / "c.json", clock, fetch=lambda d: cal.get(d, []))
    clock.t = at(12, 0)
    assert e.next_dates({"MSFT", "AAPL"}) == {"MSFT": (date(2026, 10, 28), "post-market"),
                                              "AAPL": (date(2026, 10, 30), "pre-market")}
    clock.t = at(12, 0, date(2026, 10, 29))
    assert e.next_dates({"MSFT", "AAPL"}) == {"AAPL": (date(2026, 10, 30), "pre-market")}
    # cached: a second load makes no requests
    NasdaqEarnings({"MSFT"}, date(2026, 10, 26), date(2026, 10, 31), tmp_path / "c.json", clock,
                   fetch=lambda d: pytest.fail("fetched again"))


def test_variants_differ_only_in_the_holding_paragraph():
    assert CAUTIOUS_PARAGRAPH in VARIANTS["cautious"]
    assert ACTIVE_PARAGRAPH in VARIANTS["active"] and CAUTIOUS_PARAGRAPH not in VARIANTS["active"]
    assert VARIANTS["active"].replace(ACTIVE_PARAGRAPH, CAUTIOUS_PARAGRAPH) == VARIANTS["cautious"]


def test_max_drawdown():
    assert max_drawdown([300, 330, 297, 310]) == pytest.approx(-10.0)
    assert max_drawdown([300, 301, 302]) == 0.0


def test_checkpoint_roundtrip(tmp_path):
    from backtest import Variant, load_checkpoint, save_checkpoint
    from guardrails import Guardrails, Limits
    p, today = {"SPY": 100.0}, {"d": DAY}
    def variant():
        a = account(p, today)
        return Variant("cautious", None, a, Guardrails(a, Limits()))
    v = variant()
    v.account.buy("SPY", 20.0)
    v.account.unsettled.append((DAY + timedelta(days=1), 5.0))
    v.rows.append({"time": "t", "outcome": "filled"})
    v.counts["buy"] = 1
    v.g._order_times = [123.0]
    replay = [DAY, DAY + timedelta(days=1)]
    save_checkpoint(tmp_path / "c.json", [v], p, 2.9, [(DAY, {"cautious": 299.9}, 100.0)], replay, 5)
    w = variant()
    prices = {}
    spy, curve, done = load_checkpoint(tmp_path / "c.json", [w], prices, replay, 5)
    assert spy == 2.9 and done == {DAY} and prices == {"SPY": 100.0}
    assert w.account.qty == v.account.qty and w.account.cash == v.account.cash
    assert w.account.unsettled == [(DAY + timedelta(days=1), 5.0)]
    assert w.rows == v.rows and w.counts["buy"] == 1 and w.g._order_times == [123.0]
    with pytest.raises(SystemExit):
        load_checkpoint(tmp_path / "c.json", [variant()], {}, replay, 15)
