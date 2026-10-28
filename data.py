"""Data layer: one compact, validated snapshot of market + account state per cycle.

Market data comes from Alpaca's free (Basic) plan, which is real-time for the
IEX exchange only, so every request asks for feed=iex. Prices are the IEX
bid/ask midpoint; the spread is reported so the bot can account for the fact
that fractional orders fill at the quote with no price improvement.

Failure rules (from CLAUDE.md: stale data or unreadable account -> hold):
- Account reads raise BrokerError/BrokerDataError; this module lets them propagate.
- Market closed -> MarketClosed.
- A ticker with a missing, invalid or stale quote is left out of the snapshot
  and listed in `excluded`, so it can't be traded this cycle.
- If every ticker is excluded -> StaleDataError.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockSnapshotRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

ET = ZoneInfo("America/New_York")


class MarketClosed(Exception):
    pass


class StaleDataError(Exception):
    pass


@dataclass(frozen=True)
class TickerView:
    symbol: str
    price: float          # bid/ask midpoint
    spread_pct: float     # (ask - bid) / mid * 100
    chg_today_pct: float | None
    chg_1h_pct: float | None
    chg_5d_pct: float | None
    quote_age_s: float


@dataclass(frozen=True)
class Snapshot:
    as_of: datetime
    next_close: datetime
    equity: float
    spendable_cash: float
    positions: dict           # {symbol: {"usd": float, "pl_pct": float}}
    tickers: dict             # {symbol: TickerView}
    excluded: dict = field(default_factory=dict)  # {symbol: reason}
    context: str = ""                              # history / earnings / news section
    context_problems: tuple = ()                   # parts of the context that failed

    def to_prompt(self) -> str:
        """Short, fixed-format text for the decision step."""
        t = self.as_of.astimezone(ET)
        close = self.next_close.astimezone(ET)
        lines = [
            f"time {t:%Y-%m-%d %H:%M} ET, market open, closes {close:%H:%M} ET "
            f"({_until(self.next_close - self.as_of)} from now)",
            f"equity ${self.equity:.2f}, spendable cash ${self.spendable_cash:.2f}",
        ]
        if self.positions:
            held = ", ".join(f"{s} ${p['usd']:.2f} ({p['pl_pct']:+.1f}%)"
                             for s, p in sorted(self.positions.items()))
            lines.append(f"positions: {held}")
        else:
            lines.append("positions: none")
        lines.append("ticker price spread% today% 1h% 5d%")
        for s in sorted(self.tickers):
            v = self.tickers[s]
            lines.append(f"{s} {v.price:.2f} {v.spread_pct:.2f} {_pct(v.chg_today_pct)} "
                         f"{_pct(v.chg_1h_pct)} {_pct(v.chg_5d_pct)}")
        if self.excluded:
            lines.append("not tradable now: " + ", ".join(sorted(self.excluded)))
        if self.context:
            lines.append(self.context)
        return "\n".join(lines)


def _until(d: timedelta) -> str:
    """'2 h 05 min' / '35 min': stated outright so the model doesn't do clock math."""
    m = max(0, int(d.total_seconds() // 60))
    return f"{m // 60} h {m % 60:02d} min" if m >= 60 else f"{m} min"


def _pct(x):
    return "n/a" if x is None else f"{x:+.2f}"


def _pos(x) -> float | None:
    """Finite positive float, else None."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) and x > 0 else None


def _change(now: float, then) -> float | None:
    then = _pos(then)
    return None if then is None else (now / then - 1) * 100


class MarketData:
    def __init__(self, client: StockHistoricalDataClient, *,
                 max_quote_age: timedelta = timedelta(minutes=2),
                 max_spread_pct: float = 0.5, context=None,
                 now=lambda: datetime.now(timezone.utc)):
        self.client = client
        self.context = context   # optional ContextProvider (history, earnings, news)
        self.max_quote_age = max_quote_age
        self.max_spread_pct = max_spread_pct
        self.now = now

    @classmethod
    def from_env(cls, env=os.environ, **kw) -> "MarketData":
        return cls(StockHistoricalDataClient(env["ALPACA_API_KEY"], env["ALPACA_SECRET_KEY"]), **kw)

    def snapshot(self, broker, tickers) -> Snapshot:
        clock = broker.clock()
        if not clock.is_open:
            raise MarketClosed(f"market closed, next open {clock.next_open}")
        now = self.now()
        tickers = sorted(tickers)

        # account first: if this fails nothing else matters
        equity = broker.equity()
        cash = broker.settled_cash()
        positions = broker.positions()

        snaps = self.client.get_stock_snapshot(
            StockSnapshotRequest(symbol_or_symbols=tickers, feed=DataFeed.IEX))
        intraday = self._bars(tickers, TimeFrame(5, TimeFrameUnit.Minute), now - timedelta(minutes=70))
        daily = self._bars(tickers, TimeFrame.Day, now - timedelta(days=10))

        views, excluded = {}, {}
        for s in tickers:
            view, why = self._view(s, snaps.get(s), intraday.get(s, []), daily.get(s, []), now)
            if view:
                views[s] = view
            else:
                excluded[s] = why
        if not views:
            raise StaleDataError("no usable quotes: " + "; ".join(f"{s}: {w}" for s, w in excluded.items()))
        ctx, problems = "", ()
        if self.context is not None:
            ctx = self.context.text({s: v.price for s, v in views.items()})
            problems = tuple(self.context.problems)
        return Snapshot(now, clock.next_close, equity, cash, positions, views, excluded,
                        ctx, problems)

    def _bars(self, tickers, timeframe, start) -> dict:
        bs = self.client.get_stock_bars(StockBarsRequest(
            symbol_or_symbols=tickers, timeframe=timeframe, start=start, feed=DataFeed.IEX))
        return getattr(bs, "data", bs) or {}

    def _view(self, s, snap, intraday, daily, now):
        q = getattr(snap, "latest_quote", None)
        if q is None:
            return None, "no quote"
        bid, ask = _pos(q.bid_price), _pos(q.ask_price)
        if bid is None or ask is None or ask < bid:
            return None, f"bad quote bid={q.bid_price} ask={q.ask_price}"
        age = (now - q.timestamp).total_seconds()
        if age > self.max_quote_age.total_seconds():
            return None, f"quote {age:.0f}s old"
        mid = (bid + ask) / 2
        spread = (ask - bid) / mid * 100
        if spread > self.max_spread_pct:
            return None, f"spread {spread:.2f}% too wide"

        prev = getattr(snap, "previous_daily_bar", None)
        today = _change(mid, prev.close) if prev else None
        hour = _change(mid, intraday[0].open) if intraday else None
        # use only completed sessions, whether or not Alpaca includes today's partial bar
        today_et = now.astimezone(ET).date()
        past = [b for b in daily if b.timestamp.astimezone(ET).date() < today_et]
        week = _change(mid, past[-5].close) if len(past) >= 5 else None
        return TickerView(s, round(mid, 4), round(spread, 4), today, hour, week, age), ""
