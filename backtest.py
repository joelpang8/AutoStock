"""Replay test: run the bot's real decision step over past trading days.

    python backtest.py --days 20 --every 15 --out results/

For each past session and each decision time (every --every minutes from 9:45
to 15:45 ET) it builds the same prompt the live bot would have built at that
moment, from data available at that moment only, asks Claude, and runs the
reply through the real Guardrails against a simulated cash account. Two prompt
variants run side by side on identical market data:

  cautious  the live prompt ("Holding is the default...")
  active    the same prompt with that paragraph replaced by one that counts
            missed market gains as a cost too

and both are compared with holding SPY and with holding cash.

Simulation, not a broker:
- Prices are the close of the last completed 5-minute IEX bar. Buys fill at
  that price plus half the assumed spread, sells at it minus half (fractional
  orders fill at the quote, with no price improvement).
- Sale proceeds settle the next trading day (cash account, T+1), and the
  guardrails only let settled cash buy.
- No look-ahead: every data request is clipped to bars complete by the decision
  time, and news to headlines published before it. Past earnings dates come from
  Nasdaq's public earnings calendar.
- The model may have seen some of this period in training; it can't see the
  future of the replay itself, but treat the result as a rough check, not proof.

A checkpoint is saved in the output folder after every session; running the
same command again resumes after the last finished one.

Costs real Claude calls: about $0.0004 each, so 20 days x 25 times x 2
variants is about $0.40. Places no orders anywhere.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace as NS

from alpaca.data.timeframe import TimeFrameUnit

import guardrails as guardrails_module
from context import ContextProvider
from data import ET, Snapshot, TickerView
from decision import SYSTEM_PROMPT, DailyBudget, Decider, DecisionUnavailable
from guardrails import Guardrails, Limits

log = logging.getLogger("backtest")

SPREAD_PCT = 0.05            # assumed full bid/ask spread, % of price (IEX showed 0.01-0.18%)
BAR = timedelta(minutes=5)
START_CASH = 300.0
SELL_ALL_MARGIN_USD = 0.50   # same rule as engine.py

CAUTIOUS_PARAGRAPH = """\
Holding is the default. Trading costs money through spreads, and frequent
trading usually underperforms simply holding an index ETF. Only buy or sell
when you have a specific reason you can state in one line."""
ACTIVE_PARAGRAPH = """\
Each turn, pick the action you expect to leave the account best off over the
coming days after spread costs. Holding cash is not free either: it misses
whatever the market gains. Trade whenever you have a reason you can state in
one line."""
assert CAUTIOUS_PARAGRAPH in SYSTEM_PROMPT, "live prompt changed: update backtest.py"
VARIANTS = {"cautious": SYSTEM_PROMPT,
            "active": SYSTEM_PROMPT.replace(CAUTIOUS_PARAGRAPH, ACTIVE_PARAGRAPH)}


class SimClock:
    def __init__(self):
        self.t: datetime | None = None

    def now(self) -> datetime:
        return self.t


# --- market data, clipped to the simulated clock --------------------------------
def _aware(t):
    """alpaca-py request models store times as naive UTC."""
    return t.replace(tzinfo=timezone.utc) if t is not None and t.tzinfo is None else t


class ReplayBars:
    """Stands in for StockHistoricalDataClient.get_stock_bars. Everything is
    loaded up front; each request only sees bars complete by clock.now()."""

    def __init__(self, daily: dict, five: dict, clock: SimClock):
        self.daily, self.five, self.clock = daily, five, clock

    def get_stock_bars(self, req):
        now = self.clock.now()
        symbols = req.symbol_or_symbols
        symbols = [symbols] if isinstance(symbols, str) else symbols
        is_day = req.timeframe.unit_value == TimeFrameUnit.Day
        start, end = _aware(req.start), _aware(req.end)
        out = {}
        for s in symbols:
            bars = (self.daily if is_day else self.five).get(s, [])
            keep = []
            for b in bars:
                if start and b.timestamp < start:
                    continue
                if end and b.timestamp >= end:
                    continue
                if is_day:
                    if b.timestamp.astimezone(ET).date() >= now.astimezone(ET).date():
                        continue   # today's daily bar isn't complete until the close
                elif b.timestamp + BAR > now:
                    continue
                keep.append(b)
            if keep:
                out[s] = keep
        return NS(data=out)


class ReplayNews:
    """Real Alpaca news, cached per request window (both variants share them)."""

    def __init__(self, client):
        self.client, self._cache, self._lock = client, {}, threading.Lock()

    def get_news(self, req):
        key = (req.symbols, req.start, req.end)
        with self._lock:
            if key not in self._cache:
                self._cache[key] = self.client.get_news(req)
            return self._cache[key]


class NasdaqEarnings:
    """Past and upcoming report dates from Nasdaq's public earnings calendar,
    one request per weekday, cached to a JSON file."""
    URL = "https://api.nasdaq.com/api/calendar/earnings?date={}"
    TIMES = {"time-pre-market": "pre-market", "time-after-hours": "post-market"}

    def __init__(self, tickers, first: date, last: date, cache_path: Path, clock: SimClock,
                 fetch=None):
        self.clock, self.fetch = clock, fetch or self._fetch
        cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
        self.reports: dict = {}   # {ticker: [(date, when)]}
        d = first
        while d <= last:
            if d.weekday() < 5:
                k = d.isoformat()
                if k not in cache:
                    cache[k] = self.fetch(d)
                for sym, when in cache[k]:
                    if sym in tickers:
                        self.reports.setdefault(sym, []).append((d, self.TIMES.get(when, "")))
            d += timedelta(days=1)
        cache_path.write_text(json.dumps(cache))

    def _fetch(self, d: date) -> list:
        req = urllib.request.Request(self.URL.format(d.isoformat()), headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            rows = ((json.load(r).get("data") or {}).get("rows")) or []
        return [(row["symbol"], row.get("time", "")) for row in rows]

    def next_dates(self, tickers) -> dict:
        """Same shape as context.EarningsCalendar.next_dates, as of the sim clock.
        A report today before the open still counts for today, like the live feed."""
        today = self.clock.now().astimezone(ET).date()
        out = {}
        for s in tickers:
            upcoming = sorted(r for r in self.reports.get(s, []) if r[0] >= today)
            if upcoming:
                out[s] = upcoming[0]
        return out


# --- simulated cash account ------------------------------------------------------
@dataclass
class SimAccount:
    """The broker interface guardrails.py expects, plus positions(), with
    T+1 settlement of sale proceeds."""
    prices: callable                     # () -> {ticker: mid}
    today: callable                      # () -> date
    next_session: callable               # (date) -> date
    cash: float = START_CASH             # settled
    unsettled: list = field(default_factory=list)   # [(settles_on, usd)]
    qty: dict = field(default_factory=dict)
    cost: dict = field(default_factory=dict)        # usd paid for current qty
    spread_paid: float = 0.0
    trades: int = 0

    def _settle(self):
        today = self.today()
        due = [u for u in self.unsettled if u[0] <= today]
        self.unsettled = [u for u in self.unsettled if u[0] > today]
        self.cash += sum(u for _, u in due)

    def equity(self) -> float:
        self._settle()
        p = self.prices()
        return self.cash + sum(u for _, u in self.unsettled) + sum(q * p[s] for s, q in self.qty.items())

    def settled_cash(self) -> float:
        self._settle()
        return self.cash

    def position_value(self, ticker) -> float:
        return self.qty.get(ticker, 0.0) * self.prices()[ticker]

    def positions(self) -> dict:
        p = self.prices()
        return {s: {"usd": q * p[s], "qty": q, "pl_pct": (q * p[s] / self.cost[s] - 1) * 100}
                for s, q in self.qty.items()}

    def cancel_all_orders(self):
        pass   # fills are instant: nothing is ever open

    def close_all_positions(self):
        for s in list(self.qty):
            self.sell(s, qty=self.qty[s])

    def buy(self, ticker, usd):
        self._settle()
        price = self.prices()[ticker] * (1 + SPREAD_PCT / 200)
        q = usd / price
        self.cash -= usd
        self.qty[ticker] = self.qty.get(ticker, 0.0) + q
        self.cost[ticker] = self.cost.get(ticker, 0.0) + usd
        self.spread_paid += q * (price - self.prices()[ticker])
        self.trades += 1

    def sell(self, ticker, usd=None, qty=None):
        mid = self.prices()[ticker]
        price = mid * (1 - SPREAD_PCT / 200)
        held = self.qty[ticker]
        q = held if qty is not None else min(held, usd / price)
        proceeds = q * price
        self.cost[ticker] *= (held - q) / held
        self.qty[ticker] = held - q
        if self.qty[ticker] <= 1e-9:
            del self.qty[ticker], self.cost[ticker]
        self.unsettled.append((self.next_session(self.today()), proceeds))
        self.spread_paid += q * (mid - price)
        self.trades += 1


# --- the replay ---------------------------------------------------------------------
@dataclass
class Variant:
    name: str
    decider: Decider
    account: SimAccount
    g: Guardrails
    rows: list = field(default_factory=list)
    counts: dict = field(default_factory=lambda: {"buy": 0, "sell": 0, "hold": 0, "rejected": 0,
                                                  "invalid": 0, "unavailable": 0})
    ai_cost: float = 0.0


def ticker_views(five: dict, daily: dict, t: datetime, tickers) -> dict:
    day = t.astimezone(ET).date()
    open_ = datetime.combine(day, time(9, 30), tzinfo=ET)
    views = {}
    for s in tickers:
        today = [b for b in five.get(s, []) if open_ <= b.timestamp and b.timestamp + BAR <= t]
        if not today:
            continue
        mid = float(today[-1].close)
        past = [b for b in daily.get(s, []) if b.timestamp.astimezone(ET).date() < day]
        hour = [b for b in today if b.timestamp >= t - timedelta(minutes=70)]
        chg = lambda ref: (mid / float(ref) - 1) * 100
        views[s] = TickerView(s, mid, SPREAD_PCT, chg(past[-1].close) if past else None,
                              chg(hour[0].open) if hour else None,
                              chg(past[-5].close) if len(past) >= 5 else None, 0.0)
    return views


def decide_and_trade(v: Variant, snap: Snapshot, t: datetime):
    row = {"time": t.astimezone(ET).strftime("%Y-%m-%d %H:%M"), "action": "", "ticker": "",
           "usd": "", "outcome": "", "reason": ""}
    try:
        r = v.decider.decide(snap)
    except DecisionUnavailable as e:
        v.counts["unavailable"] += 1
        row.update(outcome="unavailable", reason=str(e))
        return row
    v.ai_cost += r.cost_usd
    if r.decision is None:
        v.counts["invalid"] += 1
        row.update(outcome="invalid", reason=r.error)
        return row
    d = r.decision
    row.update(action=d.action, ticker=d.ticker, usd=d.usd, reason=d.reason)
    if d.action == "hold":
        v.counts["hold"] += 1
        row["outcome"] = "hold"
        return row
    ok, why = v.g.check_order(d.ticker, d.action, d.usd)
    if not ok:
        v.counts["rejected"] += 1
        row["outcome"] = f"rejected: {why}"
        return row
    if d.action == "buy":
        v.account.buy(d.ticker, d.usd)
    else:
        held = snap.positions[d.ticker]
        if d.usd >= held["usd"] - SELL_ALL_MARGIN_USD:
            v.account.sell(d.ticker, qty=held["qty"])
        else:
            v.account.sell(d.ticker, usd=d.usd)
    v.counts[d.action] += 1
    row["outcome"] = "filled"
    return row


def run(days: int, every: int, out: Path, tickers=None):
    from alpaca.data.enums import DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.historical.news import NewsClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    import anthropic

    out.mkdir(parents=True, exist_ok=True)
    key, secret = os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"]
    tickers = sorted(tickers or Limits().allowed_tickers)
    client = StockHistoricalDataClient(key, secret)
    end = datetime.now(timezone.utc)

    log.info("loading a year of daily bars and ~%d days of 5-minute bars", days + 95)
    daily = client.get_stock_bars(StockBarsRequest(
        symbol_or_symbols=tickers, timeframe=TimeFrame.Day,
        start=end - timedelta(days=380 + days * 2), feed=DataFeed.IEX)).data
    sessions = sorted({b.timestamp.astimezone(ET).date() for b in daily["SPY"]})
    today = end.astimezone(ET).date()
    if end.astimezone(ET).time() < time(16, 0):
        sessions = [s for s in sessions if s < today]   # only completed sessions
    replay = sessions[-days:]
    first = datetime.combine(replay[0], time(0), tzinfo=ET)
    five = client.get_stock_bars(StockBarsRequest(
        symbol_or_symbols=tickers, timeframe=TimeFrame(5, TimeFrameUnit.Minute),
        start=first - timedelta(days=95), feed=DataFeed.IEX)).data
    log.info("replaying %s to %s (%d sessions)", replay[0], replay[-1], len(replay))

    clock = SimClock()
    nxt = {d: sessions[i + 1] if i + 1 < len(sessions) else d + timedelta(days=1)
           for i, d in enumerate(sessions)}
    earnings = NasdaqEarnings(set(tickers), replay[0], replay[-1] + timedelta(days=95),
                              out / "earnings_cache.json", clock)
    ctx = ContextProvider(ReplayBars(daily, five, clock), ReplayNews(NewsClient(key, secret)),
                          earnings, now=clock.now)
    guardrails_module.time = NS(time=lambda: clock.now().timestamp())   # hourly cap on sim time

    prices: dict = {}
    api = anthropic.Anthropic(timeout=60.0, max_retries=3)
    variants = []
    for name, system in VARIANTS.items():
        acct = SimAccount(prices=lambda: prices, today=lambda: clock.now().astimezone(ET).date(),
                          next_session=lambda d: nxt[d])
        variants.append(Variant(name, Decider(api, os.getenv("CLAUDE_MODEL", "claude-haiku-5-5"),
                                              DailyBudget(1000.0), system=system),
                                acct, Guardrails(acct, Limits())))

    curve = []   # (date, {variant: equity}, spy_close)
    spy_shares = None
    ckpt = out / "checkpoint.json"
    done = set()
    if ckpt.exists():
        spy_shares, curve, done = load_checkpoint(ckpt, variants, prices, replay, every)
        log.info("resuming after %s (%d sessions already done)", max(done), len(done))
    with ThreadPoolExecutor(len(variants)) as pool:
        for day in replay:
            if day in done:
                continue
            t = datetime.combine(day, time(9, 30), tzinfo=ET) + timedelta(minutes=15)
            close = datetime.combine(day, time(16, 0), tzinfo=ET)
            while t < close:
                clock.t = t
                views = ticker_views(five, daily, t, tickers)
                # carried forward: a ticker with no fresh bar keeps its last price
                prices.update({s: v.price for s, v in views.items()})
                if spy_shares is None:
                    spy_shares = START_CASH / (prices["SPY"] * (1 + SPREAD_PCT / 200))
                text = ctx.text(prices)
                jobs = []
                for v in variants:
                    if v.g.halted or not v.g.check_floor():
                        continue
                    snap = Snapshot(t, close, v.account.equity(), v.account.settled_cash(),
                                    v.account.positions(), views, {}, text, tuple(ctx.problems))
                    jobs.append((v, pool.submit(decide_and_trade, v, snap, t)))
                for v, job in jobs:
                    row = job.result()
                    row["equity"] = round(v.account.equity(), 2)
                    v.rows.append(row)
                t += timedelta(minutes=every)
            clock.t = close
            last = {s: float([b for b in five[s] if b.timestamp + BAR <= close][-1].close) for s in tickers}
            prices.update(last)
            curve.append((day, {v.name: v.account.equity() for v in variants}, last["SPY"]))
            log.info("%s  %s  SPY hold $%.2f", day,
                     "  ".join(f"{v.name} ${v.account.equity():.2f}" for v in variants),
                     spy_shares * last["SPY"])
            save_checkpoint(ckpt, variants, prices, spy_shares, curve, replay, every)

    report(variants, curve, spy_shares, replay, every, out)


# --- checkpoint: a restart resumes after the last finished session -------------------
def save_checkpoint(path: Path, variants, prices, spy_shares, curve, replay, every):
    state = {
        "replay": [d.isoformat() for d in replay], "every": every, "spy_shares": spy_shares,
        "prices": prices, "curve": [(d.isoformat(), eq, spy) for d, eq, spy in curve],
        "variants": {v.name: {
            "cash": v.account.cash, "unsettled": [(d.isoformat(), u) for d, u in v.account.unsettled],
            "qty": v.account.qty, "cost": v.account.cost, "spread_paid": v.account.spread_paid,
            "trades": v.account.trades, "rows": v.rows, "counts": v.counts, "ai_cost": v.ai_cost,
            "halted": v.g.halted, "halt_reason": v.g.halt_reason, "order_times": v.g._order_times,
        } for v in variants},
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state))
    os.replace(tmp, path)


def load_checkpoint(path: Path, variants, prices, replay, every):
    state = json.loads(path.read_text())
    if state["replay"] != [d.isoformat() for d in replay] or state["every"] != every:
        raise SystemExit(f"{path} is from a different run (days or --every differ): "
                         "use another --out folder or delete it")
    prices.update(state["prices"])
    for v in variants:
        st = state["variants"][v.name]
        a = v.account
        a.cash, a.qty, a.cost = st["cash"], st["qty"], st["cost"]
        a.unsettled = [(date.fromisoformat(d), u) for d, u in st["unsettled"]]
        a.spread_paid, a.trades = st["spread_paid"], st["trades"]
        v.rows, v.counts, v.ai_cost = st["rows"], st["counts"], st["ai_cost"]
        v.g.halted, v.g.halt_reason, v.g._order_times = st["halted"], st["halt_reason"], st["order_times"]
    curve = [(date.fromisoformat(d), eq, spy) for d, eq, spy in state["curve"]]
    return state["spy_shares"], curve, {d for d, _, _ in curve}


def max_drawdown(values) -> float:
    peak, worst = values[0], 0.0
    for x in values:
        peak = max(peak, x)
        worst = min(worst, x / peak - 1)
    return worst * 100


def report(variants, curve, spy_shares, replay, every, out: Path):
    for v in variants:
        with open(out / f"decisions_{v.name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["time", "action", "ticker", "usd", "outcome",
                                              "equity", "reason"])
            w.writeheader()
            w.writerows(v.rows)
    with open(out / "equity_by_day.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date"] + [v.name for v in variants] + ["spy_hold", "cash"])
        for day, eq, spy in curve:
            w.writerow([day] + [f"{eq[v.name]:.2f}" for v in variants]
                       + [f"{spy_shares * spy:.2f}", f"{START_CASH:.2f}"])

    spy_curve = [spy_shares * spy for _, _, spy in curve]
    lines = [f"# Replay test: {replay[0]} to {replay[-1]} ({len(replay)} sessions, "
             f"a decision every {every} min)", "",
             f"Start ${START_CASH:.2f} cash. Assumed spread {SPREAD_PCT}%. Sale proceeds settle T+1.", "",
             "| | end equity | return | max drawdown | buys | sells | holds | rejected | invalid | "
             "spread paid | AI cost |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for v in variants:
        eq = [e[v.name] for _, e, _ in curve]
        c = v.counts
        lines.append(f"| {v.name} prompt | ${eq[-1]:.2f} | {(eq[-1] / START_CASH - 1) * 100:+.2f}% | "
                     f"{max_drawdown([START_CASH] + eq):.2f}% | {c['buy']} | {c['sell']} | "
                     f"{c['hold']} | {c['rejected']} | {c['invalid'] + c['unavailable']} | "
                     f"${v.account.spread_paid:.2f} | ${v.ai_cost:.3f} |")
    lines.append(f"| hold SPY | ${spy_curve[-1]:.2f} | {(spy_curve[-1] / START_CASH - 1) * 100:+.2f}% | "
                 f"{max_drawdown([START_CASH] + spy_curve):.2f}% | 1 | 0 | | | | | |")
    lines.append(f"| hold cash | ${START_CASH:.2f} | +0.00% | 0.00% | | | | | | | |")
    lines += ["", "Equity at each close: equity_by_day.csv. Every decision with its reason: "
              "decisions_<variant>.csv.", "",
              "Positions at the end:"]
    for v in variants:
        held = ", ".join(f"{s} ${p['usd']:.2f}" for s, p in sorted(v.account.positions().items()))
        lines.append(f"- {v.name}: {held or 'none'}")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--days", type=int, default=20, help="past sessions to replay")
    ap.add_argument("--every", type=int, default=15, help="minutes between decisions")
    ap.add_argument("--out", default="backtest_results", help="folder for the report and CSVs")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    for noisy in ("httpx", "httpx2", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    run(a.days, a.every, Path(a.out))


if __name__ == "__main__":
    sys.exit(main())
