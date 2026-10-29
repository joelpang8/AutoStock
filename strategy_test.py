"""Test simple, rule-based strategies on years of daily prices (no AI calls).

    python strategy_test.py --out results/

Each strategy starts with $300 and follows one fixed rule. Signals use the
close of day t and trade at the close of day t+1 (no same-day look-ahead),
paying half an assumed 0.05% spread on every dollar traded. The owner's $200
floor is applied: the first close below $200 sells everything and the strategy
stays in cash (the live bot halts until the owner resumes it).

Reported for the full period and for every 1-year window starting each month,
because one lucky start date can make any rule look good.

Hindsight warning: these tickers were chosen in 2026, after a decade in which
US tech (and NVDA above all) did unusually well. A rule that leans on them looks
better here than it can be expected to do next decade.
"""
from __future__ import annotations

import argparse
import math
import os
import statistics
from datetime import datetime
from pathlib import Path

START_CASH, FLOOR, SPREAD_PCT = 300.0, 200.0, 0.05
ALLOWED = ["SPY", "QQQ", "VOO", "IWM", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]
STOCKS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]
EXTRA = ["QLD", "TQQQ", "SSO"]


def sma(xs, n):
    return sum(xs[-n:]) / n if len(xs) >= n else None


# --- strategies: (prices so far by ticker) -> target weights {ticker: weight} ---------
def hold(ticker):
    return lambda hist, i: {ticker: 1.0}


def trend(lev, signal="QQQ", n=200):
    """Hold `lev` while `signal` closes above its n-day average, else cash."""
    def rule(hist, i):
        closes = hist[signal][:i + 1]
        m = sma(closes, n)
        return {lev: 1.0} if m is not None and closes[-1] > m else {}
    return rule


def momentum(universe, top, lookback=126, every=21):
    """Every `every` trading days, hold the `top` best performers over `lookback`
    days (only those that are up), equal weight. Stateless: the pick is always
    made as of the last rebalance day, so every window sees the same schedule."""
    def rule(hist, i):
        r = i - i % every
        if r < lookback:
            return {}
        rets = {s: hist[s][r] / hist[s][r - lookback] - 1 for s in universe}
        best = [s for s, x in sorted(rets.items(), key=lambda kv: -kv[1])[:top] if x > 0]
        return {s: 1 / len(best) for s in best} if best else {}
    return rule


def mix(weights):
    return lambda hist, i: dict(weights)


STRATEGIES = {
    "SPY (benchmark)": hold("SPY"),
    "QQQ": hold("QQQ"),
    "QLD 2x Nasdaq": hold("QLD"),
    "TQQQ 3x Nasdaq": hold("TQQQ"),
    "QLD when QQQ > 200d avg": trend("QLD"),
    "TQQQ when QQQ > 200d avg": trend("TQQQ"),
    "SSO 2x S&P when SPY > 200d avg": trend("SSO", "SPY"),
    "70% SPY + 30% TQQQ": mix({"SPY": 0.7, "TQQQ": 0.3}),
    "Momentum: best 2 of the 9": momentum(ALLOWED, 2),
    "Momentum: best 1 of 5 stocks": momentum(STOCKS, 1, lookback=63),
}


# --- simulation ------------------------------------------------------------------------
def simulate(rule, hist, dates, start, end):
    """Equity at each close from index start..end. Rebalances to the target
    weights at the next close; the floor liquidates for good."""
    cash, qty, halted = START_CASH, {}, False
    curve, pending = [], None
    for i in range(start, end + 1):
        px = {s: hist[s][i] for s in hist}
        equity = cash + sum(q * px[s] for s, q in qty.items())
        if not halted and pending is not None:
            target = {s: w * equity for s, w in pending.items()}
            for s in set(qty) | set(target):
                have = qty.get(s, 0.0) * px[s]
                trade = target.get(s, 0.0) - have
                if abs(trade) < 1.0:   # skip dust
                    continue
                cost = abs(trade) * SPREAD_PCT / 200
                qty[s] = qty.get(s, 0.0) + trade / px[s]
                cash -= trade + cost
            qty = {s: q for s, q in qty.items() if q * px[s] > 1e-6}
            equity = cash + sum(q * px[s] for s, q in qty.items())
        if not halted and equity < FLOOR:
            cash -= sum(q * px[s] for s, q in qty.items()) * SPREAD_PCT / 200
            cash += sum(q * px[s] for s, q in qty.items())
            qty, halted = {}, True
            equity = cash
        curve.append(equity)
        pending = None if halted else rule(hist, i)
    return curve, halted


def stats(curve, days):
    years = days / 252
    cagr = (curve[-1] / START_CASH) ** (1 / years) - 1 if years > 0 else 0
    peak, dd = curve[0], 0.0
    for x in curve:
        peak = max(peak, x)
        dd = min(dd, x / peak - 1)
    return cagr * 100, dd * 100


def load(start: datetime):
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    c = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
    data = c.get_stock_bars(StockBarsRequest(symbol_or_symbols=ALLOWED + EXTRA, timeframe=TimeFrame.Day,
                                             start=start, feed=DataFeed.SIP,
                                             adjustment=Adjustment.ALL)).data
    by_date = {}
    for s, bars in data.items():
        for b in bars:
            by_date.setdefault(b.timestamp.date(), {})[s] = float(b.close)
    want = set(ALLOWED + EXTRA)
    dates = sorted(d for d, row in by_date.items() if set(row) >= want)
    return {s: [by_date[d][s] for d in dates] for s in want}, dates


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="strategy_results")
    a = ap.parse_args(argv)
    hist, dates = load(datetime(2016, 1, 1))
    first = 210                    # room for the 200-day average and the momentum lookback
    last = len(dates) - 1
    lines = [f"# Strategy test: {dates[first]} to {dates[-1]} ({(last - first) / 252:.1f} years)", "",
             f"Start ${START_CASH:.0f}, $200 floor (sells all and stops), {SPREAD_PCT}% spread, "
             "signal at one close, trade at the next.", "",
             "| strategy | yearly return | worst drop | $300 became | hit the floor | "
             "1-yr windows: median | worst | beat SPY | hit floor |",
             "|---|---|---|---|---|---|---|---|---|"]
    windows = list(range(first, last - 252, 21))
    spy_1y = [simulate(STRATEGIES["SPY (benchmark)"], hist, dates, w, w + 252)[0][-1] for w in windows]
    for name, rule in STRATEGIES.items():
        curve, halted = simulate(rule, hist, dates, first, last)
        cagr, dd = stats(curve, last - first)
        hit = next((dates[first + k] for k, x in enumerate(curve) if x < FLOOR), None)
        ones = [simulate(rule, hist, dates, w, w + 252) for w in windows]
        rets = [(c[-1] / START_CASH - 1) * 100 for c, _ in ones]
        beat = sum(c[-1] > s for (c, _), s in zip(ones, spy_1y)) / len(ones) * 100
        floors = sum(h for _, h in ones) / len(ones) * 100
        lines.append(f"| {name} | {cagr:+.1f}% | {dd:.0f}% | ${curve[-1]:,.0f} | "
                     f"{hit or 'no'} | {statistics.median(rets):+.0f}% | {min(rets):+.0f}% | "
                     f"{beat:.0f}% | {floors:.0f}% |")
    lines += ["", f"1-year windows: {len(windows)} of them, one starting every month.",
              "Hindsight warning: tickers picked in 2026 after a strong decade for US tech."]
    text = "\n".join(lines) + "\n"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "strategy_report.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
