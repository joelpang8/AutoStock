"""Which strategy gives the most return per unit of risk? (No AI calls.)

    python efficiency_test.py --out results/

Builds on strategy_test.py (same data, $300 start, $200 floor, signal at one
close and trade at the next) and adds what "efficient" needs:

- Sharpe ratio (yearly return over yearly volatility) and return / worst drop.
- Robustness: the same rule in each half of the period separately, and the
  trend rule with other average lengths (100 to 250 days). A rule that only
  works in one half, or only with one exact setting, is luck, not an edge.
- Costs at 0.10% spread (double the earlier assumption) and turnover: how
  many dollars traded per year, since each order is capped at $20.
- Volatility targeting: hold less QLD when the market is choppy, more when it
  is calm, aiming for a fixed yearly volatility. Rebalanced weekly, and only
  when the target moves by more than 10% of the account, to keep trades low.

The same hindsight warning applies: a decade picked in 2026, good for US tech.
"""
from __future__ import annotations

import argparse
import math
import statistics
from datetime import datetime
from pathlib import Path

import strategy_test as st
from strategy_test import FLOOR, START_CASH, hold, load, momentum, sma, trend

SPREAD_PCT = 0.10
WEEK = 5
MIN_SHIFT = 0.10    # vol-target: ignore target changes under 10% of equity


def vol20(closes) -> float | None:
    if len(closes) < 21:
        return None
    rets = [closes[i] / closes[i - 1] - 1 for i in range(len(closes) - 20, len(closes))]
    return statistics.stdev(rets) * math.sqrt(252)


def vol_target(target, lev="QLD", lev_x=2.0, signal="QQQ", n=200):
    """Trend filter as before; when on, QLD weight = target vol / (2 x QQQ's
    20-day vol), at most 100%. Decided weekly, as of the last rebalance day."""
    def rule(hist, i):
        r = i - i % WEEK
        closes = hist[signal][:r + 1]
        m = sma(closes, n)
        if m is None or closes[-1] <= m:
            return {}
        v = vol20(closes)
        w = 1.0 if not v else min(1.0, target / (lev_x * v))
        return {lev: round(w, 2)}
    return rule


STRATEGIES = {
    "SPY (benchmark)": hold("SPY"),
    "QQQ": hold("QQQ"),
    "QQQ when QQQ > 200d avg": trend("QQQ"),
    "QLD when QQQ > 200d avg (current)": trend("QLD"),
    "  same, 100-day average": trend("QLD", n=100),
    "  same, 150-day average": trend("QLD", n=150),
    "  same, 250-day average": trend("QLD", n=250),
    "QLD trend + vol target 20%": vol_target(0.20),
    "QLD trend + vol target 25%": vol_target(0.25),
    "QLD trend + vol target 30%": vol_target(0.30),
    "Momentum: best 2 of the 9": momentum(st.ALLOWED, 2),
}


def simulate(rule, hist, start, end):
    """Like strategy_test.simulate, plus dollars traded. A target change smaller
    than MIN_SHIFT of equity in every ticker is skipped (fewer, bigger trades)."""
    cash, qty, halted, pending = START_CASH, {}, False, None
    curve, traded = [], 0.0
    for i in range(start, end + 1):
        px = {s: hist[s][i] for s in hist}
        equity = cash + sum(q * px[s] for s, q in qty.items())
        if not halted and pending is not None:
            target = {s: w * equity for s, w in pending.items()}
            moves = {s: target.get(s, 0.0) - qty.get(s, 0.0) * px[s] for s in set(qty) | set(target)}
            exit_all = not pending and qty
            if exit_all or any(abs(m) > MIN_SHIFT * equity for m in moves.values()):
                for s, m in moves.items():
                    if abs(m) < 1.0:
                        continue
                    qty[s] = qty.get(s, 0.0) + m / px[s]
                    cash -= m + abs(m) * SPREAD_PCT / 200
                    traded += abs(m)
                qty = {s: q for s, q in qty.items() if q * px[s] > 1e-6}
                equity = cash + sum(q * px[s] for s, q in qty.items())
        if not halted and equity < FLOOR:
            value = sum(q * px[s] for s, q in qty.items())
            cash += value * (1 - SPREAD_PCT / 200)
            traded += value
            qty, halted, equity = {}, True, cash
        curve.append(equity)
        pending = None if halted else rule(hist, i)
    return curve, halted, traded


def metrics(curve, traded):
    years = (len(curve) - 1) / 252
    cagr = (curve[-1] / curve[0]) ** (1 / years) - 1
    rets = [curve[i] / curve[i - 1] - 1 for i in range(1, len(curve))]
    vol = statistics.stdev(rets) * math.sqrt(252)
    peak, dd = curve[0], 0.0
    for x in curve:
        peak = max(peak, x)
        dd = min(dd, x / peak - 1)
    avg_equity = statistics.fmean(curve)
    return {"cagr": cagr * 100, "vol": vol * 100, "sharpe": (cagr / vol) if vol else 0.0,
            "dd": dd * 100, "calmar": cagr / -dd if dd else 0.0,
            "turnover": traded / avg_equity / years}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="strategy_results")
    a = ap.parse_args(argv)
    hist, dates = load(datetime(2016, 1, 1))
    first, last = 260, len(dates) - 1   # room for the 250-day average
    mid = (first + last) // 2
    windows = list(range(first, last - 252, 21))

    rows = []
    for name, rule in STRATEGIES.items():
        curve, _, traded = simulate(rule, hist, first, last)
        m = metrics(curve, traded)
        h1 = metrics(*[simulate(rule, hist, first, mid)[k] for k in (0, 2)])
        h2 = metrics(*[simulate(rule, hist, mid, last)[k] for k in (0, 2)])
        floors = sum(simulate(rule, hist, w, w + 252)[1] for w in windows) / len(windows) * 100
        rows.append((name, m, h1, h2, floors))

    lines = [f"# Efficiency test: {dates[first]} to {dates[-1]} ({(last - first) / 252:.1f} years)", "",
             f"$300 start, $200 floor, {SPREAD_PCT}% spread, signal at one close, trade at the next. "
             f"Halves: {dates[first]} to {dates[mid]}, then {dates[mid]} to {dates[-1]}.", "",
             "| strategy | per year | volatility | Sharpe | worst drop | return / drop | "
             "1st half | 2nd half | 1-yr floor hits | traded per year |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, m, h1, h2, floors in rows:
        lines.append(f"| {name} | {m['cagr']:+.1f}% | {m['vol']:.0f}% | {m['sharpe']:.2f} | "
                     f"{m['dd']:.0f}% | {m['calmar']:.2f} | {h1['cagr']:+.1f}% | {h2['cagr']:+.1f}% | "
                     f"{floors:.0f}% | {m['turnover']:.1f}x |")
    lines += ["", "Sharpe here is yearly return / yearly volatility (no risk-free rate). "
              "'Traded per year' is dollars traded over the average account value.",
              "Hindsight warning: tickers and period picked in 2026, after a strong decade for US tech."]
    text = "\n".join(lines) + "\n"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "efficiency_report.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
