"""Strategy Time Machine: 10,000 possible next years for each strategy.

    python strategy_timemachine.py --out results/

The day-trading Time Machine (Jake Schincariol's day-trader-skill, MIT) asks:
if today plays out like the stock's recent past, how often does the trade hit
its target before its stop? This asks the same question at the scale this bot
trades: if the next year plays out like a shuffled mix of past years, how often
does each strategy hit the $200 floor, and where does $300 end up?

How a replay is built:
- It starts from the real market state today (the last 260 real closes, so
  the 200-day average and the volatility estimate are today's).
- The next 252 trading days are stitched from random 63-day blocks (about a
  quarter each) of real daily moves since 2016. SPY, QQQ and QLD keep each
  day's moves together, so they stay as correlated as they really were, and
  blocks keep trends and crashes intact for a few months at a time.
- Each strategy then trades that year exactly as in efficiency_test.py:
  signal at one close, trade at the next, 0.10% spread, $300 start, and the
  $200 floor sells everything and stops for good.

Two worlds:
- "history mix": the real daily moves. 2016-2026 was a strong decade for US
  tech, so this world leans optimistic.
- "flat market": the same daily swings with the decade's upward drift taken
  out (SPY and QQQ average 0% a year). QLD is rebuilt as 2x QQQ minus its real
  measured drag, so leverage still costs what it cost. This is the stress
  test: what the strategy does when the market goes nowhere.

A simulation, not a forecast: the future can do things the last decade never
did. Same seed, same numbers.
"""
from __future__ import annotations

import argparse
import math
import random
import statistics
from datetime import datetime
from pathlib import Path

from strategy_test import FLOOR, START_CASH, load

SPREAD_PCT = 0.10
YEAR, BLOCK, HISTORY = 252, 63, 260
REPLAYS, SEED = 10_000, 19
TICKERS = ("SPY", "QQQ", "QLD")
SMA_DAYS, VOL_DAYS, WEEK, MIN_SHIFT = 200, 20, 5, 0.10


# --- the replay worlds ------------------------------------------------------------
def daily_returns(hist) -> dict:
    """{ticker: [r_1, ...]}: simple daily returns, aligned across tickers."""
    return {s: [hist[s][i] / hist[s][i - 1] - 1 for i in range(1, len(hist[s]))] for s in TICKERS}


def flat_world(rets) -> dict:
    """Same swings, no drift: SPY and QQQ de-meaned in log terms (so they
    compound to about 0% a year), QLD = 2 x QQQ minus QLD's measured drag."""
    out = {}
    for s in ("SPY", "QQQ"):
        mu = statistics.fmean(math.log1p(r) for r in rets[s])
        out[s] = [math.expm1(math.log1p(r) - mu) for r in rets[s]]
    drag = statistics.fmean(2 * q - l for q, l in zip(rets["QQQ"], rets["QLD"]))
    out["QLD"] = [2 * q - drag for q in out["QQQ"]]
    return out


def replay_year(rets, rng) -> dict:
    """252 days of returns from random 63-day blocks, all tickers together."""
    n = len(rets["SPY"])
    days = []
    while len(days) < YEAR:
        # wrap around the end, so every day is equally likely to be drawn
        start = rng.randrange(n)
        days.extend((start + k) % n for k in range(BLOCK))
    days = days[:YEAR]
    return {s: [rets[s][d] for d in days] for s in TICKERS}


def prices(today: dict, year: dict) -> dict:
    """Real history up to today, then the replayed year."""
    out = {}
    for s in TICKERS:
        p = list(today[s])
        for r in year[s]:
            p.append(p[-1] * (1 + r))
        out[s] = p
    return out


# --- the strategies: target weights from prices up to index i ---------------------
class Signals:
    """O(1) rolling 200-day average and 20-day volatility of QQQ."""

    def __init__(self, qqq):
        self.qqq = qqq
        self.cum = [0.0]
        for x in qqq:
            self.cum.append(self.cum[-1] + x)
        rets = [0.0] + [qqq[i] / qqq[i - 1] - 1 for i in range(1, len(qqq))]
        self.c1, self.c2 = [0.0], [0.0]
        for r in rets:
            self.c1.append(self.c1[-1] + r)
            self.c2.append(self.c2[-1] + r * r)

    def above_sma(self, i) -> bool:
        return self.qqq[i] > (self.cum[i + 1] - self.cum[i + 1 - SMA_DAYS]) / SMA_DAYS

    def vol(self, i) -> float:
        n = VOL_DAYS
        s1 = self.c1[i + 1] - self.c1[i + 1 - n]
        s2 = self.c2[i + 1] - self.c2[i + 1 - n]
        var = max(0.0, (s2 - s1 * s1 / n) / (n - 1))
        return math.sqrt(var * 252)


def hold_spy(sig, i):
    return {"SPY": 1.0}


def qld_trend(sig, i):
    return {"QLD": 1.0} if sig.above_sma(i) else {}


def qld_vol_target(target):
    def rule(sig, i):
        r = i - i % WEEK
        if not sig.above_sma(r):
            return {}
        v = sig.vol(r)
        return {"QLD": round(min(1.0, target / (2 * v)) if v else 1.0, 2)}
    return rule


STRATEGIES = {
    "Hold SPY": hold_spy,
    "QLD trend (current)": qld_trend,
    "QLD trend + 25% vol target": qld_vol_target(0.25),
}


def run_year(rule, px, start) -> tuple[float, bool, float]:
    """(end equity, hit the floor, lowest equity) from index start to the end."""
    sig = Signals(px["QQQ"])
    cash, qty, halted, pending, low = START_CASH, {}, False, None, START_CASH
    for i in range(start, len(px["SPY"])):
        p = {s: px[s][i] for s in TICKERS}
        equity = cash + sum(q * p[s] for s, q in qty.items())
        if not halted and pending is not None:
            target = {s: w * equity for s, w in pending.items()}
            moves = {s: target.get(s, 0.0) - qty.get(s, 0.0) * p[s] for s in set(qty) | set(target)}
            if (not pending and qty) or any(abs(m) > MIN_SHIFT * equity for m in moves.values()):
                for s, m in moves.items():
                    if abs(m) >= 1.0:
                        qty[s] = qty.get(s, 0.0) + m / p[s]
                        cash -= m + abs(m) * SPREAD_PCT / 200
                qty = {s: q for s, q in qty.items() if q * p[s] > 1e-6}
                equity = cash + sum(q * p[s] for s, q in qty.items())
        if not halted and equity < FLOOR:
            cash += sum(q * p[s] for s, q in qty.items()) * (1 - SPREAD_PCT / 200)
            qty, halted, equity = {}, True, cash
        low = min(low, equity)
        pending = None if halted else rule(sig, i)
    return equity, halted, low


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def run(hist, replays=REPLAYS, seed=SEED):
    today = {s: hist[s][-HISTORY:] for s in TICKERS}
    real = daily_returns(hist)
    results = {}
    for world, rets in (("history mix", real), ("flat market", flat_world(real))):
        rng = random.Random(seed)
        ends = {name: [] for name in STRATEGIES}
        floors = {name: 0 for name in STRATEGIES}
        lows = {name: [] for name in STRATEGIES}
        for _ in range(replays):
            px = prices(today, replay_year(rets, rng))
            for name, rule in STRATEGIES.items():
                end, hit, low = run_year(rule, px, HISTORY - 1)
                ends[name].append(end)
                floors[name] += hit
                lows[name].append(low)
        spy = ends["Hold SPY"]
        results[world] = {name: {
            "floor": floors[name] / replays * 100,
            "median": statistics.median(ends[name]),
            "p5": pct(ends[name], 0.05), "p95": pct(ends[name], 0.95),
            "loss": sum(e < START_CASH for e in ends[name]) / replays * 100,
            "beat_spy": sum(e > s for e, s in zip(ends[name], spy)) / replays * 100,
            "low_p5": pct(lows[name], 0.05),
        } for name in STRATEGIES}
    return results


def report(results, as_of, replays, seed) -> str:
    lines = [f"# Strategy Time Machine: {replays:,} possible next years from {as_of} (seed {seed})", "",
             "$300 start, $200 floor (sells all and stops), 0.10% spread. Each year is stitched "
             "from random 63-day blocks of real daily moves since 2016, starting from today's "
             "real market state.", ""]
    for world, res in results.items():
        lines += [f"## {world}", "",
                  "| strategy | hit $200 floor | ends below $300 | median end | 90% of years between | "
                  "beats holding SPY | bad-year low point (5%) |",
                  "|---|---|---|---|---|---|---|"]
        for name, r in res.items():
            beat = "-" if name == "Hold SPY" else f"{r['beat_spy']:.0f}%"
            lines.append(f"| {name} | {r['floor']:.1f}% | {r['loss']:.0f}% | ${r['median']:.0f} | "
                         f"${r['p5']:.0f} - ${r['p95']:.0f} | {beat} | ${r['low_p5']:.0f} |")
        lines.append("")
    lines += ["A simulation built from 2016-2026, not a forecast. 'Flat market' removes that "
              "decade's upward drift to show what happens when the market goes nowhere.",
              f"Same data, same seed ({seed}): same numbers."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="strategy_results")
    ap.add_argument("--replays", type=int, default=REPLAYS)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args(argv)
    hist, dates = load(datetime(2016, 1, 1))
    text = report(run(hist, a.replays, a.seed), dates[-1], a.replays, a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "timemachine_report.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
