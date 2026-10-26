"""Chart setup tags for the decision step, after the Day Trader skill's Chart
Scanner (github.com/Jakeschincariol/day-trader-skill, MIT, Jake Schincariol).

Same rules as its daily scan, on completed daily closes (oldest first):
  breakout_20d     the last close is above every close of the 20 before it
  bull_flag        up more than 10% over the 10 sessions before the last 5,
                   then the last 5 closes inside a 5% range
  reclaim_ma20     closed under the 20-day average, then closed above it
  oversold         RSI 14 (Wilder) under 30
  oversold_bounce  RSI 14 was under 30 the session before and the last close was up

and its opening-range rule on today's 5-minute bars: the range is the 9:30,
9:35 and 9:40 bars; a breakout is the first completed bar to close above the
range high. We also report the mirror case (a close below the range low), since
the bot can sell what it holds, and where the last completed bar closed, since a
morning break can reverse.

Tags are information for Claude, nothing more: a matched rule is not a
prediction, and every order still goes through the guardrails.
"""
from __future__ import annotations

from datetime import datetime, time, timedelta

from data import ET

MIN_DAILY = 30
BAR = timedelta(minutes=5)
RANGE_BARS = 3   # 9:30, 9:35, 9:40 -> the first 15 minutes


def rsi(closes, n=14):
    """Wilder's RSI, or None with too few closes."""
    if len(closes) < n + 1:
        return None
    gains = [max(0.0, closes[i] - closes[i - 1]) for i in range(1, len(closes))]
    losses = [max(0.0, closes[i - 1] - closes[i]) for i in range(1, len(closes))]
    ag, al = sum(gains[:n]) / n, sum(losses[:n]) / n
    for i in range(n, len(gains)):
        ag = (ag * (n - 1) + gains[i]) / n
        al = (al * (n - 1) + losses[i]) / n
    return 100 - 100 / (1 + ag / al) if al else 100.0


def daily_setups(closes) -> tuple[list[str], float | None]:
    """(tags, rsi14) from completed daily closes. Too little history -> ([], None)."""
    c = closes
    if len(c) < MIN_DAILY:
        return [], None
    last = c[-1]
    r, r_prev = rsi(c), rsi(c[:-1])
    tags = []
    if last > max(c[-21:-1]):
        tags.append("breakout_20d")
    if c[-6] / c[-16] - 1 > 0.10 and (max(c[-5:]) - min(c[-5:])) / last < 0.05:
        tags.append("bull_flag")
    if c[-2] < sum(c[-21:-1]) / 20 and last > sum(c[-20:]) / 20:
        tags.append("reclaim_ma20")
    if r is not None and r < 30:
        tags.append("oversold")
    if r_prev is not None and r_prev < 30 and last > c[-2]:
        tags.append("oversold_bounce")
    return tags, r


def range_state(bars, day, now: datetime):
    """The opening range of one session from its 5-minute bars (objects with
    timestamp=bar start, high, low, close), using only bars complete by `now`.
    None when a range bar is missing. Otherwise a dict: hi, lo, first =
    (side "above"/"below", close, close time ET) of the first completed bar to
    close outside the range or None, and last_close of the last completed bar
    before 4:00 PM."""
    open_ = datetime.combine(day, time(9, 30), tzinfo=ET)
    close = datetime.combine(day, time(16, 0), tzinfo=ET)
    # regular session only: Alpaca's bars include pre-market and after-hours trading
    done = sorted((b for b in bars
                   if open_ <= b.timestamp < close and b.timestamp + BAR <= now),
                  key=lambda b: b.timestamp)
    rng = [b for b in done if b.timestamp < open_ + RANGE_BARS * BAR]
    if {b.timestamp.astimezone(ET) for b in rng} != {open_ + k * BAR for k in range(RANGE_BARS)}:
        return None
    hi, lo = max(float(b.high) for b in rng), min(float(b.low) for b in rng)
    first = None
    for b in done[len(rng):]:
        c = float(b.close)
        if c > hi or c < lo:
            first = ("above" if c > hi else "below", c, (b.timestamp + BAR).astimezone(ET))
            break
    return {"hi": hi, "lo": lo, "first": first, "last_close": float(done[-1].close)}


def opening_range(bars, now: datetime) -> str:
    """Today's opening range as a short phrase for the prompt."""
    day = now.astimezone(ET).date()
    st = range_state(bars, day, now)
    if st is None:
        ready = datetime.combine(day, time(9, 30), tzinfo=ET) + RANGE_BARS * BAR
        return "opening range n/a" if now >= ready else "opening range forming"
    hi, lo = st["hi"], st["lo"]
    text = f"opening range {lo:.2f}-{hi:.2f}"
    if st["first"] is None:
        return f"{text}: no close outside it yet"
    side, c, at = st["first"]
    ref = hi if side == "above" else lo
    last = st["last_close"]
    now_side = "above" if last > hi else "below" if last < lo else "inside"
    return (f"{text}: first close {side} it {at:%H:%M} ({(c / ref - 1) * 100:+.2f}%), "
            f"last 5-min close {now_side} it")


def setup_lines(closes_by_ticker: dict, bars_by_ticker: dict | None, now: datetime) -> list[str]:
    """One line per ticker. bars_by_ticker None means intraday bars were unavailable."""
    out = []
    for s in sorted(closes_by_ticker):
        tags, r = daily_setups(closes_by_ticker[s])
        parts = [", ".join(tags) or "no daily setup",
                 f"rsi14 {r:.0f}" if r is not None else "rsi14 n/a"]
        if bars_by_ticker is not None:
            parts.append(opening_range(bars_by_ticker.get(s, []), now))
        out.append(f"{s} " + "; ".join(parts))
    return out
