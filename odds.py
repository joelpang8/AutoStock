"""Historical odds for the setups a ticker shows right now, after the history
check in the Day Trader skill's Time Machine (github.com/Jakeschincariol/
day-trader-skill, MIT, Jake Schincariol): before acting on a setup, look at
every past time the same rule fired on the same ticker and count what followed.

- Daily setups (setups.daily_setups) over the past year of closes: the return
  5 sessions after each day the setup fired, against all days as a baseline.
- Opening-range breaks over the past ~60 sessions of 5-minute bars: the return
  from the first close outside the range to that session's close.

These are small samples from one ticker's own past, not predictions; the prompt
says so. There is no simulation here (the skill's 10,000 replays need a stop and
a target, which this bot doesn't use).
"""
from __future__ import annotations

import statistics
from collections import defaultdict
from datetime import datetime, time

from data import ET
from setups import MIN_DAILY, daily_setups, range_state

HORIZON = 5                   # sessions after a daily setup
LAST_BAR = time(15, 30)       # a session counts for the opening-range check only if
                              # its bars reach at least this late (no half-day gaps)
SMALL = 10


def _stats(rets) -> tuple[int, int, float]:
    return len(rets), sum(r > 0 for r in rets), statistics.median(rets) * 100


def daily_odds(closes) -> tuple[dict, tuple | None]:
    """({tag: (n, ups, median %)}, baseline over all days) for 5-session-forward returns."""
    by_tag, every = defaultdict(list), []
    for i in range(MIN_DAILY - 1, len(closes) - HORIZON):
        ret = closes[i + HORIZON] / closes[i] - 1
        every.append(ret)
        for tag in daily_setups(closes[:i + 1])[0]:
            by_tag[tag].append(ret)
    return ({t: _stats(r) for t, r in by_tag.items()},
            _stats(every) if every else None)


def orb_odds(bars) -> dict:
    """{"sessions": n, "above": (n, ups, median %), "below": ...} from past sessions'
    5-minute bars: return from the first close outside the opening range to the
    session's last close."""
    days = defaultdict(list)
    for b in bars:
        days[b.timestamp.astimezone(ET).date()].append(b)
    out, sessions = {"above": [], "below": []}, 0
    for day, day_bars in days.items():
        end = datetime.combine(day, time(16, 0), tzinfo=ET)
        regular = [b.timestamp.astimezone(ET).time() for b in day_bars
                   if b.timestamp.astimezone(ET).time() < time(16, 0)]
        st = range_state(day_bars, day, end)
        if st is None or not regular or max(regular) < LAST_BAR:
            continue
        sessions += 1
        if st["first"]:
            side, price, _ = st["first"]
            out[side].append(st["last_close"] / price - 1)
    return {"sessions": sessions,
            **{side: _stats(r) for side, r in out.items() if r}}


def _fmt(n, ups, med) -> str:
    small = ", small sample" if n < SMALL else ""
    return f"up in {ups}/{n} ({ups / n * 100:.0f}%{small}), median {med:+.2f}%"


def odds_lines(closes_by_ticker: dict, orb_by_ticker: dict | None, today_side: dict) -> list[str]:
    """Lines only for setups active now. today_side: {ticker: "above"/"below"} for
    today's first opening-range break. orb_by_ticker None: history unavailable."""
    out = []
    for s in sorted(closes_by_ticker):
        closes = closes_by_ticker[s]
        tags = daily_setups(closes)[0]
        if tags:
            per_tag, base = daily_odds(closes)
            for t in tags:
                if t in per_tag:
                    out.append(f"{s} {t}, past year, 5 sessions later: {_fmt(*per_tag[t])}"
                               + (f" (all days: {_fmt(*base)})" if base else ""))
                else:
                    out.append(f"{s} {t}: never fired before in the past year")
        side = today_side.get(s)
        if side and orb_by_ticker is not None and s in orb_by_ticker:
            o = orb_by_ticker[s]
            if side in o:
                n = o[side][0]
                out.append(f"{s} first close {side} the opening range: happened {n} of the last "
                           f"{o['sessions']} sessions; from that close to the 4pm close "
                           f"{_fmt(*o[side])}")
            else:
                out.append(f"{s} first close {side} the opening range: not seen in the last "
                           f"{o['sessions']} sessions")
    return out
