"""Historical odds: daily-setup event study and the opening-range history."""
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace as NS

from data import ET
from odds import daily_odds, odds_lines, orb_odds


def test_daily_odds_counts_forward_returns():
    # flat, then a breakout day followed by a rise: breakout_20d fires once
    closes = [100.0] * 40 + [101.0, 102.0, 103.0, 104.0, 105.0, 106.0]
    per_tag, base = daily_odds(closes)
    n, ups, med = per_tag["breakout_20d"]
    assert n >= 1 and ups == n
    assert base[0] == len(closes) - 29 - 5   # every day with enough history and 5 days ahead


def test_daily_odds_short_history_is_empty():
    assert daily_odds([100.0] * 20) == ({}, None)


# --- opening-range history ------------------------------------------------------
def session(day, path, hi=1.0, lo=1.0):
    """5-minute bars 9:30..15:55 for `day`, closes from `path` (padded with its last value)."""
    path = path + [path[-1]] * (78 - len(path))
    start = datetime(day.year, day.month, day.day, 9, 30, tzinfo=ET)
    return [NS(timestamp=(start + timedelta(minutes=5 * i)).astimezone(timezone.utc),
               high=c + hi, low=c - lo, close=c) for i, c in enumerate(path)]


def test_orb_odds_up_break_that_holds_and_one_that_fails():
    up_holds = session(date(2026, 10, 19), [100, 100, 100, 102, 103] + [104] * 73)
    up_fails = session(date(2026, 10, 20), [100, 100, 100, 102] + [99] * 74)
    no_break = session(date(2026, 10, 21), [100] * 78)
    o = orb_odds(up_holds + up_fails + no_break)
    assert o["sessions"] == 3
    n, ups, _ = o["above"]
    assert (n, ups) == (2, 1)
    assert "below" not in o


def test_orb_odds_ignores_premarket_and_after_hours_bars():
    day = date(2026, 10, 19)
    regular = session(day, [100, 100, 100, 102] + [101] * 74)
    extra = [NS(timestamp=(datetime(2026, 10, 19, h, m, tzinfo=ET)).astimezone(timezone.utc),
                high=200, low=1, close=150) for h, m in ((8, 0), (9, 0), (16, 0), (16, 30))]
    o = orb_odds(regular + extra)
    assert o["sessions"] == 1
    n, ups, med = o["above"]
    assert (n, ups) == (1, 0) and round(med, 2) == -0.98   # 102 -> 101, after-hours 150 ignored


def test_orb_odds_skips_short_sessions():
    half_day = session(date(2026, 11, 27), [100, 100, 100, 102])[:40]   # ends 12:50
    assert orb_odds(half_day)["sessions"] == 0


# --- prompt lines -----------------------------------------------------------------
def test_odds_lines_only_for_active_setups():
    breakout = [100.0] * 40 + [101.0]
    flat = [100.0] * 41
    orb = {"SPY": {"sessions": 58, "above": (21, 11, 0.04)}, "QQQ": {"sessions": 58}}
    lines = odds_lines({"SPY": breakout, "QQQ": flat}, orb, {"SPY": "above", "QQQ": "below"})
    assert any(l.startswith("SPY breakout_20d") for l in lines)
    assert ("SPY first close above the opening range: happened 21 of the last 58 sessions; "
            "from that close to the 4pm close up in 11/21 (52%), median +0.04%") in lines
    assert "QQQ first close below the opening range: not seen in the last 58 sessions" in lines
    assert not any(l.startswith("QQQ ") and "breakout" in l for l in lines)


def test_small_samples_are_flagged():
    lines = odds_lines({"SPY": [100.0] * 30}, {"SPY": {"sessions": 5, "above": (3, 2, 0.1)}},
                       {"SPY": "above"})
    assert "small sample" in lines[0]


def test_no_history_means_no_orb_line():
    assert odds_lines({"SPY": [100.0] * 30}, None, {"SPY": "above"}) == []
