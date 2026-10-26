"""Chart setup tags: the daily rules, the opening range, and the prompt section.
(The daily rules were also checked against the Day Trader skill's scanner on
20,000 random price series with no mismatch.)"""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS

from data import ET
from setups import daily_setups, opening_range, rsi, setup_lines

FLAT = [100.0] * 40


def test_too_little_history_gives_no_tags():
    assert daily_setups([100.0] * 29) == ([], None)


def test_breakout_20d():
    tags, _ = daily_setups(FLAT[:-1] + [101.0])
    assert "breakout_20d" in tags


def test_no_breakout_when_equal_to_high():
    assert "breakout_20d" not in daily_setups(FLAT)[0]


def test_bull_flag():
    c = [100.0] * 20 + [100 + 1.5 * i for i in range(1, 11)] + [115.0, 114.0, 115.5, 114.5, 115.0]
    assert "bull_flag" in daily_setups(c)[0]


def test_reclaim_ma20():
    c = [100.0] * 30 + [98.0, 102.0]
    assert "reclaim_ma20" in daily_setups(c)[0]


def test_oversold_and_bounce():
    falling = [100 * 0.98 ** i for i in range(40)]
    tags, r = daily_setups(falling)
    assert "oversold" in tags and r < 30
    tags, _ = daily_setups(falling + [falling[-1] * 1.01])
    assert "oversold_bounce" in tags


def test_rsi_all_gains_is_100():
    assert rsi([float(i) for i in range(20)]) == 100.0


# --- opening range --------------------------------------------------------------
DAY = datetime(2026, 10, 22, tzinfo=ET)


def bar(hh, mm, h, l, c):
    return NS(timestamp=DAY.replace(hour=hh, minute=mm).astimezone(timezone.utc), high=h, low=l, close=c)


RANGE = [bar(9, 30, 101, 99, 100), bar(9, 35, 102, 99.5, 101), bar(9, 40, 101.5, 98.5, 100)]


def at(hh, mm):
    return DAY.replace(hour=hh, minute=mm).astimezone(timezone.utc)


def test_breakout_above_range_reports_close_time():
    bars = RANGE + [bar(9, 45, 101.8, 100, 101.5), bar(9, 50, 103, 101, 102.5)]
    assert opening_range(bars, at(10, 0)) == \
        "opening range 98.50-102.00: first close above it 09:55 (+0.49%), last 5-min close above it"


def test_reversal_after_first_break_is_reported():
    bars = RANGE + [bar(9, 45, 99, 97, 98.0), bar(9, 50, 103, 98, 102.5)]
    assert opening_range(bars, at(10, 0)).endswith(
        "first close below it 09:50 (-0.51%), last 5-min close above it")


def test_breakdown_below_range():
    bars = RANGE + [bar(9, 45, 99, 97, 98.0)]
    assert "first close below it 09:50" in opening_range(bars, at(9, 51))


def test_bar_still_forming_is_ignored():
    bars = RANGE + [bar(9, 45, 105, 100, 104)]
    assert opening_range(bars, at(9, 48)).endswith("no close outside it yet")


def test_range_forming_before_945():
    assert opening_range(RANGE[:2], at(9, 42)) == "opening range forming"


def test_missing_range_bar_is_na():
    assert opening_range([RANGE[0], RANGE[2]], at(10, 30)) == "opening range n/a"


def test_yesterdays_bars_are_ignored():
    old = [NS(timestamp=b.timestamp - timedelta(days=1), high=b.high, low=b.low, close=b.close)
           for b in RANGE]
    assert opening_range(old, at(10, 30)) == "opening range n/a"


# --- prompt lines ---------------------------------------------------------------
def test_setup_lines():
    lines = setup_lines({"SPY": FLAT[:-1] + [101.0], "QQQ": [1.0] * 5},
                        {"SPY": RANGE + [bar(9, 45, 103, 101, 102.5)]}, at(10, 0))
    assert lines[0] == "QQQ no daily setup; rsi14 n/a; opening range n/a"
    assert lines[1].startswith("SPY breakout_20d; rsi14 100; opening range 98.50-102.00: "
                               "first close above it 09:50")


def test_setup_lines_without_intraday():
    assert setup_lines({"SPY": FLAT}, None, at(10, 0)) == ["SPY no daily setup; rsi14 100"]
