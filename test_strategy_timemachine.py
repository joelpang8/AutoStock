"""Strategy Time Machine plumbing: the replayed years, the flat world, the signals
and the floor. No network."""
import math
import random
import statistics

import pytest

import strategy_timemachine as tm


def toy_returns(n=600, seed=1):
    rng = random.Random(seed)
    q = [rng.gauss(0.0008, 0.012) for _ in range(n)]
    return {"SPY": [x * 0.7 for x in q], "QQQ": q, "QLD": [2 * x - 0.00005 for x in q]}


def test_replay_year_keeps_tickers_on_the_same_days():
    rets = toy_returns()
    year = tm.replay_year(rets, random.Random(3))
    assert all(len(year[s]) == tm.YEAR for s in tm.TICKERS)
    for k in range(tm.YEAR):
        assert year["SPY"][k] == pytest.approx(year["QQQ"][k] * 0.7)


def test_flat_world_has_no_drift_but_keeps_swings():
    rets = toy_returns()
    flat = tm.flat_world(rets)
    assert statistics.fmean(math.log1p(r) for r in flat["QQQ"]) == pytest.approx(0, abs=1e-12)
    assert statistics.stdev(flat["QQQ"]) == pytest.approx(statistics.stdev(rets["QQQ"]), rel=0.01)
    # QLD keeps its measured drag against 2x QQQ
    drag = statistics.fmean(2 * q - l for q, l in zip(rets["QQQ"], rets["QLD"]))
    assert flat["QLD"][0] == pytest.approx(2 * flat["QQQ"][0] - drag)


def test_signals_match_slow_versions():
    qqq = [100 + math.sin(i / 9) * 5 + i * 0.05 for i in range(400)]
    sig = tm.Signals(qqq)
    for i in (250, 333, 399):
        assert sig.above_sma(i) == (qqq[i] > sum(qqq[i - 199:i + 1]) / 200)
        rets = [qqq[j] / qqq[j - 1] - 1 for j in range(i - 19, i + 1)]
        assert sig.vol(i) == pytest.approx(statistics.stdev(rets) * math.sqrt(252), rel=1e-6)


def flat_prices(n, qld_path):
    return {"SPY": [100.0] * n, "QQQ": [100.0 + 0.1 * k for k in range(n)], "QLD": qld_path}


def test_trend_strategy_follows_qld_and_pays_the_spread():
    n = 300
    qld = [50.0] * 260 + [50.0 * 1.01 ** k for k in range(1, 41)]
    end, hit, _ = tm.run_year(tm.qld_trend, flat_prices(n, qld), 259)
    # fully in QLD from the first replayed close; QLD then rises 1% a day
    assert not hit
    # the spread is paid once, in cash, when the $300 goes in
    assert end == pytest.approx(300 * 1.01 ** 39 - 300 * tm.SPREAD_PCT / 200, rel=1e-9)


def test_floor_sells_everything_and_stops():
    n = 300
    qld = [50.0] * 261 + [30.0] * 39          # -40% the day after buying
    end, hit, low = tm.run_year(tm.qld_trend, flat_prices(n, qld), 259)
    assert hit and end < tm.FLOOR and low == end


def test_risk_off_holds_cash():
    n = 300
    px = {"SPY": [100.0] * n, "QQQ": [200.0 - 0.1 * k for k in range(n)], "QLD": [50.0] * n}
    assert tm.run_year(tm.qld_trend, px, 259)[0] == tm.START_CASH


def test_run_is_repeatable_with_the_same_seed():
    rets = toy_returns(900)
    hist = {s: [100.0] for s in tm.TICKERS}
    for s in tm.TICKERS:
        for r in rets[s]:
            hist[s].append(hist[s][-1] * (1 + r))
    a = tm.run(hist, replays=30, seed=5)
    b = tm.run(hist, replays=30, seed=5)
    assert a == b
    assert set(a) == {"history mix", "flat market"}
