"""Integration tests against a real Alpaca PAPER account.

Skipped unless ALPACA_API_KEY / ALPACA_SECRET_KEY are set (env or .env).
Refuses to run if the broker isn't in paper mode.

    python -m pytest -v test_broker_paper.py                         # read-only
    RUN_PAPER_ORDER_TESTS=1 python -m pytest -v test_broker_paper.py # also places paper orders

The order tests cancel and liquidate everything in the paper account.
"""
import os
import time

import pytest
from dotenv import load_dotenv

load_dotenv()

from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest

from broker import AlpacaBroker
from guardrails import Guardrails, Limits

pytestmark = pytest.mark.skipif(
    not (os.getenv("ALPACA_API_KEY") and os.getenv("ALPACA_SECRET_KEY")),
    reason="no Alpaca paper keys")

orders_enabled = pytest.mark.skipif(
    os.getenv("RUN_PAPER_ORDER_TESTS") != "1", reason="RUN_PAPER_ORDER_TESTS!=1")


@pytest.fixture(scope="module")
def broker():
    b = AlpacaBroker.from_env()
    assert b.paper, "integration tests only run against paper"
    return b


def wait_for(cond, timeout=20):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(1)
    return False


def test_reads(broker):
    eq, cash = broker.equity(), broker.settled_cash()
    acct = broker.client.get_account()
    print(f"\nequity={eq} settled_cash={cash} cash={acct.cash} "
          f"buying_power={acct.buying_power} nmbp={acct.non_marginable_buying_power} "
          f"multiplier={acct.multiplier}")
    assert eq >= 0 and cash >= 0
    assert cash <= eq + 1e-6 or eq == 0
    assert broker.position_value("SPY") >= 0
    print("safety problems:", broker.safety_problems())


def test_position_value_of_unheld_ticker_is_zero(broker):
    assert broker.position_value("ZZZZ") == 0.0  # 404 path must not raise


@orders_enabled
def test_pending_order_counts_and_reserves_cash(broker):
    """Answers an open question: does Alpaca reduce buying power for open orders?"""
    broker.cancel_all_orders()
    before_cash = broker.settled_cash()
    before_pos = broker.position_value("SPY")
    o = broker.client.submit_order(LimitOrderRequest(
        symbol="SPY", qty=1, side=OrderSide.BUY, limit_price=1.00,
        time_in_force=TimeInForce.DAY))
    try:
        assert wait_for(lambda: broker.position_value("SPY") >= before_pos + 1.0)
        after_cash = broker.settled_cash()
        print(f"\nsettled_cash before={before_cash} with $1 open buy={after_cash}")
    finally:
        broker.client.cancel_order_by_id(o.id)
    assert wait_for(lambda: not broker.open_orders("SPY"))


@orders_enabled
def test_forced_floor_breach_liquidates_and_halts(broker):
    if not broker.client.get_clock().is_open:
        pytest.skip("market closed: a market buy wouldn't fill")
    broker.client.submit_order(MarketOrderRequest(
        symbol="SPY", notional=2, side=OrderSide.BUY, time_in_force=TimeInForce.DAY))
    assert wait_for(lambda: broker.position_value("SPY") > 0 and not broker.open_orders())

    g = Guardrails(broker, Limits(equity_floor=broker.equity() + 1_000_000))
    ok, why = g.check_order("SPY", "buy", 1)
    assert not ok and g.halted, why
    assert wait_for(lambda: not broker.client.get_all_positions(), timeout=60)
    assert not g.check_order("SPY", "buy", 1)[0]  # stays halted
