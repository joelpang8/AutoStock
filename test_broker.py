"""Unit tests for AlpacaBroker against a fake TradingClient. No network.

Run: python -m pytest test_broker.py
"""
import json
from types import SimpleNamespace as NS

import pytest
from alpaca.common.exceptions import APIError
from alpaca.trading.enums import AccountStatus, OrderSide

from broker import (LIVE_CONFIRM_VALUE, AlpacaBroker, BrokerDataError,
                    BrokerError)
from guardrails import Guardrails


def api_error(status):
    http = NS(response=NS(status_code=status), request=None)
    return APIError(json.dumps({"code": status, "message": "x"}), http)


def account(**over):
    a = dict(status=AccountStatus.ACTIVE, equity="300.00", cash="250.00",
             buying_power="250.00", non_marginable_buying_power="250.00",
             multiplier="1", trading_blocked=False, account_blocked=False,
             trade_suspended_by_user=False)
    a.update(over)
    return NS(**a)


class FakeClient:
    def __init__(self):
        self.acct = account()
        self.positions = {}
        self.orders = []
        self.cfg = NS(max_margin_multiplier="1", no_shorting=True, fractional_trading=True)
        self.cancel_result = []
        self.close_result = []
        self.close_kwargs = None
        self.fail_account = []  # exceptions to raise on successive get_account calls

    def get_account(self):
        if self.fail_account:
            raise self.fail_account.pop(0)
        return self.acct

    def get_open_position(self, symbol):
        if symbol not in self.positions:
            raise api_error(404)
        return NS(market_value=self.positions[symbol])

    def get_orders(self, filter=None):
        syms = filter.symbols if filter else None
        return [o for o in self.orders if not syms or o.symbol in syms]

    def cancel_orders(self):
        return self.cancel_result

    def close_all_positions(self, cancel_orders=None):
        self.close_kwargs = {"cancel_orders": cancel_orders}
        return self.close_result

    def get_account_configurations(self):
        return self.cfg


def order(symbol="SPY", side=OrderSide.BUY, notional=None, qty=None,
          limit_price=None, filled_qty="0", filled_avg_price=None):
    return NS(id="o1", symbol=symbol, side=side, notional=notional, qty=qty,
              limit_price=limit_price, filled_qty=filled_qty,
              filled_avg_price=filled_avg_price)


@pytest.fixture
def fc():
    return FakeClient()


@pytest.fixture
def broker(fc):
    return AlpacaBroker(fc, paper=True, retry_delay=0)


# --- equity ---------------------------------------------------------------
def test_equity_parses(broker):
    assert broker.equity() == 300.0


@pytest.mark.parametrize("bad", [None, "", "abc", "nan", "inf", "-1"])
def test_equity_rejects_bad_values(broker, fc, bad):
    fc.acct = account(equity=bad)
    with pytest.raises(BrokerDataError):
        broker.equity()


def test_equity_rejects_inactive_account(broker, fc):
    fc.acct = account(status=AccountStatus.ACCOUNT_CLOSED)
    with pytest.raises(BrokerDataError):
        broker.equity()


def test_transient_error_retried(broker, fc):
    fc.fail_account = [api_error(503), ConnectionError("reset")]
    assert broker.equity() == 300.0


def test_persistent_error_raises(broker, fc):
    fc.fail_account = [api_error(503)] * 3
    with pytest.raises(BrokerError):
        broker.equity()


def test_client_error_not_retried(broker, fc):
    fc.fail_account = [api_error(401), None]
    with pytest.raises(BrokerError):
        broker.equity()
    assert fc.fail_account == [None]  # only one attempt was made


# --- settled cash ---------------------------------------------------------
def test_settled_cash_takes_minimum(broker, fc):
    fc.acct = account(cash="250", buying_power="500", non_marginable_buying_power="240")
    assert broker.settled_cash() == 240.0


def test_settled_cash_never_negative(broker, fc):
    fc.acct = account(cash="-5", buying_power="0", non_marginable_buying_power="0")
    assert broker.settled_cash() == 0.0


def test_settled_cash_zero_when_blocked(broker, fc):
    fc.acct = account(trading_blocked=True)
    assert broker.settled_cash() == 0.0


def test_settled_cash_missing_field_raises(broker, fc):
    fc.acct = account(non_marginable_buying_power=None)
    with pytest.raises(BrokerDataError):
        broker.settled_cash()


# --- position value -------------------------------------------------------
def test_position_value_no_position(broker):
    assert broker.position_value("SPY") == 0.0


def test_position_value_includes_pending_buys(broker, fc):
    fc.positions["SPY"] = "40.00"
    fc.orders = [order(notional="15"),
                 order(qty="0.1", limit_price="500"),            # $50 bound
                 order(side=OrderSide.SELL, notional="10"),        # sells ignored
                 order(symbol="QQQ", notional="20")]               # other ticker
    assert broker.position_value("SPY") == pytest.approx(40 + 15 + 50)


def test_partially_filled_notional_counts_remainder(broker, fc):
    fc.orders = [order(notional="20", filled_qty="0.02", filled_avg_price="500")]
    assert broker.position_value("SPY") == pytest.approx(10)


def test_unpriceable_open_order_raises(broker, fc):
    fc.orders = [order(qty="1")]  # market order by qty: no price bound
    with pytest.raises(BrokerDataError):
        broker.position_value("SPY")


def test_position_lookup_error_raises(broker, fc):
    fc.get_open_position = lambda s: (_ for _ in ()).throw(api_error(500))
    with pytest.raises(BrokerError):
        broker.position_value("SPY")


# --- cancel / close -------------------------------------------------------
def test_cancel_all_ok(broker, fc):
    fc.cancel_result = [NS(id="a", status=200), NS(id="b", status=204)]
    broker.cancel_all_orders()


def test_cancel_all_reports_failures(broker, fc):
    fc.cancel_result = [NS(id="a", status=200), NS(id="b", status=500)]
    with pytest.raises(BrokerError, match="b=500"):
        broker.cancel_all_orders()


def test_close_all_cancels_orders_too(broker, fc):
    fc.close_result = [NS(symbol="SPY", status=200)]
    broker.close_all_positions()
    assert fc.close_kwargs == {"cancel_orders": True}


def test_close_all_reports_failures(broker, fc):
    fc.close_result = [NS(symbol="SPY", status=200), NS(symbol="QQQ", status=403)]
    with pytest.raises(BrokerError, match="QQQ=403"):
        broker.close_all_positions()


# --- account safety -------------------------------------------------------
def test_safety_ok(broker):
    assert broker.safety_problems() == []


def test_safety_flags_margin_and_shorting(broker, fc):
    fc.acct = account(multiplier="2")
    fc.cfg = NS(max_margin_multiplier="4", no_shorting=False, fractional_trading=False)
    assert len(broker.safety_problems()) == 4


# --- live-trading gate ----------------------------------------------------
KEYS = {"ALPACA_API_KEY": "k", "ALPACA_SECRET_KEY": "s"}


def test_from_env_defaults_to_paper():
    assert AlpacaBroker.from_env(KEYS).paper is True


def test_from_env_requires_keys():
    with pytest.raises(BrokerError):
        AlpacaBroker.from_env({})


def test_live_refused_without_confirmation():
    with pytest.raises(BrokerError, match="live trading refused"):
        AlpacaBroker.from_env({**KEYS, "ALPACA_PAPER": "false"})
    with pytest.raises(BrokerError):
        AlpacaBroker.from_env({**KEYS, "ALPACA_PAPER": "false", "ALLOW_LIVE_TRADING": "yes"})


def test_live_allowed_with_confirmation():
    b = AlpacaBroker.from_env({**KEYS, "ALPACA_PAPER": "false",
                               "ALLOW_LIVE_TRADING": LIVE_CONFIRM_VALUE})
    assert b.paper is False


def test_typo_in_paper_flag_refused():
    with pytest.raises(BrokerError):
        AlpacaBroker.from_env({**KEYS, "ALPACA_PAPER": "flase"})


# --- wired through the guardrails ----------------------------------------
def test_guardrails_with_real_broker_class(broker, fc):
    g = Guardrails(broker)
    assert g.check_order("SPY", "buy", 15) == (True, "ok")
    fc.positions["SPY"] = "50"
    assert not g.check_order("SPY", "buy", 15)[0]  # per-ticker cap


def test_guardrails_floor_breach_liquidates(broker, fc):
    fc.acct = account(equity="199.50")
    fc.close_result = [NS(symbol="SPY", status=200)]
    g = Guardrails(broker)
    ok, why = g.check_order("SPY", "buy", 10)
    assert not ok and g.halted and "below floor" in why
    assert fc.close_kwargs == {"cancel_orders": True}


def test_guardrails_unreadable_equity_blocks_trading(broker, fc):
    fc.acct = account(equity="nan")
    with pytest.raises(BrokerDataError):
        Guardrails(broker).check_order("SPY", "buy", 10)
