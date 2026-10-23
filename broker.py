"""Alpaca broker adapter implementing the interface guardrails.py expects.

    equity(), settled_cash(), position_value(ticker),
    cancel_all_orders(), close_all_positions()

Design rules:
- Every read goes to Alpaca fresh. If a value is missing, non-numeric, NaN or
  negative, raise BrokerDataError instead of returning a guess. The caller must
  treat that as "account state unreadable: don't trade".
- Paper is the default. Live requires ALPACA_PAPER=false *and*
  ALLOW_LIVE_TRADING=yes-i-have-owner-approval, so it can't happen by accident.

Alpaca account facts (checked against docs.alpaca.markets, Oct 2026):
- Alpaca opens every individual account as a margin account; under $2,000
  equity it is "limited margin" (multiplier 1, buying_power == cash). There is
  no cash-account type for individuals and no settled-cash field in the API.
  Alpaca fronts the T+1 float, so sale proceeds are spendable immediately and
  there are no good-faith violations to avoid.
- So settled_cash() returns the most conservative spendable cash Alpaca
  reports: min(cash, buying_power, non_marginable_buying_power), never below 0.
  This guarantees the guardrail never approves a buy that would borrow.
"""
from __future__ import annotations

import logging
import math
import os
import time

from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AccountStatus, OrderSide, QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest

log = logging.getLogger(__name__)

LIVE_CONFIRM_VALUE = "yes-i-have-owner-approval"


class BrokerError(Exception):
    """An Alpaca call failed or returned something we can't act on."""


class BrokerDataError(BrokerError):
    """Account data was missing or invalid. Don't trade on it."""


def _money(value, name: str) -> float:
    """Parse an Alpaca money string. Raise instead of guessing."""
    if value is None:
        raise BrokerDataError(f"{name} missing from Alpaca response")
    try:
        x = float(value)
    except (TypeError, ValueError):
        raise BrokerDataError(f"{name} not a number: {value!r}") from None
    if not math.isfinite(x):
        raise BrokerDataError(f"{name} not finite: {value!r}")
    return x


def _is_transient(err: Exception) -> bool:
    if isinstance(err, APIError):
        code = err.status_code
        return code is None or code == 429 or code >= 500
    return isinstance(err, (ConnectionError, TimeoutError, OSError))


class AlpacaBroker:
    def __init__(self, client: TradingClient, *, paper: bool,
                 retries: int = 3, retry_delay: float = 1.0):
        self.client = client
        self.paper = paper
        self.retries = retries
        self.retry_delay = retry_delay

    @classmethod
    def from_env(cls, env=os.environ) -> "AlpacaBroker":
        key = env.get("ALPACA_API_KEY")
        secret = env.get("ALPACA_SECRET_KEY")
        if not key or not secret:
            raise BrokerError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set")
        paper_flag = env.get("ALPACA_PAPER", "true").strip().lower()
        if paper_flag not in ("true", "false"):
            raise BrokerError(f"ALPACA_PAPER must be true or false, got {paper_flag!r}")
        paper = paper_flag == "true"
        if not paper and env.get("ALLOW_LIVE_TRADING") != LIVE_CONFIRM_VALUE:
            raise BrokerError(
                "live trading refused: set ALLOW_LIVE_TRADING="
                f"{LIVE_CONFIRM_VALUE} only after the owner approves")
        log.info("connecting to Alpaca (%s)", "paper" if paper else "LIVE")
        return cls(TradingClient(key, secret, paper=paper), paper=paper)

    # --- plumbing ----------------------------------------------------------
    def _call(self, what: str, fn, *args, **kwargs):
        """Retry transient failures. Only use for reads and idempotent deletes,
        never for order submission."""
        for attempt in range(1, self.retries + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                if attempt < self.retries and _is_transient(e):
                    log.warning("%s failed (attempt %d): %s; retrying", what, attempt, e)
                    time.sleep(self.retry_delay * attempt)
                    continue
                raise BrokerError(f"{what} failed: {e}") from e

    def _account(self):
        acct = self._call("get_account", self.client.get_account)
        if acct.status != AccountStatus.ACTIVE:
            raise BrokerDataError(f"account status is {acct.status}, not ACTIVE")
        return acct

    # --- interface used by guardrails.py ----------------------------------
    def equity(self) -> float:
        x = _money(self._account().equity, "equity")
        if x < 0:
            raise BrokerDataError(f"equity negative: {x}")
        return x

    def settled_cash(self) -> float:
        """Conservative spendable cash. See module docstring for why this is not
        literally 'settled' cash: Alpaca has no cash account or settled field."""
        acct = self._account()
        if acct.trading_blocked or acct.account_blocked or acct.trade_suspended_by_user:
            return 0.0
        vals = [_money(acct.cash, "cash"),
                _money(acct.buying_power, "buying_power"),
                _money(acct.non_marginable_buying_power, "non_marginable_buying_power")]
        return max(0.0, min(vals))

    def position_value(self, ticker: str) -> float:
        """Market value held in `ticker` plus any open buy orders for it, so the
        per-ticker cap also counts orders that haven't filled yet."""
        held = 0.0
        try:
            pos = self.client.get_open_position(ticker)
            held = _money(pos.market_value, f"{ticker} market_value")
        except APIError as e:
            if e.status_code != 404:  # 404 = no position, anything else is a failure
                raise BrokerError(f"get_open_position({ticker}) failed: {e}") from e
        except Exception as e:
            raise BrokerError(f"get_open_position({ticker}) failed: {e}") from e
        return held + self.pending_buy_usd(ticker)

    def cancel_all_orders(self) -> None:
        results = self._call("cancel_orders", self.client.cancel_orders) or []
        bad = [r for r in results if not 200 <= int(r.status) < 300]
        if bad:
            raise BrokerError(f"cancel failed for {len(bad)} orders: "
                              + ", ".join(f"{r.id}={r.status}" for r in bad))
        log.warning("cancelled %d open orders", len(results))

    def close_all_positions(self) -> None:
        """Market-sell every position. Outside regular hours Alpaca queues these
        for the next open, so callers must not assume the account is flat yet."""
        results = self._call("close_all_positions", self.client.close_all_positions,
                             cancel_orders=True) or []
        bad = [r for r in results if r.status is None or not 200 <= int(r.status) < 300]
        if bad:
            raise BrokerError(f"liquidation failed for: "
                              + ", ".join(f"{r.symbol}={r.status}" for r in bad))
        log.warning("liquidation orders placed for %d positions", len(results))

    # --- extras ------------------------------------------------------------
    def open_orders(self, ticker: str | None = None):
        req = GetOrdersRequest(status=QueryOrderStatus.OPEN,
                               symbols=[ticker] if ticker else None)
        return self._call("get_orders", self.client.get_orders, filter=req) or []

    def pending_buy_usd(self, ticker: str | None = None) -> float:
        """Upper-bound dollar value of unfilled buy orders."""
        total = 0.0
        for o in self.open_orders(ticker):
            if o.side != OrderSide.BUY:
                continue
            filled = _money(o.filled_avg_price, "filled_avg_price") * _money(o.filled_qty, "filled_qty") \
                if o.filled_avg_price else 0.0
            if o.notional is not None:
                total += max(0.0, _money(o.notional, "notional") - filled)
            elif o.qty is not None and o.limit_price is not None:
                remaining = _money(o.qty, "qty") - _money(o.filled_qty or 0, "filled_qty")
                total += max(0.0, remaining) * _money(o.limit_price, "limit_price")
            else:
                # market order by quantity: no price bound we can trust
                raise BrokerDataError(f"can't value open order {o.id} for {o.symbol}")
        return total

    def safety_problems(self) -> list[str]:
        """Account settings that would let the bot borrow or short. Empty = ok.
        Read-only: it never changes account settings."""
        acct = self._account()
        cfg = self._call("get_account_configurations", self.client.get_account_configurations)
        problems = []
        if str(acct.multiplier) != "1":
            problems.append(f"account multiplier is {acct.multiplier}, expected 1 (no margin)")
        if str(cfg.max_margin_multiplier) != "1":
            problems.append(f"max_margin_multiplier is {cfg.max_margin_multiplier}, set it to 1")
        if not cfg.no_shorting:
            problems.append("no_shorting is off, turn it on (long-only)")
        if not cfg.fractional_trading:
            problems.append("fractional_trading is off; dollar-amount orders need it")
        if acct.trading_blocked or acct.account_blocked or acct.trade_suspended_by_user:
            problems.append("account is blocked or suspended from trading")
        return problems
