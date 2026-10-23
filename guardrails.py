"""Guardrails for the trading bot.

Every order the AI proposes must pass through Guardrails.check_order().
These rules are plain code: the AI cannot override them.
"""
import time
from dataclasses import dataclass, field


@dataclass
class Limits:
    equity_floor: float = 200.0       # halt + liquidate if equity falls below this
    max_trade_usd: float = 20.0       # max dollars per single order
    max_ticker_usd: float = 60.0      # max dollars held in one ticker
    max_orders_per_hour: int = 12     # safety net on order rate
    allowed_tickers: set = field(default_factory=lambda: {
        "SPY", "QQQ", "VOO", "IWM", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
    })


class Guardrails:
    def __init__(self, broker, limits: Limits | None = None):
        # broker must provide: equity(), settled_cash(), position_value(ticker),
        # cancel_all_orders(), close_all_positions()
        self.broker = broker
        self.limits = limits or Limits()
        self.halted = False
        self.halt_reason = ""
        self._order_times: list[float] = []

    # --- kill switch -------------------------------------------------------
    def check_floor(self) -> bool:
        """Call before every decision cycle. Returns True if trading may continue."""
        if self.halted:
            return False
        equity = self.broker.equity()
        if equity < self.limits.equity_floor:
            self.halt(f"equity {equity:.2f} below floor {self.limits.equity_floor:.2f}")
            return False
        return True

    def halt(self, reason: str):
        self.halted = True
        self.halt_reason = reason
        self.broker.cancel_all_orders()
        self.broker.close_all_positions()

    def resume(self):
        """Manual restart only (e.g. from the check-in command)."""
        self.halted = False
        self.halt_reason = ""

    # --- per-order checks --------------------------------------------------
    def check_order(self, ticker: str, side: str, usd: float):
        """Returns (ok, reason). Never raises on a bad AI proposal."""
        if not self.check_floor():
            return False, f"halted: {self.halt_reason}"
        if side not in ("buy", "sell"):
            return False, f"invalid side {side!r}"
        if not (usd > 0):
            return False, "amount must be positive"
        if ticker not in self.limits.allowed_tickers:
            return False, f"{ticker} not in allowed universe"
        if usd > self.limits.max_trade_usd:
            return False, f"{usd:.2f} exceeds per-trade cap {self.limits.max_trade_usd:.2f}"

        now = time.time()
        self._order_times = [t for t in self._order_times if now - t < 3600]
        if len(self._order_times) >= self.limits.max_orders_per_hour:
            return False, "hourly order cap reached"

        if side == "buy":
            if usd > self.broker.settled_cash():
                return False, "not enough settled cash"
            held = self.broker.position_value(ticker)
            if held + usd > self.limits.max_ticker_usd:
                return False, f"would exceed per-ticker cap {self.limits.max_ticker_usd:.2f}"

        self._order_times.append(now)
        return True, "ok"
