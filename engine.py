"""Execution engine: one decision cycle every few minutes during market hours.

Cycle order (each step can end the cycle with a hold):
  1. halted -> keep liquidating until flat, nothing else.   paused -> nothing.
  2. floor check (Guardrails) -> may halt + liquidate.
  3. snapshot (data layer)   -> market closed / stale / unreadable -> hold.
  4. this slot already has an order (restart mid-cycle) -> skip.
  5. an earlier order still open -> wait for it, cancel if stuck.
  6. ask Claude -> invalid / unavailable / over budget -> hold.
  7. Guardrails.check_order -> rejected -> hold.
  8. submit one market order by dollars, client_order_id = this slot's id.
  9. confirm the fill, log slippage against the quote.

No double orders: the client_order_id is derived from the time slot, Alpaca
refuses a second order with the same id, step 4 skips a slot that already has
one, and a failed submit is looked up by id before anything else happens.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta, timezone

from broker import BrokerError
from data import MarketClosed, StaleDataError
from decision import DecisionUnavailable
from state import BotState

log = logging.getLogger(__name__)

TERMINAL = {"filled", "canceled", "expired", "rejected", "done_for_day", "replaced", "stopped", "suspended"}
SELL_ALL_MARGIN_USD = 0.50   # a sell within this of the whole position sells it all, by qty


def _status(order) -> str:
    return str(getattr(order.status, "value", order.status))


class Engine:
    def __init__(self, broker, data, decider, guardrails, store, tickers, *,
                 interval_min: int = 5, notify=lambda kind, msg: None, record=lambda event: None,
                 now=lambda: datetime.now(timezone.utc), sleep=time.sleep,
                 fill_timeout_s: float = 30, stuck_order_min: float = 10):
        self.broker, self.data, self.decider, self.g = broker, data, decider, guardrails
        self.store, self.tickers, self.interval = store, sorted(tickers), interval_min
        self.notify, self.record, self.now, self.sleep = notify, record, now, sleep
        self.fill_timeout_s, self.stuck_order_min = fill_timeout_s, stuck_order_min
        self._last_alert: dict = {}
        # owner commands arrive from the check-in bot's thread: they wait for a running cycle
        self.lock = threading.RLock()

        self.state: BotState = store.load()
        # restore what guardrails.py keeps in memory, so a restart can't reset it
        self.g.halted, self.g.halt_reason = self.state.halted, self.state.halt_reason
        self.g._order_times = list(self.state.order_times)
        self.decider.budget.spent = {datetime.fromisoformat(k).date(): v
                                     for k, v in self.state.ai_spend.items()}

    # --- helpers -----------------------------------------------------------
    def slot(self, t: datetime) -> datetime:
        return t.replace(minute=t.minute - t.minute % self.interval, second=0, microsecond=0)

    def client_id(self, slot: datetime) -> str:
        return f"ab-{slot:%Y%m%dT%H%M}"

    def _event(self, kind: str, **fields):
        e = {"ts": self.now().isoformat(), "kind": kind, **fields}
        log.info("%s %s", kind, fields)
        self.record(e)
        return e

    def _alert(self, kind: str, msg: str, dedupe_min: float = 60):
        """Push to the owner; identical messages at most once per dedupe window."""
        key, t = (kind, msg), self.now()
        last = self._last_alert.get(key)
        if last and t - last < timedelta(minutes=dedupe_min):
            return
        self._last_alert[key] = t
        self.notify(kind, msg)

    def _save(self):
        self.state.halted, self.state.halt_reason = self.g.halted, self.g.halt_reason
        self.state.order_times = list(self.g._order_times)
        self.state.ai_spend = {d.isoformat(): v for d, v in self.decider.budget.spent.items()}
        self.store.save(self.state)

    # --- owner commands (from the check-in bot, see checkin.py) -------------
    def pause(self):
        with self.lock:
            self.state.paused = True
            self._save()

    def resume(self):
        """Clears pause and halt. The only way out of a floor halt."""
        with self.lock:
            self.state.paused = False
            self.g.resume()
            self._save()

    def halt(self, reason: str):
        """Cancels open orders, sells everything, and stays halted until resume()."""
        with self.lock:
            try:
                self.g.halt(reason)
            finally:
                self._save()

    # --- one cycle -----------------------------------------------------------
    def run_cycle(self) -> dict:
        with self.lock:
            return self._locked_cycle()

    def _locked_cycle(self) -> dict:
        try:
            return self._cycle()
        except Exception as e:  # never crash the loop on one bad cycle
            log.exception("cycle failed")
            self._alert("error", f"cycle failed: {e}")
            return self._event("error", error=str(e))
        finally:
            try:
                self._save()
            except Exception as e:
                self._alert("error", f"could not save state: {e}")

    def _cycle(self) -> dict:
        if self.g.halted:
            return self._keep_liquidating()
        if self.state.paused:
            return self._event("paused")

        try:
            ok = self.g.check_floor()
        except BrokerError as e:
            if self.g.halted:   # floor breached but liquidation errored: retried next cycle
                self._alert("halt", f"HALTED: {self.g.halt_reason}. Liquidation error, "
                                    f"retrying: {e}", dedupe_min=0)
                return self._event("halt", why=self.g.halt_reason, error=str(e))
            self._alert("error", f"can't read equity, not trading: {e}")
            return self._event("hold", why=f"account unreadable: {e}")
        if not ok:
            self._save()
            self._alert("halt", f"HALTED: {self.g.halt_reason}. Sold everything. "
                                "Send resume to restart.", dedupe_min=0)
            return self._event("halt", why=self.g.halt_reason)

        try:
            snap = self.data.snapshot(self.broker, self.tickers)
        except MarketClosed as e:
            return self._event("idle", why=str(e))
        except (StaleDataError, BrokerError) as e:
            self._alert("error", f"no trade this cycle: {e}")
            return self._event("hold", why=str(e))
        if snap.excluded:
            log.info("excluded this cycle: %s", snap.excluded)
        for problem in snap.context_problems:
            self._alert("error", f"deciding without full context: {problem}")

        cid = self.client_id(self.slot(snap.as_of))
        if self.broker.order_by_client_id(cid) is not None:
            return self._event("skip", why=f"slot already has order {cid}")
        if self._waiting_on_open_orders():
            return self._event("skip", why="earlier order still open")

        try:
            res = self.decider.decide(snap)
        except DecisionUnavailable as e:
            self._alert("error", f"no AI decision: {e}")
            return self._event("hold", why=str(e))
        base = {"cost_usd": round(res.cost_usd, 6), "usage": res.usage,
                "request_id": res.request_id, "raw": res.raw_text}
        if res.decision is None:
            self._alert("error", f"invalid AI reply, holding: {res.error}")
            return self._event("invalid", error=res.error, **base)
        d = res.decision
        decision = {"action": d.action, "ticker": d.ticker, "usd": d.usd, "reason": d.reason}
        if d.action == "hold":
            return self._event("decision", decision=decision, **base)

        try:
            ok, why = self.g.check_order(d.ticker, d.action, d.usd)
        except BrokerError as e:
            ok, why = False, f"account unreadable during checks: {e}"
        self._save()   # check_order may have halted, and it records the order time
        if not ok:
            if self.g.halted:
                self._alert("halt", f"HALTED: {self.g.halt_reason}", dedupe_min=0)
            else:
                self._alert("rejected", f"guardrail blocked {d.action} {d.ticker} ${d.usd:.2f}: {why}")
            return self._event("rejected", decision=decision, why=why, **base)

        return self._execute(d, snap, cid, decision, base)

    # --- execution -----------------------------------------------------------
    def _execute(self, d, snap, cid, decision, base) -> dict:
        kw = {"usd": d.usd}
        held = snap.positions.get(d.ticker)
        if d.action == "sell" and held and d.usd >= held["usd"] - SELL_ALL_MARGIN_USD:
            kw = {"qty": held["qty"]}   # whole position: by shares, so nothing is left over
        try:
            order = self.broker.submit_market(d.ticker, d.action, cid, **kw)
        except Exception as e:
            # the request may have reached Alpaca anyway: look before concluding
            try:
                order = self.broker.order_by_client_id(cid)
            except BrokerError:
                order = None
            if order is None:
                self._alert("error", f"order {d.action} {d.ticker} ${d.usd:.2f} failed: {e}")
                return self._event("order_failed", decision=decision, error=str(e), **base)
            log.warning("submit raised but order %s exists: %s", cid, e)

        order = self._await_fill(order)
        status = _status(order)
        ev = {"decision": decision, "order_id": str(order.id), "client_order_id": cid,
              "status": status, **base}
        if status == "filled":
            price = float(order.filled_avg_price)
            mid = snap.tickers[d.ticker].price
            ev.update(fill_price=price, filled_qty=float(order.filled_qty),
                      slippage_pct=round((price / mid - 1) * 100 * (1 if d.action == "buy" else -1), 4))
            self._alert("trade", f"{d.action.upper()} {d.ticker} ${d.usd:.2f} @ {price:.2f} "
                                 f"(slippage {ev['slippage_pct']:+.3f}%): {d.reason}", dedupe_min=0)
        elif status in TERMINAL:
            self._alert("error", f"order {d.action} {d.ticker} ended {status}")
        else:
            self._alert("trade", f"{d.action.upper()} {d.ticker} ${d.usd:.2f} submitted, "
                                 f"still {status}: {d.reason}", dedupe_min=0)
        return self._event("order", **ev)

    def _await_fill(self, order):
        for _ in range(int(self.fill_timeout_s)):
            if _status(order) in TERMINAL:
                break
            self.sleep(1)
            order = self.broker.order(order.id)
        return order

    def _waiting_on_open_orders(self) -> bool:
        open_orders = self.broker.open_orders()
        if not open_orders:
            return False
        cutoff = self.now() - timedelta(minutes=self.stuck_order_min)
        for o in open_orders:
            if o.submitted_at and o.submitted_at < cutoff:
                self.broker.cancel_order(o.id)
                self._alert("error", f"cancelled stuck order {o.symbol} {o.client_order_id}")
        return True

    def _keep_liquidating(self) -> dict:
        """While halted, make sure everything really gets sold."""
        try:
            positions = self.broker.positions()
            if positions and not any(str(getattr(o.side, "value", o.side)) == "sell"
                                     for o in self.broker.open_orders()):
                self.broker.close_all_positions()
                self._alert("halt", f"halted, retrying liquidation of {sorted(positions)}")
        except BrokerError as e:
            self._alert("error", f"halted, liquidation check failed: {e}")
        return self._event("halted", why=self.g.halt_reason)

    # --- main loop -------------------------------------------------------------
    def run_forever(self):
        self._alert("status", "bot started" + (" (HALTED)" if self.g.halted else ""), dedupe_min=0)
        while True:
            ev = self.run_cycle()
            t = self.now()
            nxt = self.slot(t) + timedelta(minutes=self.interval, seconds=5)
            if ev.get("kind") == "idle":
                nxt = t + timedelta(minutes=15)   # market closed: check the clock now and then
            self.sleep(max(1.0, (nxt - t).total_seconds()))
