"""Owner check-in: the status / pause / resume / halt commands, the alert text,
and the daily summary. Chat-platform agnostic; discord_bot.py is the wiring.

Only the owner's user id may run commands. Every method that touches the
broker runs blocking calls, so the chat bot calls them off its event loop.
"""
from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime, time, timezone

from data import ET

log = logging.getLogger(__name__)

SUMMARY_AFTER = time(16, 10)   # US/Eastern, a few minutes after the close
ALERT_ICONS = {"trade": "💸", "halt": "🛑", "error": "⚠️", "rejected": "🚫", "status": "ℹ️"}
LOUD = {"halt", "error"}       # these @mention the owner so the phone buzzes


class CheckIn:
    def __init__(self, engine, broker, owner_id: int, *,
                 now=lambda: datetime.now(timezone.utc)):
        self.engine, self.broker, self.owner_id, self.now = engine, broker, int(owner_id), now
        self.day = self._et_date()
        self.tally: Counter = Counter()   # today's event kinds and actions
        self.ai_cost = 0.0
        self.trades: list[str] = []
        self.last_decision = None         # (ts, decision dict)
        # a restart after the close shouldn't send a second (empty) summary
        self.summary_sent_for = self.day if self._past_summary_time() else None

    def _et_now(self) -> datetime:
        return self.now().astimezone(ET)

    def _et_date(self):
        return self._et_now().date()

    def _past_summary_time(self) -> bool:
        return self._et_now().time() >= SUMMARY_AFTER

    def authorized(self, user_id) -> bool:
        return int(user_id) == self.owner_id

    # --- event feed (wraps the engine's record callback) --------------------
    def record(self, event: dict):
        """Count what happened today. Never raises into the engine."""
        try:
            self._roll_day()
            kind = event.get("kind", "?")
            self.tally[kind] += 1
            self.ai_cost += float(event.get("cost_usd") or 0)
            d = event.get("decision")
            if d:
                self.tally["action:" + d["action"]] += 1
                self.last_decision = (event.get("ts", ""), d)
            if kind == "order" and event.get("status") == "filled":
                self.trades.append(f"{d['action'].upper()} {d['ticker']} ${d['usd']:.2f} "
                                   f"@ {event['fill_price']:.2f}")
        except Exception:
            log.exception("check-in could not record event")

    def replay(self, events):
        """Rebuild today's counts after a restart, from the stored events."""
        for e in events:
            self.record(e)

    def day_start_utc(self) -> str:
        start = datetime.combine(self._et_date(), time(0), tzinfo=ET)
        return start.astimezone(timezone.utc).isoformat()

    def _roll_day(self):
        today = self._et_date()
        if today != self.day:
            self.day, self.tally, self.ai_cost, self.trades = today, Counter(), 0.0, []

    # --- text ---------------------------------------------------------------
    def alert_text(self, kind: str, msg: str) -> str:
        text = f"{ALERT_ICONS.get(kind, '🔔')} **{kind}**: {msg}"
        if kind in LOUD:
            text = f"<@{self.owner_id}> " + text
        return text[:1900]

    def _mode_line(self) -> str:
        e = self.engine
        mode = "paper" if getattr(self.broker, "paper", True) else "**LIVE**"
        if e.g.halted:
            state = f"🛑 HALTED: {e.g.halt_reason}"
        elif e.state.paused:
            state = "⏸️ paused"
        else:
            state = "▶️ running"
        return f"{state} ({mode})"

    def _account_lines(self) -> list[str]:
        try:
            equity, cash = self.broker.equity(), self.broker.settled_cash()
            positions = self.broker.positions()
        except Exception as e:
            return [f"account: unreadable ({e})"]
        floor = self.engine.g.limits.equity_floor
        lines = [f"equity ${equity:.2f} (floor ${floor:.2f}), settled cash ${cash:.2f}"]
        if positions:
            lines += [f"• {t}: ${p['usd']:.2f} ({p['pl_pct']:+.2f}%)"
                      for t, p in sorted(positions.items())]
        else:
            lines.append("no positions")
        return lines

    def _today_line(self) -> str:
        self._roll_day()
        t = self.tally
        return (f"today: {t['action:buy']} buys, {t['action:sell']} sells, "
                f"{t['action:hold']} holds, {t['order']} orders, {t['rejected']} rejected, "
                f"{t['invalid']} invalid replies, {t['error']} errors, AI ${self.ai_cost:.4f}")

    def status(self) -> str:
        with self.engine.lock:   # a consistent view, not one taken mid-cycle
            lines = [self._mode_line()] + self._account_lines()
            g, budget = self.engine.g, self.engine.decider.budget
            recent = [x for x in g._order_times if self.now().timestamp() - x < 3600]
            lines.append(f"orders last hour {len(recent)}/{g.limits.max_orders_per_hour}, "
                         f"AI spend today ${budget.spent_today():.4f}/${budget.limit_usd:.2f}")
            lines.append(self._today_line())
            if self.last_decision:
                ts, d = self.last_decision
                lines.append(f"last decision {ts[11:16]} UTC: {d['action']} {d['ticker']} "
                             f"${d['usd']:.2f}: {d['reason']}")
        return "\n".join(lines)

    # --- commands -----------------------------------------------------------
    def pause(self) -> str:
        if self.engine.g.halted:
            return "Already halted. Send /resume to restart trading."
        self.engine.pause()
        return "⏸️ Paused. Positions are kept, no new trades. /resume to continue."

    def resume(self) -> str:
        was = "halt" if self.engine.g.halted else "pause" if self.engine.state.paused else None
        self.engine.resume()
        if was is None:
            return "Already running."
        return f"▶️ Resumed after {was}. Trading again from the next cycle."

    def halt(self) -> str:
        if self.engine.g.halted:
            return f"Already halted: {self.engine.g.halt_reason}"
        try:
            self.engine.halt("halted by owner from check-in")
        except Exception as e:
            # still halted; the engine keeps retrying the liquidation every cycle
            return f"🛑 Halted, but selling hit an error (retried every cycle): {e}"
        return "🛑 Halted. Open orders cancelled and all positions sold. /resume to restart."

    # --- daily summary --------------------------------------------------------
    def summary_due(self) -> bool:
        now = self._et_now()
        return (now.weekday() < 5 and now.time() >= SUMMARY_AFTER
                and self.summary_sent_for != now.date())

    def daily_summary(self) -> str:
        """Builds the summary and marks today's as sent."""
        with self.engine.lock:
            self.summary_sent_for = self._et_date()
            lines = [f"📊 **Daily summary {self.summary_sent_for}**", self._mode_line()]
            lines += self._account_lines()
            lines.append(self._today_line())
            lines += [f"• {t}" for t in self.trades] or ["no trades today"]
        return "\n".join(lines)[:1900]
