"""Check-in commands, alert text and daily summary, on the fake engine parts.
No network: the Discord wiring is tested without connecting."""
import asyncio
import threading
from datetime import datetime, timezone
from types import SimpleNamespace as NS

import pytest

from checkin import CheckIn
from discord_bot import AlertQueue, CheckInBot
from test_engine import NOW, make, parts  # noqa: F401  (parts is a fixture)

OWNER = 111
AFTER_CLOSE = datetime(2026, 10, 22, 20, 15, tzinfo=timezone.utc)   # Thu 16:15 ET
SATURDAY = datetime(2026, 10, 24, 20, 15, tzinfo=timezone.utc)


def wire(p, now=NOW):
    """Engine whose record() also feeds a CheckIn, like main.py."""
    p.broker.paper = True
    holder = {}
    e = make(p)
    e.record = lambda ev: (p.events.append(ev), holder["c"].record(ev))
    c = holder["c"] = CheckIn(e, p.broker, OWNER, now=lambda: now)
    return e, c


# --- authorization -------------------------------------------------------------
def test_only_owner_is_authorized(parts):
    _, c = wire(parts)
    assert c.authorized(OWNER) and c.authorized(str(OWNER))
    assert not c.authorized(222)


# --- commands ---------------------------------------------------------------------
def test_pause_keeps_positions_and_stops_trading(parts):
    parts.broker.held = {"SPY": {"usd": 18.0, "pl_pct": 1.0, "qty": 0.18}}
    e, c = wire(parts)
    assert "Paused" in c.pause()
    assert e.run_cycle()["kind"] == "paused"
    assert parts.broker.held and not parts.broker.submitted
    assert parts.store.load().paused   # survives a restart


def test_halt_sells_everything_and_persists(parts):
    parts.broker.held = {"SPY": {"usd": 18.0, "pl_pct": 1.0, "qty": 0.18}}
    e, c = wire(parts)
    assert "all positions sold" in c.halt()
    assert parts.broker.closed_all == 1 and not parts.broker.held
    st = parts.store.load()
    assert st.halted and "owner" in st.halt_reason
    assert "Already halted" in c.halt() and parts.broker.closed_all == 1
    assert e.run_cycle()["kind"] == "halted" and not parts.broker.submitted


def test_halt_reports_liquidation_error_but_stays_halted(parts):
    e, c = wire(parts)
    def boom():
        raise RuntimeError("api down")
    parts.broker.close_all_positions = boom
    assert "error" in c.halt()
    assert e.g.halted and parts.store.load().halted


def test_resume_clears_halt_and_pause(parts):
    e, c = wire(parts)
    assert c.resume() == "Already running."
    c.halt()
    assert "after halt" in c.resume()
    assert not e.g.halted and not parts.store.load().halted
    c.pause()
    assert "after pause" in c.resume()
    assert e.run_cycle()["kind"] == "order"


def test_pause_while_halted_says_so(parts):
    e, c = wire(parts)
    c.halt()
    assert "Already halted" in c.pause() and not e.state.paused


def test_command_waits_for_running_cycle(parts):
    e, c = wire(parts)
    e.lock.acquire()   # a cycle is in progress
    done = threading.Event()
    t = threading.Thread(target=lambda: (c.pause(), done.set()))
    t.start()
    assert not done.wait(0.2)
    e.lock.release()
    assert done.wait(2) and e.state.paused


# --- status ---------------------------------------------------------------------------
def test_status_shows_account_and_today(parts):
    parts.broker.held = {"SPY": {"usd": 18.0, "pl_pct": 1.5, "qty": 0.18}}
    e, c = wire(parts)
    e.run_cycle()   # buys QQQ
    s = c.status()
    assert "running (paper)" in s
    assert "equity $300.00 (floor $200.00)" in s
    assert "SPY: $18.00 (+1.50%)" in s
    assert "1 buys" in s and "1 orders" in s
    assert "orders last hour 1/12" in s
    assert "last decision 15:07 UTC: buy QQQ $15.00: test buy" in s


def test_status_survives_unreadable_account(parts):
    _, c = wire(parts)
    parts.broker._equity = None
    assert "account: unreadable" in c.status()


def test_status_flags_live_and_halt(parts):
    _, c = wire(parts)
    parts.broker.paper = False
    c.halt()
    s = c.status()
    assert "HALTED" in s and "LIVE" in s


# --- alerts ---------------------------------------------------------------------------
def test_loud_alerts_mention_owner(parts):
    _, c = wire(parts)
    assert c.alert_text("halt", "x").startswith(f"<@{OWNER}>")
    assert c.alert_text("error", "x").startswith(f"<@{OWNER}>")
    assert "<@" not in c.alert_text("trade", "BUY QQQ")
    assert len(c.alert_text("error", "y" * 5000)) <= 1900


def test_record_never_raises_into_engine(parts):
    _, c = wire(parts)
    c.record({"kind": "order", "status": "filled", "decision": {"action": "buy"}})   # malformed
    c.record("not a dict")


# --- daily summary ---------------------------------------------------------------
def test_summary_due_once_after_close_on_weekdays(parts):
    e, c = wire(parts)
    e.run_cycle()
    clock = {"t": AFTER_CLOSE}
    c.now = lambda: clock["t"]
    assert c.summary_due()
    s = c.daily_summary()
    assert "Daily summary 2026-10-22" in s and "BUY QQQ $15.00 @ 100.05" in s
    assert not c.summary_due()
    clock["t"] = SATURDAY
    assert not c.summary_due()


def test_no_duplicate_summary_after_restart_past_close(parts):
    _, c = wire(parts, now=AFTER_CLOSE)
    assert not c.summary_due()


def test_tally_resets_on_new_day(parts):
    e, c = wire(parts)
    e.run_cycle()
    c.now = lambda: datetime(2026, 10, 23, 14, 0, tzinfo=timezone.utc)
    assert "0 buys" in c._today_line() and c.trades == []


# --- Discord wiring (no connection) -----------------------------------------------
def test_alert_queue_drains_in_order():
    q = AlertQueue()
    q.notify("trade", "a")
    q.notify("error", "b")
    assert q.drain() == [("trade", "a"), ("error", "b")] and q.drain() == []


class FakeInteraction:
    def __init__(self, user_id):
        self.user = NS(id=user_id)
        self.sent = []
        self.response = NS(send_message=self._send, defer=self._defer)
        self.followup = NS(send=self._send)
    async def _send(self, text, **kw): self.sent.append(text)
    async def _defer(self, **kw): pass


@pytest.fixture
def bot(parts):
    _, c = wire(parts)
    return CheckInBot(c, AlertQueue(), guild_id=1, channel_id=2)


def run(bot, name, user):
    cmd = bot.tree.get_command(name, guild=bot.guild)
    i = FakeInteraction(user)
    asyncio.run(cmd.callback(i))
    return i.sent


def test_bot_registers_the_four_commands(bot):
    names = {c.name for c in bot.tree.get_commands(guild=bot.guild)}
    assert names == {"status", "pause", "resume", "halt"}


def test_bot_refuses_strangers(bot, parts):
    assert run(bot, "halt", 999) == ["Not authorized."]
    assert parts.broker.closed_all == 0


def test_bot_runs_owner_command(bot, parts):
    assert "Paused" in run(bot, "pause", OWNER)[0]
    assert parts.store.load().paused
