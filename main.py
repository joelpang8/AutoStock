"""Run the bot: python main.py

Reads settings from the environment / .env (see .env.example). Paper unless
ALPACA_PAPER=false and ALLOW_LIVE_TRADING is set (see broker.py).

With the DISCORD_* settings, the engine runs in a background thread and the
Discord check-in bot (status / pause / resume / halt, alerts, daily summary)
runs in the main thread. If either one dies, the whole process exits.
"""
import logging
import os
import threading

from dotenv import load_dotenv

load_dotenv()

from broker import AlpacaBroker
from checkin import CheckIn
from context import ContextProvider
from data import MarketData
from decision import Decider
from engine import Engine
from guardrails import Guardrails
from storage import Database, SqliteStateStore

log = logging.getLogger("bot")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler("bot.log")])
    broker = AlpacaBroker.from_env()
    problems = broker.safety_problems()
    if problems:
        raise SystemExit("refusing to start, fix account settings first: " + "; ".join(problems))

    db = Database(os.getenv("DB_PATH", "autostock.db"))
    store = SqliteStateStore(db)
    if store.import_json(os.getenv("STATE_PATH", "state.json")):
        log.info("imported old state.json into %s", db.path)

    discord_cfg = [os.getenv(k) for k in
                   ("DISCORD_BOT_TOKEN", "DISCORD_GUILD_ID", "DISCORD_CHANNEL_ID", "DISCORD_OWNER_ID")]
    if any(discord_cfg) and not all(discord_cfg):
        raise SystemExit("set all of DISCORD_BOT_TOKEN, DISCORD_GUILD_ID, DISCORD_CHANNEL_ID, "
                         "DISCORD_OWNER_ID, or none of them")
    alerts = None
    if all(discord_cfg):
        from discord_bot import AlertQueue
        alerts = AlertQueue()
        send = alerts.notify
    else:
        log.warning("no Discord settings: alerts only go to the log")

        def send(kind, msg):
            log.warning("ALERT [%s] %s", kind, msg)

    def notify(kind, msg):
        db.record_alert(kind, msg)
        send(kind, msg)

    checkin = None

    def record(event):
        db.record(event)
        if checkin:
            checkin.record(event)

    heartbeat = lambda: None
    if os.getenv("HEALTHCHECK_URL"):
        from heartbeat import Heartbeat
        heartbeat = Heartbeat(os.environ["HEALTHCHECK_URL"]).ping
    else:
        log.warning("no HEALTHCHECK_URL: nobody will notice if the bot goes silent")

    guardrails = Guardrails(broker)
    strategy = os.getenv("STRATEGY", "qld_trend")
    if strategy == "qld_trend":
        from trend import HOLD, WATCHDOG_PROMPT, TrendStrategy
        if HOLD not in guardrails.limits.allowed_tickers:
            raise SystemExit(f"STRATEGY=qld_trend needs {HOLD} in the allowed tickers in "
                             "guardrails.py (an owner decision), or set STRATEGY=claude")
        watchdog = Decider.from_env()
        watchdog.system = WATCHDOG_PROMPT
        decider = TrendStrategy.from_env(watchdog, guardrails.limits)
    elif strategy == "claude":
        decider = Decider.from_env()
    else:
        raise SystemExit(f"unknown STRATEGY={strategy!r}: use qld_trend or claude")
    log.info("strategy: %s", strategy)
    engine = Engine(broker, MarketData.from_env(context=ContextProvider.from_env()), decider, guardrails,
                    store, guardrails.limits.allowed_tickers,
                    interval_min=int(os.getenv("DECISION_INTERVAL_MINUTES", "5")),
                    notify=notify, record=record, heartbeat=heartbeat)
    checkin = CheckIn(engine, broker, int(discord_cfg[3] or 0))
    checkin.replay(db.events_since(checkin.day_start_utc()))
    log.info("starting (%s)", "paper" if broker.paper else "LIVE")
    if alerts is None:
        engine.run_forever()
        return

    def run_engine():
        try:
            engine.run_forever()
        except BaseException:
            log.exception("engine loop died, exiting so the supervisor restarts the bot")
        os._exit(1)   # don't leave the chat bot running without the engine

    threading.Thread(target=run_engine, name="engine", daemon=True).start()
    from discord_bot import CheckInBot
    CheckInBot(checkin, alerts, int(discord_cfg[1]), int(discord_cfg[2])).run(
        discord_cfg[0], log_handler=None)


if __name__ == "__main__":
    main()
