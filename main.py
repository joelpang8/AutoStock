"""Run the bot: python main.py

Reads settings from the environment / .env (see .env.example). Paper unless
ALPACA_PAPER=false and ALLOW_LIVE_TRADING is set (see broker.py).

With the DISCORD_* settings, the engine runs in a background thread and the
Discord check-in bot (status / pause / resume / halt, alerts, daily summary)
runs in the main thread. If either one dies, the whole process exits.
"""
import json
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
from state import StateStore

log = logging.getLogger("bot")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler("bot.log")])
    broker = AlpacaBroker.from_env()
    problems = broker.safety_problems()
    if problems:
        raise SystemExit("refusing to start, fix account settings first: " + "; ".join(problems))

    events = open("events.jsonl", "a")

    def write_event(event):   # replaced by SQLite in step 5
        events.write(json.dumps(event, default=str) + "\n")
        events.flush()

    discord_cfg = [os.getenv(k) for k in
                   ("DISCORD_BOT_TOKEN", "DISCORD_GUILD_ID", "DISCORD_CHANNEL_ID", "DISCORD_OWNER_ID")]
    if any(discord_cfg) and not all(discord_cfg):
        raise SystemExit("set all of DISCORD_BOT_TOKEN, DISCORD_GUILD_ID, DISCORD_CHANNEL_ID, "
                         "DISCORD_OWNER_ID, or none of them")
    alerts = None
    if all(discord_cfg):
        from discord_bot import AlertQueue
        alerts = AlertQueue()
        notify = alerts.notify
    else:
        log.warning("no Discord settings: alerts only go to the log")

        def notify(kind, msg):
            log.warning("ALERT [%s] %s", kind, msg)

    checkin = None

    def record(event):
        write_event(event)
        if checkin:
            checkin.record(event)

    guardrails = Guardrails(broker)
    engine = Engine(broker, MarketData.from_env(context=ContextProvider.from_env()), Decider.from_env(), guardrails,
                    StateStore(os.getenv("STATE_PATH", "state.json")), guardrails.limits.allowed_tickers,
                    interval_min=int(os.getenv("DECISION_INTERVAL_MINUTES", "5")),
                    notify=notify, record=record)
    checkin = CheckIn(engine, broker, int(discord_cfg[3] or 0))
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
