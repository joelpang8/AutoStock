"""Run the bot: python main.py

Reads settings from the environment / .env (see .env.example). Paper unless
ALPACA_PAPER=false and ALLOW_LIVE_TRADING is set (see broker.py).
"""
import json
import logging
import os

from dotenv import load_dotenv

load_dotenv()

from broker import AlpacaBroker
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

    def record(event):   # replaced by SQLite in step 5
        events.write(json.dumps(event, default=str) + "\n")
        events.flush()

    def notify(kind, msg):   # replaced by the phone check-in bot in step 6
        log.warning("ALERT [%s] %s", kind, msg)

    guardrails = Guardrails(broker)
    engine = Engine(broker, MarketData.from_env(context=ContextProvider.from_env()), Decider.from_env(), guardrails,
                    StateStore(os.getenv("STATE_PATH", "state.json")), guardrails.limits.allowed_tickers,
                    interval_min=int(os.getenv("DECISION_INTERVAL_MINUTES", "5")),
                    notify=notify, record=record)
    log.info("starting (%s)", "paper" if broker.paper else "LIVE")
    engine.run_forever()


if __name__ == "__main__":
    main()
