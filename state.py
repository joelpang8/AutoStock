"""Bot state that must survive a crash or restart: halt, pause, recent order
times (for the hourly cap) and today's AI spend.

Stored in SQLite (storage.SqliteStateStore). If it exists but can't be read,
loading fails loudly instead of starting with a blank (unhalted) state.
"""
from __future__ import annotations

from dataclasses import dataclass, field


class StateError(Exception):
    pass


@dataclass
class BotState:
    halted: bool = False
    halt_reason: str = ""
    paused: bool = False
    order_times: list = field(default_factory=list)   # unix times of approved orders
    ai_spend: dict = field(default_factory=dict)      # {"YYYY-MM-DD": usd}
