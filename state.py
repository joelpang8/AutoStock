"""Bot state that must survive a crash or restart: halt, pause, recent order
times (for the hourly cap) and today's AI spend.

Stored as a small JSON file, written atomically (temp file + rename) so a crash
mid-write can't leave it half-written. If the file exists but can't be read,
loading fails loudly instead of starting with a blank (unhalted) state.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field


class StateError(Exception):
    pass


@dataclass
class BotState:
    halted: bool = False
    halt_reason: str = ""
    paused: bool = False
    order_times: list = field(default_factory=list)   # unix times of approved orders
    ai_spend: dict = field(default_factory=dict)      # {"YYYY-MM-DD": usd}


class StateStore:
    def __init__(self, path: str):
        self.path = path

    def load(self) -> BotState:
        if not os.path.exists(self.path):
            return BotState()
        try:
            with open(self.path) as f:
                raw = json.load(f)
            return BotState(**raw)
        except Exception as e:
            raise StateError(f"can't read state file {self.path}: {e}. "
                             "Refusing to start: fix or remove it deliberately.") from e

    def save(self, state: BotState) -> None:
        d = os.path.dirname(os.path.abspath(self.path))
        fd, tmp = tempfile.mkstemp(dir=d, prefix=".state-")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(asdict(state), f, indent=2, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
