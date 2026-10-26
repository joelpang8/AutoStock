"""Dead-man heartbeat: ping an outside URL after every engine cycle.

If the bot crashes, hangs, or its server goes down, the pings stop and the
outside service (e.g. healthchecks.io) alerts the owner. The bot can't send
that alert itself, which is why it lives outside.

Pings run in a background thread with a short timeout, so a slow or down
heartbeat service never delays a trading cycle.
"""
from __future__ import annotations

import logging
import threading
import urllib.request

log = logging.getLogger(__name__)


class Heartbeat:
    def __init__(self, url: str, timeout: float = 10.0, opener=urllib.request.urlopen):
        self.url, self.timeout, self.opener = url, timeout, opener
        self._busy = threading.Lock()

    def ping(self):
        if not self._busy.acquire(blocking=False):   # previous ping still running: skip
            return
        threading.Thread(target=self._send, name="heartbeat", daemon=True).start()

    def _send(self):
        try:
            with self.opener(self.url, timeout=self.timeout):
                pass
        except Exception as e:
            log.warning("heartbeat ping failed: %s", e)
        finally:
            self._busy.release()
