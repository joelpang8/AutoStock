"""Heartbeat pings: one per cycle, never blocking, never raising."""
import threading

from heartbeat import Heartbeat
from test_engine import make, parts  # noqa: F401  (parts is a fixture)


class Opener:
    def __init__(self, fail=False, gate=None):
        self.calls, self.fail, self.gate = [], fail, gate
        self.done = threading.Event()
    def __call__(self, url, timeout):
        self.calls.append((url, timeout))
        if self.gate:
            self.gate.wait(2)
        self.done.set()
        if self.fail:
            raise OSError("down")
        return self
    def __enter__(self): return self
    def __exit__(self, *a): pass


def test_ping_hits_url():
    o = Opener()
    Heartbeat("https://hc/abc", opener=o).ping()
    assert o.done.wait(2) and o.calls == [("https://hc/abc", 10.0)]


def test_failed_ping_is_swallowed_and_next_ping_still_runs():
    o = Opener(fail=True)
    hb = Heartbeat("u", opener=o)
    hb.ping()
    assert o.done.wait(2)
    o.done.clear()
    for _ in range(100):           # wait for the lock to be released
        if hb._busy.acquire(blocking=False):
            hb._busy.release()
            break
        threading.Event().wait(0.01)
    hb.ping()
    assert o.done.wait(2) and len(o.calls) == 2


def test_slow_ping_does_not_block_or_pile_up():
    gate = threading.Event()
    o = Opener(gate=gate)
    hb = Heartbeat("u", opener=o)
    hb.ping()
    hb.ping()                       # returns at once, skipped while the first is stuck
    gate.set()
    assert o.done.wait(2)
    assert len(o.calls) == 1


def test_engine_beats_after_every_cycle(parts):
    beats = []
    e = make(parts)
    e.heartbeat = lambda: beats.append(1)
    sleeps = []
    def sleep(s):
        sleeps.append(s)
        if len(sleeps) == 3:
            raise KeyboardInterrupt
    e.sleep = sleep
    try:
        e.run_forever()
    except KeyboardInterrupt:
        pass
    assert len(beats) == 3
