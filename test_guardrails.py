import sys
sys.path.insert(0, ".")
from guardrails import Guardrails, Limits

class FakeBroker:
    def __init__(self, equity=300.0, cash=300.0):
        self._equity, self._cash = equity, cash
        self.held = {}
        self.cancelled = False
        self.closed = False
    def equity(self): return self._equity
    def settled_cash(self): return self._cash
    def position_value(self, t): return self.held.get(t, 0.0)
    def cancel_all_orders(self): self.cancelled = True
    def close_all_positions(self): self.closed = True

b = FakeBroker()
g = Guardrails(b)
assert g.check_order("SPY", "buy", 15) == (True, "ok")
assert not g.check_order("SPY", "buy", 25)[0]            # over per-trade cap
assert not g.check_order("GME", "buy", 10)[0]            # not allowed
assert not g.check_order("SPY", "short", 10)[0]          # bad side
assert not g.check_order("SPY", "buy", -5)[0]            # bad amount
b.held["SPY"] = 295
assert not g.check_order("SPY", "buy", 10)[0]            # per-ticker cap ($300)
assert g.check_order("QLD", "buy", 10)[0]                # QLD is allowed
b._cash = 5
assert not g.check_order("QQQ", "buy", 10)[0]            # unsettled/no cash

# kill switch
b2 = FakeBroker(equity=199.99)
g2 = Guardrails(b2)
ok, why = g2.check_order("SPY", "buy", 10)
assert not ok and g2.halted and b2.cancelled and b2.closed
b2._equity = 300
assert not g2.check_order("SPY", "buy", 10)[0]           # stays halted until resume
g2.resume()
assert g2.check_order("SPY", "buy", 10)[0]

# rate cap
b3 = FakeBroker()
g3 = Guardrails(b3, Limits(max_orders_per_hour=3))
for _ in range(3):
    assert g3.check_order("SPY", "buy", 1)[0]
assert not g3.check_order("SPY", "buy", 1)[0]
print("all guardrail tests passed")
