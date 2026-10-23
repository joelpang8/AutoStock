"""Decision step tests with a fake Anthropic client. No network, no spend."""
import json
from datetime import date, datetime, timezone
from types import SimpleNamespace as NS

import anthropic
import pytest

from data import Snapshot, TickerView
from decision import (MAX_TOKENS, BudgetExhausted, DailyBudget, Decider,
                      DecisionUnavailable, InvalidReply, call_cost, validate)

NOW = datetime(2026, 10, 22, 15, 0, tzinfo=timezone.utc)


def tv(sym, price=100.0):
    return TickerView(sym, price, 0.02, 0.1, 0.0, 1.0, 3.0)


SNAP = Snapshot(NOW, NOW.replace(hour=20), 300.0, 250.0,
                {"SPY": {"usd": 40.0, "pl_pct": 1.0}},
                {"SPY": tv("SPY"), "QQQ": tv("QQQ")}, {"NVDA": "quote 300s old"})


def reply(**kw):
    d = {"action": "buy", "ticker": "QQQ", "usd": 15, "reason": "momentum"}
    d.update(kw)
    return json.dumps(d)


# --- validation: good replies ---------------------------------------------
def test_valid_buy():
    assert validate(reply(), SNAP).usd == 15.0


def test_valid_sell_within_position():
    d = validate(reply(action="sell", ticker="SPY", usd=40), SNAP)
    assert (d.action, d.ticker, d.usd) == ("sell", "SPY", 40.0)


def test_valid_hold():
    d = validate(reply(action="hold", ticker="NONE", usd=0, reason="nothing compelling"), SNAP)
    assert d.action == "hold"


# --- validation: malformed replies ----------------------------------------
BAD = [
    ("", "not JSON"),
    ("buy QQQ $15", "not JSON"),
    ("```json\n" + reply() + "\n```", "not JSON"),
    ("[1,2]", "wrong fields"),
    (json.dumps({"action": "buy", "ticker": "QQQ", "usd": 15}), "wrong fields"),
    (reply()[:-1] + ', "confidence": 0.9}', "wrong fields"),
    (reply(action="short"), "bad action"),
    (reply(action="BUY"), "bad action"),
    (reply(ticker="GME"), "not tradable"),
    (reply(ticker="NVDA"), "not tradable"),            # excluded this cycle
    (reply(ticker="qqq"), "not tradable"),
    (reply(ticker=None), "ticker not a string"),
    (reply(usd="15"), "not a number"),
    (reply(usd=True), "not a number"),
    (reply(usd=None), "not a number"),
    (reply(usd=0), "positive"),
    (reply(usd=-5), "positive"),
    (reply(usd=15.123), "2 decimals"),
    (reply(reason=""), "missing reason"),
    (reply(reason="   "), "missing reason"),
    (reply(reason="line one\nline two"), "one short line"),
    (reply(reason="x" * 400), "one short line"),
    (reply(reason=5), "missing reason"),
    (reply(action="sell", ticker="QQQ", usd=10), "no position"),
    (reply(action="sell", ticker="SPY", usd=50), "only $40.00 held"),
    (reply(action="hold", ticker="SPY", usd=0), "hold must use"),
    (reply(action="hold", ticker="NONE", usd=10), "hold must use"),
]


@pytest.mark.parametrize("text,why", BAD)
def test_malformed_reply_rejected(text, why):
    with pytest.raises(InvalidReply, match=why.replace("$", r"\$")):
        validate(text, SNAP)


def test_nan_and_infinity_rejected():
    for v in ("NaN", "Infinity", "-Infinity"):
        text = '{"action": "buy", "ticker": "QQQ", "usd": %s, "reason": "x"}' % v
        with pytest.raises(InvalidReply):
            validate(text, SNAP)


# --- cost and budget --------------------------------------------------------
def usage(i=1000, o=500, read=0, write=0):
    return NS(input_tokens=i, output_tokens=o,
              cache_read_input_tokens=read, cache_creation_input_tokens=write)


def test_haiku_cost():
    # 1000 in * $0.10/M + 500 out * $0.50/M
    assert call_cost("claude-haiku-5-5", usage()) == pytest.approx(0.00035)


def test_cache_tokens_priced():
    assert call_cost("claude-haiku-5-5", usage(0, 0, read=1_000_000)) == pytest.approx(0.01)
    assert call_cost("claude-haiku-5-5", usage(0, 0, write=1_000_000)) == pytest.approx(0.125)


def test_budget_resets_daily():
    day = [date(2026, 10, 22)]
    b = DailyBudget(1.0, today=lambda: day[0])
    b.add(0.99)
    assert not b.can_afford(0.02)
    day[0] = date(2026, 10, 23)
    assert b.can_afford(0.02)


# --- Decider with a fake client --------------------------------------------
class FakeMessages:
    def __init__(self, resp=None, exc=None):
        self.resp, self.exc, self.calls = resp, exc, []

    def create(self, **kw):
        self.calls.append(kw)
        if self.exc:
            raise self.exc
        return self.resp


def response(text=reply(), stop="end_turn", u=None):
    content = [NS(type="thinking", thinking=""), NS(type="text", text=text)]
    return NS(content=content, stop_reason=stop, usage=u or usage(), _request_id="req_1")


def decider(resp=None, exc=None, limit=1.0):
    msgs = FakeMessages(resp, exc)
    d = Decider(NS(messages=msgs), "claude-haiku-5-5",
                DailyBudget(limit, today=lambda: date(2026, 10, 22)))
    return d, msgs


def test_decide_happy_path_and_request_shape():
    d, msgs = decider(response())
    r = d.decide(SNAP)
    assert r.decision.action == "buy" and r.error == ""
    assert r.cost_usd == pytest.approx(0.00035) and r.request_id == "req_1"
    assert d.budget.spent_today() == pytest.approx(0.00035)
    kw = msgs.calls[0]
    assert kw["model"] == "claude-haiku-5-5" and kw["max_tokens"] == MAX_TOKENS
    assert kw["output_config"]["effort"] == "low"
    assert kw["output_config"]["format"]["type"] == "json_schema"
    assert "temperature" not in kw and "thinking" not in kw
    assert "SPY" in kw["messages"][0]["content"]


def test_invalid_reply_is_reported_not_raised_and_still_costs():
    d, _ = decider(response(text="I think you should buy QQQ"))
    r = d.decide(SNAP)
    assert r.decision is None and "not JSON" in r.error
    assert d.budget.spent_today() > 0


@pytest.mark.parametrize("stop", ["max_tokens", "refusal"])
def test_bad_stop_reason_is_invalid(stop):
    r = decider(response(stop=stop))[0].decide(SNAP)
    assert r.decision is None and stop in r.error


def test_no_text_block_is_invalid():
    resp = response()
    resp.content = [NS(type="thinking", thinking="")]
    r = decider(resp)[0].decide(SNAP)
    assert r.decision is None and "not JSON" in r.error


def test_api_error_raises_unavailable():
    err = anthropic.APIConnectionError(request=NS(method="POST", url="x"))
    d, _ = decider(exc=err)
    with pytest.raises(DecisionUnavailable):
        d.decide(SNAP)


def test_budget_exhausted_makes_no_call():
    d, msgs = decider(response(), limit=0.0005)
    d.budget.add(0.0004)
    with pytest.raises(BudgetExhausted):
        d.decide(SNAP)
    assert msgs.calls == []


def test_unknown_model_refused():
    with pytest.raises(ValueError):
        Decider(None, "claude-mystery-1", DailyBudget(1.0))
