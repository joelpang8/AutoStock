"""One real Claude call on a made-up snapshot (costs well under $0.01).

Skipped unless ANTHROPIC_API_KEY is set (env or .env).
    python -m pytest -v -s test_decision_live.py
"""
import os

import pytest
from dotenv import load_dotenv

load_dotenv()

from decision import Decider
from test_decision import SNAP

pytestmark = pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="no Anthropic key")


def test_one_real_decision():
    d = Decider.from_env()
    r = d.decide(SNAP)
    print(f"\nmodel={d.model} cost=${r.cost_usd:.5f} usage={r.usage} "
          f"decision={r.decision} error={r.error!r}\nraw={r.raw_text}")
    assert r.cost_usd < 0.01
    assert r.decision is not None, r.error
