# Project brief: autonomous AI stock trading bot

## Goal
Build a bot that trades US stocks by itself, all the time, using Claude (via the Anthropic API) to make trading decisions. The owner can check in at any moment and can pause or halt it.

## Owner's requirements (do not change without asking)
- Market: US stocks and ETFs only, through Alpaca (cash account, fractional shares by dollar amount).
- Starting budget: $300 of real money.
- Live trading from the start. No long paper-trading phase. Paper keys are fine for building and testing the connection, but the first live run uses very small positions ($10 to $20) instead.
- Trades are unlimited in principle. The hourly order cap in `guardrails.py` is only a safety net.
- Hard floor: if total equity falls below **$200**, the bot must cancel open orders, sell everything, and halt until the owner manually resumes it.
- Runs always-on without supervision, but the owner can check in any time.

## Architecture
1. **Data layer:** every few minutes during market hours, pull prices, recent bars, and account state (equity, settled cash, positions) into a compact summary.
2. **Decision step:** send the summary to Claude through the Anthropic API. Require a strict structured reply: `action` (buy/sell/hold), `ticker`, `usd`, and a one-line `reason`. Reject anything malformed. Never guess at a bad reply.
3. **Guardrail layer (`guardrails.py`, already written and tested):** every proposed order must pass `Guardrails.check_order()` before it reaches the broker. This is plain code and the AI must not be able to bypass or modify it.
4. **Execution engine:** place orders by dollar amount, confirm fills, handle API errors and disconnects, never double-submit after a restart, and log everything.
5. **Check-in channel:** a Telegram or Discord bot with commands `status`, `pause`, `resume`, `halt`. It pushes an alert for every trade, every error, and a daily summary of equity and decisions.
6. **Deployment:** a systemd service on a small cloud server with auto-restart, a heartbeat alert if the bot goes silent, and the bot idle outside market hours (owner chose systemd over Docker; see DEPLOY.md).
7. **Storage:** SQLite for trades, decisions with reasoning, and state.

Stack: Python, `alpaca-py`, the Anthropic SDK, SQLite, systemd.

## Existing files
- `guardrails.py`: floor kill-switch ($200), per-trade cap ($20), per-ticker cap ($300, raised from $60 by the owner on 2026-10-28 for the QLD trend strategy), 12 orders per hour, allowed-ticker list (QLD added by the owner the same day), settled-cash check. Expects a broker object with `equity()`, `settled_cash()`, `position_value(ticker)`, `cancel_all_orders()`, `close_all_positions()`.
- `test_guardrails.py`: tests against a fake broker. Run with `python test_guardrails.py`. They must keep passing.

## Build order
1. Alpaca broker class implementing the interface `guardrails.py` expects, tested against Alpaca **paper** keys.
2. Data layer.
3. Decision step with strict output validation, plus tests that feed it malformed replies.
4. Execution engine wired through the guardrails.
5. SQLite logging.
6. Check-in bot.
7. Deployment (systemd, DEPLOY.md).
8. Only then, a live run, after asking the owner (see rules below).

## Things to verify, not assume
- How Alpaca reports **settled cash** for a small cash account. The guardrails depend on this, so find the right field and test it.
- Current Alpaca rules for fractional orders (allowed order types, time-in-force, trading hours), and how they apply day trading rules to a small cash account. The SEC approved removing the $25,000 pattern day trader minimum in 2026, but brokers have until October 2027 to implement it, so confirm what Alpaca enforces today.
- Fractional orders fill at the quote at submission with no price improvement. Account for this slippage.
- Check Alpaca's current docs instead of relying on memory for any API detail.

## Rules for you (Claude Code)
- **Never** put API keys in code, logs, or version control. Read them from environment variables or a `.env` file listed in `.gitignore`. Provide a `.env.example` with placeholder values.
- **Never** switch to live trading, or place a real order, without asking the owner first and getting a clear yes.
- **Never** raise the limits in `guardrails.py` or remove a check without asking. The floor is $200 unless the owner says otherwise.
- The floor and size limits must stay enforced in code, not in prompts. The AI's reasoning is never a substitute for a guardrail.
- Add tests for each new component. Keep `python test_guardrails.py` passing.
- Handle failure safely: if data is stale, the API is down, or the AI's reply is invalid, do nothing (hold) and alert the owner. If the account state can't be read, don't trade.
- Log every decision and every order with a timestamp and the reason.

## Done means
- All tests pass.
- The bot runs end to end against Alpaca paper keys, including a forced floor breach that correctly liquidates and halts.
- Check-in commands and alerts work from the owner's phone.
- It restarts cleanly after a crash without duplicating orders.
- The owner has reviewed it and approved the first live run.

## Reality check
Most active strategies lose money after costs, and an LLM picking trades has no proven edge. The $300 is money the owner can afford to lose. The first month's goal is a system that runs reliably, respects the floor, and produces logs the owner understands, not profit. Compare results against simply holding an index ETF.
