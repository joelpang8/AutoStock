# Running the bot on an always-on server

The bot runs as a systemd service on a small Linux server: it starts at boot,
restarts 10 seconds after any crash, and idles outside market hours. An outside
heartbeat check (healthchecks.io) alerts you if it goes silent, because a dead
bot can't warn you itself. No Docker needed.

Restarts are safe: halt, pause, the hourly order count and every order live in
`autostock.db`, and each order's id is tied to its 5-minute slot, so a restart
never sends an order twice.

## 1. Get a server (about $4 to $6 a month)

Any provider works. Pick:
- **Ubuntu 24.04**, the smallest plan (1 vCPU, 1 GB RAM is plenty).
- A **US East** region (close to Alpaca and the exchanges), e.g. New York or Ashburn.
- **SSH key** login rather than a password, if the provider offers it.

DigitalOcean (a "Basic" droplet), Hetzner (US locations) and Vultr all have
plans in that range.

## 2. Set up the free dead-man alert

1. Sign up at healthchecks.io and **Add Check**.
2. Set **Period 15 minutes, Grace 10 minutes** (the bot pings every 5 minutes
   in market hours and every 15 outside them).
3. Under Integrations, add your phone: the Discord webhook (same channel as
   the bot), email, or the mobile app.
4. Copy the check's **ping URL** (`https://hc-ping.com/...`). It goes in `.env`
   as `HEALTHCHECK_URL`.

## 3. Install

Log in (`ssh root@YOUR_SERVER_IP`) and run, for the branch you deploy
(`main` once the work is merged):

```bash
bash <(curl -fsSL https://raw.githubusercontent.com/joelpang8/AutoStock/main/deploy/install.sh) main
```

It installs Python, creates an `autostock` user, puts the code in
`/opt/autostock/app` and installs the service, but doesn't start it yet.

Turn on the firewall (the bot needs no open ports, only SSH):

```bash
ufw allow OpenSSH && ufw --force enable
```

## 4. Fill in the keys

```bash
nano /opt/autostock/app/.env
```

Fill in the Alpaca **paper** keys, `ANTHROPIC_API_KEY`, `ALPHAVANTAGE_API_KEY`,
the four `DISCORD_*` values and `HEALTHCHECK_URL`. Keep `ALPACA_PAPER=true`.
Save with Ctrl+O, Enter, Ctrl+X. The file is readable only by root and the bot.

On the server the Anthropic key is plain `ANTHROPIC_API_KEY` (the
`AUTOSTOCK_` name was only a workaround for Claude Code's cloud sessions).

## 5. Start it

```bash
systemctl start autostock
journalctl -u autostock -f      # watch it start, Ctrl+C to stop watching
```

You should see "bot started" in Discord and the healthchecks.io check turn green.

## Everyday commands (on the server)

| what | command |
| --- | --- |
| is it running? | `systemctl status autostock` |
| live log | `journalctl -u autostock -f` |
| stop / start / restart | `systemctl stop autostock` (or `start`, `restart`) |
| update to the latest code | `bash /opt/autostock/app/deploy/update.sh` |
| look at the decisions | `sqlite3 /opt/autostock/app/autostock.db "select ts, action, ticker, usd, reason from decisions order by id desc limit 20"` |

`stop` on the server stops the process. `/pause` and `/halt` in Discord are the
everyday controls, and a halt stays in force across restarts until `/resume`.

## Going live

Not yet. Live trading needs the owner's explicit yes (see CLAUDE.md), new live
Alpaca keys in `.env`, and `ALPACA_PAPER=false` plus
`ALLOW_LIVE_TRADING=yes-i-have-owner-approval`.
