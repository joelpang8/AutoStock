#!/usr/bin/env bash
# One-time setup on a fresh Ubuntu 24.04 server, run as root (see DEPLOY.md):
#   bash <(curl -fsSL https://raw.githubusercontent.com/joelpang8/AutoStock/BRANCH/deploy/install.sh) BRANCH
# BRANCH defaults to main.
# It installs Python, creates an "autostock" user, puts the code in
# /opt/autostock/app, and installs (but does not start) the service.
set -euo pipefail

BRANCH="${1:-main}"
REPO="https://github.com/joelpang8/AutoStock.git"
BASE=/opt/autostock

[ "$(id -u)" = 0 ] || { echo "run as root"; exit 1; }

apt-get update -qq
apt-get install -y -qq git python3 python3-venv sqlite3
timedatectl set-timezone America/New_York || true   # logs in market time; the bot itself uses UTC/ET explicitly

id autostock >/dev/null 2>&1 || useradd --system --create-home --home-dir "$BASE" --shell /usr/sbin/nologin autostock
mkdir -p "$BASE"
if [ -d "$BASE/app/.git" ]; then
    echo "$BASE/app already exists, leaving it as is (use deploy/update.sh to update)"
else
    git clone --branch "$BRANCH" "$REPO" "$BASE/app"
fi
python3 -m venv "$BASE/venv"
"$BASE/venv/bin/pip" install -q --upgrade pip
"$BASE/venv/bin/pip" install -q -r "$BASE/app/requirements.txt"

if [ ! -f "$BASE/app/.env" ]; then
    cp "$BASE/app/.env.example" "$BASE/app/.env"
    echo "created $BASE/app/.env from the example: fill it in before starting"
fi
chown -R autostock:autostock "$BASE"
chmod 600 "$BASE/app/.env"

cp "$BASE/app/deploy/autostock.service" /etc/systemd/system/autostock.service
systemctl daemon-reload
systemctl enable autostock >/dev/null

cat <<MSG

Installed. Next:
  1. nano $BASE/app/.env      (paste your keys; keep ALPACA_PAPER=true)
  2. systemctl start autostock
  3. journalctl -u autostock -f   (watch it start; Ctrl+C to stop watching)
MSG
