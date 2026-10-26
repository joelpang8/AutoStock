#!/usr/bin/env bash
# Pull the latest code and restart, run as root: bash /opt/autostock/app/deploy/update.sh
# The restart is safe: halt/pause and order history live in autostock.db, and
# order ids are tied to the 5-minute slot, so nothing is sent twice.
set -euo pipefail
BASE=/opt/autostock
sudo -u autostock git -C "$BASE/app" pull --ff-only
sudo -u autostock "$BASE/venv/bin/pip" install -q -r "$BASE/app/requirements.txt"
sudo -u autostock bash -c "cd $BASE/app && $BASE/venv/bin/python test_guardrails.py"
cp "$BASE/app/deploy/autostock.service" /etc/systemd/system/autostock.service
systemctl daemon-reload
systemctl restart autostock
systemctl --no-pager status autostock | head -5
