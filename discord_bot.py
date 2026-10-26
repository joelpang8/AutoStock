"""Discord wiring for the check-in channel (logic lives in checkin.py).

Slash commands /status /pause /resume /halt in one server, owner only.
Alerts and the daily summary go to one channel. Needs no privileged intents.

The engine notifies from its own thread; alerts go through a thread-safe queue
that the bot drains, so an alert raised before Discord connects (or while it
reconnects) is sent once it's back rather than lost.

Setup (once):
  1. discord.com/developers/applications > New Application > Bot > Reset Token:
     that's DISCORD_BOT_TOKEN. No privileged intents needed.
  2. OAuth2 > URL Generator: scopes "bot" and "applications.commands",
     permissions "Send Messages" and "View Channels". Open the URL, add the bot
     to your own private server.
  3. Discord Settings > Advanced > Developer Mode on. Right-click the server,
     the alerts channel and your own name > Copy ID: DISCORD_GUILD_ID,
     DISCORD_CHANNEL_ID, DISCORD_OWNER_ID.
  4. Set that channel's notifications to "All Messages" on your phone so every
     trade alert arrives; halts and errors also @mention you.
"""
from __future__ import annotations

import asyncio
import logging
import queue

import discord
from discord import app_commands

from checkin import CheckIn

log = logging.getLogger(__name__)


class AlertQueue:
    """notify(kind, msg) is safe to call from any thread."""

    def __init__(self):
        self.q: queue.Queue = queue.Queue()

    def notify(self, kind: str, msg: str):
        log.warning("ALERT [%s] %s", kind, msg)
        self.q.put((kind, msg))

    def drain(self) -> list:
        out = []
        while True:
            try:
                out.append(self.q.get_nowait())
            except queue.Empty:
                return out


class CheckInBot(discord.Client):
    def __init__(self, checkin: CheckIn, alerts: AlertQueue, guild_id: int, channel_id: int):
        super().__init__(intents=discord.Intents.default(),
                         allowed_mentions=discord.AllowedMentions(users=True, everyone=False,
                                                                  roles=False))
        self.checkin, self.alerts = checkin, alerts
        self.guild = discord.Object(id=int(guild_id))
        self.channel_id = int(channel_id)
        self.tree = app_commands.CommandTree(self)
        self._register_commands()

    def _register_commands(self):
        c = self.checkin
        commands = {
            "status": ("Equity, positions, today's decisions, paused/halted", c.status),
            "pause": ("Stop new trades, keep positions", c.pause),
            "resume": ("Clear a pause or a halt and trade again", c.resume),
            "halt": ("Cancel orders, SELL EVERYTHING, stop until /resume", c.halt),
        }
        for name, (desc, fn) in commands.items():
            self.tree.add_command(app_commands.Command(
                name=name, description=desc, callback=self._handler(name, fn)), guild=self.guild)

    def _handler(self, name, fn):
        async def handler(interaction: discord.Interaction):
            if not self.checkin.authorized(interaction.user.id):
                log.warning("refused /%s from %s (%s)", name, interaction.user, interaction.user.id)
                await interaction.response.send_message("Not authorized.", ephemeral=True)
                return
            log.info("owner command /%s", name)
            await interaction.response.defer(thinking=True)   # broker calls can take seconds
            try:
                text = await asyncio.to_thread(fn)
            except Exception as e:
                log.exception("/%s failed", name)
                text = f"⚠️ /{name} failed: {e}"
            await interaction.followup.send(text[:1900])
        return handler

    async def setup_hook(self):
        await self.tree.sync(guild=self.guild)
        self.loop.create_task(self._pump())

    async def _channel(self):
        return self.get_channel(self.channel_id) or await self.fetch_channel(self.channel_id)

    async def _send(self, text: str):
        try:
            await (await self._channel()).send(text)
        except Exception:
            log.exception("could not send to Discord: %s", text)

    async def _pump(self):
        """Sends queued alerts and the daily summary, forever."""
        await self.wait_until_ready()
        while not self.is_closed():
            for kind, msg in self.alerts.drain():
                await self._send(self.checkin.alert_text(kind, msg))
            if self.checkin.summary_due():
                try:
                    await self._send(await asyncio.to_thread(self.checkin.daily_summary))
                except Exception:
                    log.exception("daily summary failed")
            await asyncio.sleep(1)
