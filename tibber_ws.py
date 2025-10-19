from __future__ import annotations
import asyncio
import json
import os
from typing import Optional, Callable

import aiohttp


class TibberWSClient:
    """Minimal GraphQL websocket client for Tibber realtime liveMeasurement.

    Tries graphql-transport-ws first, falls back to legacy graphql-ws.
    """

    def __init__(self, token: str, home_id: str, *, session: Optional[aiohttp.ClientSession] = None, logger=None, prefer_legacy: bool = False, user_agent: Optional[str] = None, ws_url: Optional[str] = None):
        self.token = token
        self.home_id = home_id
        self.session = session
        self.logger = logger
        self.ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._protocol: Optional[str] = None
        self._recv_task: Optional[asyncio.Task] = None
        self._running = False
        self._prefer_legacy = prefer_legacy
        self._user_agent = user_agent or os.getenv("SAX_USER_AGENT") or "sax"
        self._ws_url = ws_url
        try:
            self._log_every_n = max(1, int(os.getenv("SAX_TIBBER_WS_LOG_EVERY_N", "20")))
        except Exception:
            self._log_every_n = 20

    @property
    def running(self) -> bool:
        return self._running and self.ws is not None and not self.ws.closed

    async def start(self, callback: Callable[[dict], None]) -> None:
        if self.running:
            return

        # Require dynamically provided websocket URL to avoid legacy endpoints
        if not self._ws_url:
            raise RuntimeError("No websocketSubscriptionUrl provided; refusing to use legacy endpoint")
        url = self._ws_url
        headers = {
            "User-Agent": str(self._user_agent),
            # Authorization header is optional; Tibber expects token in connection_init payload
            "Authorization": f"Bearer {self.token}",
        }
        owns_session = False
        sess = self.session
        if sess is None or sess.closed:
            sess = aiohttp.ClientSession()
            owns_session = True
        try:
            # Tibber expects graphql-transport-ws subprotocol
            protocols = ["graphql-transport-ws"]
            self.ws = await sess.ws_connect(
                url,
                headers=headers,
                autoping=True,
                protocols=protocols,
            )
            # Determine negotiated subprotocol
            try:
                chosen = self.ws.headers.get("Sec-WebSocket-Protocol")
                if not chosen:
                    # Fallback to first offered if server doesn't echo header
                    chosen = protocols[0]
                self._protocol = chosen
            except Exception:
                self._protocol = protocols[0]
            if self.logger:
                self.logger.info(f"Raw WS connected using protocol={self._protocol}")
            # Init with auth in payload as required by Tibber
            init_payload = {"token": self.token}
            await self.ws.send_json({"type": "connection_init", "payload": init_payload})

            # Wait for ack (explicit connection_ack required)
            ack_ok = False
            handshake_logged = 0
            # Wait up to 10s (100 * 0.1s) for an explicit ack; reply to ping if required by server
            for _ in range(100):
                try:
                    msg = await asyncio.wait_for(self.ws.receive(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                    except Exception:
                        continue
                    t = data.get("type")
                    if self.logger and handshake_logged < 3:
                        self.logger.info(f"Raw WS frame (handshake): {data}")
                        handshake_logged += 1
                    if t == "ping":
                        try:
                            await self.ws.send_json({"type": "pong"})
                        except Exception:
                            pass
                        continue
                    if t in ("connection_ack", "ka", "pong"):
                        if t == "connection_ack":
                            ack_ok = True
                            break
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    break
            if not ack_ok:
                if self.logger:
                    self.logger.warning("Raw WS: did not receive connection_ack within timeout; closing.")
                try:
                    await self.ws.close()
                except Exception:
                    pass
                raise RuntimeError("Tibber WS did not send connection_ack")

            # Use inline argument to mirror Tibber Explorer behavior
            query = (
                "subscription($homeId: ID!) {\n"
                "  liveMeasurement(homeId: $homeId) {\n"
                "    timestamp\n"
                "    power\n"
                "    powerProduction\n"
                "    accumulatedConsumption\n"
                "    accumulatedCost\n"
                "    currency\n"
                "    minPower\n"
                "    averagePower\n"
                "    maxPower\n"
                "  }\n"
                "}"
            )
            payload = {"query": query, "operationName": None, "variables": {"homeId": self.home_id}}
            sub_id = "1"
            # graphql-transport-ws uses 'subscribe'
            await self.ws.send_json({"id": sub_id, "type": "subscribe", "payload": payload})

            self._running = True
            loop = asyncio.get_running_loop()
            self._recv_task = loop.create_task(self._recv_loop(callback, sub_id))
        except Exception:
            if owns_session:
                await sess.close()
            raise

    async def _recv_loop(self, callback: Callable[[dict], None], sub_id: str):
        assert self.ws is not None
        frame_idx = 0
        try:
            async for msg in self.ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                    except Exception:
                        continue
                    t = data.get("type")
                    # Log errors and non-data frames at INFO; for data frames, log every Nth to reduce noise
                    if t not in ("next", "data"):
                        if self.logger:
                            self.logger.info(f"Raw WS frame: {data}")
                    if t in ("ka", "ping"):
                        # keep-alive; respond to ping
                        if t == "ping":
                            await self.ws.send_json({"type": "pong"})
                        continue
                    if t in ("next", "data"):
                        # Some servers omit id for single subscription streams
                        if data.get("id") not in (None, sub_id):
                            continue
                        frame_idx += 1
                        if self.logger and (frame_idx == 1 or frame_idx % self._log_every_n == 0):
                            self.logger.info(f"Raw WS frame: {data}")
                        payload = data.get("payload") or {}
                        # Normalize to shape {"data": {...}} expected by existing callback
                        if "data" in payload:
                            callback({"data": payload["data"]})
                    elif t in ("error", "complete"):
                        if t == "error" and self.logger:
                            self.logger.warning(f"Raw WS server error: {data}")
                        # subscription ended or errored; stop
                        break
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    break
        finally:
            await self.stop()

    async def stop(self):
        self._running = False
        try:
            if self._recv_task:
                self._recv_task.cancel()
        except Exception:
            pass
        try:
            if self.ws and not self.ws.closed:
                await self.ws.close()
        except Exception:
            pass
        self.ws = None
