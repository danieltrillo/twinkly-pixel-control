"""A fake Twinkly controller, for tests and for dry runs without hardware.

    twinkly-bridge fake --pixels 300 --http-port 8081 --rt-port 7778

It speaks enough of the local API (login, verify, gestalt, fw/version,
led/mode, led/config) and the realtime UDP protocol (v1, v2 and v3) for the
bridge to stream to it, and prints the frame rate it receives.
"""

from __future__ import annotations

import asyncio
import base64
import os
import time

from aiohttp import web

from .twinkly import CODE_OK


class _Udp(asyncio.DatagramProtocol):
    def __init__(self, fake: FakeTwinkly) -> None:
        self.fake = fake

    def datagram_received(self, data: bytes, addr) -> None:
        self.fake.on_udp(data)


class FakeTwinkly:
    def __init__(
        self,
        *,
        pixels: int = 250,
        profile: str = "RGBW",
        firmware: str = "2.8.18",
        product: str = "TWPRO1500ETHP",
        host: str = "127.0.0.1",
        http_port: int = 0,
        rt_port: int = 0,
        rt_timeout: float = 3.0,
    ) -> None:
        self.pixels = pixels
        self.profile = profile.upper()
        self.bpl = 4 if self.profile == "RGBW" else 3
        self.firmware = firmware
        self.product = product
        self.host = host
        self.http_port = http_port
        self.rt_port = rt_port
        self.rt_timeout = rt_timeout
        self.mode = "movie"
        self.modes: list[str] = []
        self.token_b64 = ""
        self.token = b""
        self.logins = 0
        self.frames = 0
        self.packets = 0
        self.bad_packets = 0
        self.last_frame = b""
        self.last_udp = 0.0
        self._partial = bytearray()
        self._runner: web.AppRunner | None = None
        self._udp: asyncio.DatagramTransport | None = None
        self._watch: asyncio.Task | None = None

    # -- HTTP -----------------------------------------------------------------

    def _authed(self, request: web.Request) -> bool:
        return bool(self.token_b64) and request.headers.get("X-Auth-Token") == self.token_b64

    @staticmethod
    def _denied() -> web.Response:
        return web.Response(status=401, text="Invalid Token.")

    async def _login(self, request: web.Request) -> web.Response:
        body = await request.json()
        if len(base64.b64decode(body.get("challenge", ""))) != 32:
            return web.json_response({"code": 1101})
        self.token = os.urandom(8)
        self.token_b64 = base64.b64encode(self.token).decode()
        self.logins += 1
        return web.json_response(
            {
                "authentication_token": self.token_b64,
                "authentication_token_expires_in": 14400,
                "challenge-response": "ab" * 20,
                "code": CODE_OK,
            }
        )

    async def _verify(self, request: web.Request) -> web.Response:
        if not self._authed(request):
            return self._denied()
        return web.json_response({"code": CODE_OK})

    async def _fw(self, request: web.Request) -> web.Response:
        return web.json_response({"version": self.firmware, "code": CODE_OK})

    async def _gestalt(self, request: web.Request) -> web.Response:
        return web.json_response(
            {
                "product_name": "Twinkly",
                "product_code": self.product,
                "device_name": f"Twinkly_FAKE{self.http_port}",
                "fw_family": "G",
                "led_profile": self.profile,
                "bytes_per_led": self.bpl,
                "number_of_led": self.pixels,
                "max_supported_led": 1500,
                "frame_rate": 24,
                "measured_frame_rate": 24.0,
                "mac": "02:00:00:00:00:01",
                "code": CODE_OK,
            }
        )

    async def _get_mode(self, request: web.Request) -> web.Response:
        if not self._authed(request):
            return self._denied()
        return web.json_response({"mode": self.mode, "code": CODE_OK})

    async def _set_mode(self, request: web.Request) -> web.Response:
        if not self._authed(request):
            return self._denied()
        body = await request.json()
        self.mode = str(body.get("mode"))
        self.modes.append(self.mode)
        if self.mode == "rt":
            self.last_udp = time.monotonic()
        return web.json_response({"code": CODE_OK})

    async def _led_config(self, request: web.Request) -> web.Response:
        if not self._authed(request):
            return self._denied()
        return web.json_response({"strings": [{"first_led_id": 0, "length": self.pixels}], "code": CODE_OK})

    # -- UDP ------------------------------------------------------------------

    def on_udp(self, data: bytes) -> None:
        self.packets += 1
        if len(data) < 10 or data[1:9] != self.token:
            self.bad_packets += 1
            return
        if self.mode != "rt":
            return  # a real device ignores frames outside rt mode
        self.last_udp = time.monotonic()
        need = self.pixels * self.bpl
        version = data[0]
        if version == 3:
            if len(data) < 12:
                self.bad_packets += 1
                return
            if data[11] == 0:
                self._partial = bytearray()
            self._partial += data[12:]
            if len(self._partial) >= need:
                self.last_frame = bytes(self._partial[:need])
                self.frames += 1
                self._partial = bytearray()
        else:
            self.last_frame = bytes(data[10 : 10 + need])
            self.frames += 1

    def pixel(self, index: int) -> tuple[int, ...]:
        """Pixel as sent on the wire (WRGB order for RGBW)."""
        start = index * self.bpl
        return tuple(self.last_frame[start : start + self.bpl])

    async def _rt_watchdog(self) -> None:
        while True:
            await asyncio.sleep(0.1)
            if self.mode == "rt" and time.monotonic() - self.last_udp > self.rt_timeout:
                self.mode = "movie"

    # -- lifetime -------------------------------------------------------------

    async def start(self) -> None:
        app = web.Application()
        app.router.add_post("/xled/v1/login", self._login)
        app.router.add_post("/xled/v1/verify", self._verify)
        app.router.add_get("/xled/v1/fw/version", self._fw)
        app.router.add_get("/xled/v1/gestalt", self._gestalt)
        app.router.add_get("/xled/v1/led/mode", self._get_mode)
        app.router.add_post("/xled/v1/led/mode", self._set_mode)
        app.router.add_get("/xled/v1/led/config", self._led_config)
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, self.http_port)
        await site.start()
        self.http_port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
        loop = asyncio.get_running_loop()
        self._udp, _ = await loop.create_datagram_endpoint(lambda: _Udp(self), local_addr=(self.host, self.rt_port))
        self.rt_port = self._udp.get_extra_info("sockname")[1]
        self._watch = loop.create_task(self._rt_watchdog())

    async def stop(self) -> None:
        if self._watch is not None:
            self._watch.cancel()
        if self._udp is not None:
            self._udp.close()
        if self._runner is not None:
            await self._runner.cleanup()


async def run_fake(args) -> None:
    fake = FakeTwinkly(
        pixels=args.pixels,
        profile=args.profile,
        firmware=args.firmware,
        host=args.host,
        http_port=args.http_port,
        rt_port=args.rt_port,
    )
    await fake.start()
    print(
        f"fake Twinkly on {fake.host}: HTTP {fake.http_port}, realtime UDP {fake.rt_port}, "
        f"{fake.pixels} {fake.profile} pixels. Ctrl+C to stop."
    )
    last = 0
    try:
        while True:
            await asyncio.sleep(1)
            fps = fake.frames - last
            last = fake.frames
            first = fake.pixel(0) if fake.last_frame else ()
            print(f"mode {fake.mode:<6} {fps:>3} fps  frames {fake.frames:<7} bad {fake.bad_packets}  pixel0 {first}")
    finally:
        await fake.stop()


def add_fake_command(sub) -> None:
    p = sub.add_parser("fake", help="run a fake Twinkly controller for dry runs")
    p.add_argument("--pixels", type=int, default=250)
    p.add_argument("--profile", choices=["RGB", "RGBW"], default="RGBW")
    p.add_argument("--firmware", default="2.8.18")
    p.add_argument("--host", default="127.0.0.1", help="address to listen on (default 127.0.0.1)")
    p.add_argument("--http-port", type=int, default=8081)
    p.add_argument("--rt-port", type=int, default=7778)

    def _run(args) -> int:
        try:
            asyncio.run(run_fake(args))
        except KeyboardInterrupt:
            pass
        return 0

    p.set_defaults(func=_run)
