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
        api: str = "v1",
        group_slave: bool = False,
        ports: int = 1,
        artnet: bool = False,
    ) -> None:
        self.ports = max(1, ports)  # v2 gestalt lists the pixels split evenly over this many outputs
        self.artnet = artnet  # v2: the built-in Art-Net mode (artnet_en in the gestalt, artnet/config)
        self.artnet_config = {"model": 0, "port_address": 0, "dmx_address": 1}
        self.api = api  # "v1" (consumer) or "v2" (Pro firmware 4.x: /xled/v2, result codes nested)
        self.group_slave = group_slave
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
        self.reject_tokens = False  # every authenticated call answers 401, like a device that lost our session
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
        if self.reject_tokens:
            return False
        return bool(self.token_b64) and request.headers.get("X-Auth-Token") == self.token_b64

    def _ok(self, payload: dict, code: int = CODE_OK) -> web.Response:
        """v1 puts the code at the top level; v2 nests it under result."""
        if self.api == "v1":
            return web.json_response({**payload, "code": code})
        return web.json_response({**payload, "result": {"code": code}})

    def _denied(self) -> web.Response:
        if self.api == "v1":
            return web.Response(status=401, text="Invalid Token.")
        return web.json_response({"result": {"code": 2401}}, status=401)

    async def _not_found(self, request: web.Request) -> web.Response:
        if self.api == "v1":
            return web.Response(status=404, text="Resource not found.")
        return web.json_response({"result": {"code": 2404}}, status=404)

    async def _login(self, request: web.Request) -> web.Response:
        body = await request.json()
        if len(base64.b64decode(body.get("challenge", ""))) != 32:
            return web.json_response({"code": 1101})
        self.token = os.urandom(8)
        self.token_b64 = base64.b64encode(self.token).decode()
        self.logins += 1
        extra = {"auth-role": "user"} if self.api == "v2" else {}
        return self._ok(
            {
                "authentication_token": self.token_b64,
                "authentication_token_expires_in": 14400,
                "challenge-response": "ab" * 20,
                **extra,
            }
        )

    async def _verify(self, request: web.Request) -> web.Response:
        if not self._authed(request):
            return self._denied()
        return self._ok({})

    async def _fw(self, request: web.Request) -> web.Response:
        if self.api == "v1":
            return self._ok({"version": self.firmware})
        if not self._authed(request):
            return self._denied()  # Pro firmware 4.0.1 wants a token for fw/ct1/version
        return web.json_response({"version": self.firmware, "desc": "Release build (fake)"})

    async def _gestalt(self, request: web.Request) -> web.Response:
        if self.api == "v1":
            return self._ok(
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
                }
            )
        # shaped like a real TWPROCTRLPLC21 on firmware 4.0.1
        group = (
            {"mode": "slave", "uid": "FAKE-GROUP", "size": 36000, "offset": 34560}
            if self.group_slave
            else {"mode": "none", "uid": "", "size": 0, "offset": 0}
        )
        return self._ok(
            {
                "product_code": self.product,
                "device_name": f"TwPRO_FAKE{self.http_port}",
                "uptime": 1,
                "max_supported_led": 1500,
                "number_of_led": self.pixels,
                "device_config": {
                    "led_drv": "d9865c",
                    "led_id": 182,
                    "led_profile": self.profile,
                    # like the real device: every port's string starts at 1 again
                    "ports": [
                        {"port_id": i, "strings": [{"start": 1, "len": self.pixels // self.ports}]}
                        for i in range(self.ports)
                    ],
                },
                "group": group,
                "frame_rate": 10,
                "artnet_en": self.artnet,
                "network": {"dhcp": True, "ip": self.host, "netmask": "255.255.255.0", "gateway": self.host},
                "cloud": True,
                "rest_locked": False,
            }
        )

    async def _get_mode(self, request: web.Request) -> web.Response:
        if not self._authed(request):
            return self._denied()
        return self._ok({"mode": self.mode})

    async def _set_mode(self, request: web.Request) -> web.Response:
        if not self._authed(request):
            return self._denied()
        if self.group_slave:
            return web.json_response({"result": {"code": 2403}}, status=403)
        body = await request.json()
        self.mode = str(body.get("mode"))
        self.modes.append(self.mode)
        if self.mode == "rt":
            self.last_udp = time.monotonic()
        return self._ok({})

    async def _get_artnet(self, request: web.Request) -> web.Response:
        if not self._authed(request):
            return self._denied()
        return self._ok({"enable": self.artnet, **self.artnet_config})

    async def _set_artnet(self, request: web.Request) -> web.Response:
        if not self._authed(request):
            return self._denied()
        body = await request.json()
        if not isinstance(body.get("enable"), bool):
            return web.json_response({"result": {"code": 1104}}, status=400)
        self.artnet = body["enable"]
        for key in self.artnet_config:
            if isinstance(body.get(key), int):
                self.artnet_config[key] = body[key]
        return self._ok({})

    async def _led_config(self, request: web.Request) -> web.Response:
        if not self._authed(request):
            return self._denied()
        return self._ok({"strings": [{"first_led_id": 0, "length": self.pixels}]})

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
        v = self.api
        app.router.add_post(f"/xled/{v}/login", self._login)
        app.router.add_post(f"/xled/{v}/verify", self._verify)
        app.router.add_get(f"/xled/{v}/gestalt", self._gestalt)
        if v == "v1":
            app.router.add_get("/xled/v1/fw/version", self._fw)
            app.router.add_get("/xled/v1/led/mode", self._get_mode)
            app.router.add_post("/xled/v1/led/mode", self._set_mode)
            app.router.add_get("/xled/v1/led/config", self._led_config)
        else:
            app.router.add_get("/xled/v2/fw/ct1/version", self._fw)
            app.router.add_get("/xled/v2/application/mode", self._get_mode)
            app.router.add_post("/xled/v2/application/mode", self._set_mode)
            app.router.add_get("/xled/v2/artnet/config", self._get_artnet)
            app.router.add_post("/xled/v2/artnet/config", self._set_artnet)
        app.router.add_route("*", "/{tail:.*}", self._not_found)
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
        api=args.api,
        group_slave=args.group_slave,
        ports=args.ports,
        artnet=args.artnet,
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
    p.add_argument("--api", choices=["v1", "v2"], default="v1", help="v2 = behave like Pro firmware 4.x")
    p.add_argument("--group-slave", action="store_true", help="v2 only: refuse output writes like a group slave")
    p.add_argument("--ports", type=int, default=1, help="v2 only: how many outputs the gestalt lists")
    p.add_argument("--artnet", action="store_true", help="v2 only: start with the built-in Art-Net mode on")

    def _run(args) -> int:
        try:
            asyncio.run(run_fake(args))
        except KeyboardInterrupt:
            pass
        return 0

    p.set_defaults(func=_run)
