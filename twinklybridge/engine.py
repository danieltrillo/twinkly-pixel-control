"""The bridge: Art-Net / sACN in, one realtime stream per Twinkly out.

    ArtDmx / sACN  ->  universe map  ->  device pixel buffer (console layout)
    device frame loop (fixed fps)  ->  reorder to Twinkly layout  ->  UDP 7777

Each device also runs a control loop over HTTP: log in, read what the device
is, put it in realtime ("rt") mode when data arrives, check every few seconds
that it is still in rt mode, and log in again when the token runs out.
"""

from __future__ import annotations

import asyncio
import logging
import socket
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from . import __version__, artnet, mapcode, sacn
from .config import ARTNET, SACN, Config, DeviceConfig, Segment, format_universe
from .pixels import convert
from .twinkly import AuthError, DeviceInfo, TwinklyClient, TwinklyError, build_packets, protocol_for_firmware

log = logging.getLogger(__name__)

KEEPALIVE = 0.5  # resend the last frame this often when nothing changes; rt mode times out without frames
MODE_CHECK = 5.0  # seconds between "are you still in rt mode?" checks
RELOGIN_AFTER = 3 * 3600.0  # tokens last about 4 hours
CONTROL_TICK = 0.25
ACTIVE_WINDOW = 3.0
BACKOFF = (1, 2, 4, 8, 15)

Sender = Callable[[bytes, tuple[str, int]], bool]


class RateCounter:
    __slots__ = ("total", "rate", "last", "_window", "_window_start")

    def __init__(self) -> None:
        self.total = 0
        self.rate = 0.0
        self.last = 0.0
        self._window = 0
        self._window_start = time.monotonic()

    def hit(self, now: float) -> None:
        self.total += 1
        self._window += 1
        self.last = now

    def tick(self, now: float) -> None:
        elapsed = now - self._window_start
        if elapsed > 0:
            self.rate = self._window / elapsed
            self._window = 0
            self._window_start = now

    def to_dict(self, now: float) -> dict:
        return {
            "total": self.total,
            "fps": round(self.rate, 1),
            "age": round(now - self.last, 1) if self.last else None,
        }


# --------------------------------------------------------------------------
# Camera mapping scan
# --------------------------------------------------------------------------


class ScanError(Exception):
    """A mapping scan could not start. The message is user facing."""


@dataclass
class ScanJob:
    """The camera-mapping blink code (mapcode.py), pre-rendered in the device's wire order.

    While a scan runs the device shows these frames instead of its Art-Net input;
    the input keeps landing in the buffer and takes over again when the scan ends.
    """

    frames: list[bytes]
    slot: float
    repeat: int
    started: float
    pixels: int
    brightness: int
    color: str
    stopped: bool = False

    @property
    def pass_seconds(self) -> float:
        return len(self.frames) * self.slot

    @property
    def total_seconds(self) -> float:
        return self.pass_seconds * self.repeat

    def elapsed(self, now: float) -> float:
        return max(0.0, now - self.started)

    def done(self, now: float) -> bool:
        return self.stopped or self.elapsed(now) >= self.total_seconds

    def frame_at(self, now: float) -> bytes:
        return self.frames[int(self.elapsed(now) / self.slot) % len(self.frames)]

    def status(self, now: float) -> dict:
        t = self.elapsed(now)
        return {
            "pass": min(int(t / self.pass_seconds) + 1, self.repeat),
            "passes": self.repeat,
            "elapsed": round(t, 1),
            "remaining": round(max(0.0, self.total_seconds - t), 1),
            "pass_seconds": round(self.pass_seconds, 1),
            "pixels": self.pixels,
            "slot": self.slot,
            "brightness": self.brightness,
            "color": self.color,
        }


def render_scan_frames(pixels: int, order: str, brightness: int, color: str) -> list[bytes]:
    """One wire-order frame per slot of the blink code."""
    v = max(1, min(int(brightness), 255))
    order = order.upper()
    if color == "white" and "W" in order:
        lit_rgbw = {"R": 0, "G": 0, "B": 0, "W": v}
    elif color == "green":
        lit_rgbw = {"R": 0, "G": v, "B": 0, "W": 0}
    else:  # "rgb": white from the colour chips (also what "white" means on an RGB strand)
        lit_rgbw = {"R": v, "G": v, "B": v, "W": 0}
    lit = bytes(lit_rgbw[ch] for ch in order)
    black = bytes(len(order))
    words = [mapcode.codeword(i, pixels) for i in range(pixels)]
    frames = []
    for slot in mapcode.schedule(pixels):
        mask = mapcode.lit_mask(slot, pixels, words)
        frames.append(b"".join(lit if on else black for on in mask))
    return frames


# --------------------------------------------------------------------------
# One Twinkly device
# --------------------------------------------------------------------------


@dataclass
class DeviceRunner:
    cfg: DeviceConfig
    sender: Sender
    session: object = None  # aiohttp.ClientSession
    clock: Callable[[], float] = time.monotonic
    state: str = "starting"
    error: str = ""
    info: DeviceInfo | None = None
    order: str = ""
    protocol: int = 3
    previous_mode: str = ""
    rt_armed: bool = False
    ever_had_data: bool = False
    last_rx: float = 0.0
    last_send: float = 0.0
    dirty: bool = False
    rx: RateCounter = field(default_factory=RateCounter)
    tx: RateCounter = field(default_factory=RateCounter)
    send_errors: int = 0
    warnings: list[str] = field(default_factory=list)
    scan: ScanJob | None = None

    def __post_init__(self) -> None:
        self.buffer = bytearray(self.cfg.pixels * self.cfg.channels)
        self.client: TwinklyClient | None = None
        self._ready = False
        self._login_at = 0.0
        self._mode_checked = 0.0
        self._tasks: list[asyncio.Task] = []

    @property
    def name(self) -> str:
        return self.cfg.name

    @property
    def ready(self) -> bool:
        return self._ready

    # -- input ---------------------------------------------------------------

    def write(self, segment: Segment, data: bytes, now: float) -> None:
        ch = self.cfg.channels
        chunk = data[segment.channel : segment.channel + segment.count * ch]
        start = segment.first_pixel * ch
        self.buffer[start : start + len(chunk)] = chunk
        self.dirty = True
        self.last_rx = now
        self.ever_had_data = True
        self.rx.hit(now)

    def has_data(self, now: float) -> bool:
        return self.ever_had_data and now - self.last_rx < self.cfg.loss_timeout

    def wants_rt(self, now: float) -> bool:
        if self.scanning(now) or self.has_data(now):
            return True
        return self.ever_had_data and self.cfg.on_loss in ("hold", "black")

    # -- camera mapping scan ---------------------------------------------------

    def scanning(self, now: float) -> bool:
        return self.scan is not None and not self.scan.done(now)

    def start_scan(self, *, slot: float, brightness: int, color: str, repeat: int) -> ScanJob:
        """Show the blink code on this device for `repeat` passes (replaces the input meanwhile)."""
        if not self._ready or self.client is None or not self.order:
            raise ScanError(f"{self.name} is {self.state}; it has to be online first")
        if self.info is not None and self.info.group_mode == "slave":
            raise ScanError(f"{self.name} is a group slave; take it out of its group in the Twinkly app first")
        pixels = self.out_pixels
        frames = render_scan_frames(pixels, self.order, brightness, color)
        self.scan = ScanJob(
            frames=frames,
            slot=float(slot),
            repeat=max(1, int(repeat)),
            started=self.clock(),
            pixels=pixels,
            brightness=max(1, min(int(brightness), 255)),
            color=color,
        )
        log.info(
            "%s: mapping scan started, %d pixels, %d pass(es) of %.0f s",
            self.name,
            pixels,
            self.scan.repeat,
            self.scan.pass_seconds,
        )
        return self.scan

    def stop_scan(self) -> None:
        if self.scan is not None and not self.scan.stopped:
            self.scan.stopped = True
            log.info("%s: mapping scan stopped", self.name)

    # -- output --------------------------------------------------------------

    @property
    def out_pixels(self) -> int:
        if self.info and self.info.number_of_led > 0:
            return self.info.number_of_led
        return self.cfg.pixels

    def render(self, black: bool = False) -> bytes:
        n = self.out_pixels
        ch = self.cfg.channels
        if black:
            return bytes(n * len(self.order))
        src = self.buffer
        if n != self.cfg.pixels:
            src = bytes(src[: n * ch]).ljust(n * ch, b"\x00")
        return bytes(convert(src, self.cfg.input, self.order, n))

    def send_frame(self, now: float, black: bool = False, frame: bytes | None = None) -> None:
        if self.client is None or not self.client.token:
            return
        if frame is None:
            frame = self.render(black)
        ok = True
        for packet in build_packets(self.protocol, self.client.token, frame, self.out_pixels):
            if not self.sender(packet, (self.cfg.ip, self.cfg.rt_port)):
                ok = False
        if ok:
            self.tx.hit(now)
        else:
            self.send_errors += 1
        self.last_send = now
        self.dirty = False

    def frame_step(self, now: float) -> None:
        """Called at cfg.fps. Sends a frame when there is something to send."""
        if not self._ready or not self.rt_armed:
            return
        if self.scan is not None:
            if not self.scan.done(now):
                self.send_frame(now, frame=self.scan.frame_at(now))
                return
            log.info("%s: mapping scan finished", self.name)
            self.scan = None
            self.dirty = True  # show the input (or the held look) again right away
        if self.has_data(now):
            if self.dirty or now - self.last_send >= KEEPALIVE:
                self.send_frame(now)
        elif self.cfg.on_loss == "hold":
            if now - self.last_send >= KEEPALIVE:
                self.send_frame(now)
        elif self.cfg.on_loss == "black":
            if now - self.last_send >= KEEPALIVE:
                self.send_frame(now, black=True)

    # -- control -------------------------------------------------------------

    async def connect(self) -> None:
        self.state = "connecting"
        client = TwinklyClient(self.session, self.cfg.ip, port=self.cfg.http_port)
        await client.login()  # first: Pro firmware 4.x answers fw/ct1/version only with a token
        info = await client.info()
        self.client = client
        self.info = info
        self.protocol = protocol_for_firmware(info.firmware)
        self.order = self.cfg.order if self.cfg.order != "auto" else ("WRGB" if info.led_profile == "RGBW" else "RGB")
        self.warnings = []
        if info.number_of_led and info.number_of_led != self.cfg.pixels:
            self.warnings.append(f"config says {self.cfg.pixels} pixels, device reports {info.number_of_led}")
        if len(self.order) != info.bytes_per_led:
            self.warnings.append(f"order {self.order} has {len(self.order)} bytes, device is {info.led_profile}")
        if info.group_mode == "slave":
            self.warnings.append(
                f"controller is a group slave (size {info.group_size}, offset {info.group_offset}); "
                "Pro firmware refuses realtime until it leaves the group"
            )
        for w in self.warnings:
            log.warning("%s (%s): %s", self.name, self.cfg.ip, w)
        self._login_at = self.clock()
        self._ready = True
        self.rt_armed = False
        self.state = "ready"
        self.error = ""
        log.info(
            "%s (%s): %s fw %s, %d %s LEDs, sending %s with protocol v%d",
            self.name,
            self.cfg.ip,
            info.product_code or "Twinkly",
            info.firmware or "?",
            info.number_of_led,
            info.led_profile,
            self.order,
            self.protocol,
        )

    async def control_step(self, now: float) -> None:
        assert self.client is not None
        if now - self._login_at >= RELOGIN_AFTER:
            await self.client.login()
            self._login_at = now
        want = self.wants_rt(now)
        if want and not self.rt_armed:
            mode = await self.client.get_mode()
            if mode and mode != "rt":
                self.previous_mode = mode
            await self.client.set_mode("rt")
            self.rt_armed = True
            self._mode_checked = now
            self.send_frame(now)
            log.info("%s: streaming (was in '%s' mode)", self.name, self.previous_mode or "?")
        elif not want and self.rt_armed:
            back = self.previous_mode or "movie"
            await self.client.set_mode(back)
            self.rt_armed = False
            log.info("%s: no data for %.0fs, handed back to '%s' mode", self.name, self.cfg.loss_timeout, back)
        elif self.rt_armed and now - self._mode_checked >= MODE_CHECK:
            self._mode_checked = now
            mode = await self.client.get_mode()
            if mode != "rt":
                log.warning("%s: device left rt mode (now '%s'), switching back", self.name, mode)
                await self.client.set_mode("rt")
        if self.scanning(now) and self.rt_armed:
            self.state = f"mapping scan (pass {self.scan.status(now)['pass']}/{self.scan.repeat})"
        elif self.rt_armed:
            self.state = "streaming" if self.has_data(now) else f"no input ({self.cfg.on_loss})"
        else:
            self.state = "released" if self.ever_had_data else "ready, waiting for input"

    async def control_loop(self) -> None:
        failures = 0
        while True:
            try:
                if not self._ready:
                    await self.connect()
                    failures = 0
                await self.control_step(self.clock())
                await asyncio.sleep(CONTROL_TICK)
            except asyncio.CancelledError:
                raise
            except TwinklyError as exc:
                if isinstance(exc, AuthError) and self._ready:
                    # the token expired (or the device rebooted) mid-run: log in again right away
                    log.info("%s: token expired, logging in again", self.name)
                    self._ready = False
                    self.rt_armed = False
                    continue
                # a 401 while connecting is not an expired token; back off like any other failure
                if self._ready or failures == 0:
                    log.warning("%s (%s): %s", self.name, self.cfg.ip, exc)
                self._ready = False
                self.rt_armed = False
                self.state = "offline"
                self.error = str(exc)
                delay = BACKOFF[min(failures, len(BACKOFF) - 1)]
                failures += 1
                await asyncio.sleep(delay)
            except Exception as exc:  # never let one device kill the bridge
                log.exception("%s: unexpected error", self.name)
                self._ready = False
                self.rt_armed = False
                self.state = "error"
                self.error = f"{exc.__class__.__name__}: {exc}"
                await asyncio.sleep(BACKOFF[-1])

    async def frame_loop(self) -> None:
        period = 1.0 / self.cfg.fps
        loop = asyncio.get_running_loop()
        next_at = loop.time()
        while True:
            next_at += period
            delay = next_at - loop.time()
            if delay < 0:
                next_at = loop.time()
                delay = 0
            await asyncio.sleep(delay)
            try:
                self.frame_step(self.clock())
            except Exception:
                log.exception("%s: frame send failed", self.name)

    def start(self) -> None:
        loop = asyncio.get_running_loop()
        self._tasks = [loop.create_task(self.control_loop()), loop.create_task(self.frame_loop())]

    async def stop(self, release: bool = True) -> None:
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []
        if release and self.rt_armed and self.client is not None:
            try:
                await asyncio.wait_for(self.client.set_mode(self.previous_mode or "movie"), 2.0)
            except (TwinklyError, TimeoutError):
                pass
        self.rt_armed = False

    def status(self, now: float) -> dict:
        return {
            "name": self.name,
            "ip": self.cfg.ip,
            "enabled": self.cfg.enabled,
            "state": self.state,
            "error": self.error,
            "warnings": list(self.warnings),
            "protocol": self.cfg.protocol,
            "universes": self.cfg.universe_label,
            "universe_count": len(self.cfg.segments),
            "pixels": self.cfg.pixels,
            "input": self.cfg.input,
            "order": self.order or self.cfg.order,
            "fps": self.cfg.fps,
            "on_loss": self.cfg.on_loss,
            "udp_protocol": self.protocol if self.info else None,
            "previous_mode": self.previous_mode,
            "info": self.info.to_dict() if self.info else None,
            "rx": self.rx.to_dict(now),
            "tx": self.tx.to_dict(now),
            "send_errors": self.send_errors,
            "scan": self.scan.status(now) if self.scanning(now) else None,
        }


# --------------------------------------------------------------------------
# Sockets
# --------------------------------------------------------------------------


class _Protocol(asyncio.DatagramProtocol):
    def __init__(self, handler: Callable[[bytes, str], None]) -> None:
        self.handler = handler

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        try:
            self.handler(data, addr[0])
        except Exception:
            log.exception("error handling a packet from %s", addr[0])


def _listen_socket(port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if hasattr(socket, "SO_REUSEPORT"):
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except OSError:
            pass
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
    except OSError:
        pass
    sock.bind(("", port))
    sock.setblocking(False)
    return sock


def guess_local_ip() -> str:
    """Best guess at this box's main IPv4 address (no packet is sent)."""
    for probe in ("10.255.255.255", "2.255.255.255", "192.168.255.255"):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect((probe, 1))
            ip = s.getsockname()[0]
            if ip and not ip.startswith("127."):
                return ip
        except OSError:
            pass
        finally:
            s.close()
    return "0.0.0.0"


class UdpSender:
    """One unbound socket for every Twinkly frame."""

    def __init__(self) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        try:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
        except OSError:
            pass
        self.errors = 0
        self.last_error = ""

    def __call__(self, packet: bytes, addr: tuple[str, int]) -> bool:
        try:
            self.sock.sendto(packet, addr)
            return True
        except (BlockingIOError, InterruptedError):
            self.errors += 1
            self.last_error = "send buffer full"
            return False
        except OSError as exc:
            self.errors += 1
            self.last_error = f"{addr[0]}: {exc.strerror or exc}"
            return False

    def close(self) -> None:
        self.sock.close()


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------


class Engine:
    def __init__(self, config: Config, *, sender: Sender | None = None, hostname: str | None = None) -> None:
        self.config = config
        self.hostname = hostname or socket.gethostname()
        self._own_sender = sender is None
        self.sender: Sender = sender or UdpSender()
        self.devices = [DeviceRunner(cfg=d, sender=self.sender) for d in config.enabled_devices]
        self._map: dict[tuple[str, int], list[tuple[DeviceRunner, Segment]]] = {}
        self._rebuild_map()
        self.ip = config.input.ip if config.input.ip != "0.0.0.0" else guess_local_ip()
        self.last_rx: dict[tuple[str, int], float] = {}
        self.unmapped: dict[tuple[str, int], tuple[str, float]] = {}
        self.artnet_rx = RateCounter()
        self.sacn_rx = RateCounter()
        self.polls_rx = 0
        self.dropped: dict[str, int] = {}
        self._artnet_transport: asyncio.DatagramTransport | None = None
        self._sacn_transport: asyncio.DatagramTransport | None = None
        self._sacn_sock: socket.socket | None = None
        self._session = None
        self._tasks: list[asyncio.Task] = []
        self._poll_counter = 0
        self.started_at = 0.0
        self.running = False

    # -- lifetime --------------------------------------------------------------

    async def start(self, *, open_sockets: bool = True) -> None:
        import aiohttp

        loop = asyncio.get_running_loop()
        self._session = aiohttp.ClientSession()
        if open_sockets:
            if self.config.input.artnet:
                transport, _ = await loop.create_datagram_endpoint(
                    lambda: _Protocol(self.handle_artnet), sock=_listen_socket(artnet.ARTNET_PORT)
                )
                self._artnet_transport = transport
            if self.config.input.sacn:
                sock = _listen_socket(sacn.SACN_PORT)
                self._sacn_sock = sock
                self._join_sacn_groups(sock)
                transport, _ = await loop.create_datagram_endpoint(lambda: _Protocol(self.handle_sacn), sock=sock)
                self._sacn_transport = transport
        for dev in self.devices:
            dev.session = self._session
            dev.start()
        self._tasks = [loop.create_task(self._tick_loop())]
        self.running = True
        self.started_at = time.monotonic()
        self.announce()
        log.info(
            "bridge started: %d device(s), %d universe(s), listening as %s (Art-Net %s, sACN %s)",
            len(self.devices),
            len(self._map),
            self.ip,
            "on" if self.config.input.artnet else "off",
            "on" if self.config.input.sacn else "off",
        )

    async def stop(self) -> None:
        self.running = False
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []
        await asyncio.gather(*(d.stop() for d in self.devices), return_exceptions=True)
        for transport in (self._artnet_transport, self._sacn_transport):
            if transport is not None:
                transport.close()
        self._artnet_transport = self._sacn_transport = None
        if self._session is not None:
            await self._session.close()
            self._session = None
        if self._own_sender and isinstance(self.sender, UdpSender):
            self.sender.close()
        log.info("bridge stopped")

    def _rebuild_map(self) -> None:
        self._map = {}
        for dev in self.devices:
            for seg in dev.cfg.segments:
                self._map.setdefault((dev.cfg.protocol, seg.universe), []).append((dev, seg))

    async def apply_config(self, config: Config) -> None:
        """Update to a new config in place, touching only the controllers that changed.

        Controllers whose config is unchanged keep their connection, login and
        stream; only added, removed and changed ones are started or stopped, and
        the listen sockets stay open. The caller must use this only while
        config.input is unchanged (same sockets, same protocols); otherwise it
        restarts the whole engine.
        """
        old = {d.cfg.name: d for d in self.devices}
        wanted = {c.name: c for c in config.enabled_devices}
        kept: dict[str, DeviceRunner] = {}
        to_stop: list[DeviceRunner] = []
        for name, runner in old.items():
            new_cfg = wanted.get(name)
            if new_cfg is not None and new_cfg == runner.cfg:
                kept[name] = runner
            else:
                to_stop.append(runner)  # removed, disabled, or its settings changed
        started: list[DeviceRunner] = []
        new_devices: list[DeviceRunner] = []
        for c in config.enabled_devices:
            runner = kept.get(c.name)
            if runner is None:
                runner = DeviceRunner(cfg=c, sender=self.sender)
                runner.session = self._session
                started.append(runner)
            new_devices.append(runner)
        # Swap the device set and map in one synchronous step, so feed() never
        # sees a half-built map, then release the old controllers and start the new.
        self.devices = new_devices
        self._rebuild_map()
        self.config = config
        await asyncio.gather(*(r.stop(release=True) for r in to_stop), return_exceptions=True)
        for runner in started:
            runner.start()
        self.announce()
        log.info(
            "config applied in place: %d device(s), %d universe(s) (%d started, %d stopped, %d kept streaming)",
            len(new_devices),
            len(self._map),
            len(started),
            len(to_stop),
            len(kept),
        )

    def _join_sacn_groups(self, sock: socket.socket) -> None:
        local = self.config.input.ip if self.config.input.ip != "0.0.0.0" else "0.0.0.0"
        for protocol, universe in self._map:
            if protocol != SACN:
                continue
            group = sacn.multicast_address(universe)
            try:
                sock.setsockopt(
                    socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, socket.inet_aton(group) + socket.inet_aton(local)
                )
            except OSError as exc:
                log.warning("cannot join sACN universe %d (%s): %s", universe, group, exc)

    async def _tick_loop(self) -> None:
        while True:
            await asyncio.sleep(1.0)
            now = time.monotonic()
            self.artnet_rx.tick(now)
            self.sacn_rx.tick(now)
            for dev in self.devices:
                dev.rx.tick(now)
                dev.tx.tick(now)
            for key, (_src, seen) in list(self.unmapped.items()):
                if now - seen > 20:
                    del self.unmapped[key]

    # -- input -----------------------------------------------------------------

    def _drop(self, reason: str) -> None:
        self.dropped[reason] = self.dropped.get(reason, 0) + 1

    def feed(self, protocol: str, universe: int, data: bytes, source: str, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        self.last_rx[(protocol, universe)] = now
        targets = self._map.get((protocol, universe))
        if not targets:
            key = (protocol, universe)
            if key not in self.unmapped:
                log.info(
                    "%s universe %s from %s is not mapped to a device",
                    protocol,
                    format_universe(protocol, universe),
                    source,
                )
            self.unmapped[key] = (source, now)
            return
        for dev, seg in targets:
            dev.write(seg, data, now)

    def handle_artnet(self, data: bytes, src_ip: str) -> None:
        if src_ip == self.ip:
            return  # our own ArtPollReply coming back
        op = artnet.opcode(data)
        if op is None:
            self._drop("not art-net")
            return
        now = time.monotonic()
        self.artnet_rx.hit(now)
        if op == artnet.OP_DMX:
            dmx = artnet.decode_dmx(data)
            if dmx is None:
                self._drop("bad ArtDmx")
                return
            self.feed(ARTNET, dmx.universe, dmx.data, src_ip, now)
        elif op == artnet.OP_POLL:
            poll = artnet.decode_poll(data)
            if poll is not None:
                self.polls_rx += 1
                dest = src_ip if self.config.input.reply == "unicast" else self.config.input.broadcast
                for packet in self.build_poll_replies(poll, now):
                    self._send_artnet(packet, dest)

    def handle_sacn(self, data: bytes, src_ip: str) -> None:
        packet = sacn.decode_data(data)
        if packet is None or packet.preview or packet.terminated or packet.start_code != 0:
            return
        now = time.monotonic()
        self.sacn_rx.hit(now)
        self.feed(SACN, packet.universe, packet.data, src_ip, now)

    # -- ArtPollReply ----------------------------------------------------------

    def _send_artnet(self, packet: bytes, dest: str) -> None:
        if self._artnet_transport is not None:
            try:
                self._artnet_transport.sendto(packet, (dest, artnet.ARTNET_PORT))
            except OSError as exc:
                log.debug("ArtPollReply to %s failed: %s", dest, exc)

    def announce(self) -> None:
        for packet in self.build_poll_replies(None, time.monotonic()):
            self._send_artnet(packet, self.config.input.broadcast)

    @property
    def artnet_universes(self) -> list[int]:
        return sorted(u for p, u in self._map if p == ARTNET)

    def build_poll_replies(self, poll: artnet.ArtPoll | None, now: float) -> list[bytes]:
        """One ArtPollReply per group of up to four universes sharing Net and Sub-Net."""
        ports = [u for u in self.artnet_universes if poll is None or poll.wants(u)]
        if not ports and poll is not None and poll.target_top is not None:
            return []
        groups: dict[tuple[int, int], list[int]] = {}
        for pa in ports:
            net, sub, _ = artnet.split_port_address(pa)
            groups.setdefault((net, sub), []).append(pa)
        chunks = [(n, s, items[i : i + 4]) for (n, s), items in groups.items() for i in range(0, len(items), 4)]
        if not chunks:
            chunks = [(0, 0, [])]
        self._poll_counter += 1
        major, _, minor = __version__.partition(".")
        version = ((int(major) & 0xFF) << 8) | (int(minor.split(".")[0] or 0) & 0xFF)
        live = sum(1 for d in self.devices if d.rt_armed)
        report = artnet.node_report(artnet.RC_POWER_OK, self._poll_counter, f"{live}/{len(self.devices)} Twinkly live")
        replies = []
        for index, (net, sub, items) in enumerate(chunks, start=1):
            port_types = bytearray(4)
            good_out = bytearray(4)
            sw_out = bytearray(4)
            for i, pa in enumerate(items):
                port_types[i] = artnet.PORT_TYPE_OUTPUT
                sw_out[i] = pa & 0x0F
                if now - self.last_rx.get((ARTNET, pa), -1e9) < ACTIVE_WINDOW:
                    good_out[i] = artnet.GOOD_OUTPUT_DATA_TRANSMITTED
            reply = artnet.ArtPollReply(
                ip=self.ip,
                version=version,
                net_switch=net,
                sub_switch=sub,
                oem=0x00FF,
                status1=0xE0,
                esta_man=0x7FF0,
                short_name=self.config.input.short_name,
                long_name=self.config.input.long_name or f"Twinkly Bridge on {self.hostname}",
                node_report=report,
                num_ports=len(items),
                port_types=bytes(port_types),
                good_output=bytes(good_out),
                sw_out=bytes(sw_out),
                style=artnet.STYLE_NODE,
                bind_ip=self.ip,
                bind_index=index,
                status2=artnet.STATUS2_PORT_ADDRESS_15BIT | artnet.STATUS2_WEB_CONFIG,
            )
            replies.append(artnet.encode_poll_reply(reply))
        return replies

    # -- status ----------------------------------------------------------------

    def status(self) -> dict:
        now = time.monotonic()
        return {
            "version": __version__,
            "hostname": self.hostname,
            "running": self.running,
            "uptime": round(now - self.started_at) if self.started_at else 0,
            "input": {
                "ip": self.ip,
                "broadcast": self.config.input.broadcast,
                "artnet": self.config.input.artnet,
                "sacn": self.config.input.sacn,
                "artnet_rx": self.artnet_rx.to_dict(now),
                "sacn_rx": self.sacn_rx.to_dict(now),
                "polls_rx": self.polls_rx,
                "live_universes": len([1 for t in self.last_rx.values() if now - t < ACTIVE_WINDOW]),
            },
            "devices": [d.status(now) for d in self.devices],
            "unmapped": [
                {"protocol": p, "universe": format_universe(p, u), "source": src, "age": round(now - seen, 1)}
                for (p, u), (src, seen) in sorted(self.unmapped.items())
            ],
            "dropped": dict(self.dropped),
            "send_errors": getattr(self.sender, "errors", 0),
            "last_send_error": getattr(self.sender, "last_error", ""),
        }
