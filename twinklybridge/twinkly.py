"""Twinkly local API: HTTP control on port 80, realtime pixel frames on UDP 7777.

This is Twinkly's unofficial, community documented protocol:
https://xled-docs.readthedocs.io/en/latest/protocol_details.html

    1. POST /xled/v1/login   {"challenge": base64(32 random bytes)}
       -> authentication_token (base64 of 8 bytes), challenge-response
    2. POST /xled/v1/verify  (header X-Auth-Token) -> code 1000
    3. POST /xled/v1/led/mode {"mode": "rt"}
    4. UDP frames to port 7777. The device drops back to its stored effect a
       few seconds after the last frame, so frames have to keep coming.

Pixel byte order on the wire: RGB strands are R,G,B. RGBW strands are
W,R,G,B (white first).
"""

from __future__ import annotations

import base64
import os
import socket
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

HTTP_PORT = 80
RT_PORT = 7777
DISCOVERY_PORT = 5555
FRAGMENT_BYTES = 900  # pixel bytes per UDP datagram in protocol v3
CODE_OK = 1000


class TwinklyError(Exception):
    """A request to a Twinkly device failed. The message is user facing."""


class AuthError(TwinklyError):
    """The device rejected our token (HTTP 401)."""


class NotFound(TwinklyError):
    """The endpoint does not exist on this firmware (HTTP 404 / code 2404)."""


def result_code(reply: dict) -> int | None:
    """v1 replies carry {"code": 1000}; v2 (Pro firmware) carry {"result": {"code": 1000}}."""
    if isinstance(reply.get("code"), int):
        return reply["code"]
    result = reply.get("result")
    if isinstance(result, dict) and isinstance(result.get("code"), int):
        return result["code"]
    return None


# --------------------------------------------------------------------------
# Wire format
# --------------------------------------------------------------------------


def version_tuple(text: str) -> tuple[int, ...]:
    out = []
    for part in str(text).split("."):
        digits = "".join(ch for ch in part if ch.isdigit())
        out.append(int(digits) if digits else 0)
    return tuple(out)


def protocol_for_firmware(fw: str) -> int:
    """UDP realtime protocol version: v3 (fragmented) for 2.4.14 and newer.

    Firmware that hides its version (newer Pro, /xled/v2 only) is modern: v3.
    """
    if not str(fw).strip():
        return 3
    v = version_tuple(fw)
    if v >= (2, 4, 14):
        return 3
    if v >= (2, 0, 0):
        return 2
    return 1


def default_order(led_profile: str) -> str:
    return "WRGB" if str(led_profile).upper() == "RGBW" else "RGB"


def valid_order(order: str) -> bool:
    order = order.upper()
    if len(order) == 3:
        return sorted(order) == sorted("RGB")
    if len(order) == 4:
        return sorted(order) == sorted("RGBW")
    return False


def build_packets(protocol: int, token: bytes, frame: bytes, n_leds: int) -> list[bytes]:
    """Split one frame into the UDP datagrams for the given protocol version."""
    if len(token) != 8:
        raise ValueError("token must be 8 raw bytes")
    if protocol == 3:
        packets = []
        for index, start in enumerate(range(0, max(len(frame), 1), FRAGMENT_BYTES)):
            header = b"\x03" + token + b"\x00\x00" + bytes([index & 0xFF])
            packets.append(header + frame[start : start + FRAGMENT_BYTES])
        return packets
    if protocol == 2:
        return [b"\x02" + token + b"\x00" + frame]
    return [b"\x01" + token + bytes([min(n_leds, 255)]) + frame]


def black_frame(n_leds: int, bytes_per_led: int) -> bytes:
    return bytes(n_leds * bytes_per_led)


# --------------------------------------------------------------------------
# Device info
# --------------------------------------------------------------------------


@dataclass
class DeviceInfo:
    firmware: str = ""
    product_code: str = ""
    device_name: str = ""
    led_profile: str = "RGB"
    number_of_led: int = 0
    max_supported_led: int = 0
    frame_rate: float = 0.0
    measured_frame_rate: float | None = None
    mac: str = ""
    fw_family: str = ""
    api: str = "v1"
    group_mode: str = ""
    group_offset: int = 0
    group_size: int = 0
    artnet_enabled: bool | None = None

    @property
    def bytes_per_led(self) -> int:
        return 4 if self.led_profile.upper() == "RGBW" else 3

    @classmethod
    def from_api(cls, gestalt: dict[str, Any], firmware: str, api: str = "v1") -> DeviceInfo:
        def num(key: str, default: float = 0, src: dict | None = None) -> float:
            try:
                return float((src if src is not None else gestalt).get(key, default))
            except (TypeError, ValueError):
                return default

        measured = gestalt.get("measured_frame_rate")
        device_config = gestalt.get("device_config") if isinstance(gestalt.get("device_config"), dict) else {}
        group = gestalt.get("group") if isinstance(gestalt.get("group"), dict) else {}
        profile = gestalt.get("led_profile") or device_config.get("led_profile") or "RGB"
        artnet = gestalt.get("artnet_en")
        return cls(
            firmware=str(firmware or ""),
            product_code=str(gestalt.get("product_code", "")),
            device_name=str(gestalt.get("device_name", "")),
            led_profile=str(profile).upper(),
            number_of_led=int(num("number_of_led")),
            max_supported_led=int(num("max_supported_led")),
            frame_rate=num("frame_rate"),
            measured_frame_rate=float(measured) if isinstance(measured, (int, float)) else None,
            mac=str(gestalt.get("mac", "")),
            fw_family=str(gestalt.get("fw_family", "")),
            api=api,
            group_mode=str(group.get("mode", "")),
            group_offset=int(num("offset", 0, group)),
            group_size=int(num("size", 0, group)),
            artnet_enabled=artnet if isinstance(artnet, bool) else None,
        )

    def to_dict(self) -> dict:
        return {
            "firmware": self.firmware,
            "product_code": self.product_code,
            "device_name": self.device_name,
            "led_profile": self.led_profile,
            "number_of_led": self.number_of_led,
            "max_supported_led": self.max_supported_led,
            "frame_rate": self.frame_rate,
            "measured_frame_rate": self.measured_frame_rate,
            "mac": self.mac,
            "api": self.api,
            "group_mode": self.group_mode,
            "group_offset": self.group_offset,
            "group_size": self.group_size,
            "artnet_enabled": self.artnet_enabled,
        }


# --------------------------------------------------------------------------
# Async HTTP client (aiohttp)
# --------------------------------------------------------------------------


class TwinklyClient:
    """Login and mode control for one device. Uses an aiohttp ClientSession."""

    def __init__(self, session: Any, ip: str, *, port: int = HTTP_PORT, timeout: float = 3.0) -> None:
        self.session = session
        self.ip = ip
        self.root = f"http://{ip}:{port}/xled"
        self.api = ""  # "v1" (consumer / older Pro firmware) or "v2" (newer Pro firmware); found by detect()
        self.timeout = timeout
        self.token_b64 = ""
        self.token = b""

    @property
    def base(self) -> str:
        return f"{self.root}/{self.api or 'v1'}"

    async def _request(self, method: str, path: str, body: dict | None = None, auth: bool = True) -> dict:
        import aiohttp

        headers = {}
        if auth and self.token_b64:
            headers["X-Auth-Token"] = self.token_b64
        try:
            async with self.session.request(
                method,
                self.base + path,
                json=body,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as resp:
                if resp.status == 401:
                    raise AuthError(f"{path}: token rejected")
                if resp.status == 404:
                    raise NotFound(f"{method} /xled/{self.api or 'v1'}{path}: not on this firmware")
                if resp.status >= 400:
                    text = (await resp.text())[:120]
                    raise TwinklyError(f"{method} {path}: HTTP {resp.status} {text}")
                try:
                    data = await resp.json(content_type=None)
                except ValueError:
                    raise TwinklyError(f"{method} {path}: reply was not JSON") from None
        except TwinklyError:
            raise
        except TimeoutError:
            raise TwinklyError(f"{method} {path}: no answer from {self.ip}") from None
        except aiohttp.ClientError as exc:
            raise TwinklyError(f"{method} {path}: {exc.__class__.__name__}: {exc}") from None
        except OSError as exc:
            raise TwinklyError(f"{method} {path}: {exc.strerror or exc}") from None
        if not isinstance(data, dict):
            raise TwinklyError(f"{method} {path}: unexpected reply {data!r}")
        return data

    async def login(self) -> None:
        if not self.api:
            await self.detect()
        challenge = base64.b64encode(os.urandom(32)).decode()
        reply = await self._request("POST", "/login", {"challenge": challenge}, auth=False)
        token = reply.get("authentication_token")
        if not token:
            raise TwinklyError(f"login refused: {reply}")
        try:
            raw = base64.b64decode(token)
        except ValueError:
            raise TwinklyError(f"login returned a bad token {token!r}") from None
        if len(raw) != 8:
            raise TwinklyError(f"login returned a {len(raw)} byte token, expected 8")
        self.token_b64 = token
        self.token = raw
        verify = await self._request("POST", "/verify", {"challenge-response": reply.get("challenge-response", "")})
        if result_code(verify) != CODE_OK:
            raise TwinklyError(f"verify refused: {verify}")

    async def detect(self) -> dict:
        """Find which API generation the device speaks and return its gestalt."""
        for api in ("v1", "v2"):
            self.api = api
            try:
                return await self._request("GET", "/gestalt", auth=False)
            except NotFound:
                continue
        self.api = ""
        raise TwinklyError(f"{self.ip} answers HTTP but has no /xled/v1 or /xled/v2 gestalt; is it a Twinkly?")

    # Endpoint names that differ between the two API generations. v2 (Pro
    # firmware 4.x) was verified on a TWPROCTRLPLC21; the names come from
    # the ttls library and from probing the device.
    @property
    def _mode_path(self) -> str:
        return "/application/mode" if self.api == "v2" else "/led/mode"

    @property
    def _fw_path(self) -> str:
        return "/fw/ct1/version" if self.api == "v2" else "/fw/version"

    async def info(self) -> DeviceInfo:
        """Gestalt plus firmware version. Call it after login(): Pro firmware 4.x
        answers fw/ct1/version only with a token, and this returns "" for the
        version rather than failing when there is none."""
        gestalt = await self.detect()
        try:
            fw = (await self._request("GET", self._fw_path)).get("version", "")
        except (NotFound, AuthError):
            fw = ""
        return DeviceInfo.from_api(gestalt, fw, self.api)

    async def get_mode(self) -> str:
        """Current LED mode, or "" when the firmware doesn't expose it."""
        try:
            reply = await self._request("GET", self._mode_path)
        except NotFound:
            return ""
        return str(reply.get("mode", ""))

    async def set_mode(self, mode: str) -> None:
        try:
            reply = await self._request("POST", self._mode_path, {"mode": mode})
        except TwinklyError as exc:
            if "HTTP 403" in str(exc):
                raise TwinklyError(
                    f"set mode {mode!r} refused (403). On Pro firmware this means the controller is a group "
                    "slave; take it out of its group in the Twinkly Pro app"
                ) from None
            raise
        if result_code(reply) != CODE_OK:
            raise TwinklyError(f"set mode {mode!r} refused: {reply}")

    async def artnet_config(self) -> dict:
        """The built-in Art-Net mode's settings on Pro firmware: enable, model, port_address, dmx_address.

        Consumer firmware has no such endpoint (NotFound)."""
        reply = await self._request("GET", "/artnet/config")
        return {k: v for k, v in reply.items() if k not in ("code", "result")}

    async def set_artnet(self, enable: bool) -> dict:
        """Switch the built-in Art-Net mode on or off, keeping its other settings; returns the new settings.

        The bridge needs it off: a controller in Art-Net mode fights the realtime stream."""
        current = await self.artnet_config()
        body = {**current, "enable": bool(enable)}
        reply = await self._request("POST", "/artnet/config", body)
        if result_code(reply) != CODE_OK:
            raise TwinklyError(f"artnet/config refused: {reply}")
        after = await self.artnet_config()
        if bool(after.get("enable")) != bool(enable):
            state = "on" if after.get("enable") else "off"
            raise TwinklyError(f"the controller accepted the change but still reports built-in Art-Net {state}")
        return after

    async def led_config(self) -> list[dict]:
        try:
            reply = await self._request("GET", "/led/config")
        except NotFound:
            return []
        strings = reply.get("strings")
        return strings if isinstance(strings, list) else []


# --------------------------------------------------------------------------
# Discovery (UDP 5555)
# --------------------------------------------------------------------------

DISCOVERY_REQUEST = b"\x01discover"


@dataclass
class Discovered:
    ip: str
    name: str
    http_port: int = HTTP_PORT


def parse_discovery_reply(data: bytes, sender_ip: str) -> Discovered | None:
    """Reply: 4 bytes IP, 2 status bytes, device name, NUL.

    Consumer devices answer b"OK"; Pro controllers answer b"TP". The docs say
    the IP is byte-reversed, but a Pro sends it in network order, so the UDP
    sender address is used when we have it.
    """
    if len(data) < 7 or data[:9] == DISCOVERY_REQUEST or not data[4:6].isalpha():
        return None
    name = data[6:].split(b"\x00", 1)[0].decode("utf-8", "replace")
    ip = sender_ip or socket.inet_ntoa(data[0:4])
    return Discovered(ip=ip, name=name)


def discover(
    broadcast: str | Sequence[str] = "255.255.255.255", timeout: float = 2.0, bind_ip: str = ""
) -> list[Discovered]:
    """Broadcast a discovery request (to one or more addresses) and collect replies (blocking)."""
    targets = [broadcast] if isinstance(broadcast, str) else list(dict.fromkeys(broadcast))
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.bind((bind_ip, 0))
    sock.settimeout(0.2)
    found: dict[str, Discovered] = {}
    try:
        deadline = time.monotonic() + timeout
        next_send = 0.0
        while time.monotonic() < deadline:
            if time.monotonic() >= next_send:
                for target in targets:
                    try:
                        sock.sendto(DISCOVERY_REQUEST, (target, DISCOVERY_PORT))
                    except OSError:
                        continue  # no route to that subnet; the others may still answer
                next_send = time.monotonic() + 0.5
            try:
                data, addr = sock.recvfrom(512)
            except TimeoutError:
                continue
            except OSError:
                continue
            item = parse_discovery_reply(data, addr[0])
            if item is not None:
                found[item.ip] = item
    finally:
        sock.close()
    return sorted(found.values(), key=lambda d: socket.inet_aton(d.ip))


def local_networks() -> list[dict]:
    """This box's IPv4 interfaces: [{"name", "ip", "broadcast"}], without loopback (Linux)."""
    out: list[dict] = []
    try:
        import fcntl
        import struct
    except ImportError:
        return out
    try:
        names = sorted(os.listdir("/sys/class/net"))  # no netlink needed (systemd sandboxes block it)
    except OSError:
        try:
            names = [name for _, name in socket.if_nameindex()]
        except (OSError, AttributeError):
            return out
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        for name in names:
            packed = struct.pack("256s", name.encode()[:15])
            try:
                ip = socket.inet_ntoa(fcntl.ioctl(sock.fileno(), 0x8915, packed)[20:24])  # SIOCGIFADDR
                bcast = socket.inet_ntoa(fcntl.ioctl(sock.fileno(), 0x8919, packed)[20:24])  # SIOCGIFBRDADDR
            except OSError:
                continue
            if ip.startswith("127.") or bcast in ("0.0.0.0", ip):
                continue
            out.append({"name": name, "ip": ip, "broadcast": bcast})
    finally:
        sock.close()
    return out


__all__ = [
    "AuthError",
    "DeviceInfo",
    "Discovered",
    "TwinklyClient",
    "TwinklyError",
    "build_packets",
    "default_order",
    "discover",
    "local_networks",
    "parse_discovery_reply",
    "protocol_for_firmware",
]
