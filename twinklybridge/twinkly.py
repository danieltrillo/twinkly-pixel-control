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
    """UDP realtime protocol version: v3 (fragmented) for 2.4.14 and newer."""
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

    @property
    def bytes_per_led(self) -> int:
        return 4 if self.led_profile.upper() == "RGBW" else 3

    @classmethod
    def from_api(cls, gestalt: dict[str, Any], firmware: str) -> DeviceInfo:
        def num(key: str, default: float = 0) -> float:
            try:
                return float(gestalt.get(key, default))
            except (TypeError, ValueError):
                return default

        measured = gestalt.get("measured_frame_rate")
        return cls(
            firmware=str(firmware or ""),
            product_code=str(gestalt.get("product_code", "")),
            device_name=str(gestalt.get("device_name", "")),
            led_profile=str(gestalt.get("led_profile", "RGB")).upper(),
            number_of_led=int(num("number_of_led")),
            max_supported_led=int(num("max_supported_led")),
            frame_rate=num("frame_rate"),
            measured_frame_rate=float(measured) if isinstance(measured, (int, float)) else None,
            mac=str(gestalt.get("mac", "")),
            fw_family=str(gestalt.get("fw_family", "")),
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
        }


# --------------------------------------------------------------------------
# Async HTTP client (aiohttp)
# --------------------------------------------------------------------------


class TwinklyClient:
    """Login and mode control for one device. Uses an aiohttp ClientSession."""

    def __init__(self, session: Any, ip: str, *, port: int = HTTP_PORT, timeout: float = 3.0) -> None:
        self.session = session
        self.ip = ip
        self.base = f"http://{ip}:{port}/xled/v1"
        self.timeout = timeout
        self.token_b64 = ""
        self.token = b""

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
        if verify.get("code") != CODE_OK:
            raise TwinklyError(f"verify refused: {verify}")

    async def info(self) -> DeviceInfo:
        fw = await self._request("GET", "/fw/version", auth=False)
        gestalt = await self._request("GET", "/gestalt", auth=False)
        return DeviceInfo.from_api(gestalt, fw.get("version", ""))

    async def get_mode(self) -> str:
        reply = await self._request("GET", "/led/mode")
        return str(reply.get("mode", ""))

    async def set_mode(self, mode: str) -> None:
        reply = await self._request("POST", "/led/mode", {"mode": mode})
        if reply.get("code") != CODE_OK:
            raise TwinklyError(f"set mode {mode!r} refused: {reply}")

    async def led_config(self) -> list[dict]:
        reply = await self._request("GET", "/led/config")
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


def parse_discovery_reply(data: bytes, sender_ip: str) -> Discovered | None:
    """Reply: 4 bytes IP (reversed), b"OK", device name, NUL."""
    if len(data) < 6 or data[4:6] != b"OK":
        return None
    name = data[6:].split(b"\x00", 1)[0].decode("utf-8", "replace")
    ip = socket.inet_ntoa(bytes(reversed(data[0:4])))
    if ip == "0.0.0.0":
        ip = sender_ip
    return Discovered(ip=ip, name=name)


def discover(broadcast: str = "255.255.255.255", timeout: float = 2.0, bind_ip: str = "") -> list[Discovered]:
    """Broadcast a discovery request and collect replies (blocking)."""
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
                sock.sendto(DISCOVERY_REQUEST, (broadcast, DISCOVERY_PORT))
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


def pack_ip_reversed(ip: str) -> bytes:
    """Helper for tests and fakes: the IP field of a discovery reply."""
    return bytes(reversed(socket.inet_aton(ip)))


__all__ = [
    "AuthError",
    "DeviceInfo",
    "Discovered",
    "TwinklyClient",
    "TwinklyError",
    "build_packets",
    "default_order",
    "discover",
    "parse_discovery_reply",
    "protocol_for_firmware",
]
