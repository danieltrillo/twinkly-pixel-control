"""Configuration: one YAML file. Every error names the exact key that is wrong."""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass, field
from typing import Any

import yaml

from . import artnet, sacn
from .twinkly import HTTP_PORT, RT_PORT, valid_order

DEFAULT_CONFIG_PATH = "/etc/twinkly-bridge/config.yaml"

ARTNET = "artnet"
SACN = "sacn"
PROTOCOLS = (ARTNET, SACN)
ON_LOSS = ("hold", "black", "release")
LOG_LEVELS = ("debug", "info", "warning", "error")
DMX_SLOTS = 512


class ConfigError(Exception):
    """Raised for an invalid configuration. The message is user facing."""


# --------------------------------------------------------------------------
# Data classes
# --------------------------------------------------------------------------


@dataclass
class InputConfig:
    ip: str = "0.0.0.0"  # this box's IP on the console network ("" / 0.0.0.0 = any)
    prefix: int = 0
    broadcast: str = "255.255.255.255"
    artnet: bool = True
    sacn: bool = False
    reply: str = "broadcast"  # how ArtPollReply is sent: broadcast | unicast
    short_name: str = "TwinklyBridge"
    long_name: str = ""

    @property
    def cidr(self) -> str:
        return f"{self.ip}/{self.prefix}"


@dataclass(frozen=True)
class Segment:
    """Part of a device's pixels carried by one universe."""

    universe: int
    channel: int  # 0-based slot of the first pixel in that universe
    first_pixel: int
    count: int


@dataclass
class DeviceConfig:
    name: str
    ip: str
    universe: int
    pixels: int
    protocol: str = ARTNET
    start_channel: int = 1
    input: str = "RGBW"  # channel layout of one pixel as the console sends it
    pixels_per_universe: int = 0  # 0 = as many as fit (128 for RGBW, 170 for RGB)
    order: str = "auto"  # byte order on the wire to the Twinkly; auto = from the device
    fps: float = 25.0
    on_loss: str = "hold"
    loss_timeout: float = 5.0
    enabled: bool = True
    http_port: int = HTTP_PORT
    rt_port: int = RT_PORT

    @property
    def channels(self) -> int:
        return len(self.input)

    @property
    def per_universe(self) -> int:
        return self.pixels_per_universe or DMX_SLOTS // self.channels

    @property
    def segments(self) -> list[Segment]:
        return build_segments(self.universe, self.pixels, self.channels, self.per_universe, self.start_channel)

    @property
    def universes(self) -> list[int]:
        return [s.universe for s in self.segments]

    @property
    def universe_label(self) -> str:
        us = self.universes
        first, last = format_universe(self.protocol, us[0]), format_universe(self.protocol, us[-1])
        return first if first == last else f"{first} - {last}"


@dataclass
class WebConfig:
    enabled: bool = True
    bind: str = "0.0.0.0"
    port: int = 8080


MAP_COLORS = ("white", "rgb", "green")


@dataclass
class MappingConfig:
    """Camera pixel mapping from the web page (/map)."""

    dir: str = ""  # clips and results; "" = $STATE_DIRECTORY/mapping under systemd, else ./mapping-data
    keep_clips: bool = False  # keep uploaded videos after a good decode (results are tiny, clips are not)
    min_free_mb: int = 500  # refuse uploads when the disk has less than this left
    max_upload_mb: int = 4000
    slot: float = 0.4  # seconds per code slot
    brightness: int = 96  # 1-255 for the lit pixels
    color: str = "white"  # white (W chip) | rgb | green
    passes: int = 10  # how often the code repeats per scan (about 18 s each)
    decode_width: int = 1280  # decode videos at this width


@dataclass
class Config:
    input: InputConfig
    devices: list[DeviceConfig]
    web: WebConfig = field(default_factory=WebConfig)
    mapping: MappingConfig = field(default_factory=MappingConfig)
    log_level: str = "info"

    @property
    def enabled_devices(self) -> list[DeviceConfig]:
        return [d for d in self.devices if d.enabled]


def format_universe(protocol: str, universe: int) -> str:
    return artnet.format_universe(universe) if protocol == ARTNET else str(universe)


def build_segments(
    first_universe: int, pixels: int, channels: int, per_universe: int, start_channel: int
) -> list[Segment]:
    """Pixels fill universes in order and never straddle two universes.

    start_channel only shifts the first universe; the rest start at channel 1.
    """
    segments = []
    offset = start_channel - 1
    universe = first_universe
    pixel = 0
    while pixel < pixels:
        fit = min(per_universe, (DMX_SLOTS - offset) // channels)
        if fit <= 0:
            raise ValueError("no pixel fits in the first universe after start_channel")
        count = min(fit, pixels - pixel)
        segments.append(Segment(universe=universe, channel=offset, first_pixel=pixel, count=count))
        pixel += count
        universe += 1
        offset = 0
    return segments


# --------------------------------------------------------------------------
# Parsing helpers
# --------------------------------------------------------------------------


def _mapping(value: Any, path: str) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"{path}: expected a mapping of keys and values")
    return value


def _bool(value: Any, path: str, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "yes", "on", "1"):
        return True
    if isinstance(value, str) and value.strip().lower() in ("false", "no", "off", "0"):
        return False
    raise ConfigError(f"{path}: expected true or false, got {value!r}")


def _int(value: Any, path: str, default: int | None, lo: int, hi: int) -> int:
    if value is None:
        if default is None:
            raise ConfigError(f"{path}: is required")
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ConfigError(f"{path}: expected a number, got {value!r}")
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise ConfigError(f"{path}: expected a whole number, got {value!r}") from None
    if not lo <= number <= hi:
        raise ConfigError(f"{path}: {number} is out of range ({lo}-{hi})")
    return number


def _float(value: Any, path: str, default: float, lo: float, hi: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ConfigError(f"{path}: expected a number, got {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ConfigError(f"{path}: expected a number, got {value!r}") from None
    if not lo <= number <= hi:
        raise ConfigError(f"{path}: {number} is out of range ({lo}-{hi})")
    return number


def _str(value: Any, path: str, default: str = "") -> str:
    if value is None:
        return default
    if not isinstance(value, (str, int, float)):
        raise ConfigError(f"{path}: expected text, got {value!r}")
    return str(value)


def _choice(value: Any, path: str, choices: tuple[str, ...], default: str) -> str:
    if value is None:
        return default
    text = str(value).strip().lower()
    if text not in choices:
        raise ConfigError(f"{path}: {value!r} is not one of {', '.join(choices)}")
    return text


def _ipv4(value: Any, path: str) -> str:
    text = _str(value, path).strip()
    try:
        return str(ipaddress.IPv4Address(text))
    except ValueError:
        raise ConfigError(f"{path}: {value!r} is not an IPv4 address") from None


def _unknown_keys(mapping: dict, allowed: set[str], path: str) -> None:
    extra = [k for k in mapping if k not in allowed]
    if extra:
        raise ConfigError(f"{path}: unknown key '{extra[0]}' (allowed: {', '.join(sorted(allowed))})")


def _parse_universe(value: Any, protocol: str, path: str) -> int:
    if value is None:
        raise ConfigError(f"{path}: is required")
    try:
        if protocol == ARTNET:
            return artnet.parse_universe(value)
        return sacn.parse_universe(value)
    except ValueError as exc:
        raise ConfigError(f"{path}: {exc}") from None


def _layout(value: Any, path: str, default: str, allow_auto: bool) -> str:
    if value is None:
        return default
    text = _str(value, path).strip().upper()
    if allow_auto and text == "AUTO":
        return "auto"
    if not valid_order(text):
        what = "auto or " if allow_auto else ""
        raise ConfigError(f"{path}: {value!r} must be {what}a channel order like RGBW, WRGB, GRB or RGB")
    return text


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------


def _parse_input(raw: Any) -> InputConfig:
    spec = _mapping(raw, "input")
    _unknown_keys(spec, {"cidr", "ip", "broadcast", "artnet", "sacn", "reply", "short_name", "long_name"}, "input")
    if spec.get("cidr") is not None and spec.get("ip") is not None:
        raise ConfigError("input: use either 'cidr' or 'ip', not both")
    ip, prefix = "0.0.0.0", 0
    if spec.get("cidr") is not None:
        text = str(spec["cidr"]).strip()
        if "/" not in text:
            raise ConfigError(f"input.cidr: {text!r} needs a prefix length, like 2.0.0.20/8")
        try:
            iface = ipaddress.IPv4Interface(text)
        except ValueError:
            raise ConfigError(f"input.cidr: {text!r} is not like 2.0.0.20/8") from None
        ip, prefix = str(iface.ip), iface.network.prefixlen
    elif spec.get("ip") is not None:
        ip = _ipv4(spec["ip"], "input.ip")
    if spec.get("broadcast") is not None:
        broadcast = _ipv4(spec["broadcast"], "input.broadcast")
    elif prefix and prefix < 31:
        broadcast = str(ipaddress.IPv4Network(f"{ip}/{prefix}", strict=False).broadcast_address)
    else:
        broadcast = "255.255.255.255"
    cfg = InputConfig(
        ip=ip,
        prefix=prefix,
        broadcast=broadcast,
        artnet=_bool(spec.get("artnet"), "input.artnet", True),
        sacn=_bool(spec.get("sacn"), "input.sacn", False),
        reply=_choice(spec.get("reply"), "input.reply", ("broadcast", "unicast"), "broadcast"),
        short_name=_str(spec.get("short_name"), "input.short_name", "TwinklyBridge")[:17] or "TwinklyBridge",
        long_name=_str(spec.get("long_name"), "input.long_name", "")[:63],
    )
    if not cfg.artnet and not cfg.sacn:
        raise ConfigError("input: turn on at least one of artnet or sacn")
    return cfg


DEVICE_KEYS = {
    "name",
    "ip",
    "universe",
    "protocol",
    "pixels",
    "start_channel",
    "input",
    "pixels_per_universe",
    "order",
    "fps",
    "on_loss",
    "loss_timeout",
    "enabled",
    "http_port",
    "rt_port",
}
DEFAULT_KEYS = DEVICE_KEYS - {"name", "ip", "universe", "pixels", "enabled"}


def _parse_device(index: int, raw: Any, defaults: dict, inp: InputConfig) -> DeviceConfig:
    path = f"devices[{index}]"
    spec = _mapping(raw, path)
    _unknown_keys(spec, DEVICE_KEYS, path)
    merged = {**defaults, **{k: v for k, v in spec.items() if v is not None}}

    def where(key: str) -> str:
        return f"{path}.{key}" if key in spec else f"defaults.{key}"

    name = _str(spec.get("name"), f"{path}.name", f"Twinkly {index + 1}").strip() or f"Twinkly {index + 1}"
    path = f"devices[{index}] ({name})"
    if spec.get("ip") is None:
        raise ConfigError(f"{path}.ip: is required")
    ip = _ipv4(spec.get("ip"), f"{path}.ip")
    default_protocol = ARTNET if inp.artnet else SACN
    protocol = _choice(merged.get("protocol"), where("protocol"), PROTOCOLS, default_protocol)
    if protocol == ARTNET and not inp.artnet:
        raise ConfigError(f"{path}: uses Art-Net but input.artnet is off")
    if protocol == SACN and not inp.sacn:
        raise ConfigError(f"{path}: uses sACN but input.sacn is off")
    universe = _parse_universe(spec.get("universe"), protocol, f"{path}.universe")
    pixels = _int(spec.get("pixels"), f"{path}.pixels", None, 1, 20000)
    layout = _layout(merged.get("input"), where("input"), "RGBW", allow_auto=False)
    channels = len(layout)
    start = _int(merged.get("start_channel"), where("start_channel"), 1, 1, DMX_SLOTS - channels + 1)
    ppu = _int(merged.get("pixels_per_universe"), where("pixels_per_universe"), 0, 0, DMX_SLOTS // channels)
    order = _layout(merged.get("order"), where("order"), "auto", allow_auto=True)

    device = DeviceConfig(
        name=name,
        ip=ip,
        universe=universe,
        pixels=pixels,
        protocol=protocol,
        start_channel=start,
        input=layout,
        pixels_per_universe=ppu,
        order=order,
        fps=_float(merged.get("fps"), where("fps"), 25.0, 1, 60),
        on_loss=_choice(merged.get("on_loss"), where("on_loss"), ON_LOSS, "hold"),
        loss_timeout=_float(merged.get("loss_timeout"), where("loss_timeout"), 5.0, 0.5, 3600),
        enabled=_bool(spec.get("enabled"), f"{path}.enabled", True),
        http_port=_int(merged.get("http_port"), where("http_port"), HTTP_PORT, 1, 65535),
        rt_port=_int(merged.get("rt_port"), where("rt_port"), RT_PORT, 1, 65535),
    )
    last = device.universes[-1]
    limit = artnet.MAX_PORT_ADDRESS if protocol == ARTNET else sacn.MAX_UNIVERSE
    if last > limit:
        raise ConfigError(f"{path}: {pixels} pixels starting at {spec.get('universe')!r} run past the last universe")
    return device


MAPPING_KEYS = {
    "dir",
    "keep_clips",
    "min_free_mb",
    "max_upload_mb",
    "slot",
    "brightness",
    "color",
    "passes",
    "decode_width",
}


def parse_config(raw: Any) -> Config:
    top = _mapping(raw, "config")
    _unknown_keys(top, {"input", "defaults", "devices", "web", "mapping", "log_level"}, "config")
    inp = _parse_input(top.get("input"))

    defaults = _mapping(top.get("defaults"), "defaults")
    _unknown_keys(defaults, DEFAULT_KEYS, "defaults")

    raw_devices = top.get("devices")
    if raw_devices is None:
        raw_devices = []
    if not isinstance(raw_devices, list):
        raise ConfigError("devices: expected a list")
    devices = [_parse_device(i, d, defaults, inp) for i, d in enumerate(raw_devices)]

    seen_ips: dict[str, str] = {}
    seen_names: set[str] = set()
    for d in devices:
        if d.ip in seen_ips:
            raise ConfigError(f"devices ({d.name}): IP {d.ip} is already used by '{seen_ips[d.ip]}'")
        seen_ips[d.ip] = d.name
        if d.name in seen_names:
            raise ConfigError(f"devices: the name '{d.name}' is used twice")
        seen_names.add(d.name)

    web_raw = _mapping(top.get("web"), "web")
    _unknown_keys(web_raw, {"enabled", "bind", "port"}, "web")
    web = WebConfig(
        enabled=_bool(web_raw.get("enabled"), "web.enabled", True),
        bind=_str(web_raw.get("bind"), "web.bind", "0.0.0.0").strip() or "0.0.0.0",
        port=_int(web_raw.get("port"), "web.port", 8080, 1, 65535),
    )
    map_raw = _mapping(top.get("mapping"), "mapping")
    _unknown_keys(map_raw, MAPPING_KEYS, "mapping")
    mapping = MappingConfig(
        dir=_str(map_raw.get("dir"), "mapping.dir", "").strip(),
        keep_clips=_bool(map_raw.get("keep_clips"), "mapping.keep_clips", False),
        min_free_mb=_int(map_raw.get("min_free_mb"), "mapping.min_free_mb", 500, 0, 1_000_000),
        max_upload_mb=_int(map_raw.get("max_upload_mb"), "mapping.max_upload_mb", 4000, 1, 100_000),
        slot=_float(map_raw.get("slot"), "mapping.slot", 0.4, 0.1, 2.0),
        brightness=_int(map_raw.get("brightness"), "mapping.brightness", 96, 1, 255),
        color=_choice(map_raw.get("color"), "mapping.color", MAP_COLORS, "white"),
        passes=_int(map_raw.get("passes"), "mapping.passes", 10, 1, 100),
        decode_width=_int(map_raw.get("decode_width"), "mapping.decode_width", 1280, 320, 4096),
    )
    return Config(
        input=inp,
        devices=devices,
        web=web,
        mapping=mapping,
        log_level=_choice(top.get("log_level"), "log_level", LOG_LEVELS, "info"),
    )


def overlaps(config: Config) -> list[str]:
    """Human readable warnings for universes used by more than one device."""
    users: dict[tuple[str, int], list[str]] = {}
    for d in config.enabled_devices:
        for u in d.universes:
            users.setdefault((d.protocol, u), []).append(d.name)
    out = []
    for (protocol, u), names in sorted(users.items()):
        if len(names) > 1:
            out.append(f"{protocol} universe {format_universe(protocol, u)} feeds {', '.join(names)}")
    return out


def parse_yaml(text: str) -> Config:
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        where = f" (line {mark.line + 1})" if mark is not None else ""
        raise ConfigError(f"YAML syntax error{where}: {getattr(exc, 'problem', exc)}") from None
    if raw is None:
        raise ConfigError("the config file is empty")
    return parse_config(raw)


def load_config(path: str) -> Config:
    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    except OSError as exc:
        raise ConfigError(f"cannot read {path}: {exc.strerror}") from None
    return parse_yaml(text)
