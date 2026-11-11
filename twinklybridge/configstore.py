"""The config file as the setup page edits it: load, change, validate, write with a backup.

The page works on the raw YAML mapping (what the file says), never on the
parsed dataclasses, so keys the page does not know about survive a save.
Every change is validated with parse_config before anything is written.
"""

from __future__ import annotations

import glob
import math
import os
import time
from typing import Any

import yaml

from . import artnet, sacn
from .config import (
    ARTNET,
    DEFAULT_KEYS,
    DEVICE_KEYS,
    DMX_SLOTS,
    MAPPING_KEYS,
    SACN,
    Config,
    ConfigError,
    format_universe,
    overlaps,
    parse_config,
)

INPUT_KEYS = ("cidr", "ip", "broadcast", "artnet", "sacn", "reply", "short_name", "long_name")
WEB_KEYS = ("enabled", "bind", "port")
DEVICE_ORDER = (
    "name",
    "ip",
    "universe",
    "pixels",
    "protocol",
    "input",
    "order",
    "start_channel",
    "pixels_per_universe",
    "fps",
    "on_loss",
    "loss_timeout",
    "enabled",
    "http_port",
    "rt_port",
)
DEFAULTS_ORDER = tuple(k for k in DEVICE_ORDER if k in DEFAULT_KEYS)
MAPPING_ORDER = (
    "dir",
    "keep_clips",
    "min_free_mb",
    "max_upload_mb",
    "slot",
    "brightness",
    "color",
    "passes",
    "decode_width",
)
SECTIONS = {
    "input": INPUT_KEYS,
    "defaults": DEFAULTS_ORDER,
    "web": WEB_KEYS,
    "mapping": MAPPING_ORDER,
}

_INT_KEYS = {
    "pixels",
    "start_channel",
    "pixels_per_universe",
    "http_port",
    "rt_port",
    "port",
    "min_free_mb",
    "max_upload_mb",
    "brightness",
    "passes",
    "decode_width",
}
_FLOAT_KEYS = {"fps", "loss_timeout", "slot"}
_BOOL_KEYS = {"enabled", "artnet", "sacn", "keep_clips"}
_LAYOUT_KEYS = {"input", "order"}
_TRUE = ("true", "yes", "on", "1")
_FALSE = ("false", "no", "off", "0")

BACKUPS = 5


# --------------------------------------------------------------------------
# Reading and writing the file
# --------------------------------------------------------------------------


def raw_from_text(text: str) -> dict:
    """The YAML mapping in the file, or a ConfigError that names the line."""
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        where = f" (line {mark.line + 1})" if mark is not None else ""
        raise ConfigError(f"YAML syntax error{where}: {getattr(exc, 'problem', exc)}") from None
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError("the config file must be a mapping of keys and values")
    return raw


def load_raw(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise ConfigError(f"cannot read {path}: {exc.strerror}") from None
    return raw_from_text(text)


def write_problem(path: str) -> str:
    """Why the bridge could not save the file, or "" when it can."""
    directory = os.path.dirname(os.path.abspath(path)) or "."
    if os.path.exists(path) and not os.access(path, os.W_OK):
        return f"{path} is not writable by the bridge's user"
    if not os.access(directory, os.W_OK):
        return f"the folder {directory} is not writable by the bridge's user (needed for the backup copy)"
    return ""


def backups_of(path: str) -> list[str]:
    """Previous versions of the file, newest first."""
    found = glob.glob(glob.escape(path) + ".*.bak")
    return sorted(found, key=lambda f: (os.path.getmtime(f), f), reverse=True)


def write_text(path: str, text: str, keep: int = BACKUPS) -> str:
    """Replace the file atomically, keeping the previous version as <path>.<time>.bak.

    Returns the backup's path ("" when there was no file to back up).
    """
    backup = ""
    mode = 0o640
    if os.path.exists(path):
        mode = os.stat(path).st_mode & 0o777
        stamp = time.strftime("%Y%m%d-%H%M%S")
        backup = f"{path}.{stamp}.bak"
        n = 1
        while os.path.exists(backup):  # several saves in one second
            n += 1
            backup = f"{path}.{stamp}-{n}.bak"
        with open(path, "rb") as src, open(backup, "wb") as dst:
            dst.write(src.read())
        os.chmod(backup, mode)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.chmod(tmp, mode)
    os.replace(tmp, path)
    for old in backups_of(path)[keep:]:
        try:
            os.remove(old)
        except OSError:
            pass
    return backup


# --------------------------------------------------------------------------
# Writing YAML the way the example config looks
# --------------------------------------------------------------------------


def _scalar(value: Any) -> str:
    """One YAML scalar, quoted only when the plain form would read back as something else."""
    text = yaml.safe_dump(value, default_flow_style=True, allow_unicode=True, width=10**6).strip()
    if text.endswith("\n..."):
        text = text[:-4].strip()
    return text


def _block(mapping: dict, order: tuple[str, ...], indent: str) -> list[str]:
    keys = [k for k in order if k in mapping] + [k for k in mapping if k not in order]
    lines = []
    for key in keys:
        value = mapping[key]
        if isinstance(value, (dict, list)):
            dumped = yaml.safe_dump({key: value}, default_flow_style=False, sort_keys=False, allow_unicode=True)
            lines.extend(indent + line for line in dumped.rstrip("\n").split("\n"))
        else:
            lines.append(f"{indent}{key}: {_scalar(value)}")
    return lines


def _section(name: str, mapping: dict, order: tuple[str, ...]) -> str:
    if not mapping:
        return f"{name}: {{}}\n"
    return f"{name}:\n" + "\n".join(_block(mapping, order, "  ")) + "\n"


def dump(raw: dict) -> str:
    """The whole file in a fixed, commented layout (see example-config.yaml)."""
    text = (
        "# Twinkly Bridge configuration\n"
        f"# Written by the setup page on {time.strftime('%Y-%m-%d %H:%M:%S')}. Editing by hand is fine too;\n"
        "# the next save from the page rewrites the file in this layout. `twinkly-bridge example`\n"
        "# prints the annotated example with every key explained.\n"
        "\n"
        "# The console side: this box's IP on the Art-Net / sACN network and what it listens for.\n"
    )
    text += _section("input", dict(raw.get("input") or {}), INPUT_KEYS)
    text += "\n# Applied to every device unless the device says otherwise.\n"
    text += _section("defaults", dict(raw.get("defaults") or {}), DEFAULTS_ORDER)
    text += (
        "\n# One entry per Twinkly controller. Art-Net universes are net.subnet.universe (0.0.0 is\n"
        "# the first one); sACN universes are numbers from 1. Pixels fill the following universes.\n"
    )
    devices = raw.get("devices") or []
    if not devices:
        text += "devices: []\n\n"
    else:
        text += "devices:\n"
        for dev in devices:
            lines = _block(dict(dev), DEVICE_ORDER, "")
            text += "  - " + lines[0] + "\n" + "".join("    " + line + "\n" for line in lines[1:]) + "\n"
    text += "# The status / setup / mapping pages. Changing bind or port takes effect after a restart.\n"
    text += _section("web", dict(raw.get("web") or {}), WEB_KEYS)
    mapping = dict(raw.get("mapping") or {})
    if mapping:
        text += "\n# Camera pixel mapping (the /map page).\n" + _section("mapping", mapping, MAPPING_ORDER)
    text += f"\nlog_level: {_scalar(raw.get('log_level') or 'info')}\n"
    for key, value in raw.items():
        if key not in ("input", "defaults", "devices", "web", "mapping", "log_level"):
            text += "\n" + "\n".join(_block({key: value}, (), "")) + "\n"
    return text


# --------------------------------------------------------------------------
# Changing the raw mapping
# --------------------------------------------------------------------------


def _coerce(section: str, spec: dict, allowed: tuple[str, ...] | set[str]) -> dict:
    """Form values (mostly strings) into the types the YAML should carry; empty values are dropped."""
    if not isinstance(spec, dict):
        raise ConfigError(f"{section}: expected a mapping of keys and values")
    out: dict[str, Any] = {}
    for key, value in spec.items():
        key = str(key)
        if key not in allowed:
            raise ConfigError(f"{section}: unknown key '{key}'")
        if isinstance(value, str):
            value = value.strip()
        if value is None or value == "":
            continue
        where = f"{section}.{key}"
        if key in _BOOL_KEYS or (section == "web" and key == "enabled"):
            if isinstance(value, str):
                if value.lower() in _TRUE:
                    value = True
                elif value.lower() in _FALSE:
                    value = False
                else:
                    raise ConfigError(f"{where}: expected true or false, got {value!r}")
            elif not isinstance(value, bool):
                raise ConfigError(f"{where}: expected true or false, got {value!r}")
        elif key in _INT_KEYS:
            try:
                number = float(value)
            except (TypeError, ValueError):
                raise ConfigError(f"{where}: expected a whole number, got {value!r}") from None
            if isinstance(value, bool) or not number.is_integer():
                raise ConfigError(f"{where}: expected a whole number, got {value!r}")
            value = int(number)
        elif key in _FLOAT_KEYS:
            try:
                number = float(value)
            except (TypeError, ValueError):
                raise ConfigError(f"{where}: expected a number, got {value!r}") from None
            value = int(number) if number.is_integer() else number
        elif key in _LAYOUT_KEYS:
            value = "auto" if str(value).lower() == "auto" else str(value).upper()
        elif key == "universe":
            pass  # canonical form depends on the protocol; done in coerce_device
        else:
            value = str(value)
        out[key] = value
    return out


def _default_protocol(raw: dict) -> str:
    defaults = raw.get("defaults") if isinstance(raw.get("defaults"), dict) else {}
    inp = raw.get("input") if isinstance(raw.get("input"), dict) else {}
    if defaults.get("protocol"):
        return str(defaults["protocol"]).strip().lower()
    artnet_on = inp.get("artnet", True)
    if isinstance(artnet_on, str):
        artnet_on = artnet_on.strip().lower() in _TRUE
    return ARTNET if artnet_on else SACN


def canonical_universe(value: Any, protocol: str) -> str | int:
    """Art-Net universes are stored as net.subnet.universe text, sACN ones as numbers."""
    try:
        if protocol == SACN:
            return sacn.parse_universe(value)
        return artnet.format_universe(artnet.parse_universe(value))
    except ValueError as exc:
        raise ConfigError(str(exc)) from None


def coerce_device(spec: dict, raw: dict) -> dict:
    dev = _coerce("device", spec, DEVICE_KEYS)
    protocol = dev.get("protocol") or _default_protocol(raw)
    if "universe" in dev:
        label = f"device {dev.get('name') or dev.get('ip') or ''}".strip()
        try:
            dev["universe"] = canonical_universe(dev["universe"], protocol)
        except ConfigError as exc:
            raise ConfigError(f"{label}.universe: {exc}") from None
    return {k: dev[k] for k in DEVICE_ORDER if k in dev}


def _devices(raw: dict) -> list:
    devices = raw.get("devices")
    if devices is None:
        devices = raw["devices"] = []
    if not isinstance(devices, list):
        raise ConfigError("devices: expected a list")
    return devices


def _find(raw: dict, name: str) -> int:
    for index, dev in enumerate(_devices(raw)):
        if isinstance(dev, dict) and str(dev.get("name", "")).strip() == name:
            return index
    raise ConfigError(f"no device named {name!r} in the config")


def _span(dev: dict, raw: dict) -> int:
    """How many universes a device spec takes (its own keys, then defaults)."""
    defaults = raw.get("defaults") if isinstance(raw.get("defaults"), dict) else {}
    layout = str(dev.get("input") or defaults.get("input") or "RGBW")
    channels = max(1, len(layout))
    per = int(dev.get("pixels_per_universe") or defaults.get("pixels_per_universe") or 0) or DMX_SLOTS // channels
    start = int(dev.get("start_channel") or defaults.get("start_channel") or 1)
    pixels = int(dev.get("pixels") or 1)
    first_fit = max(1, min(per, (DMX_SLOTS - (start - 1)) // channels))
    if pixels <= first_fit:
        return 1
    return 1 + math.ceil((pixels - first_fit) / per)


def next_universe(raw: dict, protocol: str, skip: int | None = None) -> str | int:
    """The first universe after everything configured for that protocol."""
    last = -1 if protocol == ARTNET else 0
    for index, dev in enumerate(_devices(raw)):
        if index == skip or not isinstance(dev, dict) or dev.get("universe") in (None, ""):
            continue
        if (dev.get("protocol") or _default_protocol(raw)) != protocol:
            continue
        try:
            first = (sacn if protocol == SACN else artnet).parse_universe(dev["universe"])
            last = max(last, first + _span(dev, raw) - 1)
        except (ValueError, TypeError):
            continue
    return canonical_universe(last + 1, protocol)


def add_devices(raw: dict, specs: list[dict]) -> list[dict]:
    """Append devices; one without a universe gets the next free one. Validates the result."""
    added = []
    for spec in specs:
        dev = coerce_device(spec, raw)
        if "universe" not in dev:
            dev["universe"] = next_universe(raw, dev.get("protocol") or _default_protocol(raw))
            dev = {k: dev[k] for k in DEVICE_ORDER if k in dev}
        _devices(raw).append(dev)
        added.append(dev)
    parse_config(raw)
    return added


def replace_device(raw: dict, name: str, spec: dict) -> dict:
    """Swap the device called name for spec (same position). Validates the result."""
    index = _find(raw, name)
    dev = coerce_device(spec, raw)
    if "universe" not in dev:
        dev["universe"] = next_universe(raw, dev.get("protocol") or _default_protocol(raw), skip=index)
        dev = {k: dev[k] for k in DEVICE_ORDER if k in dev}
    _devices(raw)[index] = dev
    parse_config(raw)
    return dev


def patch_device(raw: dict, name: str, changes: dict) -> dict:
    """Change some keys of a device (None or "" removes a key). Validates the result."""
    index = _find(raw, name)
    current = dict(_devices(raw)[index])
    for key in changes:
        if str(key) not in DEVICE_KEYS:
            raise ConfigError(f"device: unknown key '{key}'")
    for key, value in changes.items():
        if value is None or (isinstance(value, str) and not value.strip()):
            current.pop(key, None)
        else:
            current[key] = value
    dev = coerce_device(current, raw)
    _devices(raw)[index] = dev
    parse_config(raw)
    return dev


def remove_device(raw: dict, name: str) -> None:
    index = _find(raw, name)
    del _devices(raw)[index]
    parse_config(raw)


def set_sections(raw: dict, changes: dict) -> None:
    """Replace whole sections (input, defaults, web, mapping) and/or log_level. Validates the result."""
    for section, values in changes.items():
        if section == "log_level":
            if values in (None, ""):
                raw.pop("log_level", None)
            else:
                raw["log_level"] = str(values).strip().lower()
            continue
        if section not in SECTIONS:
            raise ConfigError(f"config: unknown section '{section}'")
        cleaned = _coerce(section, values or {}, SECTIONS[section])
        if section == "input" and "cidr" in cleaned:
            cleaned.pop("ip", None)
        if cleaned or section == "input":
            raw[section] = cleaned
        else:
            raw.pop(section, None)
    parse_config(raw)


# --------------------------------------------------------------------------
# What the page shows
# --------------------------------------------------------------------------


def describe(raw: dict, path: str) -> dict:
    """The raw sections plus what they mean, for the setup page."""
    error = ""
    config: Config | None = None
    try:
        config = parse_config(raw)
    except ConfigError as exc:
        error = str(exc)
    devices = []
    raw_devices = raw.get("devices") if isinstance(raw.get("devices"), list) else []
    for index, spec in enumerate(raw_devices):
        entry: dict[str, Any] = {"spec": dict(spec) if isinstance(spec, dict) else {}, "derived": None}
        if config is not None and index < len(config.devices):
            d = config.devices[index]
            entry["derived"] = {
                "name": d.name,
                "protocol": d.protocol,
                "universes": d.universe_label,
                "universe_count": len(d.segments),
                "first_universe": format_universe(d.protocol, d.universes[0]),
                "last_universe": format_universe(d.protocol, d.universes[-1]),
                "pixels": d.pixels,
                "input": d.input,
                "order": d.order,
                "fps": d.fps,
                "on_loss": d.on_loss,
                "loss_timeout": d.loss_timeout,
                "start_channel": d.start_channel,
                "pixels_per_universe": d.pixels_per_universe or d.per_universe,
                "enabled": d.enabled,
            }
        devices.append(entry)
    next_free = {}
    for protocol in (ARTNET, SACN):
        try:
            next_free[protocol] = next_universe(raw, protocol)
        except ConfigError:
            next_free[protocol] = "0.0.0" if protocol == ARTNET else 1
    return {
        "path": path,
        "write_problem": write_problem(path),
        "error": error,
        "input": dict(raw.get("input") or {}) if isinstance(raw.get("input"), dict) else {},
        "input_effective": (
            {"ip": config.input.ip, "prefix": config.input.prefix, "broadcast": config.input.broadcast}
            if config
            else None
        ),
        "defaults": dict(raw.get("defaults") or {}) if isinstance(raw.get("defaults"), dict) else {},
        "devices": devices,
        "web": dict(raw.get("web") or {}) if isinstance(raw.get("web"), dict) else {},
        "mapping": dict(raw.get("mapping") or {}) if isinstance(raw.get("mapping"), dict) else {},
        "log_level": raw.get("log_level") or "info",
        "warnings": overlaps(config) if config else [],
        "next_universe": next_free,
        "backups": [os.path.basename(b) for b in backups_of(path)],
    }


__all__ = [
    "BACKUPS",
    "DEVICE_ORDER",
    "MAPPING_KEYS",
    "add_devices",
    "backups_of",
    "canonical_universe",
    "coerce_device",
    "describe",
    "dump",
    "load_raw",
    "next_universe",
    "patch_device",
    "raw_from_text",
    "remove_device",
    "replace_device",
    "set_sections",
    "write_problem",
    "write_text",
]
