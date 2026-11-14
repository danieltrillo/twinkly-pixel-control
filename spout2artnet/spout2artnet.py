"""spout2artnet: Resolume's composition over Spout in, the tree's pixels over Art-Net out.

Replaces Resolume Arena's Advanced Output (Lumiverses + thousands of fixture
slices) for the Twinkly tree. Arena only has to send its composition over
Spout; this app samples the colour under every mapped pixel and sends the
Art-Net itself, addressed exactly the way the Advanced Output preset would.

The pixel map is the same Resolume Advanced Output XML the bridge's /map page
exports (or any Arena 7 preset made of DMX fixture slices): every slice's
input rect, pixel grid, colour format and start channel, and every
lumiverse's Sub-Net / Universe / target IP.

    python spout2artnet.py info  --map tree.xml
    python spout2artnet.py run   --map tree.xml --target 1=10.0.0.1 --target 2=10.0.0.2
    python spout2artnet.py senders

Windows only for Spout (pip install SpoutGL numpy). `--source test` runs a
built-in pattern instead, on any OS, to check the tree without Arena.

Art-Net(TM) Designed by and Copyright Artistic Licence Holdings Ltd.
"""

from __future__ import annotations

import argparse
import ipaddress
import math
import socket
import struct
import sys
import time
import xml.etree.ElementTree as ET
import zlib
from dataclasses import dataclass, field

import numpy as np

__version__ = "0.1.0"

DMX = 512
ARTNET_PORT = 6454
_ARTDMX = struct.Struct("<8sHBBBBBBBB")
_COLS = {"r": 0, "g": 1, "b": 2, "w": 3}
DEFAULT_GAMMA = 2.5  # what an Arena 7.24.3 pixel fixture saves as its Gamma


# --------------------------------------------------------------------------
# The map: a Resolume Advanced Output preset
# --------------------------------------------------------------------------


@dataclass
class Slice:
    name: str
    start: int  # 1-based start channel inside its lumiverse
    corners: np.ndarray  # (4, 2) composition px: v0, v1, v2, v3 as Resolume stores them
    width: int  # fixture pixels across v0 -> v1
    height: int  # fixture pixels along v0 -> v3
    fmt: str  # colour format, e.g. "rgbw"
    gamma: float | None


@dataclass
class Lumiverse:
    number: int  # 1-based, in file order (what --target 1=... refers to)
    name: str
    subnet: int
    universe: int
    target: str  # TargetIP as saved: an IP, or TT_BROADCAST / TT_DISABLED
    enabled: bool
    slices: list[Slice] = field(default_factory=list)

    @property
    def base(self) -> int:
        return (self.subnet << 4) | self.universe


@dataclass
class Rig:
    comp_w: int
    comp_h: int
    lumiverses: list[Lumiverse]

    @property
    def pixels(self) -> int:
        return sum(s.width * s.height for lu in self.lumiverses for s in lu.slices)


def _named(el: ET.Element, name: str) -> ET.Element | None:
    for e in el.iter():
        if e.get("name") == name and e.tag in ("Param", "ParamRange", "ParamChoice"):
            return e
    return None


def _value(el: ET.Element, name: str, default=None):
    e = _named(el, name)
    return default if e is None else e.get("value", default)


def _num(text, default: float) -> float:
    try:
        return float(text)
    except (TypeError, ValueError):
        return default


def load_resolume(path: str) -> Rig:
    """Read every DMX lumiverse and fixture slice from an Arena 7 Advanced Output preset."""
    root = ET.parse(path).getroot()
    size = root.find(".//CurrentCompositionTextureSize")
    comp_w = int(_num(size.get("width") if size is not None else None, 1920))
    comp_h = int(_num(size.get("height") if size is not None else None, 1080))
    lumis = []
    for n, screen in enumerate(root.iter("DmxScreen"), start=1):
        params = next((p for p in screen.findall("Params") if p.get("name") == "Params"), None)
        enabled = True if params is None else _value(params, "Enabled", "1") != "0"
        out = screen.find(".//DmxOutputParams")
        lu = Lumiverse(
            number=n,
            name=screen.get("name") or f"Lumiverse {n}",
            subnet=int(_num(_value(out, "Subnet") if out is not None else None, 0)),
            universe=int(_num(_value(out, "Universe") if out is not None else None, 0)),
            target=(_value(out, "TargetIP", "") if out is not None else "") or "",
            enabled=enabled,
        )
        for sl in screen.iter("DmxSlice"):
            common = next((p for p in sl.findall("Params") if p.get("name") == "Common"), sl)
            if _value(common, "Enabled", "1") == "0":
                continue
            rect = sl.find("InputRect")
            if rect is None:
                continue
            corners = np.array([[_num(v.get("x"), 0), _num(v.get("y"), 0)] for v in rect.findall("v")], np.float64)
            if corners.shape != (4, 2):
                continue
            pix = sl.find(".//ParamFixturePixels")
            pix = pix if pix is not None else sl
            gamma = _value(pix, "Gamma")
            lu.slices.append(
                Slice(
                    name=_value(common, "Name", "") or "",
                    start=int(_num(_value(sl, "Start Channel"), 1)),
                    corners=corners,
                    width=max(1, int(_num(_value(pix, "Width"), 1))),
                    height=max(1, int(_num(_value(pix, "Height"), 1))),
                    fmt=(_value(pix, "Color Format", "rgb") or "rgb").lower(),
                    gamma=None if gamma is None else _num(gamma, DEFAULT_GAMMA),
                )
            )
        lumis.append(lu)
    if not lumis:
        raise SystemExit(f"{path}: no DMX lumiverses in this file (is it an Advanced Output preset?)")
    return Rig(comp_w, comp_h, lumis)


def cell_centres(s: Slice) -> np.ndarray:
    """Centre of every fixture pixel in DMX order, in composition px.

    Pixel 0 sits in the v0 corner; Width runs v0 -> v1, Height runs v0 -> v3,
    row by row. A 1xN line slice is one column from the v0/v1 end to the v2/v3 end.
    """
    v0, v1, _, v3 = s.corners
    i = (np.arange(s.width) + 0.5) / s.width
    j = (np.arange(s.height) + 0.5) / s.height
    jj, ii = np.meshgrid(j, i, indexing="ij")  # row-major: j outer, i inner
    pts = v0 + ii.reshape(-1, 1) * (v1 - v0) + jj.reshape(-1, 1) * (v3 - v0)
    return pts


# --------------------------------------------------------------------------
# Plan: where to sample and where each colour goes, for one input size
# --------------------------------------------------------------------------


@dataclass
class Group:
    """Pixels of one lumiverse that share a colour format."""

    lumi: int  # index into Plan.outputs
    iy: np.ndarray  # (P,) or (P, K) sample rows
    ix: np.ndarray
    chan: np.ndarray  # (P, C) flat channel index in the lumiverse buffer
    cols: np.ndarray  # (C,) which of R, G, B, W each channel carries
    names: list[str] = field(default_factory=list)


@dataclass
class Output:
    lumiverse: Lumiverse
    ip: str
    net: int
    buffer: np.ndarray  # flat DMX buffer, a whole number of universes
    used: list[int]  # bytes to send per universe (even, >= 2)
    seq: list[int]

    def port_address(self, u: int) -> int:
        return ((self.net << 8) + self.lumiverse.base + u) & 0x7FFF


def resolve_targets(rig: Rig, targets: dict[str, str], nets: dict[str, int]) -> list[tuple[Lumiverse, str, int]]:
    """(lumiverse, ip, net) for every lumiverse that will send."""
    out = []
    missing = []
    for lu in rig.lumiverses:
        if not lu.slices:
            continue
        key = str(lu.number)
        ip = targets.get(key) or targets.get("all") or ""
        if not ip:
            try:
                ipaddress.IPv4Address(lu.target)
                ip = lu.target
            except ValueError:
                missing.append(lu)
                continue
        if ip.lower() == "off":
            continue
        out.append((lu, ip, nets.get(key, nets.get("all", 0))))
    if missing:
        names = ", ".join(f"{lu.number} ({lu.name}, TargetIP {lu.target or 'unset'})" for lu in missing)
        raise SystemExit(f"no destination IP for lumiverse {names}. Add --target N=IP (or --target N=off to skip it).")
    if not out:
        raise SystemExit("nothing to send: every lumiverse is empty or switched off")
    for i, (a, ip_a, net_a) in enumerate(out):
        for b, ip_b, net_b in out[i + 1 :]:
            if ip_a == ip_b and (net_a << 8) + a.base == (net_b << 8) + b.base:
                print(
                    f"warning: lumiverses {a.number} and {b.number} both start at the same Art-Net universe"
                    f" on {ip_a}; give them different --target IPs or a different --net",
                    file=sys.stderr,
                )
    return out


def build_plan(rig: Rig, sends, width: int, height: int, area: int = 1) -> tuple[list[Output], list[Group]]:
    """Sample positions scaled to a width x height input, and each pixel's DMX channels."""
    sx, sy = width / rig.comp_w, height / rig.comp_h
    k = max(1, int(area))
    offs = np.arange(k) - (k - 1) / 2.0
    oy, ox = (a.ravel() for a in np.meshgrid(offs, offs, indexing="ij"))
    outputs, groups = [], []
    for lu, ip, net in sends:
        by_fmt: dict[str, list] = {}
        top = 0
        for s in lu.slices:
            fmt = "".join(c for c in s.fmt if c in _COLS) or "rgb"
            n = s.width * s.height
            pts = cell_centres(s)
            first = s.start - 1 + np.arange(n) * len(fmt)
            chan = first[:, None] + np.arange(len(fmt))[None, :]
            top = max(top, int(chan.max()) + 1)
            by_fmt.setdefault(fmt, []).append((pts, chan, s.name))
        nuni = max(1, math.ceil(top / DMX))
        buf = np.zeros(nuni * DMX, np.uint8)
        li = len(outputs)
        used = [0] * nuni
        for fmt, items in by_fmt.items():
            pts = np.concatenate([p for p, _, _ in items])
            chan = np.concatenate([c for _, c, _ in items])
            px = pts[:, 0:1] * sx + ox[None, :]
            py = pts[:, 1:2] * sy + oy[None, :]
            ix = np.clip(np.floor(px), 0, width - 1).astype(np.intp)
            iy = np.clip(np.floor(py), 0, height - 1).astype(np.intp)
            if k == 1:
                ix, iy = ix[:, 0], iy[:, 0]
            groups.append(Group(li, iy, ix, chan, np.array([_COLS[c] for c in fmt]), [n for _, _, n in items]))
            for u, hi in _universe_tops(chan):
                used[u] = max(used[u], hi)
        used = [min(DMX, max(2, h + (h & 1))) for h in used]
        outputs.append(Output(lu, ip, int(net), buf, used, [0] * nuni))
    return outputs, groups


def _universe_tops(chan: np.ndarray):
    flat = chan.ravel()
    uni = flat // DMX
    for u in np.unique(uni):
        yield int(u), int(flat[uni == u].max() % DMX) + 1


# --------------------------------------------------------------------------
# Colour
# --------------------------------------------------------------------------


def make_lut(gamma: float, brightness: float) -> np.ndarray:
    x = np.arange(256) / 255.0
    return np.clip(np.round(255.0 * np.power(x, gamma) * brightness), 0, 255).astype(np.uint8)


def to_rgbw(rgb: np.ndarray, white: str) -> np.ndarray:
    """(P, 3) uint8 -> (P, 4) R, G, B, W.

    none: W stays off (straight pass-through).
    min:  the grey part of the colour moves to the W chip (RGB minus it).
    add:  W gets the grey part too, RGB unchanged (brightest).
    """
    out = np.zeros((rgb.shape[0], 4), np.uint8)
    out[:, :3] = rgb
    if white in ("min", "add"):
        w = rgb.min(axis=1)
        out[:, 3] = w
        if white == "min":
            out[:, :3] -= w[:, None]
    return out


class Renderer:
    def __init__(self, rig: Rig, sends, *, gamma: float | None, brightness: float, white: str, area: int):
        self.rig, self.sends, self.area, self.white = rig, sends, area, white
        if gamma is None:
            found = [s.gamma for lu, _, _ in sends for s in lu.slices if s.gamma is not None]
            gamma = found[0] if found else DEFAULT_GAMMA
        self.gamma = gamma
        self.lut = make_lut(gamma, brightness)
        self.size: tuple[int, int] | None = None
        self.outputs: list[Output] = []
        self.groups: list[Group] = []

    def _prepare(self, w: int, h: int) -> None:
        if self.size == (w, h):
            return
        old = {id(o.lumiverse): o.seq for o in self.outputs}
        self.outputs, self.groups = build_plan(self.rig, self.sends, w, h, self.area)
        for o in self.outputs:
            if id(o.lumiverse) in old and len(old[id(o.lumiverse)]) == len(o.seq):
                o.seq = old[id(o.lumiverse)]
        self.size = (w, h)

    def render(self, img: np.ndarray) -> None:
        """Fill every output buffer from one frame (H x W x 3 or 4, uint8)."""
        h, w = img.shape[:2]
        self._prepare(w, h)
        for g in self.groups:
            px = img[g.iy, g.ix, :3]
            if px.ndim == 3:  # area sampling: average the K samples
                px = (px.astype(np.uint16).sum(axis=1) + px.shape[1] // 2) // px.shape[1]
                px = px.astype(np.uint8)
            rgbw = to_rgbw(self.lut[px], self.white)
            self.outputs[g.lumi].buffer[g.chan] = rgbw[:, g.cols]

    def blackout(self) -> None:
        for o in self.outputs:
            o.buffer[:] = 0

    def packets(self) -> list[tuple[bytes, tuple[str, int]]]:
        pk = []
        for o in self.outputs:
            for u, n in enumerate(o.used):
                o.seq[u] = o.seq[u] % 255 + 1
                data = o.buffer[u * DMX : u * DMX + n].tobytes()
                pk.append((artdmx(o.port_address(u), data, o.seq[u]), (o.ip, ARTNET_PORT)))
        return pk


def artdmx(port_address: int, data: bytes, sequence: int) -> bytes:
    if len(data) % 2:
        data += b"\x00"
    head = _ARTDMX.pack(
        b"Art-Net\x00",
        0x5000,
        0,
        14,
        sequence & 0xFF,
        0,
        port_address & 0xFF,
        (port_address >> 8) & 0x7F,
        len(data) >> 8,
        len(data) & 0xFF,
    )
    return head + data


# --------------------------------------------------------------------------
# Sending, paced
# --------------------------------------------------------------------------


class Sender:
    """Sends one frame's packets spread over a short window instead of one burst.

    A whole tree is a few hundred universes per frame. Spacing them out keeps the
    receiver's socket buffer from overflowing; the bridge then does its own
    per-controller pacing towards the Twinklys.
    """

    def __init__(self, gap_us: float, bind: str = ""):
        self.gap = max(0.0, gap_us) / 1e6
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        try:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 20)
        except OSError:
            pass
        if bind:
            self.sock.bind((bind, 0))
        self.sent = 0
        self.errors = 0

    def send(self, packets, budget: float) -> None:
        gap = self.gap if self.gap * len(packets) <= budget else budget / max(1, len(packets))
        t0 = time.perf_counter()
        for k, (pkt, addr) in enumerate(packets):
            if gap:
                wait = t0 + k * gap - time.perf_counter()
                if wait > 0.001:  # sleep in >= 1 ms steps; the packets in between go back to back
                    time.sleep(wait)
            try:
                self.sock.sendto(pkt, addr)
                self.sent += 1
            except OSError:
                self.errors += 1

    def close(self) -> None:
        self.sock.close()


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


class SpoutSource:
    def __init__(self, name: str, flip: bool):
        try:
            import SpoutGL
        except ImportError:
            raise SystemExit("Spout needs Windows and `pip install SpoutGL` (or try --source test)") from None
        self.SpoutGL = SpoutGL
        self.rx = SpoutGL.SpoutReceiver()
        try:
            self.rx.createOpenGL()  # a hidden GL context, so no window is needed
        except Exception:
            pass
        if name:
            self.rx.setReceiverName(name)
        self.flip = flip
        self.buf: np.ndarray | None = None
        self.fmt = SpoutGL.enums.GL_RGBA
        self.label = name or "active sender"

    def grab(self) -> np.ndarray | None:
        ok = self.rx.receiveImage(self.buf, self.fmt, self.flip, 0)
        if self.rx.isUpdated():
            w, h = self.rx.getSenderWidth(), self.rx.getSenderHeight()
            self.buf = np.zeros((h, w, 4), np.uint8) if w and h else None
            return None  # the next call fills the new buffer
        if ok and self.buf is not None and self.rx.isConnected():
            return self.buf
        return None

    def close(self) -> None:
        try:
            self.rx.releaseReceiver()
        except Exception:
            pass


class TestSource:
    """A pattern at the composition size: hue sweeps around the tree, or bands rise up it."""

    def __init__(self, w: int, h: int, pattern: str):
        self.w, self.h, self.pattern = w, h, pattern
        self.t0 = time.perf_counter()
        self.label = f"test:{pattern}"

    def grab(self) -> np.ndarray:
        t = time.perf_counter() - self.t0
        if self.pattern == "rise":
            v = (np.arange(self.h) / self.h - t * 0.25) % 1.0
            col = hue_rgb(v)  # (h, 3)
            return np.broadcast_to(col[:, None, :], (self.h, self.w, 3))
        if self.pattern == "white":
            return np.full((1, 1, 3), 255, np.uint8).repeat(self.h, 0).repeat(self.w, 1)
        u = (np.arange(self.w) / self.w + t * 0.2) % 1.0
        row = hue_rgb(u)  # (w, 3)
        return np.broadcast_to(row[None, :, :], (self.h, self.w, 3))

    def close(self) -> None:
        pass


class ImageSource:
    def __init__(self, path: str):
        try:
            from PIL import Image
        except ImportError:
            raise SystemExit("--source image:FILE needs `pip install pillow`") from None
        self.img = np.asarray(Image.open(path).convert("RGB"))
        self.label = path

    def grab(self) -> np.ndarray:
        return self.img

    def close(self) -> None:
        pass


def hue_rgb(h: np.ndarray) -> np.ndarray:
    h6 = (h % 1.0) * 6.0
    r = np.clip(np.abs(h6 - 3) - 1, 0, 1)
    g = np.clip(2 - np.abs(h6 - 2), 0, 1)
    b = np.clip(2 - np.abs(h6 - 4), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


# --------------------------------------------------------------------------
# Snapshot: what came in, with the sample points on top
# --------------------------------------------------------------------------


def write_png(path: str, rgb: np.ndarray) -> None:
    h, w = rgb.shape[:2]
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")
    with open(path, "wb") as fh:
        fh.write(png)


def snapshot(path: str, img: np.ndarray, groups: list[Group]) -> None:
    out = np.ascontiguousarray(img[:, :, :3]).copy()
    h, w = out.shape[:2]
    for g in groups:
        iy = g.iy if g.iy.ndim == 1 else g.iy[:, g.iy.shape[1] // 2]
        ix = g.ix if g.ix.ndim == 1 else g.ix[:, g.ix.shape[1] // 2]
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                out[np.clip(iy + dy, 0, h - 1), np.clip(ix + dx, 0, w - 1)] = (255, 0, 255)
    write_png(path, out)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _pairs(items, kind) -> dict[str, str]:
    out = {}
    for it in items or []:
        key, sep, val = it.partition("=")
        if not sep:
            key, val = "all", it
        out[key.strip().lower() if key.strip().lower() == "all" else key.strip()] = val.strip()
    if kind == "ip":
        for k, v in out.items():
            if v.lower() != "off":
                try:
                    ipaddress.IPv4Address(v)
                except ValueError:
                    raise SystemExit(f"--target {k}={v}: not an IPv4 address") from None
    return out


def describe(rig: Rig, sends=None) -> str:
    lines = [f"composition {rig.comp_w}x{rig.comp_h}, {rig.pixels} pixels in {len(rig.lumiverses)} lumiverse(s)"]
    by = {id(lu): (ip, net) for lu, ip, net in sends or []}
    for lu in rig.lumiverses:
        n = sum(s.width * s.height for s in lu.slices)
        fmts = sorted({s.fmt for s in lu.slices})
        top = max((s.start - 1 + s.width * s.height * max(1, len(s.fmt)) for s in lu.slices), default=0)
        nuni = math.ceil(top / DMX) if top else 0
        line = (
            f"  {lu.number}: {lu.name}: {n} px in {len(lu.slices)} slices, {'/'.join(fmts) or '-'},"
            f" {nuni} universes from Sub-Net {lu.subnet} Universe {lu.universe}, TargetIP {lu.target or 'unset'}"
        )
        if not lu.enabled:
            line += " (disabled in Arena)"
        if id(lu) in by:
            ip, net = by[id(lu)]
            first = (net << 8) + lu.base
            line += f"\n     -> {ip}, Art-Net {first >> 8}.{(first >> 4) & 15}.{first & 15} and up"
        lines.append(line)
    return "\n".join(lines)


def cmd_info(a) -> int:
    rig = load_resolume(a.map)
    sends = None
    if a.target:
        sends = resolve_targets(rig, _pairs(a.target, "ip"), {k: int(v) for k, v in _pairs(a.net, "").items()})
    print(describe(rig, sends))
    return 0


def cmd_senders(a) -> int:
    try:
        import SpoutGL
    except ImportError:
        raise SystemExit("Spout needs Windows and `pip install SpoutGL`") from None
    rx = SpoutGL.SpoutReceiver()
    names = []
    try:
        names = list(rx.getSenderList())
    except Exception:
        pass
    active = None
    try:
        active = rx.getActiveSender()
    except Exception:
        pass
    if not names and not active:
        print("no Spout senders right now (is Arena's Spout output on?)")
    for n in names:
        info = None
        try:
            info = rx.getSenderInfo(n)
        except Exception:
            pass
        size = f" {info.width}x{info.height}" if info else ""
        print(f"{n}{size}{'  (active)' if n == active else ''}")
    if active and active not in names:
        print(f"{active}  (active)")
    return 0


def cmd_run(a) -> int:
    rig = load_resolume(a.map)
    sends = resolve_targets(rig, _pairs(a.target, "ip"), {k: int(v) for k, v in _pairs(a.net, "").items()})
    print(describe(rig, sends))
    renderer = Renderer(rig, sends, gamma=a.gamma, brightness=a.brightness / 100.0, white=a.white, area=a.area)
    print(f"gamma {renderer.gamma:g}, brightness {a.brightness:g}%, white {a.white}, {a.fps:g} fps")
    if a.source == "spout":
        src = SpoutSource(a.sender, a.flip)
    elif a.source.startswith("test"):
        src = TestSource(rig.comp_w, rig.comp_h, a.source.partition(":")[2] or "sweep")
    elif a.source.startswith("image:"):
        src = ImageSource(a.source.partition(":")[2])
    else:
        raise SystemExit(f"--source {a.source}: use spout, test[:sweep|rise|white] or image:FILE")
    sender = Sender(a.gap_us, a.bind)
    period = 1.0 / a.fps
    next_t = time.perf_counter()
    last_in = 0.0
    have = False
    frames_in = frames_out = 0
    shot_due = time.perf_counter() + 2.0 if a.snapshot else None
    stat_t = time.perf_counter()
    sent0 = 0
    print(f"input {src.label}; Ctrl+C to stop")
    try:
        while True:
            now = time.perf_counter()
            img = src.grab()
            if img is not None:
                renderer.render(img)
                have, last_in = True, now
                frames_in += 1
                if shot_due and now >= shot_due:
                    snapshot(a.snapshot, img, renderer.groups)
                    print(f"\nsnapshot -> {a.snapshot}")
                    shot_due = None
            if have:  # with no new frame, the last one repeats (Art-Net likes a steady refresh)
                sender.send(renderer.packets(), budget=period * 0.6)
                frames_out += 1
            if now - stat_t >= 1.0:
                dt = now - stat_t
                size = f"{renderer.size[0]}x{renderer.size[1]}" if renderer.size else "-"
                state = "no input yet" if not have else ("INPUT LOST, holding" if now - last_in > 1.0 else "ok")
                pps = (sender.sent - sent0) / dt
                sys.stdout.write(
                    f"\r{state:<20} in {frames_in / dt:5.1f} fps {size:>9}  out {frames_out / dt:5.1f} fps"
                    f" {pps:7.0f} pkt/s  errors {sender.errors}   "
                )
                sys.stdout.flush()
                frames_in = frames_out = 0
                sent0 = sender.sent
                stat_t = now
            next_t += period
            wait = next_t - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
            else:
                next_t = time.perf_counter()  # running late: don't try to catch up
            if time.perf_counter() > a._deadline:
                break
    except KeyboardInterrupt:
        pass
    finally:
        print()
        if a.blackout_on_exit and renderer.outputs:
            renderer.blackout()
            for _ in range(3):
                sender.send(renderer.packets(), budget=0.05)
            print("sent blackout")
        src.close()
        sender.close()
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="spout2artnet", description=__doc__.split("\n\n")[0])
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--map", required=True, help="Resolume Advanced Output XML (the /map export)")
        sp.add_argument(
            "--target",
            action="append",
            metavar="N=IP",
            help="destination IP for lumiverse N (1, 2, ...; 'all=IP' for every one; N=off skips it)."
            " Default: the TargetIP saved in the XML",
        )
        sp.add_argument(
            "--net",
            action="append",
            metavar="N=NET",
            help="Art-Net Net for lumiverse N (default 0, like Arena, which can't set the Net field)",
        )

    sp = sub.add_parser("info", help="show the lumiverses, pixels and universes in a map")
    common(sp)
    sp.set_defaults(func=cmd_info)

    sp = sub.add_parser("senders", help="list Spout senders")
    sp.set_defaults(func=cmd_senders)

    sp = sub.add_parser("run", help="Spout in, Art-Net out")
    common(sp)
    sp.add_argument("--source", default="spout", help="spout (default), test[:sweep|rise|white], image:FILE")
    sp.add_argument("--sender", default="", help="Spout sender name (default: the active sender)")
    sp.add_argument("--flip", action="store_true", help="flip the Spout image upside down")
    sp.add_argument("--fps", type=float, default=30.0, help="Art-Net frames per second (default 30, like a lumiverse)")
    sp.add_argument("--gamma", type=float, default=None, help="default: the fixture Gamma in the XML, else 2.5")
    sp.add_argument("--brightness", type=float, default=100.0, help="master level, percent")
    sp.add_argument("--white", choices=("none", "min", "add"), default="none", help="how RGBW pixels use W")
    sp.add_argument("--area", type=int, default=1, help="average an N x N patch per pixel (default 1 = centre)")
    sp.add_argument("--gap-us", type=float, default=50.0, help="spacing between Art-Net packets, microseconds")
    sp.add_argument("--bind", default="", help="local IP to send from (pick the Art-Net NIC)")
    sp.add_argument("--snapshot", help="2 s in, save the input with every sample point marked (PNG)")
    sp.add_argument("--blackout-on-exit", action="store_true", help="send black when stopping")
    sp.add_argument("--duration", type=float, default=0.0, help=argparse.SUPPRESS)
    sp.set_defaults(func=cmd_run)

    a = p.parse_args(argv)
    if a.cmd == "run":
        if a.fps <= 0:
            p.error("--fps must be above 0")
        a._deadline = time.perf_counter() + a.duration if a.duration else float("inf")
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
