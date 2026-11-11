#!/usr/bin/env python3
"""Inject a frame-stamped test pattern and read it back off the wire to the
controllers, to tell a bridge data bug apart from controller firmware.

The console has been ruled out and the stall is isolated to "the same physical
output on every controller lags the other five". Two things could still cause
that: the bridge sends stale/mis-mapped bytes for that output (our bug, fixable),
or the controller renders that output late (firmware). This settles it.

We become the console. Every pixel of every frame is stamped so the OUTGOING
Twinkly frame (UDP 7777) describes itself. With RGBW and the usual WRGB wire
order each pixel carries:

    W = frame number  (0-250, counts up once per whole-tree frame)
    R = output number (1..6 -- which 240-px output this pixel belongs to)
    G = pixel index, high byte     B = pixel index, low byte

Because the input is known and always changing, the capture answers one question
with no "is it just static content?" ambiguity:

    do all six outputs of a controller carry the SAME frame number, moving
    together, frame after frame?

  * all outputs same frame, in lockstep, stamps correct
        -> the bridge is clean. A laggy output is the controller / firmware.
  * one output's range holds an old frame while the others advance,
    or its out/index stamp is wrong, or the frame is short of 1440 px
        -> the bridge is the problem, and the stamp says exactly where.

Roles
-----
  rt_probe.py --generate [--config F] [--fps N] [--seconds S] [--only NAME,..]
        Replace the console: stream the stamped pattern to the bridge's Art-Net
        input. Run with the real console shut off so nothing else writes.

  rt_probe.py --analyze CAP.pcap [--target IP] [--order WRGB]
        Reassemble the UDP-7777 frames the bridge sent to one controller and
        check the stamps. Capture first (drops far fewer packets than a live
        Python sniff):
            tcpdump -i any -w /tmp/rt.pcap 'udp port 7777 and host <IP>'

  rt_probe.py --sniff --target IP [--seconds S] [--order WRGB]
        Live AF_PACKET capture then analyze (needs root).

  rt_probe.py --probe --target IP [--seconds S] [--fps N] [--only NAME,..]
        One shot: generate + live sniff + analyze (needs root).

  rt_probe.py --selftest
"""

from __future__ import annotations

import os
import socket
import struct
import sys
import threading
import time

ARTNET_PORT = 6454
RT_PORT = 7777
OUTPUT_PIXELS = 240  # one controller output = 240 px
FRAME_MOD = 251  # frame counter cycles 0..250 (stays a single byte, never 255)

# Package imports (available from the deployed checkout). The analyzer is kept
# package-free so --analyze runs from a bare copy; --generate/--selftest need it.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
try:
    from twinklybridge.artnet import encode_dmx
    from twinklybridge.config import ARTNET, load_config
    from twinklybridge.pixels import convert
    from twinklybridge.twinkly import build_packets
except Exception:  # pragma: no cover - only outside the package
    encode_dmx = None  # type: ignore
    load_config = None  # type: ignore
    convert = None  # type: ignore
    build_packets = None  # type: ignore
    ARTNET = "artnet"


# --------------------------------------------------------------------------
# Stamp codec -- the same letter->value map on the way in and the way out
# --------------------------------------------------------------------------


def _stamp_values(frame_no: int, out_num: int, idx: int) -> dict[str, int]:
    return {
        "W": frame_no % FRAME_MOD,
        "R": out_num & 0xFF,
        "G": (idx >> 8) & 0xFF,
        "B": idx & 0xFF,
    }


def stamp_pixel(buf: bytearray, off: int, layout: str, frame_no: int, out_num: int, idx: int) -> None:
    """Write one stamped pixel at buf[off:] in channel order `layout`."""
    vals = _stamp_values(frame_no, out_num, idx)
    for k, letter in enumerate(layout):
        buf[off + k] = vals.get(letter.upper(), 0)


def output_of(idx: int) -> int:
    return idx // OUTPUT_PIXELS + 1


# --------------------------------------------------------------------------
# Generate: become the console
# --------------------------------------------------------------------------


def _generate_plan(cfg):
    """Per Art-Net segment: (device, destination IP, universe, dmx bytes builder)."""
    plan = []
    for d in cfg.enabled_devices:
        if d.protocol != ARTNET:
            continue
        feed = d.listen_ip or cfg.input.ip
        dst = feed if feed not in ("", "0.0.0.0") else "127.0.0.1"
        for seg in d.segments:
            plan.append((d, dst, seg))
    return plan


def _segment_dmx(d, seg, frame_no: int) -> bytes:
    """The DMX payload for one universe: stamped pixels placed at seg.channel."""
    ch = d.channels
    dmx = bytearray(seg.channel + seg.count * ch)
    for p in range(seg.count):
        idx = seg.first_pixel + p
        stamp_pixel(dmx, seg.channel + p * ch, d.input, frame_no, output_of(idx), idx)
    return bytes(dmx)


def generate(config_path: str, fps: float, seconds: float, only: set[str], stop: threading.Event | None = None) -> int:
    if load_config is None or encode_dmx is None:
        raise SystemExit("--generate needs the twinklybridge package on the path")
    cfg = load_config(config_path)
    plan = _generate_plan(cfg)
    if only:
        plan = [p for p in plan if p[0].name in only]
    if not plan:
        raise SystemExit("no Art-Net devices to feed (check --config / --only)")
    devices = sorted({d.name for d, _, _ in plan})
    dests = sorted({dst for _, dst, _ in plan})
    uni = len({(dst, seg.universe) for _, dst, seg in plan})
    print(
        f"generating stamped pattern: {len(devices)} device(s), {uni} universe(s) -> {', '.join(dests)}"
        f" at {fps:g} fps for {seconds:g}s",
        flush=True,
    )
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
    period = 1.0 / fps
    frame_no = 0
    seq = 0
    deadline = time.monotonic() + seconds
    sent = 0
    try:
        while time.monotonic() < deadline and not (stop and stop.is_set()):
            tick = time.monotonic()
            for d, dst, seg in plan:
                pkt = encode_dmx(seg.universe, _segment_dmx(d, seg, frame_no), sequence=seq)
                try:
                    sock.sendto(pkt, (dst, ARTNET_PORT))
                    sent += 1
                except OSError as exc:
                    print(f"  send to {dst} u{seg.universe} failed: {exc}", file=sys.stderr)
            frame_no = (frame_no + 1) % FRAME_MOD
            seq = (seq + 1) & 0xFF
            rest = period - (time.monotonic() - tick)
            if rest > 0:
                time.sleep(rest)
    finally:
        sock.close()
    print(f"generated {frame_no} frame(s), {sent} packets sent", flush=True)
    return 0


# --------------------------------------------------------------------------
# Capture: pull the UDP-7777 frames off the wire
# --------------------------------------------------------------------------


def _udp(ip: bytes):
    """(src_ip, dst_ip, dport, payload) for an IPv4/UDP packet, else None."""
    if len(ip) < 20 or (ip[0] >> 4) != 4 or ip[9] != 17:
        return None
    ihl = (ip[0] & 0x0F) * 4
    udp = ip[ihl:]
    if len(udp) < 8:
        return None
    dport = (udp[2] << 8) | udp[3]
    src = ".".join(str(b) for b in ip[12:16])
    dst = ".".join(str(b) for b in ip[16:20])
    return src, dst, dport, udp[8:]


def _strip(linktype: int, frame: bytes):
    if linktype == 1:  # EN10MB
        if len(frame) < 14:
            return None
        etype = (frame[12] << 8) | frame[13]
        off = 14
        while etype == 0x8100 and len(frame) >= off + 4:
            etype = (frame[off + 2] << 8) | frame[off + 3]
            off += 4
        return _udp(frame[off:]) if etype == 0x0800 else None
    if linktype == 113:  # LINUX_SLL
        return _udp(frame[16:]) if len(frame) >= 16 and ((frame[14] << 8) | frame[15]) == 0x0800 else None
    if linktype == 276:  # LINUX_SLL2
        return _udp(frame[20:]) if len(frame) >= 20 and ((frame[0] << 8) | frame[1]) == 0x0800 else None
    return None


def _iter_pcap(path: str):
    with open(path, "rb") as fh:
        hdr = fh.read(24)
        if len(hdr) < 24:
            raise SystemExit(f"{path}: too short to be a pcap")
        magic = hdr[:4]
        if magic in (b"\xa1\xb2\xc3\xd4", b"\xa1\xb2\x3c\x4d"):
            endian, nanos = ">", magic == b"\xa1\xb2\x3c\x4d"
        elif magic in (b"\xd4\xc3\xb2\xa1", b"\x4d\x3c\xb2\xa1"):
            endian, nanos = "<", magic == b"\x4d\x3c\xb2\xa1"
        else:
            raise SystemExit(f"{path}: not a classic pcap (magic {magic!r}); convert a pcapng with `editcap -F pcap`")
        linktype = struct.unpack(endian + "I", hdr[20:24])[0]
        rec = struct.Struct(endian + "IIII")
        while True:
            rh = fh.read(16)
            if len(rh) < 16:
                return
            ts_sec, ts_frac, incl, _orig = rec.unpack(rh)
            body = fh.read(incl)
            if len(body) < incl:
                return
            ts = ts_sec + ts_frac / (1e9 if nanos else 1e6)
            got = _strip(linktype, body)
            if got is not None:
                yield ts, got


def _iter_sniff(seconds: float):
    try:
        s = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.ntohs(0x0003))
    except (AttributeError, PermissionError) as exc:
        raise SystemExit(f"live sniff needs Linux + root: {exc}") from exc
    s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 64 * 1024 * 1024)
    s.settimeout(0.5)
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            frame = s.recv(2048)
        except TimeoutError:
            continue
        got = _strip(1, frame)
        if got is not None:
            yield time.time(), got


def _rt_datagrams(source, target: str):
    """Yield (ts, token, index, payload) for UDP-7777 frames to `target`."""
    for ts, (_src, dst, dport, payload) in source:
        if dport != RT_PORT or (target and dst != target):
            continue
        if len(payload) < 13 or payload[0] != 0x03:
            continue
        token = payload[1:9]
        index = payload[11]
        yield ts, token, index, payload[12:]


def reassemble(datagrams):
    """Group sequential fragment bursts (index resets) into whole frames.

    Yields (ts, token, frame_bytes). A fragment whose index does not advance
    beyond the current burst starts a new frame.
    """
    cur: dict[int, bytes] = {}
    cur_ts = 0.0
    cur_token = b""
    last = -1
    for ts, token, index, frag in datagrams:
        if cur and (index <= last or token != cur_token):
            yield cur_ts, cur_token, b"".join(cur[i] for i in sorted(cur))
            cur = {}
            last = -1
        if not cur:
            cur_ts, cur_token = ts, token
        cur[index] = frag
        last = index
    if cur:
        yield cur_ts, cur_token, b"".join(cur[i] for i in sorted(cur))


# --------------------------------------------------------------------------
# Analyze: read the stamps
# --------------------------------------------------------------------------


class FrameReport:
    __slots__ = ("pixels", "per_output", "bad_out", "bad_pos", "token")

    def __init__(self) -> None:
        self.pixels = 0
        self.per_output: dict[int, int] = {}  # output number -> frame number it shows
        self.bad_out = 0
        self.bad_pos = 0
        self.token = b""


def decode_frame(frame: bytes, order: str) -> FrameReport:
    dn = len(order)
    wpos, rpos, gpos, bpos = (order.find(c) for c in "WRGB")
    r = FrameReport()
    r.pixels = len(frame) // dn
    # frame number per output: take the first pixel of each output as the reading,
    # then count any pixel in that output that disagrees as a stamp error.
    seen: dict[int, int] = {}
    for i in range(r.pixels):
        base = i * dn
        fno = frame[base + wpos] if wpos >= 0 else 0
        out = frame[base + rpos] if rpos >= 0 else 0
        hi = frame[base + gpos] if gpos >= 0 else 0
        lo = frame[base + bpos] if bpos >= 0 else 0
        exp_out = output_of(i)
        if out != (exp_out & 0xFF):
            r.bad_out += 1
        if (hi << 8 | lo) != i:
            r.bad_pos += 1
        if exp_out not in seen:
            seen[exp_out] = fno
    r.per_output = seen
    return r


def analyze(frames, order: str, expect_pixels: int = 1440) -> int:
    reports = [decode_frame(f, order) for _, _, f in frames if f]
    if not reports:
        print("No UDP-7777 frames to this target were captured.")
        print("Is the bridge streaming to it, did the capture run during the pattern, and is --target right?")
        return 2

    n = len(reports)
    outs = sorted({o for r in reports for o in r.per_output})
    px = reports[len(reports) // 2].pixels
    print(f"analyzed {n} outgoing frames, {px} px/frame, outputs seen: {', '.join(f'out{o}' for o in outs)}\n")

    # 1. pixel count / mapping integrity
    short = px != expect_pixels
    bad_out = sum(r.bad_out for r in reports)
    bad_pos = sum(r.bad_pos for r in reports)
    tot_px = sum(r.pixels for r in reports) or 1

    # 2. per-frame coherence: do all outputs carry one frame number?
    tears = 0
    worst_tear = 0
    tear_by_out: dict[int, int] = {}
    # 3. per-output advancement across the capture (lag vs the fastest output)
    lag_by_out: dict[int, int] = {o: 0 for o in outs}
    for r in reports:
        if len(r.per_output) < 2:
            continue
        lead = max(r.per_output.values())
        for o, fno in r.per_output.items():
            # frame numbers wrap at FRAME_MOD; a small negative delta is a wrap
            d = (lead - fno) % FRAME_MOD
            if d > FRAME_MOD // 2:
                d = 0
            if d:
                tear_by_out[o] = max(tear_by_out.get(o, 0), d)
                lag_by_out[o] += 1
            worst_tear = max(worst_tear, d)
        if max(r.per_output.values()) != min(r.per_output.values()):
            tears += 1

    print("stamp integrity:")
    print(f"  pixels per frame     {px}   ({'OK' if not short else f'MISMATCH, expected {expect_pixels}'})")
    print(f"  wrong output stamp   {bad_out}/{tot_px} pixels ({100 * bad_out / tot_px:.2f}%)")
    print(f"  wrong index stamp    {bad_pos}/{tot_px} pixels ({100 * bad_pos / tot_px:.2f}%)\n")

    print("per-output frame coherence (are the six outputs on the same frame?):")
    print(f"  frames where outputs disagree   {tears}/{n}")
    if tear_by_out:
        for o in sorted(tear_by_out):
            print(
                f"    out{o}: trails the lead output by up to {tear_by_out[o]} frame(s), on {lag_by_out[o]}/{n} frames"
            )
    else:
        print("    none -- every captured frame had all outputs on the same frame number")
    print()

    # verdict
    clean_map = not short and bad_out == 0 and bad_pos == 0
    coherent = tears == 0
    print("=" * 68)
    if clean_map and coherent:
        print("VERDICT: BRIDGE CLEAN.")
        print("Every output carried the same frame, in lockstep, correctly mapped.")
        print("The wire the bridge sends is coherent -- a laggy output is the")
        print("CONTROLLER / FIRMWARE rendering it late, not the bridge's data.")
        rc = 0
    elif not coherent:
        who = ", ".join(f"out{o} (up to {tear_by_out[o]} frame(s))" for o in sorted(tear_by_out))
        print("VERDICT: BRIDGE TEAR on the wire.")
        print(f"These outputs lag on the outgoing frame itself: {who}.")
        print("The controller is being sent a stale frame for that output -> bridge bug.")
        rc = 1
    else:
        issues = []
        if short:
            issues.append(f"frame is {px}px, not {expect_pixels} (truncated/padded -> tail output wrong)")
        if bad_out:
            issues.append("output-number stamps are wrong (pixels land at the wrong offset)")
        if bad_pos:
            issues.append("index stamps are wrong (pixels shifted within the frame)")
        print("VERDICT: BRIDGE MAPPING problem.")
        for s in issues:
            print(f"  - {s}")
        rc = 1
    print("=" * 68)
    return rc


# --------------------------------------------------------------------------
# selftest -- full codec loop with a simulated bridge, no hardware
# --------------------------------------------------------------------------


def _simulate_bridge(cfg_device, frame_no: int, order: str, tear: dict[int, int] | None = None) -> bytes:
    """Mimic the bridge: stamp input per segment, assemble the buffer, convert to
    output order, exactly as engine.write/_latch + render do. `tear` maps an
    output number to an older frame number to fake a per-output stall."""
    d = cfg_device
    ch = d.channels
    buf = bytearray(d.pixels * ch)
    for seg in d.segments:
        dmx = _segment_dmx_tearable(d, seg, frame_no, tear)
        chunk = dmx[seg.channel : seg.channel + seg.count * ch]
        buf[seg.first_pixel * ch : seg.first_pixel * ch + len(chunk)] = chunk
    return bytes(convert(buf, d.input, order, d.pixels))


def _segment_dmx_tearable(d, seg, frame_no: int, tear: dict[int, int] | None) -> bytes:
    ch = d.channels
    dmx = bytearray(seg.channel + seg.count * ch)
    for p in range(seg.count):
        idx = seg.first_pixel + p
        out = output_of(idx)
        fno = tear.get(out, frame_no) if tear else frame_no
        stamp_pixel(dmx, seg.channel + p * ch, d.input, fno, out, idx)
    return bytes(dmx)


def _selftest() -> int:
    if convert is None or build_packets is None or load_config is None:
        print("SELFTEST SKIP: twinklybridge package not importable")
        return 0
    from twinklybridge.config import parse_config

    ok = True

    def check(name: str, cond: bool) -> None:
        nonlocal ok
        ok = ok and cond
        print(f"  [{'ok' if cond else 'XX'}] {name}")

    cfg = parse_config(
        {
            "input": {"cidr": "10.0.0.5/24"},
            "devices": [{"name": "T", "ip": "10.0.0.50", "universe": "0.0.0", "pixels": 1440, "input": "RGBW"}],
        }
    )
    dev = cfg.devices[0]
    order = "WRGB"  # Twinkly RGBW wire order

    # Healthy frame: stamp -> simulate bridge -> fragment -> reassemble -> decode.
    frame = _simulate_bridge(dev, 42, order)
    token = bytes(range(8))
    pkts = build_packets(3, token, frame, dev.pixels)
    check("fragments 1440px RGBW into 7 datagrams", len(pkts) == 7)
    datagrams = [(0.0, p[1:9], p[11], p[12:]) for p in pkts]
    reframed = list(reassemble(d for d in datagrams))
    check("reassembles one whole frame", len(reframed) == 1 and reframed[0][2] == frame)
    rep = decode_frame(reframed[0][2], order)
    check("1440 pixels decoded", rep.pixels == 1440)
    check("6 outputs seen", sorted(rep.per_output) == [1, 2, 3, 4, 5, 6])
    check("every output reads frame 42", set(rep.per_output.values()) == {42})
    check("output stamps all correct", rep.bad_out == 0)
    check("index stamps all correct", rep.bad_pos == 0)

    # Clean capture -> VERDICT clean (rc 0).
    clean = [(0.0, token, _simulate_bridge(dev, f, order)) for f in range(10)]
    print("  -- analyze(clean) --")
    check("clean capture verdicts BRIDGE CLEAN", analyze(clean, order) == 0)

    # Torn capture: output 6 stuck 3 frames behind -> VERDICT tear (rc 1).
    torn = [(0.0, token, _simulate_bridge(dev, f, order, tear={6: max(f - 3, 0)})) for f in range(10)]
    print("  -- analyze(torn: out6 lags 3) --")
    rep6 = decode_frame(torn[-1][2], order)
    check("torn frame: out6 reads an older frame than out1", rep6.per_output[6] < rep6.per_output[1])
    check("torn capture verdicts a problem", analyze(torn, order) == 1)

    # DMX offset path with start_channel shift stays aligned.
    cfg2 = parse_config(
        {
            "input": {"cidr": "10.0.0.5/24"},
            "devices": [
                {
                    "name": "S",
                    "ip": "10.0.0.51",
                    "universe": "0.0.0",
                    "pixels": 300,
                    "input": "RGBW",
                    "start_channel": 5,
                }
            ],
        }
    )
    d2 = cfg2.devices[0]
    f2 = _simulate_bridge(d2, 7, order)
    rep2 = decode_frame(f2, order)
    check("start_channel offset still maps cleanly", rep2.bad_pos == 0 and rep2.bad_out == 0)

    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str]) -> int:
    args = argv[1:]
    if "--selftest" in args:
        return _selftest()

    mode = ""
    config = "/etc/twinkly-bridge/config.yaml"
    fps = 30.0
    seconds = 20.0
    target = ""
    order = "WRGB"
    only: set[str] = set()
    path = ""
    i = 0
    while i < len(args):
        a = args[i]
        if a in ("--generate", "--sniff", "--probe", "--analyze"):
            mode = a[2:]
            if a == "--analyze":
                i += 1
                path = args[i]
        elif a == "--config":
            i += 1
            config = args[i]
        elif a == "--fps":
            i += 1
            fps = float(args[i])
        elif a == "--seconds":
            i += 1
            seconds = float(args[i])
        elif a == "--target":
            i += 1
            target = args[i]
        elif a == "--order":
            i += 1
            order = args[i].upper()
        elif a == "--only":
            i += 1
            only = {s.strip() for s in args[i].split(",") if s.strip()}
        else:
            raise SystemExit(f"unknown option {a!r}\n\n{__doc__}")
        i += 1

    if mode == "generate":
        return generate(config, fps, seconds, only)
    if mode == "analyze":
        return analyze(reassemble(_rt_datagrams(_iter_pcap(path), target)), order)
    if mode == "sniff":
        if not target:
            raise SystemExit("--sniff needs --target IP")
        print(f"sniffing UDP 7777 to {target} for {seconds:g}s (needs root)...", file=sys.stderr)
        return analyze(reassemble(_rt_datagrams(_iter_sniff(seconds), target)), order)
    if mode == "probe":
        if not target:
            raise SystemExit("--probe needs --target IP")
        stop = threading.Event()
        gen = threading.Thread(target=generate, args=(config, fps, seconds + 2, only, stop), daemon=True)
        gen.start()
        time.sleep(0.5)
        print(f"probing: generating + sniffing UDP 7777 to {target} for {seconds:g}s (needs root)...", file=sys.stderr)
        try:
            frames = list(reassemble(_rt_datagrams(_iter_sniff(seconds), target)))
        finally:
            stop.set()
            gen.join(timeout=3)
        return analyze(frames, order)

    raise SystemExit(__doc__)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
