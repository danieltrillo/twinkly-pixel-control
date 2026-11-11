#!/usr/bin/env python3
"""Tap the console's Art-Net and report, per universe, how fast it *actually*
updates. Tells apart three causes of a "laggy spot" that the bridge cannot be
blamed for:

  * a universe the source sends at a low packet rate (console/network),
  * a universe sent at full rate whose pixel values sit still for stretches
    then jump (content "hang-and-snap" — an effect/pixel-map issue),
  * a healthy universe (packets and value changes both at the frame rate).

The bridge only relays what it is sent, so if the laggy universes update slowly
here, the stall is upstream of the bridge.

Usage:
  artnet_tap.py CAPTURE.pcap [--config FILE] [--top N]   parse a tcpdump capture
  artnet_tap.py --sniff [--seconds 6] [--config FILE]     sniff live (needs root)
  artnet_tap.py --selftest                                built-in check, then exit

Capture with, e.g.:  tcpdump -i <iface> -w /tmp/art.pcap udp port 6454
(capturing to a file drops far fewer packets than a live Python sniff, so prefer
it when you care about the exact gap timing.)

--config /etc/twinkly-bridge/config.yaml annotates each universe with the device
it feeds and its index within that device, so "every device's first/last
universe" stands out.
"""

from __future__ import annotations

import socket
import statistics
import struct
import sys
import time
from collections import OrderedDict

ARTNET_PORT = 6454

# Use the bridge's own decoder when importable; fall back to a tiny inline one
# so the tool also runs from a bare checkout.
try:
    from twinklybridge import artnet as _artnet

    def decode_dmx(payload: bytes):
        d = _artnet.decode_dmx(payload)
        return (d.universe, bytes(d.data)) if d is not None else None

    def fmt_uni(u: int) -> str:
        return _artnet.format_universe(u)

except Exception:  # pragma: no cover - only when run outside the package

    def decode_dmx(payload: bytes):
        if len(payload) < 18 or payload[:8] != b"Art-Net\x00":
            return None
        if (payload[8] | (payload[9] << 8)) != 0x5000:  # OP_DMX
            return None
        length = (payload[16] << 8) | payload[17]
        if length < 1:
            return None
        uni = payload[14] | (payload[15] << 8)
        return (uni, bytes(payload[18 : 18 + length]))

    def fmt_uni(u: int) -> str:
        return f"{(u >> 8) & 0x7F}.{(u >> 4) & 0xF}.{u & 0xF}"


class UniStat:
    __slots__ = ("first", "last", "n", "changes", "_last_data", "_last_change", "gaps")

    def __init__(self, ts: float) -> None:
        self.first = ts
        self.last = ts
        self.n = 0
        self.changes = 0
        self._last_data: bytes | None = None
        self._last_change: float | None = None
        self.gaps: list[float] = []

    def add(self, ts: float, data: bytes) -> None:
        self.last = ts
        self.n += 1
        if data != self._last_data:
            if self._last_change is not None:
                self.gaps.append(ts - self._last_change)
            self.changes += 1
            self._last_change = ts
            self._last_data = data

    @property
    def elapsed(self) -> float:
        return max(self.last - self.first, 1e-9)

    @property
    def pkt_rate(self) -> float:
        return self.n / self.elapsed

    @property
    def change_rate(self) -> float:
        return self.changes / self.elapsed

    @property
    def max_gap(self) -> float:
        return max(self.gaps) if self.gaps else 0.0

    @property
    def mean_gap(self) -> float:
        return statistics.fmean(self.gaps) if self.gaps else 0.0


def summarize(packets) -> OrderedDict[int, UniStat]:
    stats: OrderedDict[int, UniStat] = OrderedDict()
    for ts, uni, data in packets:
        s = stats.get(uni)
        if s is None:
            s = stats[uni] = UniStat(ts)
        s.add(ts, data)
    return stats


# --------------------------------------------------------------------------
# pcap reader (classic libpcap format, Ethernet or Linux cooked link layers)
# --------------------------------------------------------------------------


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
            raise SystemExit(
                f"{path}: not a pcap (bad magic {magic!r}); a pcapng needs converting with `editcap -F pcap`"
            )
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
            payload = _strip_to_udp(linktype, body)
            if payload is not None:
                yield ts, payload


def _strip_to_udp(linktype: int, frame: bytes):
    """Return the UDP payload of an IPv4/UDP frame bound for the Art-Net port, else None."""
    if linktype == 1:  # DLT_EN10MB
        if len(frame) < 14:
            return None
        etype = (frame[12] << 8) | frame[13]
        off = 14
        while etype == 0x8100 and len(frame) >= off + 4:  # VLAN tag(s)
            etype = (frame[off + 2] << 8) | frame[off + 3]
            off += 4
        if etype != 0x0800:
            return None
        ip = frame[off:]
    elif linktype == 113:  # DLT_LINUX_SLL
        if len(frame) < 16 or ((frame[14] << 8) | frame[15]) != 0x0800:
            return None
        ip = frame[16:]
    elif linktype == 276:  # DLT_LINUX_SLL2
        if len(frame) < 20 or ((frame[0] << 8) | frame[1]) != 0x0800:
            return None
        ip = frame[20:]
    else:
        return None
    return _udp_payload(ip)


def _udp_payload(ip: bytes):
    if len(ip) < 20 or (ip[0] >> 4) != 4 or ip[9] != 17:  # IPv4, proto UDP
        return None
    ihl = (ip[0] & 0x0F) * 4
    udp = ip[ihl:]
    if len(udp) < 8:
        return None
    dport = (udp[2] << 8) | udp[3]
    if dport != ARTNET_PORT:
        return None
    return udp[8:]


def _iter_pcap_dmx(path: str):
    for ts, payload in _iter_pcap(path):
        d = decode_dmx(payload)
        if d is not None:
            yield ts, d[0], d[1]


# --------------------------------------------------------------------------
# live sniff (AF_PACKET; needs root; may drop under load)
# --------------------------------------------------------------------------


def _iter_sniff(seconds: float):
    try:
        s = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.ntohs(0x0003))  # ETH_P_ALL
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
        payload = _strip_to_udp(1, frame)  # AF_PACKET on an Ethernet iface = EN10MB frames
        if payload is None:
            continue
        d = decode_dmx(payload)
        if d is not None:
            yield time.time(), d[0], d[1]


# --------------------------------------------------------------------------
# universe -> device annotation (optional)
# --------------------------------------------------------------------------


def _load_annotations(config_path: str) -> dict[int, str]:
    try:
        from twinklybridge.config import load_config
    except Exception as exc:
        print(f"(could not import config to annotate: {exc})", file=sys.stderr)
        return {}
    try:
        cfg = load_config(config_path)
    except Exception as exc:
        print(f"(could not load {config_path}: {exc})", file=sys.stderr)
        return {}
    ann: dict[int, str] = {}
    for dev in cfg.devices:
        segs = dev.segments
        n = len(segs)
        for i, seg in enumerate(segs):
            # output k (1-based) covers pixels (k-1)*240..k*240-1; a universe can straddle two
            first_out = seg.first_pixel // 240 + 1
            last_out = (seg.first_pixel + seg.count - 1) // 240 + 1
            out = f"out {first_out}" if first_out == last_out else f"out {first_out}-{last_out}"
            tag = f"{dev.name} u{i + 1}/{n} {out}"
            edge = " <edge" if i == 0 or i == n - 1 else ""
            ann[seg.universe] = tag + edge
    return ann


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


def report(stats: OrderedDict[int, UniStat], top: int, ann: dict[int, str]) -> None:
    if not stats:
        print("No ArtDmx captured. Is the console unicasting to this host, and did the capture run during the effect?")
        return
    rows = list(stats.values())
    span = max(s.last for s in rows) - min(s.first for s in rows)
    pkt_rates = sorted(s.pkt_rate for s in rows)
    chg_rates = sorted(s.change_rate for s in rows)
    med_pkt = statistics.median(pkt_rates)
    med_chg = statistics.median(chg_rates)
    print(f"captured {sum(s.n for s in rows)} ArtDmx packets across {len(rows)} universes over {span:.1f}s")
    print(f"median universe: {med_pkt:.1f} pkts/s, {med_chg:.1f} value-changes/s")
    print(
        f"slow senders   = pkt rate well under {med_pkt:.0f}/s   |   content hangs = many pkts/s but few changes/s and a big max gap\n"
    )

    def line(u: int, s: UniStat) -> str:
        tag = ann.get(u, "")
        return (
            f"  {fmt_uni(u):>9}  {s.pkt_rate:6.1f} pkt/s  {s.change_rate:6.1f} chg/s  "
            f"max-gap {s.max_gap * 1000:7.0f}ms  mean-gap {s.mean_gap * 1000:6.0f}ms  {tag}"
        )

    by_gap = sorted(rows, key=lambda s: s.max_gap, reverse=True)
    uni_of = {id(s): u for u, s in stats.items()}
    print(f"longest value-hangs (top {top}) — a big max-gap at a healthy pkt rate = the content, not the bridge:")
    for s in by_gap[:top]:
        print(line(uni_of[id(s)], s))

    slow = sorted(rows, key=lambda s: s.pkt_rate)
    print(f"\nlowest packet rates (top {top}) — a low rate = the source sends this universe slowly:")
    for s in slow[:top]:
        print(line(uni_of[id(s)], s))


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------


def _selftest() -> int:
    # Build three universes with known behaviour at 30 "frames":
    #   u0 healthy  (changes every frame)
    #   u1 content hang (full packet rate, value changes only every 10th frame)
    #   u2 slow sender  (only every 6th frame sent, and it changes each time)
    pkts = []
    t = 1000.0
    dt = 1 / 30
    for f in range(30):
        ts = t + f * dt
        pkts.append((ts, 0, bytes([f & 0xFF]) * 12))  # u0 changes every frame
        pkts.append((ts, 1, bytes([(f // 10) & 0xFF]) * 12))  # u1 changes every 10 frames
        if f % 6 == 0:
            pkts.append((ts, 2, bytes([f & 0xFF]) * 12))  # u2 sent 1/6 as often
    st = summarize(pkts)
    ok = True

    def check(name, got, want, tol):
        nonlocal ok
        good = abs(got - want) <= tol
        ok = ok and good
        print(f"  [{'ok' if good else 'XX'}] {name}: {got:.2f} (want ~{want} ±{tol})")

    check("u0 pkt/s", st[0].pkt_rate, 30, 2)
    check("u0 chg/s", st[0].change_rate, 30, 2)
    check("u1 pkt/s", st[1].pkt_rate, 30, 2)
    check("u1 chg/s", st[1].change_rate, 3, 1)
    check("u1 max-gap ms", st[1].max_gap * 1000, 333, 60)
    # 5 packets over the 0.8s first->last span reads as 6.25/s; the point is it
    # is far slower than u0/u1 (~31/s), which is how a slow sender shows up.
    check("u2 pkt/s", st[2].pkt_rate, 6.25, 1)
    check("u2 much slower than u0", st[0].pkt_rate - st[2].pkt_rate, 24.8, 3)

    # Round-trip the same packets through a written pcap to exercise the parser.
    import os
    import tempfile

    path = tempfile.mktemp(suffix=".pcap")
    try:
        _write_test_pcap(path, pkts)
        reparsed = summarize(_iter_pcap_dmx(path))
        check("pcap u1 chg/s", reparsed[1].change_rate, 3, 1)
        check("pcap universes", len(reparsed), 3, 0)
    finally:
        if os.path.exists(path):
            os.remove(path)
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def _write_test_pcap(path: str, pkts) -> None:
    def artdmx(uni: int, data: bytes) -> bytes:
        return (
            b"Art-Net\x00"
            + struct.pack("<H", 0x5000)
            + struct.pack(">H", 14)
            + bytes([0, 0])
            + struct.pack("<H", uni)
            + struct.pack(">H", len(data))
            + data
        )

    def udp4(payload: bytes) -> bytes:
        udp = struct.pack(">HHHH", 40000, ARTNET_PORT, 8 + len(payload), 0) + payload
        ip = (
            bytes([0x45, 0x00])  # version 4, IHL 5; tos 0
            + struct.pack(">H", 20 + len(udp))  # total length
            + b"\x00\x00"  # identification
            + b"\x00\x00"  # flags / fragment offset
            + bytes([64, 17])  # ttl 64; protocol 17 = UDP
            + b"\x00\x00"  # header checksum (unchecked)
            + socket.inet_aton("10.0.0.64")  # src
            + socket.inet_aton("10.0.0.104")  # dst
        ) + udp
        eth = b"\x00" * 6 + b"\x00" * 6 + b"\x08\x00" + ip
        return eth

    with open(path, "wb") as fh:
        fh.write(struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1))  # EN10MB
        for ts, uni, data in pkts:
            frame = udp4(artdmx(uni, data))
            sec = int(ts)
            usec = int((ts - sec) * 1e6)
            fh.write(struct.pack("<IIII", sec, usec, len(frame), len(frame)))
            fh.write(frame)


def main(argv: list[str]) -> int:
    args = argv[1:]
    if "--selftest" in args:
        return _selftest()
    top = 40
    config = ""
    sniff = False
    seconds = 6.0
    path = ""
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--top":
            i += 1
            top = int(args[i])
        elif a == "--config":
            i += 1
            config = args[i]
        elif a == "--sniff":
            sniff = True
        elif a == "--seconds":
            i += 1
            seconds = float(args[i])
        elif not a.startswith("-"):
            path = a
        else:
            raise SystemExit(f"unknown option {a!r}")
        i += 1
    ann = _load_annotations(config) if config else {}
    if sniff:
        print(f"sniffing live for {seconds:.0f}s (ctrl-C to stop early)...", file=sys.stderr)
        stats = summarize(_iter_sniff(seconds))
    elif path:
        stats = summarize(_iter_pcap_dmx(path))
    else:
        raise SystemExit(__doc__)
    report(stats, top, ann)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
