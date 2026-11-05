#!/usr/bin/env python3
"""
pro_rt_check.py - does a Twinkly Pro (firmware 4.x, /xled/v2) take realtime frames?

Without strands attached you can't see pixels, so this looks for side effects:
  1. POST application/mode {"mode":"rt"} must be accepted (it is 403 while the
     controller is a group slave).
  2. In rt mode with NO frames, the firmware should fall back to its previous
     mode after a few seconds, like consumer firmware does.
  3. In rt mode WITH frames streaming it should stay in rt mode. If 2 falls
     back and 3 does not, the controller is consuming our UDP frames.
  4. The same with the other fragment layout (300 LEDs per datagram, as the
     ttls library sends) instead of 900-byte fragments (xled-docs).
  5. POST led/rt/frame with a full frame, in rt mode and in the previous mode.
It puts the controller back in the mode and movie it found it in.

Python 3.8+, standard library only. The only writes are application/mode,
led/rt/frame and movies/current (to restore). Nothing persistent is touched.

  python3 tools/pro_rt_check.py 10.201.82.105 [--seconds 20] [--fps 20] [--idle 15]
"""

import argparse
import base64
import json
import os
import socket
import sys
import time
import urllib.error
import urllib.request

RT_PORT = 7777
CODE_OK = 1000


class Http:
    def __init__(self, ip: str, port: int = 80, timeout: float = 3.0) -> None:
        self.ip = ip
        self.base = f"http://{ip}:{port}/xled/v2"
        self.timeout = timeout
        self.token = ""
        self.token_raw = b""

    def call(self, method, path, body=None, auth=True, ctype="application/octet-stream"):
        headers = {}
        data = None
        if isinstance(body, (bytes, bytearray)):
            data = bytes(body)
            headers["Content-Type"] = ctype
        elif body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        if auth and self.token:
            headers["X-Auth-Token"] = self.token
        req = urllib.request.Request(f"{self.base}/{path}", data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()
        except (TimeoutError, urllib.error.URLError, OSError) as e:
            return 0, str(e).encode()

    def json(self, method, path, body=None, auth=True, ctype="application/octet-stream"):
        st, raw = self.call(method, path, body, auth, ctype)
        try:
            j = json.loads(raw) if raw else {}
        except ValueError:
            j = {"_raw": raw[:120].decode(errors="replace")}
        return st, j if isinstance(j, dict) else {"_value": j}


def code_of(j):
    if "code" in j:
        return j["code"]
    result = j.get("result")
    return result.get("code") if isinstance(result, dict) else None


def login(h: Http) -> None:
    st, j = h.json("POST", "login", {"challenge": base64.b64encode(os.urandom(32)).decode()}, auth=False)
    tok = j.get("authentication_token") if st == 200 else None
    if not tok:
        sys.exit(f"login failed: HTTP {st} {j}")
    h.token = tok
    h.token_raw = base64.b64decode(tok)
    role = j.get("auth-role", "?")
    st, j = h.json("POST", "verify", {"challenge-response": j.get("challenge-response", "")})
    if st != 200 or code_of(j) != CODE_OK:
        sys.exit(f"verify failed: HTTP {st} {j}")
    print(f"  login ok (role {role}, {len(h.token_raw)}-byte token)")


def mode(h: Http) -> str:
    st, j = h.json("GET", "application/mode")
    return str(j.get("mode", f"HTTP {st}"))


def set_mode(h: Http, m: str) -> bool:
    st, j = h.json("POST", "application/mode", {"mode": m})
    ok = st == 200 and code_of(j) == CODE_OK
    print(f"  POST application/mode {m!r} -> HTTP {st} code={code_of(j)} {'ok' if ok else 'REFUSED'}")
    return ok


def frame_rate(h: Http) -> str:
    st, j = h.json("GET", "gestalt", auth=False)
    return str(j.get("frame_rate", "?"))


class Streamer:
    """Sends UDP v3 frames: 0x03 + 8 token bytes + 0x00 0x00 + fragment# + pixel bytes."""

    def __init__(self, ip: str, port: int, token_raw: bytes, bytes_per_packet: int) -> None:
        self.addr = (ip, port)
        self.token = token_raw
        self.chunk = bytes_per_packet
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sent = 0

    def send(self, frame: bytes) -> None:
        for i, start in enumerate(range(0, len(frame), self.chunk)):
            self.sock.sendto(
                b"\x03" + self.token + b"\x00\x00" + bytes([i & 0xFF]) + frame[start : start + self.chunk], self.addr
            )
        self.sent += 1


def idle_watch(h: Http, seconds: float):
    """Poll the mode with no frames going out. Returns (mode it ended in, seconds it stayed rt)."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        m = mode(h)
        if m != "rt":
            return m, time.monotonic() - t0
        time.sleep(0.5)
    return "rt", seconds


def stream_and_watch(h: Http, s: Streamer, frame: bytes, seconds: float, fps: float, label: str) -> bool:
    fragments = -(-len(frame) // s.chunk)
    print(
        f"\n-- {label}: {len(frame)} B frames in {fragments} fragments of <= {s.chunk} B at {fps:g} fps for {seconds:g} s"
    )
    period = 1.0 / fps
    t0 = time.monotonic()
    nxt = t0
    next_poll = t0 + 2.0
    seen = []
    while time.monotonic() - t0 < seconds:
        s.send(frame)
        now = time.monotonic()
        if now >= next_poll:
            seen.append((round(now - t0), mode(h)))
            next_poll += 2.0
        nxt += period
        delay = nxt - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        else:
            nxt = time.monotonic()
    stayed = bool(seen) and all(m == "rt" for _, m in seen)
    print(f"  sent {s.sent} frames; mode while streaming: " + ", ".join(f"{t}s={m}" for t, m in seen))
    print(f"  gestalt frame_rate now: {frame_rate(h)}; stayed in rt: {stayed}")
    return stayed


def rt_frame_http(h: Http, frame: bytes, label: str) -> bool:
    t0 = time.perf_counter()
    st, j = h.json("POST", "led/rt/frame", frame)
    ms = (time.perf_counter() - t0) * 1000
    ok = st == 200 and code_of(j) == CODE_OK
    print(
        f"  POST led/rt/frame {len(frame)} B {label} -> HTTP {st} code={code_of(j)} ({ms:.0f} ms){' ACCEPTED' if ok else ''}"
    )
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ip")
    ap.add_argument("--seconds", type=float, default=20, help="how long to stream in each test (default 20)")
    ap.add_argument("--fps", type=float, default=20)
    ap.add_argument("--idle", type=float, default=15, help="how long to wait for rt to time out with no frames")
    ap.add_argument("--http-port", type=int, default=80, help=argparse.SUPPRESS)
    ap.add_argument("--udp-port", type=int, default=RT_PORT, help=argparse.SUPPRESS)
    a = ap.parse_args()

    h = Http(a.ip, a.http_port)
    st, g = h.json("GET", "gestalt", auth=False)
    if st != 200:
        sys.exit(f"GET /xled/v2/gestalt -> HTTP {st} {g}")
    cfg = g.get("device_config") if isinstance(g.get("device_config"), dict) else {}
    profile = str(g.get("led_profile") or cfg.get("led_profile") or "RGB").upper()
    n = int(g.get("number_of_led") or 0)
    bpl = 4 if profile == "RGBW" else 3
    group = g.get("group") if isinstance(g.get("group"), dict) else {}
    print(f"== {g.get('device_name', '?')} @ {a.ip}: {n} x {profile}, group mode {group.get('mode', 'n/a')!r} ==")
    if group.get("mode") == "slave":
        sys.exit("still a group slave (gestalt.group.mode == 'slave'); the firmware will refuse rt mode")
    if n <= 0:
        sys.exit("controller reports 0 LEDs")
    login(h)
    prev = mode(h)
    st, cur = h.json("GET", "movies/current")
    cur_uid = cur.get("unique_id") if st == 200 else None
    print(f"  mode now {prev!r}, current movie {cur.get('name', '?')!r}, gestalt frame_rate {frame_rate(h)}")

    frame = bytes([0, 0, 8, 0] if bpl == 4 else [0, 8, 0]) * n  # dim green-ish whatever the byte order
    results = {}

    print("\n-- 1: arm rt mode")
    if not set_mode(h, "rt"):
        sys.exit("rt mode refused; nothing more to test")
    results["rt accepted"] = True

    print(f"\n-- 2: no frames for up to {a.idle:g} s; does rt time out?")
    m, t = idle_watch(h, a.idle)
    if m == "rt":
        print(f"  still in rt after {a.idle:g} s with no frames (no idle timeout; test 3 is weaker evidence then)")
    else:
        print(f"  fell back to {m!r} after {t:.1f} s with no frames")
    results["idle timeout"] = f"{m} after {t:.1f}s"

    set_mode(h, "rt")
    results["stays rt, 900 B fragments"] = stream_and_watch(
        h, Streamer(a.ip, a.udp_port, h.token_raw, 900), frame, a.seconds, a.fps, "3: 900-byte fragments (xled-docs)"
    )
    if mode(h) != "rt":
        set_mode(h, "rt")
    results["stays rt, 300 LED fragments"] = stream_and_watch(
        h, Streamer(a.ip, a.udp_port, h.token_raw, 300 * bpl), frame, a.seconds, a.fps, "4: 300-LED fragments (ttls)"
    )

    print("\n-- 5: led/rt/frame over HTTP")
    if mode(h) != "rt":
        set_mode(h, "rt")
    results["led/rt/frame in rt"] = rt_frame_http(h, frame, "in rt mode")
    print(f"  mode after: {mode(h)}")
    set_mode(h, prev if prev not in ("", "rt") else "movie")
    results[f"led/rt/frame in {prev}"] = rt_frame_http(h, frame, f"in {prev} mode")
    print(f"  mode after: {mode(h)}")

    print("\n-- restore")
    set_mode(h, prev if prev not in ("", "rt") else "movie")
    if cur_uid:
        st, j = h.json("POST", "movies/current", {"unique_id": cur_uid})
        print(f"  POST movies/current -> HTTP {st} code={code_of(j)}")
    st, cur2 = h.json("GET", "movies/current")
    print(f"  mode now {mode(h)!r}, current movie {cur2.get('name', '?')!r}")

    print("\n== RESULTS ==")
    for k, v in results.items():
        print(f"  {k:<30}: {v}")
    udp = results["stays rt, 900 B fragments"] or results["stays rt, 300 LED fragments"]
    if udp and results["idle timeout"].startswith("rt "):
        print("  verdict: rt stays up while we stream but never times out on its own, so this is only weak evidence")
    elif udp:
        print("  verdict: the controller consumes our UDP frames (rt times out idle, stays up while streaming)")
    else:
        print("  verdict: rt mode drops even while streaming; the UDP frames are not being accepted")


if __name__ == "__main__":
    main()
