#!/usr/bin/env python3
"""
twinkly_probe.py - Step 1 test: can we drive a Twinkly controller pixel-by-pixel?

Uses Twinkly's local (unofficial, community-documented) API:
  HTTP  :80   /xled/v1/...   login, device info, set mode "rt"
              /xled/v2/...   the same on Twinkly Pro firmware 4.x (found automatically)
  UDP   :7777                realtime pixel frames

Python 3.8+ standard library only. No installs needed.

Usage:
  python3 twinkly_probe.py info  <controller-ip>
  python3 twinkly_probe.py test  <controller-ip> [--fps 40] [--order WRGB]

Reference: https://xled-docs.readthedocs.io/en/latest/protocol_details.html
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

UDP_PORT = 7777
FRAGMENT_BYTES = 900  # max pixel bytes per UDP datagram (protocol v3)


# ----------------------------------------------------------------- HTTP API --
class NotFound(RuntimeError):
    """The endpoint does not exist on this firmware."""


def code(reply):
    """v1 replies carry {"code": 1000}; v2 (Pro firmware 4.x) carry {"result": {"code": 1000}}."""
    if "code" in reply:
        return reply["code"]
    result = reply.get("result")
    return result.get("code") if isinstance(result, dict) else None


class Twinkly:
    def __init__(self, ip, http_port=80, udp_port=UDP_PORT, timeout=3):
        self.ip = ip
        self.root = f"http://{ip}:{http_port}/xled"
        self.api = None  # "v1" (consumer, older Pro) or "v2" (Pro firmware 4.x); found by detect()
        self.udp_addr = (ip, udp_port)
        self.timeout = timeout
        self.token = None  # base64 string, used in HTTP header
        self.token_raw = None  # 8 raw bytes, used in UDP packets
        self.gestalt = None

    @property
    def base(self):
        return f"{self.root}/{self.api or 'v1'}"

    def _req(self, method, path, body=None, auth=True):
        headers = {"Content-Type": "application/json"}
        if auth and self.token:
            headers["X-Auth-Token"] = self.token
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                raw = r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise NotFound(f"{method} {path} -> not on this firmware (HTTP 404)") from None
            raise RuntimeError(f"{method} {path} -> HTTP {e.code}: {e.read()[:200]!r}") from None
        except (TimeoutError, urllib.error.URLError, OSError) as e:
            raise RuntimeError(f"{method} {path} -> can't reach {self.ip} ({e})") from None
        try:
            return json.loads(raw) if raw else {}
        except ValueError:
            return {"_raw": raw[:200]}

    def get(self, path, auth=True):
        return self._req("GET", path, auth=auth)

    def post(self, path, body, auth=True):
        return self._req("POST", path, body, auth=auth)

    def detect(self):
        """Find which API generation answers; returns the gestalt."""
        if self.gestalt is not None:
            return self.gestalt
        for api in ("v1", "v2"):
            self.api = api
            try:
                self.gestalt = self.get("/gestalt", auth=False)
                return self.gestalt
            except NotFound:
                continue
        self.api = None
        raise RuntimeError(f"{self.ip} answers HTTP but has no /xled/v1 or /xled/v2 gestalt; is it a Twinkly?")

    # Endpoint names that differ between the two generations. v2 was verified
    # on a TWPROCTRLPLC21 with firmware 4.0.1.
    @property
    def mode_path(self):
        return "/application/mode" if self.api == "v2" else "/led/mode"

    @property
    def fw_path(self):
        return "/fw/ct1/version" if self.api == "v2" else "/fw/version"

    def login(self):
        self.detect()
        challenge = base64.b64encode(os.urandom(32)).decode()
        r = self.post("/login", {"challenge": challenge}, auth=False)
        if "authentication_token" not in r:
            raise RuntimeError(f"login failed: {r}")
        self.token = r["authentication_token"]
        self.token_raw = base64.b64decode(self.token)
        v = self.post("/verify", {"challenge-response": r.get("challenge-response", "")})
        if code(v) != 1000:
            raise RuntimeError(f"verify failed: {v}")

    def get_mode(self):
        """Current mode, or "" when the firmware doesn't expose it."""
        try:
            return str(self.get(self.mode_path).get("mode", ""))
        except NotFound:
            return ""

    def set_mode(self, mode):
        try:
            r = self.post(self.mode_path, {"mode": mode})
        except RuntimeError as e:
            if "HTTP 403" in str(e):
                raise RuntimeError(
                    f"set mode {mode!r} refused (403). On Pro firmware this means the controller is a group "
                    "slave; take it out of its group in the Twinkly Pro app and run this again"
                ) from None
            raise
        if code(r) != 1000:
            raise RuntimeError(f"set mode {mode!r} failed: {r}")


def device_config(g):
    cfg = g.get("device_config")
    return cfg if isinstance(cfg, dict) else {}


def profile_of(g):
    """led_profile is top-level on v1 and under device_config on Pro v2."""
    return str(g.get("led_profile") or device_config(g).get("led_profile") or "RGB").upper()


def group_of(g):
    """(mode, group dict). Pro v2 only; mode is "" elsewhere."""
    group = g.get("group") if isinstance(g.get("group"), dict) else {}
    return str(group.get("mode", "")), group


# ------------------------------------------------------------- UDP frames ---
def version_tuple(v):
    out = []
    for part in str(v).split("."):
        digits = "".join(ch for ch in part if ch.isdigit())
        out.append(int(digits) if digits else 0)
    return tuple(out)


def pick_protocol(fw_version):
    """v3 (fragmented) for fw >= 2.4.14, v2 for older gen II, else v1."""
    if not any(ch.isdigit() for ch in str(fw_version)):
        return 3  # firmware that hides its version is new enough for v3
    v = version_tuple(fw_version)
    if v >= (2, 4, 14):
        return 3
    if v >= (2, 0, 0):
        return 2
    return 1


def build_packets(proto, token_raw, frame, n_leds):
    if proto == 3:
        pkts = []
        for i, start in enumerate(range(0, len(frame), FRAGMENT_BYTES)):
            hdr = b"\x03" + token_raw + b"\x00\x00" + bytes([i & 0xFF])
            pkts.append(hdr + frame[start : start + FRAGMENT_BYTES])
        return pkts
    if proto == 2:
        return [b"\x02" + token_raw + b"\x00" + frame]
    return [b"\x01" + token_raw + bytes([min(n_leds, 255)]) + frame]


class Streamer:
    """Turns a list of (r, g, b, w) pixels into Twinkly UDP frames."""

    def __init__(self, dev, n_leds, profile, proto, order=None):
        self.dev, self.n, self.proto = dev, n_leds, proto
        # Twinkly docs: RGB -> R,G,B ; RGBW -> W,R,G,B (white first)
        self.order = (order or ("WRGB" if profile == "RGBW" else "RGB")).upper()
        if sorted(self.order) not in (sorted("RGB"), sorted("RGBW")):
            raise ValueError(f"bad --order {order!r} (use e.g. WRGB, RGBW, GRB)")
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.frames_sent = 0

    def encode(self, pixels):
        idx = {"R": 0, "G": 1, "B": 2, "W": 3}
        out = bytearray()
        for p in pixels:
            for ch in self.order:
                out.append(p[idx[ch]] & 0xFF)
        return bytes(out)

    def send(self, pixels):
        frame = self.encode(pixels)
        for pkt in build_packets(self.proto, self.dev.token_raw, frame, self.n):
            self.sock.sendto(pkt, self.dev.udp_addr)
        self.frames_sent += 1

    def hold(self, pixels_fn, seconds, fps):
        """Send frames for `seconds` at `fps`. pixels_fn(t) -> pixel list."""
        period = 1.0 / fps
        t0 = time.monotonic()
        nxt = t0
        while True:
            now = time.monotonic()
            if now - t0 >= seconds:
                break
            self.send(pixels_fn(now - t0))
            nxt += period
            delay = nxt - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                nxt = time.monotonic()  # fell behind; don't spiral


# ------------------------------------------------------------------ helpers --
def hsv(h):
    """Hue 0..1 -> (r, g, b, 0) at full sat/value."""
    h = (h % 1.0) * 6
    i, f = int(h), h - int(h)
    q, t = int(255 * (1 - f)), int(255 * f)
    return [(255, t, 0, 0), (q, 255, 0, 0), (0, 255, t, 0), (0, q, 255, 0), (t, 0, 255, 0), (255, 0, q, 0)][i % 6]


def show_info(dev):
    g = dev.detect()
    try:
        fw = str(dev.get(dev.fw_path, auth=False).get("version", "?"))
    except NotFound:
        fw = "n/a"
    flat = {**device_config(g), **g}  # Pro v2 keeps led_profile etc. under device_config
    print(f"\n== Twinkly @ {dev.ip} (API {dev.api}) ==")
    print(f"  firmware        : {fw}")
    for k in (
        "product_code",
        "device_name",
        "fw_family",
        "hardware_version",
        "led_profile",
        "led_drv",
        "bytes_per_led",
        "number_of_led",
        "max_supported_led",
        "frame_rate",
        "measured_frame_rate",
        "wire_type",
        "mac",
        "artnet_en",
    ):
        if k in flat:
            print(f"  {k:<16}: {flat[k]}")
    group_mode, group = group_of(g)
    if group_mode:
        print(f"  group           : {group_mode} (size {group.get('size')}, offset {group.get('offset')})")
    if group_mode == "slave":
        print("  !! group slave  : Pro firmware refuses realtime until the controller leaves its group")
    try:
        dev.login()
        print(f"  current mode    : {dev.get_mode() or 'n/a'}")
        if dev.api == "v1":
            strings = dev.get("/led/config").get("strings", [])
            if strings:
                print(f"  outputs/strings : {len(strings)}")
                for i, st in enumerate(strings):
                    print(f"    port {i + 1}: first_led={st.get('first_led_id')} length={st.get('length')}")
        else:
            ports = device_config(g).get("ports", [])
            if ports:
                print(f"  ports           : {len(ports)}")
                for port in ports:
                    runs = ", ".join(f"start={st.get('start')} len={st.get('len')}" for st in port.get("strings", []))
                    print(f"    port {port.get('port_id')}: {runs}")
    except RuntimeError as e:
        print(f"  (login/extra info failed: {e})")
    print("\n  raw gestalt (paste this back if something looks off):")
    print("  " + json.dumps(g))
    return fw, g


def ask(prompt):
    try:
        return input(prompt).strip()
    except EOFError:
        return ""


# --------------------------------------------------------------------- test --
def run_test(dev, fps, order, proto_override, auto_yes):
    fw, g = show_info(dev)
    n = int(g.get("number_of_led", 0))
    profile = profile_of(g)
    if n <= 0:
        sys.exit("Controller reports 0 LEDs - check it in the Twinkly app.")
    if group_of(g)[0] == "slave":
        sys.exit("Controller is a group slave; Pro firmware refuses realtime. Take it out of its group first.")
    proto = proto_override or pick_protocol(fw)
    s = Streamer(dev, n, profile, proto, order)
    print(f"\n  streaming {n} px, profile {profile}, byte order {s.order}, UDP protocol v{proto}, target {fps} fps")

    prev_mode = dev.get_mode() or "movie"
    dev.set_mode("rt")
    results = {}
    black = [(0, 0, 0, 0)] * n

    def step(name, question, fn, secs):
        print(f"\n-- {name}: {question}")
        # rt mode drops back to movie after a few idle seconds (e.g. while
        # you're typing an answer), so re-arm it before every step
        dev.set_mode("rt")
        s.hold(fn, secs, fps)
        s.send(black)
        if not auto_yes:
            results[name] = ask("   what did you see? (y = as expected, or describe it): ") or "(no answer)"
        else:
            results[name] = "(auto)"

    try:
        # 1. color order: whole strand one channel at a time
        for label, px in (
            ("RED", (255, 0, 0, 0)),
            ("GREEN", (0, 255, 0, 0)),
            ("BLUE", (0, 0, 255, 0)),
            ("WHITE chip", (0, 0, 0, 255)),
        ):
            step(f"color {label}", f"whole strand should be {label}", lambda t, px=px: [px] * n, 3)

        # 2. per-pixel addressing: first pixel, last pixel, walking dot
        step(
            "first pixel",
            "ONLY the pixel nearest the controller lit (green)",
            lambda t: [(0, 255, 0, 0)] + [(0, 0, 0, 0)] * (n - 1),
            3,
        )
        step(
            "last pixel", "ONLY the far-end pixel lit (blue)", lambda t: [(0, 0, 0, 0)] * (n - 1) + [(0, 0, 255, 0)], 3
        )
        chase_secs = min(max(n / fps, 4), 20)
        step(
            "chase",
            "a single red dot walks from controller end to far end",
            lambda t: [(255, 0, 0, 0) if i == int(t / chase_secs * n) else (0, 0, 0, 0) for i in range(n)],
            chase_secs,
        )

        # 3. smoothness / frame rate: scrolling rainbow
        print("\n-- rainbow: scrolling rainbow for 10 s (watch for stutter)")
        dev.set_mode("rt")
        t0, f0 = time.monotonic(), s.frames_sent
        s.hold(lambda t: [hsv(i / max(n, 1) + t * 0.5) for i in range(n)], 10, fps)
        sent_fps = (s.frames_sent - f0) / (time.monotonic() - t0)
        s.send(black)
        try:
            g2 = dev.get("/gestalt", auth=False)
            measured = g2.get("measured_frame_rate", "n/a")
        except RuntimeError:
            measured = "n/a"
        results["rainbow"] = f"sent {sent_fps:.1f} fps; controller reports measured_frame_rate={measured}"
        if not auto_yes:
            results["rainbow"] += " | you saw: " + (ask("   smooth, or stuttery/flickery? ") or "(no answer)")
    except KeyboardInterrupt:
        print("\n  stopped early")
    finally:
        try:
            dev.set_mode(prev_mode)
            print(f"\n  restored mode: {prev_mode}")
        except RuntimeError as e:
            print(f"\n  couldn't restore mode ({e}) - pick a movie in the app")

    print("\n== RESULTS (copy/paste this whole block back to Claude) ==")
    print(
        f"ip={dev.ip} fw={fw} product={g.get('product_code')} "
        f"profile={profile} leds={n} proto=v{proto} order={s.order} fps={fps}"
    )
    for k, v in results.items():
        print(f"  {k:<14}: {v}")


# --------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["info", "test"])
    ap.add_argument("ip")
    ap.add_argument("--fps", type=float, default=40, help="frames per second to send (default 40)")
    ap.add_argument(
        "--order",
        default=None,
        help="byte order override, e.g. RGBW or GRBW (default: WRGB for RGBW strands, per Twinkly docs)",
    )
    ap.add_argument(
        "--proto",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="force UDP protocol version (default: auto from firmware)",
    )
    ap.add_argument("--http-port", type=int, default=80, help=argparse.SUPPRESS)
    ap.add_argument("--udp-port", type=int, default=UDP_PORT, help=argparse.SUPPRESS)
    ap.add_argument("--yes", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()

    dev = Twinkly(a.ip, a.http_port, a.udp_port)
    try:
        if a.command == "info":
            show_info(dev)
        else:
            run_test(dev, a.fps, a.order, a.proto, a.yes)
    except RuntimeError as e:
        sys.exit(f"\nERROR: {e}")


if __name__ == "__main__":
    main()
