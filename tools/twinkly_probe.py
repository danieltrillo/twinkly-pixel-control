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
  python3 twinkly_probe.py test  <controller-ip> [--fps 40] [--order WRGB] [--yes --hold 6]
  python3 twinkly_probe.py locate <controller-ip> [--seconds 60] [--hz 4] [--width 1] [--which both]
  python3 twinkly_probe.py chunks <controller-ip> [--hold 8]     # which output is which
  python3 twinkly_probe.py scan  <controller-ip> [--slot 0.4] [--brightness 96] [--repeat 2]
                                                  # blink code for camera mapping (tools/pixelmap.py)
  python3 twinkly_probe.py artnet <controller-ip> [--off | --on]   # the built-in Art-Net mode (Pro)

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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from twinklybridge import mapcode  # noqa: E402  (stdlib only, lives in this repo)

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
    login_error = None
    try:
        dev.login()  # before fw: Pro firmware 4.x answers fw/ct1/version only with a token
    except RuntimeError as e:
        login_error = e
    try:
        fw = str(dev.get(dev.fw_path).get("version", "?"))
    except NotFound:
        fw = "n/a"
    except RuntimeError as e:
        fw = f"n/a ({e})"
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
        if login_error is not None:
            raise login_error
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
def run_test(dev, fps, order, proto_override, auto_yes, hold=3.0):
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
        print(f"\n-- {time.strftime('%H:%M:%S')} {name} ({secs:g} s): {question}", flush=True)
        # rt mode drops back to movie after a few idle seconds (e.g. while
        # you're typing an answer), so re-arm it before every step
        dev.set_mode("rt")
        s.hold(fn, secs, fps)
        s.send(black)
        if not auto_yes:
            results[name] = ask("   what did you see? (y = as expected, or describe it): ") or "(no answer)"
        else:
            results[name] = "(auto)"
            time.sleep(1.0)  # a second of black so the steps are easy to tell apart

    try:
        # 1. color order: whole strand one channel at a time
        for label, px in (
            ("RED", (255, 0, 0, 0)),
            ("GREEN", (0, 255, 0, 0)),
            ("BLUE", (0, 0, 255, 0)),
            ("WHITE chip", (0, 0, 0, 255)),
        ):
            step(f"color {label}", f"whole strand should be {label}", lambda t, px=px: [px] * n, hold)

        # 2. per-pixel addressing: first pixel, last pixel, walking dot
        step(
            "first pixel",
            "ONLY the pixel nearest the controller lit (green)",
            lambda t: [(0, 255, 0, 0)] + [(0, 0, 0, 0)] * (n - 1),
            hold,
        )
        step(
            "last pixel",
            "ONLY the far-end pixel lit (blue)",
            lambda t: [(0, 0, 0, 0)] * (n - 1) + [(0, 0, 255, 0)],
            hold,
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


# ------------------------------------------------------------------- chunks --
OUTPUT_COLORS = [
    ("RED", (255, 0, 0, 0)),
    ("GREEN", (0, 255, 0, 0)),
    ("BLUE", (0, 0, 255, 0)),
    ("YELLOW", (255, 255, 0, 0)),
    ("CYAN", (0, 255, 255, 0)),
    ("MAGENTA", (255, 0, 255, 0)),
    ("WHITE chip", (0, 0, 0, 255)),
    ("ORANGE", (255, 80, 0, 0)),
]


def output_ranges(dev, g, n):
    """Pixel ranges per string, as (first, end, label) 0-based half-open, in output order.

    An output with a splitter and two strings in its layout gets two ranges."""
    ranges = []
    offset = 0
    for k, port in enumerate(device_config(g).get("ports", [])):
        # Pro v2: each port's strings start at 1 again (start is 1-based *within the port*);
        # the frame itself runs port by port, so lay the ports out one after another.
        strings = [st for st in port.get("strings", []) if int(st.get("len", 0)) > 0]
        port_len = 0
        for j, st in enumerate(strings):
            start, length = max(int(st.get("start", 1)) - 1, 0), int(st.get("len", 0))
            label = f"output {k + 1}" + (f" string {j + 1}" if len(strings) > 1 else "")
            ranges.append((offset + start, offset + start + length, label))
            port_len = max(port_len, start + length)
        offset += port_len
    if not ranges and dev.api == "v1":
        try:
            for k, st in enumerate(dev.get("/led/config").get("strings", [])):
                first, length = int(st.get("first_led_id", 0)), int(st.get("length", 0))
                if length > 0:
                    ranges.append((first, first + length, f"output {k + 1}"))
        except RuntimeError:
            pass
    ranges = [(a, min(b, n), label) for a, b, label in ranges if a < n]
    return ranges or [(0, n, "all pixels")]


def run_chunks(dev, fps, order, proto_override, hold):
    """Light every output in its own colour, then walk a white dot through each output in turn."""
    fw, g = show_info(dev)
    n = int(g.get("number_of_led", 0))
    profile = profile_of(g)
    if n <= 0:
        sys.exit("Controller reports 0 LEDs - check it in the Twinkly app.")
    if group_of(g)[0] == "slave":
        sys.exit("Controller is a group slave; Pro firmware refuses realtime. Take it out of its group first.")
    proto = proto_override or pick_protocol(fw)
    s = Streamer(dev, n, profile, proto, order)
    ranges = output_ranges(dev, g, n)
    black = [(0, 0, 0, 0)] * n
    print(f"\n  {len(ranges)} string(s):")
    for k, (a, b, label) in enumerate(ranges):
        print(f"    {label}: pixels {a + 1}-{b} ({b - a} px) = {OUTPUT_COLORS[k % len(OUTPUT_COLORS)][0]}")

    def all_outputs(t):
        px = list(black)
        for k, (a, b, _label) in enumerate(ranges):
            px[a:b] = [OUTPUT_COLORS[k % len(OUTPUT_COLORS)][1]] * (b - a)
        return px

    def one_output(k, secs):
        a, b, _label = ranges[k]
        name, color = OUTPUT_COLORS[k % len(OUTPUT_COLORS)]
        dim = tuple(c // 6 for c in color)

        def fn(t):
            px = list(black)
            px[a:b] = [dim] * (b - a)
            i = a + min(int(t / secs * (b - a)), b - a - 1)
            px[i] = (255, 255, 255, 255) if profile == "RGBW" else (255, 255, 255, 0)
            return px

        return fn

    prev_mode = dev.get_mode() or "movie"
    try:
        dev.set_mode("rt")
        print(f"\n-- {time.strftime('%H:%M:%S')} all outputs at once, each its own colour ({hold:g} s)", flush=True)
        s.hold(all_outputs, hold, fps)
        s.send(black)
        time.sleep(1.0)
        for k, (a, b, label) in enumerate(ranges):
            name = OUTPUT_COLORS[k % len(OUTPUT_COLORS)][0]
            print(
                f"-- {time.strftime('%H:%M:%S')} {label} only, dim {name}; a white dot walks pixel {a + 1} -> {b} "
                f"({hold:g} s)",
                flush=True,
            )
            dev.set_mode("rt")
            s.hold(one_output(k, hold), hold, fps)
            s.send(black)
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n  stopped early")
    finally:
        try:
            dev.set_mode(prev_mode)
            print(f"\n  restored mode: {prev_mode}")
        except RuntimeError as e:
            print(f"\n  couldn't restore mode ({e}) - pick a movie in the app")


# ------------------------------------------------------------------- locate --
def run_locate(dev, fps, order, proto_override, seconds, hz, width, which):
    """Strobe the first pixel(s) red and the last pixel(s) blue so you can find the ends of the run."""
    fw, g = show_info(dev)
    n = int(g.get("number_of_led", 0))
    profile = profile_of(g)
    if n <= 0:
        sys.exit("Controller reports 0 LEDs - check it in the Twinkly app.")
    if group_of(g)[0] == "slave":
        sys.exit("Controller is a group slave; Pro firmware refuses realtime. Take it out of its group first.")
    proto = proto_override or pick_protocol(fw)
    s = Streamer(dev, n, profile, proto, order)
    width = max(1, min(int(width), n))
    black = [(0, 0, 0, 0)] * n
    parts = []
    if which in ("first", "both"):
        parts.append(f"first {width} px RED (pixel 1, controller end of the first output)")
    if which in ("last", "both"):
        parts.append(f"last {width} px BLUE (pixel {n}, far end of the last output)")
    print(
        f"\n-- {time.strftime('%H:%M:%S')} locate: {' and '.join(parts)}, strobing {hz:g} Hz for {seconds:g} s",
        flush=True,
    )

    def pixels(t):
        px = list(black)
        if int(t * hz * 2) % 2 == 0:  # on half the time
            if which in ("first", "both"):
                px[:width] = [(255, 0, 0, 0)] * width
            if which in ("last", "both"):
                px[n - width :] = [(0, 0, 255, 0)] * width
        return px

    prev_mode = dev.get_mode() or "movie"
    try:
        dev.set_mode("rt")
        s.hold(pixels, seconds, fps)
        s.send(black)
    except KeyboardInterrupt:
        print("\n  stopped early")
    finally:
        try:
            dev.set_mode(prev_mode)
            print(f"\n  restored mode: {prev_mode}")
        except RuntimeError as e:
            print(f"\n  couldn't restore mode ({e}) - pick a movie in the app")


def run_scan(dev, fps, order, proto_override, slot, brightness, color, repeat, countdown):
    """Play the camera-mapping blink code (twinklybridge/mapcode.py) on every pixel."""
    fw, g = show_info(dev)
    n = int(g.get("number_of_led", 0))
    profile = profile_of(g)
    if n <= 0:
        sys.exit("Controller reports 0 LEDs - check it in the Twinkly app.")
    if group_of(g)[0] == "slave":
        sys.exit("Controller is a group slave; Pro firmware refuses realtime. Take it out of its group first.")
    proto = proto_override or pick_protocol(fw)
    s = Streamer(dev, n, profile, proto, order)
    v = max(1, min(int(brightness), 255))
    if color == "white" and profile == "RGBW":
        lit = (0, 0, 0, v)
    elif color == "green":
        lit = (0, v, 0, 0)
    else:
        lit = (v, v, v, 0)
    black_px = (0, 0, 0, 0)
    slots = mapcode.schedule(n)
    words = [mapcode.codeword(i, n) for i in range(n)]
    frames = [[lit if on else black_px for on in mapcode.lit_mask(sl, n, words)] for sl in slots]
    total = len(slots) * slot
    print(
        f"\n-- scan: {n} px, {mapcode.word_bits(n)} bits, {len(slots)} slots x {slot:g} s = {total:.1f} s"
        f" per pass, {repeat} pass(es), colour {color} at {v}",
        flush=True,
    )
    print(f"   decode with: python3 tools/pixelmap.py decode VIDEO --pixels {n} --slot {slot:g}", flush=True)

    prev_mode = dev.get_mode() or "movie"
    try:
        dev.set_mode("rt")
        for left in range(int(countdown), 0, -1):
            print(f"   recording? scan starts in {left} s", flush=True)
            s.hold(lambda t: [black_px] * n, 1.0, fps)
        for p in range(repeat):
            print(f"   pass {p + 1}/{repeat} ({time.strftime('%H:%M:%S')})", flush=True)
            s.hold(lambda t: frames[min(int(t / slot), len(frames) - 1)], total, fps)
        s.send([black_px] * n)
        print("   done - stop recording", flush=True)
    except KeyboardInterrupt:
        print("\n  stopped early")
    finally:
        try:
            dev.set_mode(prev_mode)
            print(f"\n  restored mode: {prev_mode}")
        except RuntimeError as e:
            print(f"\n  couldn't restore mode ({e}) - pick a movie in the app")


# ------------------------------------------------------------------- artnet --
def run_artnet(dev, switch, explore=False):
    """Show the built-in Art-Net setting; with --off/--on try the requests the firmware may take.

    Every request only ever carries the current settings with "enable" changed, so a
    rejected one changes nothing. --explore also probes nearby routes and methods."""
    g = dev.detect()
    dev.login()
    print(f"  {g.get('device_name', '?')} ({dev.api}), gestalt artnet_en = {g.get('artnet_en')!r}")
    try:
        cur = dev.get("/artnet/config")
    except NotFound:
        sys.exit("  no /artnet/config on this firmware")
    print(f"  GET artnet/config -> {json.dumps(cur)}")
    if switch is None:
        return
    want = switch == "on"
    settings = {k: v for k, v in cur.items() if k not in ("code", "result")}
    attempts = [
        ("POST", "/artnet/config", {"enable": want}),
        ("POST", "/artnet/config", {"enable": int(want)}),
        ("POST", "/artnet/config", {**settings, "enable": want}),
        ("POST", "/artnet/config", {**settings, "enable": int(want)}),
    ]
    if explore:
        word = "on" if want else "off"
        attempts += [
            ("POST", "/artnet/config", {"enable": word}),
            ("POST", "/artnet/config", {"enable": str(want).lower()}),
            ("POST", "/artnet/config", {"mode": word}),
            ("POST", "/artnet/config", {"artnet": {**settings, "enable": want}}),
            ("POST", "/artnet/config", {"config": {**settings, "enable": want}}),
            ("POST", "/artnet/config", {"enable": want, "model": str(settings.get("model", 0))}),
            ("PUT", "/artnet/config", {"enable": want}),
            ("PUT", "/artnet/config", {**settings, "enable": want}),
            ("PATCH", "/artnet/config", {"enable": want}),
            ("POST", "/artnet/enable", {"enable": want}),
            ("POST", "/artnet/enable", {"value": want}),
            ("POST", "/artnet", {"enable": want}),
            ("POST", "/artnet", {**settings, "enable": want}),
            ("POST", "/artnet/config/enable", {"enable": want}),
            ("POST", "/artnet/mode", {"mode": word}),
            ("POST", "/artnet/mode", {"enable": want}),
            ("POST", "/dmx/config", {**settings, "enable": want}),
            ("POST", "/dmx/config", {"enable": want}),
            ("POST", "/dmx", {"enable": want}),
            ("POST", "/device/config", {"artnet_en": want}),
            ("POST", "/artnet/en", {"enable": want}),
            ("POST", "/artnet/enabled", {"enabled": want}),
        ]
    seen = set()
    for method, path, body in attempts:
        key = (method, path, json.dumps(body, sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        label = f"{method} {path.lstrip('/')} {json.dumps(body)}"
        try:
            reply = dev._req(method, path, body)
        except NotFound:
            print(f"  {label} -> 404 (no such route)")
            continue
        except RuntimeError as e:
            print(f"  {label} -> {str(e).split('->', 1)[-1].strip()}")
            continue
        print(f"  {label} -> {json.dumps(reply)}")
        after = dev.get("/artnet/config")
        g2 = dev.get("/gestalt", auth=False)
        print(f"     now: artnet/config {json.dumps(after)}; gestalt artnet_en = {g2.get('artnet_en')!r}")
        if bool(after.get("enable")) == want and bool(g2.get("artnet_en")) == want:
            print(f"  OK: built-in Art-Net is now {'on' if want else 'off'} (request that worked: {label})")
            return
        print("     accepted, but the setting did not change; trying the next one")
    sys.exit("  nothing worked; see the replies above")


# --------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["info", "test", "locate", "chunks", "scan", "artnet"])
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
    ap.add_argument("--yes", action="store_true", help="don't ask after each step; just run the sequence")
    ap.add_argument("--hold", type=float, default=3, help="seconds to hold each colour/pixel step (default 3)")
    ap.add_argument("--seconds", type=float, default=60, help="locate: how long to strobe (default 60)")
    ap.add_argument("--hz", type=float, default=4, help="locate: strobe rate (default 4)")
    ap.add_argument("--width", type=int, default=1, help="locate: how many pixels at each end (default 1)")
    ap.add_argument("--which", choices=["first", "last", "both"], default="both", help="locate: which end(s)")
    ap.add_argument(
        "--slot", type=float, default=mapcode.DEFAULT_SLOT, help="scan: seconds per code slot (default 0.4)"
    )
    ap.add_argument("--brightness", type=int, default=96, help="scan: 0-255 level for lit pixels (default 96)")
    ap.add_argument(
        "--color", choices=["white", "rgb", "green"], default="white", help="scan: W chip, RGB white or green"
    )
    ap.add_argument("--repeat", type=int, default=2, help="scan: passes back to back (default 2)")
    ap.add_argument("--countdown", type=int, default=5, help="scan: black seconds before the first pass (default 5)")
    ap.add_argument("--off", action="store_true", help="artnet: switch the built-in Art-Net mode off")
    ap.add_argument("--on", action="store_true", help="artnet: switch it on")
    ap.add_argument("--explore", action="store_true", help="artnet: also try nearby routes, methods and body shapes")
    a = ap.parse_args()

    dev = Twinkly(a.ip, a.http_port, a.udp_port)
    try:
        if a.command == "info":
            show_info(dev)
        elif a.command == "locate":
            run_locate(dev, a.fps, a.order, a.proto, a.seconds, a.hz, a.width, a.which)
        elif a.command == "scan":
            run_scan(dev, a.fps, a.order, a.proto, a.slot, a.brightness, a.color, a.repeat, a.countdown)
        elif a.command == "chunks":
            run_chunks(dev, a.fps, a.order, a.proto, a.hold)
        elif a.command == "artnet":
            run_artnet(dev, "off" if a.off else "on" if a.on else None, a.explore)
        else:
            run_test(dev, a.fps, a.order, a.proto, a.yes, a.hold)
    except RuntimeError as e:
        sys.exit(f"\nERROR: {e}")


if __name__ == "__main__":
    main()
