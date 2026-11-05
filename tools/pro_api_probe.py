#!/usr/bin/env python3
"""
pro_api_probe.py - map what a Twinkly controller's local API can do.

Made for Pro controllers whose firmware only answers under /xled/v2 and hides
led/mode and fw/version. Finds which endpoints exist, whether the realtime
paths (UDP 7777 and POST led/rt/frame) accept frames, and whether the device
serves a web UI whose scripts name more endpoints.

Python 3.8+, standard library only.

  python3 pro_api_probe.py 10.201.82.105            # read-only: GETs only
  python3 pro_api_probe.py 10.201.82.105 --write    # also send test frames

--write sends realtime frames. With strands attached they light up dim green
for a second or two, then the controller goes back to its own effect.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import socket
import sys
import time
import urllib.error
import urllib.request

RT_PORT = 7777
FRAGMENT = 900

# GET candidates under /xled/<api>/. Reads only.
PATHS = """
gestalt status summary info device device/info device_name device_config config settings version
fw/version fw fw/update fw/0/version fw/1/version
led/mode mode led/state state led/power power led/on led/off on off
led/rt led/rt/frame led/rt/mode led/rt/status led/rt/config rt rt/frame rt/status realtime led/realtime
led/stream stream led/live live led/frame frame
led/effects led/effects/current effects
movies movies/current movies/list led/movie/config led/movie/full movie
playlist playlist/current playlists
led/color color led/brightness led/out/brightness led/out/saturation brightness
led/config led/layout led/layout/full layout led/ports ports led/outputs outputs led/strings
led/driver_params led/driver_params2 led/drv led/reset led/reset2
network network/status network/scan network/scan_results network/config
mqtt/config mqtt cloud cloud/status cloud/config
artnet artnet/config led/artnet dmx dmx/config
osc osc/config
group group/config group/status led/group sync
timer timers time timezone
music/drivers music/enabled music/config
echo logout users roles auth auth/role
ui ui/config web
""".split()

UI_PATHS = [
    "/",
    "/index.html",
    "/ui",
    "/ui/",
    "/ui/index.html",
    "/app",
    "/app/",
    "/web/",
    "/admin",
    "/login",
    "/static/",
    "/assets/",
    "/favicon.ico",
    "/manifest.json",
    "/main.js",
    "/app.js",
    "/bundle.js",
    "/index.js",
    "/xled/",
    "/xled/v2/",
    "/xled/v3/gestalt",
    "/api",
    "/api/v1",
    "/api/v2",
]


class Http:
    def __init__(self, ip: str, timeout: float = 3.0) -> None:
        self.ip = ip
        self.timeout = timeout
        self.token = ""
        self.token_raw = b""

    def call(
        self,
        method: str,
        url: str,
        body: bytes | dict | None = None,
        auth: bool = True,
        ctype: str = "application/json",
    ) -> tuple[int, dict, bytes]:
        """-> (status, headers, body). status 0 = no HTTP answer."""
        headers = {}
        data = None
        if body is not None:
            data = json.dumps(body).encode() if isinstance(body, dict) else body
            headers["Content-Type"] = ctype
        if auth and self.token:
            headers["X-Auth-Token"] = self.token
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return r.status, dict(r.headers), r.read()
        except urllib.error.HTTPError as e:
            return e.code, dict(e.headers), e.read()
        except (TimeoutError, urllib.error.URLError, OSError) as e:
            return 0, {}, str(e).encode()

    def api(self, method: str, api: str, path: str, body=None, auth=True, ctype="application/json"):
        return self.call(method, f"http://{self.ip}/xled/{api}/{path}", body, auth, ctype)


def short(b: bytes, n: int = 150) -> str:
    text = b.decode("utf-8", "replace").replace("\n", " ").strip()
    return text if len(text) <= n else text[:n] + "..."


def code_of(body: bytes):
    try:
        j = json.loads(body)
    except ValueError:
        return None
    if isinstance(j, dict):
        if isinstance(j.get("code"), int):
            return j["code"]
        r = j.get("result")
        if isinstance(r, dict):
            return r.get("code")
    return None


def section(title: str) -> None:
    print(f"\n{'=' * 8} {title} {'=' * (60 - len(title))}")


# --------------------------------------------------------------------------


def detect(h: Http) -> tuple[str, dict]:
    for api in ("v1", "v2"):
        st, _, body = h.api("GET", api, "gestalt", auth=False)
        if st == 200:
            try:
                return api, json.loads(body)
            except ValueError:
                pass
        print(f"  GET /xled/{api}/gestalt -> {st} {short(body, 80)}")
    sys.exit("no gestalt under v1 or v2; is this a Twinkly?")


def login(h: Http, api: str) -> dict:
    challenge = base64.b64encode(os.urandom(32)).decode()
    st, _, body = h.api("POST", api, "login", {"challenge": challenge}, auth=False)
    print(f"  POST login -> {st} {short(body, 200)}")
    j = json.loads(body) if st == 200 else {}
    token = j.get("authentication_token", "")
    if not token:
        sys.exit("login failed")
    h.token = token
    h.token_raw = base64.b64decode(token)
    print(
        f"  token: {len(h.token_raw)} raw bytes, expires_in={j.get('authentication_token_expires_in')}, "
        f"role={j.get('auth-role', '?')}"
    )
    st, _, body = h.api("POST", api, "verify", {"challenge-response": j.get("challenge-response", "")})
    print(f"  POST verify -> {st} {short(body, 120)}")
    return j


def enumerate_gets(h: Http, api: str) -> dict[str, int]:
    found: dict[str, int] = {}
    rows = []
    for p in PATHS:
        st, hdr, body = h.api("GET", api, p)
        allow = hdr.get("Allow") or hdr.get("allow") or ""
        if st != 404:
            found[p] = st
            rows.append((p, st, allow, short(body, 110)))
    for p, st, allow, b in rows:
        extra = f" Allow: {allow}" if allow else ""
        print(f"  {st:>3}  /xled/{api}/{p:<24}{extra}  {b}")
    print(f"  ({len(PATHS)} paths tried, {len(rows)} not 404)")
    return found


def ui_hunt(h: Http) -> None:
    scripts: set[str] = set()
    for p in UI_PATHS:
        for auth in (False, True):
            st, hdr, body = h.call("GET", f"http://{h.ip}{p}", auth=auth)
            ctype = hdr.get("Content-Type", hdr.get("content-type", "?"))
            if st == 404 or st == 0:
                if not auth:
                    continue  # try once with token
                else:
                    continue
            tag = "token" if auth else "anon"
            print(f"  {st:>3} {tag:<5} {p:<20} {ctype:<28} {len(body):>6} B  {short(body, 90)}")
            if b"<script" in body or b"<html" in body.lower():
                for m in re.finditer(rb'src=["\']([^"\']+\.js[^"\']*)', body):
                    scripts.add(m.group(1).decode("utf-8", "replace"))
            if st == 200:
                break
    server = h.call("GET", f"http://{h.ip}/xled/v2/gestalt", auth=False)[1]
    srv = {k: v for k, v in server.items() if k.lower() in ("server", "x-powered-by", "www-authenticate")}
    if srv:
        print(f"  server headers: {srv}")
    for s in sorted(scripts):
        url = s if s.startswith("http") else f"http://{h.ip}/{s.lstrip('/')}"
        st, _, body = h.call("GET", url)
        hits = sorted(set(m.decode() for m in re.findall(rb"xled/v[0-9]/[A-Za-z0-9_/.-]+", body)))
        print(f"  script {url}: {st}, {len(body)} B, endpoints named: {hits[:40]}")


def udp_probe(h: Http, ip: str, port: int, n_leds: int, bpl: int) -> None:
    frame = bytes([0, 0, 8, 0] if bpl == 4 else [0, 8, 0]) * n_leds  # dim green-ish whatever the order
    packets = []
    for i, start in enumerate(range(0, len(frame), FRAGMENT)):
        packets.append(b"\x03" + h.token_raw + b"\x00\x00" + bytes([i]) + frame[start : start + FRAGMENT])
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(1.0)
    sock.connect((ip, port))
    verdict = "open or silently dropped (no ICMP unreachable came back)"
    try:
        for _ in range(3):  # a few frames, ~10 fps
            for p in packets:
                sock.send(p)
            time.sleep(0.1)
        try:
            data = sock.recv(256)
            verdict = f"REPLIED with {len(data)} bytes: {data[:40]!r}"
        except TimeoutError:
            pass
    except ConnectionRefusedError:
        verdict = f"CLOSED (ICMP port unreachable: nothing listens on UDP {port})"
    except OSError as e:
        verdict = f"send error: {e}"
    finally:
        sock.close()
    print(f"  UDP {ip}:{port}, v3 frames of {len(frame)} B in {len(packets)} fragments x3: {verdict}")


def http_rt_probe(h: Http, api: str, n_leds: int, bpl: int) -> None:
    path = "led/rt/frame"
    full = bytes([0, 0, 8, 0] if bpl == 4 else [0, 8, 0]) * n_leds
    cases = [
        ("no token, full frame", False, full),
        (f"{bpl} B/LED x {n_leds} = {len(full)} B", True, full),
        (f"{7 - bpl} B/LED x {n_leds} = {n_leds * (7 - bpl)} B", True, bytes(n_leds * (7 - bpl))),
        ("100 B (too short)", True, bytes(100)),
        (
            "header 0x03+token+00 00 00 then frame (UDP-style body)",
            True,
            b"\x03" + h.token_raw + b"\x00\x00\x00" + full,
        ),
    ]
    for label, auth, body in cases:
        t0 = time.perf_counter()
        st, hdr, out = h.api("POST", api, path, body, auth=auth, ctype="application/octet-stream")
        ms = (time.perf_counter() - t0) * 1000
        print(f"  POST {path} [{label}] -> {st} code={code_of(out)} {short(out, 100)}  ({ms:.0f} ms)")
    # JSON-wrapped variant some firmware families use
    st, _, out = h.api("POST", api, path, {"frame": base64.b64encode(full).decode()})
    print(f"  POST {path} [json {{frame: base64}}] -> {st} code={code_of(out)} {short(out, 100)}")
    # throughput: how fast does the HTTP path go?
    ok = 0
    t0 = time.perf_counter()
    for _ in range(20):
        st, _, out = h.api("POST", api, path, full, ctype="application/octet-stream")
        ok += 1 if st == 200 and code_of(out) == 1000 else 0
    dt = time.perf_counter() - t0
    print(f"  20 x POST full frame: {ok} accepted in {dt:.2f} s = {20 / dt:.1f} fps, {dt / 20 * 1000:.0f} ms each")


def mode_probe(h: Http, api: str) -> None:
    for path, body in [
        ("led/mode", {"mode": "rt"}),
        ("led/mode", {"mode": "movie"}),
        ("mode", {"mode": "rt"}),
        ("led/rt", {"enabled": True}),
        ("led/rt/mode", {"mode": "rt"}),
        ("echo", {"probe": "hello"}),
    ]:
        st, hdr, out = h.api("POST", api, path, body)
        allow = hdr.get("Allow") or hdr.get("allow") or ""
        print(
            f"  POST {path} {json.dumps(body)} -> {st} code={code_of(out)} {short(out, 100)}"
            + (f" Allow: {allow}" if allow else "")
        )


def snapshot(h: Http, api: str, found: dict[str, int]) -> dict[str, str]:
    snap = {}
    for p, st in found.items():
        if st == 200:
            snap[p] = short(h.api("GET", api, p)[2], 400)
    return snap


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ip")
    ap.add_argument("--write", action="store_true", help="send realtime test frames (lights may flash)")
    ap.add_argument("--udp-port", type=int, default=RT_PORT)
    ap.add_argument("--no-ui", action="store_true", help="skip the web UI hunt")
    a = ap.parse_args()
    h = Http(a.ip)

    section("device")
    api, g = detect(h)
    cfg = g.get("device_config") or {}
    profile = (g.get("led_profile") or cfg.get("led_profile") or "RGB").upper()
    bpl = 4 if profile == "RGBW" else 3
    n_leds = int(g.get("number_of_led") or 0)
    print(
        f"  API /xled/{api}: {g.get('product_code')} '{g.get('device_name')}' {n_leds} {profile} LEDs, "
        f"group={g.get('group')}, artnet_en={g.get('artnet_en')}, frame_rate={g.get('frame_rate')}, "
        f"uptime={g.get('uptime')}"
    )

    section("login")
    login(h, api)

    section(f"GET enumeration under /xled/{api}")
    found = enumerate_gets(h, api)
    other = "v1" if api == "v2" else "v2"
    section(f"GET enumeration under /xled/{other} (just in case)")
    enumerate_gets(h, other)

    if not a.no_ui:
        section("web UI hunt")
        ui_hunt(h)

    if not a.write:
        print("\n(read-only run; add --write to test the realtime paths)")
        return

    before = snapshot(h, api, found)

    section("POST mode-ish endpoints")
    mode_probe(h, api)

    section("realtime over UDP")
    udp_probe(h, a.ip, a.udp_port, n_leds, bpl)

    section("realtime over HTTP")
    http_rt_probe(h, api, n_leds, bpl)

    section("realtime over UDP again (after HTTP frames, in case they armed rt mode)")
    udp_probe(h, a.ip, a.udp_port, n_leds, bpl)

    section("what changed on the device")
    after = snapshot(h, api, found)
    changed = [p for p in before if before[p] != after.get(p)]
    for p in changed:
        print(f"  {p}:\n    before {before[p]}\n    after  {after[p]}")
    if not changed:
        print("  nothing in the 200-OK GET endpoints changed")
    time.sleep(6)
    later = snapshot(h, api, found)
    changed = [p for p in after if after[p] != later.get(p)]
    print("  6 s later: " + (", ".join(changed) if changed else "still unchanged"))


if __name__ == "__main__":
    main()
