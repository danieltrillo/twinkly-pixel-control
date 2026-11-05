#!/usr/bin/env python3
"""
pro_api_probe2.py - second pass on a Twinkly Pro that speaks /xled/v2.

Read-only parts: full GET bodies, auth checks, POST-route discovery with an
empty JSON body, a TCP port scan. With --write it also runs the implicit-mode
test (led/color, then back to the current movie) and the led/rt/frame body
experiments. It never touches persistent configuration (network, mqtt,
cloud, group, artnet, osc, device name, firmware, timers, layout/mapping)
and never calls reboot/reset.

  python3 pro_api_probe2.py 10.201.82.105 --write
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import os
import socket
import sys
import time
import urllib.error
import urllib.request

FULL_GETS = [
    "summary",
    "status",
    "movies",
    "movies/current",
    "playlists",
    "led/color",
    "led/out/brightness",
    "led/out/hue",
    "mqtt/config",
    "artnet/config",
    "osc",
    "device/config",
    "gestalt",
]

POST_EMPTY = """
app_mode led/app_mode app/mode app device/mode device/app_mode mode
led/rt led/rt/start led/rt/stop led/rt/begin led/rt/end led/rt/enable led/rt/session led/rt/config
led/rt/status led/rt/frames led/rt/mode led/mode/rt rt rt/start rt/stop led/realtime realtime
led/stream stream led/live live led/play play led/stop stop led/pause pause led/resume resume
led/off led/on movies/play movies/stop movies/current playlists/current led/effects led/effects/current
led/frame frame lightmapping led/test test led/demo demo led/out/mode led/preview preview led/scene scene
summary status led/color led/out/brightness
""".split()

SECRET_WORDS = ("password", "secret", "key", "cert", "token", "pass")


class Http:
    def __init__(self, ip: str, timeout: float = 4.0) -> None:
        self.ip = ip
        self.timeout = timeout
        self.token = ""
        self.token_raw = b""

    def call(self, method, url, body=None, auth=True, ctype="application/json", headers=None):
        hdrs = dict(headers or {})
        data = None
        if body is not None:
            data = json.dumps(body).encode() if isinstance(body, dict) else body
            hdrs.setdefault("Content-Type", ctype)
        if auth and self.token and "X-Auth-Token" not in hdrs:
            hdrs["X-Auth-Token"] = self.token
        req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return r.status, dict(r.headers), r.read()
        except urllib.error.HTTPError as e:
            return e.code, dict(e.headers), e.read()
        except (TimeoutError, urllib.error.URLError, OSError) as e:
            return 0, {}, str(e).encode()

    def v2(self, method, path, body=None, auth=True, ctype="application/json", headers=None):
        return self.call(method, f"http://{self.ip}/xled/v2/{path}", body, auth, ctype, headers)

    def get_json(self, path):
        st, _, body = self.v2("GET", path)
        try:
            return st, json.loads(body)
        except ValueError:
            return st, {"_raw": body.decode("utf-8", "replace")[:300]}


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


def short(b: bytes, n: int = 160) -> str:
    t = b.decode("utf-8", "replace").replace("\n", " ").strip()
    return t if len(t) <= n else t[:n] + "..."


def mask(obj):
    if isinstance(obj, dict):
        return {
            k: ("***" if any(w in k.lower() for w in SECRET_WORDS) and isinstance(v, (str, int)) else mask(v))
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [mask(x) for x in obj]
    return obj


def section(title: str) -> None:
    print(f"\n{'=' * 8} {title} {'=' * max(1, 62 - len(title))}", flush=True)


def login(h: Http) -> None:
    challenge = base64.b64encode(os.urandom(32)).decode()
    st, _, body = h.v2("POST", "login", {"challenge": challenge}, auth=False)
    j = json.loads(body) if st == 200 else {}
    token = j.get("authentication_token", "")
    if not token:
        sys.exit(f"login failed: {st} {short(body)}")
    h.token, h.token_raw = token, base64.b64decode(token)
    st, _, body = h.v2("POST", "verify", {"challenge-response": j.get("challenge-response", "")})
    print(f"  login ok, role={j.get('auth-role')}, verify -> {st} {short(body, 60)}")


def app_mode(h: Http) -> str:
    st, j = h.get_json("summary")
    return json.dumps(j.get("app_mode")) if st == 200 else f"(summary -> {st})"


def report(label: str, st: int, hdr: dict, body: bytes) -> None:
    print(f"  {label:<70} -> {st} code={code_of(body)} {short(body, 110)}")


# --------------------------------------------------------------------------


def part_a(h: Http) -> dict:
    section("A) full GET bodies under /xled/v2")
    out = {}
    for p in FULL_GETS:
        st, j = h.get_json(p)
        out[p] = j
        shown = mask(j) if p == "mqtt/config" else j
        print(f"  GET {p} -> {st}\n    {json.dumps(shown, separators=(',', ':'))}")
    return out


def part_b(h: Http) -> None:
    section("B) auth semantics")
    st, hdr, body = h.v2("POST", "echo", {"a": 1}, auth=False)
    report("POST echo {a:1} WITHOUT token", st, hdr, body)
    st, hdr, body = h.v2("POST", "echo", {"a": 1}, headers={"X-Auth-Token": "AAAAAAAAAAA="})
    report("POST echo {a:1} with a GARBAGE token", st, hdr, body)
    st, hdr, body = h.v2("POST", "echo", {"a": 1})
    report("POST echo {a:1} with the real token", st, hdr, body)
    st, hdr, body = h.v2("POST", "led/out/brightness", {"mode": "enabled", "value": 100})
    report("POST led/out/brightness {mode:enabled,value:100} (same as current)", st, hdr, body)
    st, hdr, body = h.v2("POST", "led/out/brightness", {"bogus": 1})
    report("POST led/out/brightness {bogus:1} (bad body on a real endpoint)", st, hdr, body)
    st, hdr, body = h.v2("GET", "led/rt/frame")
    print(f"  GET led/rt/frame -> {st}; headers: {json.dumps(hdr)}")
    st, hdr, body = h.v2("POST", "led/rt/frame", bytes(16), ctype="application/octet-stream")
    print(f"  POST led/rt/frame (16 B) -> {st}; headers: {json.dumps(hdr)}; body: {short(body)}")


def rt_frame(
    h: Http,
    label: str,
    body: bytes,
    *,
    path="led/rt/frame",
    ctype="application/octet-stream",
    method="POST",
    auth=True,
    headers=None,
) -> int:
    t0 = time.perf_counter()
    st, hdr, out = h.v2(method, path, body, auth=auth, ctype=ctype, headers=headers)
    ms = (time.perf_counter() - t0) * 1000
    print(f"  {method} {path} [{label}] -> {st} code={code_of(out)} {short(out, 80)} ({ms:.0f} ms)")
    if st == 200:
        print(f"      app_mode now: {app_mode(h)}")
    return st


def part_c(h: Http, a: dict) -> dict:
    section("C) implicit-mode test (does app_mode follow the last command?)")
    cur = a.get("movies/current", {}).get("unique_id")
    movies = a.get("movies", {}).get("movies") or []
    cur_id = next((m.get("id") for m in movies if m.get("unique_id") == cur), None)
    print(f"  before: app_mode={app_mode(h)}, movies/current unique_id={cur} (id {cur_id})")
    st, hdr, body = h.v2("POST", "led/color", {"red": 0, "green": 8, "blue": 0, "white": 0})
    report("POST led/color {0,8,0,0}", st, hdr, body)
    print(f"  after led/color: app_mode={app_mode(h)}")
    n = int(a.get("gestalt", {}).get("number_of_led") or 0)
    frame = bytes([0, 0, 8, 0]) * n
    rt_frame(h, f"{len(frame)} B while in colour mode", frame)
    restored = False
    for label, method, body in [
        ("POST movies/current {unique_id}", "POST", {"unique_id": cur}),
        ("POST movies/current {id}", "POST", {"id": cur_id}),
        ("PUT movies/current {unique_id}", "PUT", {"unique_id": cur}),
        ("POST movies/current {unique_id, id}", "POST", {"unique_id": cur, "id": cur_id}),
    ]:
        if cur is None:
            break
        st, hdr, out = h.v2(method, "movies/current", body)
        report(label, st, hdr, out)
        if st == 200 and code_of(out) == 1000:
            restored = True
            break
    print(f"  after restore attempt: app_mode={app_mode(h)} (restored={restored})")
    return {"restored": restored, "unique_id": cur, "id": cur_id}


def part_d(h: Http, a: dict) -> None:
    section("D) led/rt/frame body experiments")
    g = a.get("gestalt", {})
    n = int(g.get("number_of_led") or 0)
    group = g.get("group") or {}
    gsize = int(group.get("size") or 0)
    goff = int(group.get("offset") or 0)
    px = bytes([0, 0, 8, 0])
    full = px * n
    print(f"  device: {n} LEDs x 4 B = {len(full)} B; group size {gsize}, offset {goff}")
    rt_frame(h, "5760-ish full frame, real token (baseline)", full)
    rt_frame(h, "full frame, garbage token", full, headers={"X-Auth-Token": "AAAAAAAAAAA="})
    rt_frame(h, "full frame, no token", full, auth=False)
    frag0 = b"\x03" + h.token_raw + b"\x00\x00\x00" + full[:900]
    rt_frame(h, "exact UDP v3 fragment 0 (912 B)", frag0)
    rt_frame(h, "v2-UDP style: 0x02 + token + 0x00 + full", b"\x02" + h.token_raw + b"\x00" + full)
    rt_frame(h, "4-byte LE frame number 0 + full", (0).to_bytes(4, "little") + full)
    rt_frame(h, "one 0x00 byte + full", b"\x00" + full)
    rt_frame(h, "2-byte BE led count + full", n.to_bytes(2, "big") + full)
    if gsize:
        rt_frame(h, f"whole group frame {gsize} x 4 = {gsize * 4} B", px * gsize)
        rt_frame(h, f"whole group frame {gsize} x 3 = {gsize * 3} B", bytes([0, 8, 0]) * gsize)
        rt_frame(h, f"4-byte LE offset {goff} + full", goff.to_bytes(4, "little") + full)
    rt_frame(h, "16-bit: full x 2", bytes([0, 0, 0, 0, 8, 0, 0, 0]) * n)
    rt_frame(h, "3 B/LED", bytes([0, 8, 0]) * n)
    for q in ("?frame=0", "?seq=0", f"?offset={goff}", f"?start={goff}&len={n}", "?id=0", "?index=0"):
        rt_frame(h, f"full + query {q}", full, path="led/rt/frame" + q)
    for ct in (
        "application/binary",
        "application/x-binary",
        "*/*",
        "application/x-www-form-urlencoded",
        "text/plain",
        "image/raw",
    ):
        rt_frame(h, f"full, Content-Type {ct}", full, ctype=ct)
    rt_frame(h, "PUT full", full, method="PUT")
    rt_frame(h, "PATCH full", full, method="PATCH")
    rt_frame(h, "POST full to led/rt/frame/0", full, path="led/rt/frame/0")
    rt_frame(h, "POST full to led/rt/frames", full, path="led/rt/frames")
    rt_frame(h, "POST full to led/rt", full, path="led/rt")
    rt_frame(h, "POST full to led/frame", full, path="led/frame")
    rt_frame(h, "POST full to rt/frame", full, path="rt/frame")
    rt_frame(h, "POST empty body", b"")
    rt_frame(h, "POST json {} to led/rt/frame", {})
    rt_frame(h, 'POST json {"frame": base64}', {"frame": base64.b64encode(full).decode()})
    rt_frame(h, 'POST json {"data": base64, "offset": 0}', {"data": base64.b64encode(full).decode(), "offset": 0})
    rt_frame(
        h,
        "POST json movie-like {frames:1, leds:n, data: base64}",
        {"frames_number": 1, "leds_per_frame": n, "frame_data": base64.b64encode(full).decode()},
    )


def part_e(h: Http) -> None:
    section("E) POST {} route discovery under /xled/v2 (anything that is not 404)")
    hits = 0
    for p in POST_EMPTY:
        st, hdr, body = h.v2("POST", p, {})
        if st != 404:
            hits += 1
            allow = hdr.get("Allow") or hdr.get("allow") or ""
            print(
                f"  {st:>3}  POST /xled/v2/{p:<22} code={code_of(body)} {short(body, 90)}"
                + (f"  Allow: {allow}" if allow else "")
            )
    print(f"  ({len(POST_EMPTY)} paths tried, {hits} not 404)")


def part_f(h: Http, ip: str) -> None:
    section("F) TCP port scan")

    def probe(port: int) -> int | None:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.25)
        try:
            return port if s.connect_ex((ip, port)) == 0 else None
        finally:
            s.close()

    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=400) as ex:
        open_ports = sorted(p for p in ex.map(probe, range(1, 65536)) if p)
    print(f"  open TCP ports: {open_ports} ({time.perf_counter() - t0:.0f} s)")
    for port in open_ports:
        if port == 80:
            continue
        for scheme in ("http", "https"):
            st, hdr, body = h.call("GET", f"{scheme}://{ip}:{port}/", auth=False)
            if st:
                print(f"  {scheme}://{ip}:{port}/ -> {st} {json.dumps(hdr)[:200]} {short(body, 120)}")
                break
        else:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(2)
            try:
                s.connect((ip, port))
                s.sendall(b"\r\n")
                print(f"  tcp/{port} banner: {s.recv(200)!r}")
            except OSError as e:
                print(f"  tcp/{port}: no HTTP, no banner ({e})")
            finally:
                s.close()


def part_g(h: Http, a: dict) -> None:
    section("G) persistent config unchanged?")
    for p in ("artnet/config", "osc", "mqtt/config"):
        st, j = h.get_json(p)
        print(f"  {p}: {'same' if j == a.get(p) else 'CHANGED: ' + json.dumps(mask(j))}")
    st, j = h.get_json("gestalt")
    for k in ("group", "device_config", "artnet_en", "network", "device_name"):
        print(
            f"  gestalt.{k}: {'same' if j.get(k) == a.get('gestalt', {}).get(k) else 'CHANGED: ' + json.dumps(j.get(k))}"
        )
    print(f"  final app_mode={app_mode(h)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ip")
    ap.add_argument("--write", action="store_true", help="run the output-changing parts C and D")
    ap.add_argument("--no-scan", action="store_true", help="skip the TCP port scan")
    args = ap.parse_args()
    h = Http(args.ip)
    section("login")
    login(h)
    a = part_a(h)
    part_b(h)
    if args.write:
        part_c(h, a)
        part_d(h, a)
    part_e(h)
    if not args.no_scan:
        part_f(h, args.ip)
    part_g(h, a)


if __name__ == "__main__":
    main()
