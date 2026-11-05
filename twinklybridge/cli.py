"""Command line entry point: `twinkly-bridge <command>`."""

from __future__ import annotations

import argparse
import asyncio
import colorsys
import logging
import pathlib
import socket
import sys
import time

from . import __version__, artnet
from .config import (
    ARTNET,
    DEFAULT_CONFIG_PATH,
    DMX_SLOTS,
    ConfigError,
    build_segments,
    format_universe,
    load_config,
    overlaps,
)
from .twinkly import HTTP_PORT, TwinklyClient, TwinklyError, discover

EXAMPLE_CONFIG = pathlib.Path(__file__).parent / "example-config.yaml"


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    logging.getLogger("aiohttp").setLevel(logging.WARNING)


# -- run / check / example ---------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    from .app import Supervisor

    _setup_logging(args.log_level)
    override = (args.web_bind or "0.0.0.0", args.web_port) if args.web_port else None
    try:
        return asyncio.run(Supervisor(args.config, web_override=override).run())
    except KeyboardInterrupt:
        return 0


def cmd_check(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}")
        return 1
    inp = config.input
    protocols = ", ".join(p for p, on in (("Art-Net", inp.artnet), ("sACN", inp.sacn)) if on)
    print(f"{args.config}: ok")
    print(f"  input: {inp.ip} ({protocols}), ArtPollReply to {inp.broadcast}")
    total = 0
    for d in config.devices:
        n = len(d.segments)
        total += n if d.enabled else 0
        off = "" if d.enabled else "  (disabled)"
        print(
            f"  {d.name:<20} {d.ip:<15} {d.protocol} {d.universe_label:<15} {n:>3} univ  "
            f"{d.pixels:>5} px {d.input}->{d.order} @ {d.fps:g} fps{off}"
        )
    print(f"  {len(config.enabled_devices)} device(s), {total} universe(s)")
    for warning in overlaps(config):
        print(f"  note: {warning}")
    return 0


def cmd_example(args: argparse.Namespace) -> int:
    sys.stdout.write(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    return 0


# -- info / discover ----------------------------------------------------------


async def _info(ip: str, port: int, full: bool) -> dict:
    import aiohttp

    async with aiohttp.ClientSession() as session:
        client = TwinklyClient(session, ip, port=port)
        info = await client.info()
        out = {"ip": ip, "info": info}
        if full:
            await client.login()
            out["mode"] = await client.get_mode()
            out["strings"] = await client.led_config()
        return out


def cmd_info(args: argparse.Namespace) -> int:
    try:
        r = asyncio.run(_info(args.ip, args.http_port, True))
    except TwinklyError as exc:
        print(f"{args.ip}: {exc}")
        return 1
    i = r["info"]
    print(f"{args.ip}: {i.product_code or 'Twinkly'} '{i.device_name}'")
    print(f"  API        /xled/{i.api}")
    print(f"  firmware   {i.firmware or 'n/a (not exposed by this firmware)'}")
    print(f"  LEDs       {i.number_of_led} {i.led_profile} (max {i.max_supported_led})")
    print(f"  frame rate {i.frame_rate} (measured {i.measured_frame_rate})")
    print(f"  mode       {r['mode'] or 'n/a (not exposed by this firmware)'}")
    if i.group_mode:
        print(
            f"  group      {i.group_mode}, pixels {i.group_offset}-{i.group_offset + i.number_of_led} of {i.group_size}"
        )
    if i.artnet_enabled is not None:
        print(
            f"  Art-Net    built-in DMX mode {'ON (turn off so it does not fight the bridge)' if i.artnet_enabled else 'off'}"
        )
    for n, s in enumerate(r["strings"], start=1):
        print(f"  output {n}   first LED {s.get('first_led_id')}, {s.get('length')} LEDs")
    return 0


def cmd_discover(args: argparse.Namespace) -> int:
    print(f"looking for Twinkly devices via {args.broadcast} ...", file=sys.stderr)
    found = discover(args.broadcast, args.timeout, args.bind)
    if not found:
        print(
            "nothing answered. Check the bridge is on the Twinkly VLAN, or try --broadcast with that subnet's broadcast."
        )
        return 1

    async def gather():
        return await asyncio.gather(*(_info(d.ip, HTTP_PORT, False) for d in found), return_exceptions=True)

    results = asyncio.run(gather())
    rows = []
    for d, r in zip(found, results, strict=True):
        if isinstance(r, Exception):
            rows.append((d, None, str(r)))
        else:
            rows.append((d, r["info"], ""))

    if not args.yaml:
        for d, info, err in rows:
            if info is None:
                print(f"{d.ip:<15} {d.name:<24} ? ({err})")
            else:
                print(
                    f"{d.ip:<15} {d.name:<24} {info.product_code:<16} fw {info.firmware or 'n/a':<8} "
                    f"{info.number_of_led:>5} {info.led_profile}"
                )
        print(f"{len(rows)} device(s)", file=sys.stderr)
        return 0

    # A ready-to-paste devices: block with universes assigned one after another.
    channels = len(args.input)
    universe = artnet.parse_universe(args.start_universe)
    print("devices:")
    for d, info, err in rows:
        pixels = info.number_of_led if info and info.number_of_led else 0
        if not pixels:
            print(f"  # {d.ip} {d.name}: could not read LED count ({err}); add it by hand")
            continue
        segs = build_segments(universe, pixels, channels, DMX_SLOTS // channels, 1)
        last = segs[-1].universe
        print(f"  - name: {d.name}")
        print(f"    ip: {d.ip}")
        print(
            f"    universe: {format_universe(ARTNET, universe)}   # to {format_universe(ARTNET, last)}, {len(segs)} universes"
        )
        print(
            f"    pixels: {pixels}   # {info.product_code}, {info.led_profile}"
            + (f", fw {info.firmware}" if info.firmware else "")
        )
        universe = last + 1
    return 0


# -- send: an Art-Net test pattern -------------------------------------------


def _pattern(name: str, index: int, total: int, t: float, layout: str) -> bytes:
    if name == "chase":
        on = index == int(t * 30) % max(total, 1)
        rgbw = (255, 255, 255, 0) if on else (0, 0, 0, 0)
    elif name == "white":
        rgbw = (0, 0, 0, 255)
    elif name == "solid":
        rgbw = (255, 0, 0, 0)
    else:  # rainbow
        r, g, b = colorsys.hsv_to_rgb((index / max(total, 1) + t * 0.25) % 1.0, 1.0, 1.0)
        rgbw = (int(r * 255), int(g * 255), int(b * 255), 0)
    value = dict(zip("RGBW", rgbw, strict=True))
    return bytes(value[c] for c in layout)


def cmd_send(args: argparse.Namespace) -> int:
    first, _, last = args.universe.partition("-")
    u0 = artnet.parse_universe(first.strip())
    u1 = artnet.parse_universe(last.strip()) if last else u0
    layout = args.input.upper()
    ch = len(layout)
    per = DMX_SLOTS // ch
    total = (u1 - u0 + 1) * per
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    print(
        f"sending {args.pattern} to {args.to}, universes {format_universe(ARTNET, u0)}-{format_universe(ARTNET, u1)} "
        f"({total} {layout} pixels) at {args.fps:g} fps. Ctrl+C to stop."
    )
    seq = 0
    start = time.monotonic()
    try:
        while True:
            t = time.monotonic() - start
            seq = seq % 255 + 1
            for k, u in enumerate(range(u0, u1 + 1)):
                data = b"".join(_pattern(args.pattern, k * per + i, total, t, layout) for i in range(per))
                sock.sendto(artnet.encode_dmx(u, data, seq), (args.to, artnet.ARTNET_PORT))
            time.sleep(1.0 / args.fps)
    except KeyboardInterrupt:
        return 0


# -- parser ------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="twinkly-bridge", description="Art-Net / sACN to Twinkly realtime pixel bridge"
    )
    parser.add_argument("--version", action="version", version=f"twinkly-bridge {__version__}")
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="run the bridge (default)")
    run.add_argument("-c", "--config", default=DEFAULT_CONFIG_PATH, help=f"config file (default {DEFAULT_CONFIG_PATH})")
    run.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"])
    run.add_argument("--web-port", type=int, default=0, help="override the status page port")
    run.add_argument("--web-bind", default="", help="override the status page bind address")
    run.set_defaults(func=cmd_run)

    check = sub.add_parser("check", help="validate a config file and show the universe map")
    check.add_argument("-c", "--config", default=DEFAULT_CONFIG_PATH)
    check.set_defaults(func=cmd_check)

    example = sub.add_parser("example", help="print the annotated example config")
    example.set_defaults(func=cmd_example)

    disc = sub.add_parser("discover", help="find Twinkly devices on the network")
    disc.add_argument("--broadcast", default="255.255.255.255", help="broadcast address of the Twinkly VLAN")
    disc.add_argument("--bind", default="", help="local IP to send from")
    disc.add_argument("--timeout", type=float, default=2.0)
    disc.add_argument("--yaml", action="store_true", help="print a devices: block with universes assigned in order")
    disc.add_argument("--start-universe", default="0.0.0", help="first universe for --yaml (default 0.0.0)")
    disc.add_argument("--input", default="RGBW", help="console pixel layout for --yaml (default RGBW)")
    disc.set_defaults(func=cmd_discover)

    info = sub.add_parser("info", help="show what a Twinkly device reports")
    info.add_argument("ip")
    info.add_argument("--http-port", type=int, default=HTTP_PORT, help=argparse.SUPPRESS)
    info.set_defaults(func=cmd_info)

    send = sub.add_parser("send", help="send an Art-Net test pattern (no console needed)")
    send.add_argument("--to", default="255.255.255.255", help="bridge IP or broadcast address")
    send.add_argument("--universe", default="0.0.0", help="universe or range, e.g. 0.0.0-0.0.11")
    send.add_argument("--input", default="RGBW", help="pixel layout (default RGBW)")
    send.add_argument("--pattern", choices=["rainbow", "chase", "solid", "white"], default="rainbow")
    send.add_argument("--fps", type=float, default=30)
    send.set_defaults(func=cmd_send)

    from .testtools import add_fake_command

    add_fake_command(sub)
    return parser


COMMANDS = ("run", "check", "example", "discover", "info", "send", "fake")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or (argv[0] not in COMMANDS and argv[0] not in ("-h", "--help", "--version")):
        argv = ["run"] + argv
    args = build_parser().parse_args(argv)
    return args.func(args)
