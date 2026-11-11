"""Segmenting Art-Net by destination IP so Resolume can feed more than 256 universes.

Resolume can address only Sub-Net + Universe (no Net), i.e. 256 universes per
destination IP. The bridge binds one Art-Net socket per listen IP and keys its
map by (protocol, feed, universe), so the same universe number arriving on two
different IPs reaches two different controllers.
"""

import asyncio
import socket
import time

import pytest

from twinklybridge import artnet
from twinklybridge import engine as engine_mod
from twinklybridge.app import Supervisor
from twinklybridge.config import ARTNET, load_config, parse_config
from twinklybridge.engine import Engine, UdpSender
from twinklybridge.testtools import FakeTwinkly


@pytest.fixture
def fast(monkeypatch):
    monkeypatch.setattr(engine_mod, "MODE_CHECK", 0.2)
    monkeypatch.setattr(engine_mod, "BACKOFF", (0.1,))
    monkeypatch.setattr(engine_mod, "CONTROL_TICK", 0.02)


async def wait_for(cond, timeout=4.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        await asyncio.sleep(0.02)
    return cond()


def _cfg(devices, cidr="127.0.0.1/8"):
    return parse_config({"input": {"cidr": cidr}, "devices": devices})


def _spec(name, ip, fake, **extra):
    return {
        "name": name,
        "ip": ip,
        "universe": "0.0.0",
        "pixels": fake.pixels,
        "http_port": fake.http_port,
        "rt_port": fake.rt_port,
        **extra,
    }


def test_same_universe_on_two_feeds_routes_by_ip():
    # Universe 0.0.0 arriving on .2 and on .3 must reach different controllers.
    cfg = _cfg(
        [
            {"name": "A", "ip": "127.0.0.2", "universe": "0.0.0", "pixels": 64, "listen_ip": "127.0.0.2"},
            {"name": "B", "ip": "127.0.0.3", "universe": "0.0.0", "pixels": 64, "listen_ip": "127.0.0.3"},
        ]
    )
    eng = Engine(cfg, sender=lambda p, addr: True)
    eng.feed(ARTNET, 0, bytes((1, 2, 3, 4)) * 64, "src", feed="127.0.0.2")
    eng.feed(ARTNET, 0, bytes((9, 8, 7, 6)) * 64, "src", feed="127.0.0.3")
    da = next(d for d in eng.devices if d.name == "A")
    db = next(d for d in eng.devices if d.name == "B")
    assert da.buffer[:4] == bytes((1, 2, 3, 4))
    assert db.buffer[:4] == bytes((9, 8, 7, 6))
    assert da.rx.total == 1 and db.rx.total == 1  # each saw only its own feed


def test_two_feeds_bind_and_stream_end_to_end(fast):
    async def scenario():
        a = FakeTwinkly(pixels=64, api="v2", host="127.0.0.2")
        b = FakeTwinkly(pixels=64, api="v2", host="127.0.0.3")
        await a.start()
        await b.start()
        cfg = _cfg(
            [
                _spec("A", "127.0.0.2", a, listen_ip="127.0.0.2"),
                _spec("B", "127.0.0.3", b, listen_ip="127.0.0.3"),
            ]
        )
        eng = Engine(cfg, sender=UdpSender(), hostname="box")
        await eng.start(open_sockets=True)
        # a sender whose source is neither the input IP nor a feed IP (would be
        # dropped as our own echo otherwise)
        send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        send.bind(("127.0.0.9", 0))
        try:
            # one Art-Net socket per feed, each bound to its own IP
            assert set(eng._artnet_transports) == {"127.0.0.2", "127.0.0.3"}
            for feed, transport in eng._artnet_transports.items():
                assert transport.get_extra_info("sockname")[0] == feed
            assert await wait_for(lambda: all(d.ready for d in eng.devices))

            # same universe number, different destination IPs, different colours
            send.sendto(artnet.encode_dmx(0, bytes((10, 20, 30, 40)) * 64), ("127.0.0.2", artnet.ARTNET_PORT))
            send.sendto(artnet.encode_dmx(0, bytes((1, 2, 3, 4)) * 64), ("127.0.0.3", artnet.ARTNET_PORT))
            assert await wait_for(lambda: a.frames > 0 and b.frames > 0)
            # input RGBW -> WRGB on the wire
            assert await wait_for(lambda: a.pixel(0) == (40, 10, 20, 30))
            assert await wait_for(lambda: b.pixel(0) == (4, 1, 2, 3))
            assert a.bad_packets == 0 and b.bad_packets == 0
        finally:
            send.close()
            await eng.stop()
            await a.stop()
            await b.stop()

    asyncio.run(scenario())


def test_adding_a_feed_forces_a_full_swap(tmp_path, monkeypatch):
    # Adding a device on a new listen IP needs a new socket, so the reload must
    # restart the engine rather than apply in place.
    cfg_path = tmp_path / "bridge.yaml"
    cfg_path.write_text(
        "input: {cidr: 10.0.0.2/24}\n"
        "devices:\n"
        "  - {name: A, ip: 10.0.0.5, universe: 0.0.0, pixels: 80, listen_ip: 10.0.0.2}\n"
    )
    sup = Supervisor(str(cfg_path))
    sup.config = load_config(str(cfg_path))
    calls = {"apply": 0, "swap": 0}

    class FakeEngine:
        running = True

        async def apply_config(self, config):
            calls["apply"] += 1

    sup.engine = FakeEngine()

    async def fake_swap(config):
        calls["swap"] += 1
        sup.config = config

    monkeypatch.setattr(sup, "_swap_engine", fake_swap)

    async def main():
        # add B on a second IP -> new feed -> full swap
        cfg_path.write_text(
            "input: {cidr: 10.0.0.2/24}\n"
            "devices:\n"
            "  - {name: A, ip: 10.0.0.5, universe: 0.0.0, pixels: 80, listen_ip: 10.0.0.2}\n"
            "  - {name: B, ip: 10.0.0.6, universe: 0.0.0, pixels: 80, listen_ip: 10.0.0.3}\n"
        )
        await sup.reload()
        assert calls == {"apply": 0, "swap": 1}

        # a plain edit on the same feeds -> applied in place
        cfg_path.write_text(
            "input: {cidr: 10.0.0.2/24}\n"
            "devices:\n"
            "  - {name: A, ip: 10.0.0.5, universe: 0.0.0, pixels: 80, listen_ip: 10.0.0.2, fps: 40}\n"
            "  - {name: B, ip: 10.0.0.6, universe: 0.0.0, pixels: 80, listen_ip: 10.0.0.3}\n"
        )
        await sup.reload()
        assert calls == {"apply": 1, "swap": 1}

    asyncio.run(main())
