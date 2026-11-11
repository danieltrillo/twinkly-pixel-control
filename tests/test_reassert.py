"""The "reassert realtime" button: force a controller that dropped to its own content back into rt."""

import asyncio
import time

import pytest
from aiohttp.test_utils import TestClient, TestServer

from twinklybridge import engine as engine_mod
from twinklybridge.app import Supervisor
from twinklybridge.config import parse_config
from twinklybridge.engine import Engine, UdpSender
from twinklybridge.testtools import FakeTwinkly
from twinklybridge.web import make_app


@pytest.fixture
def spin(monkeypatch):
    # Spin the control loop fast, but park the automatic mode-check far away so
    # only an explicit reassert can switch a fallen-back device back to rt.
    monkeypatch.setattr(engine_mod, "CONTROL_TICK", 0.02)
    monkeypatch.setattr(engine_mod, "MODE_CHECK", 1000.0)
    monkeypatch.setattr(engine_mod, "BACKOFF", (0.1,))


async def wait_for(cond, timeout=4.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        await asyncio.sleep(0.02)
    return cond()


def _cfg(devices):
    return parse_config({"input": {"cidr": "2.0.0.20/8"}, "devices": devices})


def _spec(name, ip, universe, fake):
    return {
        "name": name,
        "ip": ip,
        "universe": universe,
        "pixels": fake.pixels,
        "http_port": fake.http_port,
        "rt_port": fake.rt_port,
    }


def test_reassert_forces_a_fallen_back_controller_back_to_rt(spin):
    async def scenario():
        fake = FakeTwinkly(pixels=128, api="v2", host="127.0.0.2", rt_timeout=1000.0)
        await fake.start()
        eng = Engine(_cfg([_spec("A", "127.0.0.2", 0, fake)]), sender=UdpSender(), hostname="box")
        await eng.start(open_sockets=False)
        dev = eng.devices[0]
        try:
            assert await wait_for(lambda: dev.ready)
            eng.feed("artnet", 0, bytes((10, 20, 30, 40)) * 128, "2.0.0.1")
            assert await wait_for(lambda: fake.mode == "rt")
            assert fake.logins == 1

            # The controller wanders back to its own stored content, as the real Pro does.
            fake.mode = "movie"
            # The automatic check is parked, so the bridge does NOT fix it on its own.
            assert not await wait_for(lambda: fake.mode == "rt", timeout=0.3)

            # Press the button: it snaps back to rt on the next control tick, no re-login.
            assert eng.reassert() == 1
            assert await wait_for(lambda: fake.mode == "rt")
            assert fake.logins == 1
        finally:
            await eng.stop()
            await fake.stop()

    asyncio.run(scenario())


def test_reassert_targets_one_controller_by_name(spin):
    async def scenario():
        a = FakeTwinkly(pixels=64, api="v2", host="127.0.0.2", rt_timeout=1000.0)
        b = FakeTwinkly(pixels=64, api="v2", host="127.0.0.3", rt_timeout=1000.0)
        await a.start()
        await b.start()
        eng = Engine(
            _cfg([_spec("A", "127.0.0.2", 0, a), _spec("B", "127.0.0.3", 1, b)]),
            sender=UdpSender(),
            hostname="box",
        )
        await eng.start(open_sockets=False)
        try:
            assert await wait_for(lambda: all(d.ready for d in eng.devices))
            eng.feed("artnet", 0, bytes((1, 2, 3, 4)) * 64, "2.0.0.1")
            eng.feed("artnet", 1, bytes((1, 2, 3, 4)) * 64, "2.0.0.1")
            assert await wait_for(lambda: a.mode == "rt" and b.mode == "rt")

            # Both drop back; reassert only B.
            a.mode = "movie"
            b.mode = "movie"
            assert eng.reassert("B") == 1
            assert await wait_for(lambda: b.mode == "rt")
            assert a.mode == "movie"  # A was not poked

            assert eng.reassert("nope") == 0  # unknown name pokes nobody
        finally:
            await eng.stop()
            await a.stop()
            await b.stop()

    asyncio.run(scenario())


def test_reassert_endpoint(spin, tmp_path):
    async def scenario():
        fake = FakeTwinkly(pixels=64, api="v2", host="127.0.0.2", rt_timeout=1000.0)
        await fake.start()
        cfg = _cfg([_spec("A", "127.0.0.2", 0, fake)])
        eng = Engine(cfg, sender=UdpSender(), hostname="box")
        await eng.start(open_sockets=False)
        sup = Supervisor(str(tmp_path / "bridge.yaml"))
        sup.config = cfg
        sup.engine = eng
        try:
            assert await wait_for(lambda: eng.devices[0].ready)
            eng.feed("artnet", 0, bytes((1, 2, 3, 4)) * 64, "2.0.0.1")
            assert await wait_for(lambda: fake.mode == "rt")
            async with TestClient(TestServer(make_app(sup))) as client:
                r = await client.post("/api/reassert", json={"name": "ghost"})
                assert r.status == 404  # no such running device

                fake.mode = "movie"
                r = await client.post("/api/reassert", json={})
                j = await r.json()
                assert r.status == 200 and j["ok"] and j["count"] == 1
                assert await wait_for(lambda: fake.mode == "rt")  # snapped back over HTTP
        finally:
            await eng.stop()
            await fake.stop()

    asyncio.run(scenario())


def test_reassert_endpoint_503_when_not_running(tmp_path):
    async def main():
        sup = Supervisor(str(tmp_path / "bridge.yaml"))  # engine is None
        async with TestClient(TestServer(make_app(sup))) as client:
            r = await client.post("/api/reassert", json={})
            assert r.status == 503

    asyncio.run(main())
