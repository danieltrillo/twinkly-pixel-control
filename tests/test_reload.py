"""Applying a config change in place: unchanged controllers keep streaming, only the changed ones move."""

import asyncio
import time

import pytest

from twinklybridge import configstore
from twinklybridge import engine as engine_mod
from twinklybridge.app import Supervisor
from twinklybridge.config import load_config, parse_config
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


def test_disabling_one_controller_leaves_the_others_streaming(fast):
    async def scenario():
        a = FakeTwinkly(pixels=128, api="v2", host="127.0.0.2")
        b = FakeTwinkly(pixels=128, api="v2", host="127.0.0.3")
        await a.start()
        await b.start()
        spec_a = _spec("A", "127.0.0.2", 0, a)
        spec_b = _spec("B", "127.0.0.3", 1, b)
        eng = Engine(_cfg([spec_a, spec_b]), sender=UdpSender(), hostname="box")
        await eng.start(open_sockets=False)
        try:
            assert await wait_for(lambda: all(d.ready for d in eng.devices))
            eng.feed("artnet", 0, bytes((10, 20, 30, 40)) * 128, "2.0.0.1")
            eng.feed("artnet", 1, bytes((1, 2, 3, 4)) * 128, "2.0.0.1")
            assert await wait_for(lambda: a.mode == "rt" and b.mode == "rt")
            b_runner = next(d for d in eng.devices if d.name == "B")
            assert a.logins == 1 and b.logins == 1
            b_frames = b.frames

            # Disable A. B must not blink: same connection, no re-login, still in rt.
            await eng.apply_config(_cfg([spec_b]))
            assert [d.name for d in eng.devices] == ["B"]
            assert next(d for d in eng.devices if d.name == "B") is b_runner
            assert b.logins == 1 and b.mode == "rt"
            assert await wait_for(lambda: b.frames > b_frames)  # still streaming
            assert await wait_for(lambda: a.mode == "movie")  # A handed back to its effect

            # Re-add A: it reconnects, B is still never touched.
            await eng.apply_config(_cfg([spec_a, spec_b]))
            assert await wait_for(lambda: a.logins == 2 and eng.devices[0].name == "A")
            assert next(d for d in eng.devices if d.name == "B") is b_runner and b.logins == 1
        finally:
            await eng.stop()
            await a.stop()
            await b.stop()

    asyncio.run(scenario())


def test_editing_one_controller_restarts_only_it(fast):
    async def scenario():
        a = FakeTwinkly(pixels=128, api="v2", host="127.0.0.2")
        b = FakeTwinkly(pixels=128, api="v2", host="127.0.0.3")
        await a.start()
        await b.start()
        spec_a = _spec("A", "127.0.0.2", 0, a)
        spec_b = _spec("B", "127.0.0.3", 1, b)
        eng = Engine(_cfg([spec_a, spec_b]), sender=UdpSender(), hostname="box")
        await eng.start(open_sockets=False)
        try:
            assert await wait_for(lambda: all(d.ready for d in eng.devices))
            b_runner = next(d for d in eng.devices if d.name == "B")
            assert a.logins == 1 and b.logins == 1

            # Change A's fps only. A restarts (new login); B is left alone.
            await eng.apply_config(_cfg([{**spec_a, "fps": 40}, spec_b]))
            assert await wait_for(lambda: a.logins == 2)
            a_runner = next(d for d in eng.devices if d.name == "A")
            assert a_runner.cfg.fps == 40
            assert next(d for d in eng.devices if d.name == "B") is b_runner and b.logins == 1
        finally:
            await eng.stop()
            await a.stop()
            await b.stop()

    asyncio.run(scenario())


def test_reload_applies_in_place_unless_the_input_changes(tmp_path, monkeypatch):
    cfg_path = tmp_path / "bridge.yaml"
    cfg_path.write_text(
        "input: {cidr: 10.0.0.2/24}\ndevices:\n  - {name: A, ip: 10.0.0.5, universe: 0.0.0, pixels: 80}\n"
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
        # A device edit leaves the input alone -> applied in place, no engine swap.
        raw = configstore.load_raw(str(cfg_path))
        configstore.patch_device(raw, "A", {"enabled": "false"})
        cfg_path.write_text(configstore.dump(raw))
        await sup.reload()
        assert calls == {"apply": 1, "swap": 0}

        # Turning sACN on changes the listen sockets -> full engine restart.
        raw = configstore.load_raw(str(cfg_path))
        configstore.set_sections(raw, {"input": {"cidr": "10.0.0.2/24", "artnet": "true", "sacn": "true"}})
        cfg_path.write_text(configstore.dump(raw))
        await sup.reload()
        assert calls == {"apply": 1, "swap": 1}

    asyncio.run(main())
