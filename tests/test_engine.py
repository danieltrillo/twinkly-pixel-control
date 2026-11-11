import asyncio
import time

import pytest

from twinklybridge import artnet, sacn
from twinklybridge import engine as engine_mod
from twinklybridge.config import parse_config
from twinklybridge.engine import DeviceRunner, Engine, UdpSender
from twinklybridge.testtools import FakeTwinkly

BRIDGE_IP = "2.0.0.20"


def make_config(devices, **input_extra):
    return parse_config({"input": {"cidr": f"{BRIDGE_IP}/8", **input_extra}, "devices": devices})


async def wait_for(cond, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        await asyncio.sleep(0.02)
    return cond()


def rgbw_universe(pixels, color):
    return bytes(color) * pixels


# -- pure engine behaviour ----------------------------------------------------


def test_artdmx_lands_in_the_right_pixels():
    cfg = make_config([{"name": "T", "ip": "10.0.0.1", "universe": "0.0.0", "pixels": 200}])
    eng = Engine(cfg, sender=lambda p, a: True, hostname="box")
    dev = eng.devices[0]
    eng.handle_artnet(artnet.encode_dmx(0, rgbw_universe(128, (1, 2, 3, 4))), "2.0.0.1")
    eng.handle_artnet(artnet.encode_dmx(1, rgbw_universe(72, (9, 8, 7, 6))), "2.0.0.1")
    assert dev.buffer[:4] == bytes([1, 2, 3, 4])
    assert dev.buffer[127 * 4 : 128 * 4] == bytes([1, 2, 3, 4])
    assert dev.buffer[128 * 4 : 129 * 4] == bytes([9, 8, 7, 6])
    assert dev.buffer[-4:] == bytes([9, 8, 7, 6])
    assert dev.dirty and dev.rx.total == 2


def test_short_universe_only_fills_what_arrived():
    cfg = make_config([{"ip": "10.0.0.1", "universe": 0, "pixels": 128}])
    eng = Engine(cfg, sender=lambda p, a: True)
    eng.handle_artnet(artnet.encode_dmx(0, bytes([5] * 8)), "2.0.0.1")
    dev = eng.devices[0]
    assert dev.buffer[:8] == bytes([5] * 8) and dev.buffer[8:12] == bytes(4)


def test_unmapped_universe_is_listed_and_own_packets_ignored():
    cfg = make_config([{"ip": "10.0.0.1", "universe": 0, "pixels": 10}])
    eng = Engine(cfg, sender=lambda p, a: True)
    eng.handle_artnet(artnet.encode_dmx(7, b"\x01\x02"), "2.0.0.1")
    eng.handle_artnet(artnet.encode_dmx(0, b"\x01\x02"), BRIDGE_IP)  # our own echo
    assert list(eng.unmapped) == [("artnet", BRIDGE_IP, 7)]  # (protocol, feed, universe)
    assert eng.devices[0].rx.total == 0


def test_sacn_input():
    cfg = parse_config(
        {
            "input": {"cidr": f"{BRIDGE_IP}/8", "artnet": False, "sacn": True},
            "devices": [{"ip": "10.0.0.1", "universe": 5, "pixels": 2, "protocol": "sacn"}],
        }
    )
    eng = Engine(cfg, sender=lambda p, a: True)
    pkt = sacn.encode_data(5, bytes([1, 2, 3, 4, 5, 6, 7, 8]), b"\x00" * 16, "test", 100, 1)
    eng.handle_sacn(pkt, "2.0.0.1")
    assert bytes(eng.devices[0].buffer) == bytes([1, 2, 3, 4, 5, 6, 7, 8])


def test_poll_reply_lists_every_universe():
    cfg = make_config(
        [
            {"name": "A", "ip": "10.0.0.1", "universe": "0.0.0", "pixels": 1500},  # 0.0.0 - 0.0.11
            {"name": "B", "ip": "10.0.0.2", "universe": "0.0.12", "pixels": 600},  # 0.0.12 - 0.1.0
        ]
    )
    eng = Engine(cfg, sender=lambda p, a: True, hostname="box")
    replies = [artnet.decode_poll_reply(p) for p in eng.build_poll_replies(None, time.monotonic())]
    universes = sorted(u for r in replies for u in r.output_universes())
    assert universes == list(range(0, 17))
    assert [r.bind_index for r in replies] == list(range(1, len(replies) + 1))
    assert all(r.ip == BRIDGE_IP and r.short_name == "TwinklyBridge" for r in replies)


# -- against a fake Twinkly ---------------------------------------------------


@pytest.fixture
def fast(monkeypatch):
    monkeypatch.setattr(engine_mod, "MODE_CHECK", 0.2)
    monkeypatch.setattr(engine_mod, "BACKOFF", (0.1,))
    monkeypatch.setattr(engine_mod, "CONTROL_TICK", 0.02)


def run(coro):
    return asyncio.run(coro)


async def start_bridge(fake, **device):
    spec = {
        "name": "Tree",
        "ip": "127.0.0.1",
        "universe": 0,
        "pixels": fake.pixels,
        "http_port": fake.http_port,
        "rt_port": fake.rt_port,
        **device,
    }
    eng = Engine(make_config([spec]), sender=UdpSender(), hostname="box")
    await eng.start(open_sockets=False)
    return eng


def test_streams_pixels_in_wrgb_order(fast):
    async def scenario():
        fake = FakeTwinkly(pixels=200)
        await fake.start()
        eng = await start_bridge(fake)
        try:
            dev = eng.devices[0]
            assert await wait_for(lambda: dev.ready)
            assert fake.mode == "movie"  # nothing is touched until input arrives
            eng.feed("artnet", 0, rgbw_universe(128, (10, 20, 30, 40)), "2.0.0.1")
            eng.feed("artnet", 1, rgbw_universe(72, (1, 2, 3, 4)), "2.0.0.1")
            assert await wait_for(lambda: fake.mode == "rt" and fake.frames > 0 and fake.pixel(199) == (4, 1, 2, 3))
            assert fake.pixel(0) == (40, 10, 20, 30)
            assert fake.bad_packets == 0
            assert dev.state == "streaming" and dev.previous_mode == "movie"
        finally:
            await eng.stop()
            await fake.stop()
        assert fake.mode == "movie"  # handed back on shutdown

    run(scenario())


def test_keepalive_holds_rt_mode_without_changes(fast):
    async def scenario():
        fake = FakeTwinkly(pixels=10, rt_timeout=0.8)
        await fake.start()
        eng = await start_bridge(fake)
        try:
            eng.feed("artnet", 0, rgbw_universe(10, (1, 1, 1, 1)), "2.0.0.1")
            assert await wait_for(lambda: fake.mode == "rt")
            before = fake.frames
            await asyncio.sleep(1.5)  # longer than the fake's rt timeout, input "static" but still live
            assert fake.mode == "rt"
            assert fake.frames > before  # keepalive frames kept it alive
        finally:
            await eng.stop()
            await fake.stop()

    run(scenario())


def test_release_on_loss_hands_back(fast):
    async def scenario():
        fake = FakeTwinkly(pixels=10)
        await fake.start()
        eng = await start_bridge(fake, on_loss="release", loss_timeout=0.5)
        try:
            eng.feed("artnet", 0, rgbw_universe(10, (1, 1, 1, 1)), "2.0.0.1")
            assert await wait_for(lambda: fake.mode == "rt")
            assert await wait_for(lambda: fake.mode == "movie", 3.0)
            assert eng.devices[0].state == "released"
            eng.feed("artnet", 0, rgbw_universe(10, (2, 2, 2, 2)), "2.0.0.1")
            assert await wait_for(lambda: fake.mode == "rt")
        finally:
            await eng.stop()
            await fake.stop()

    run(scenario())


def test_black_on_loss(fast):
    async def scenario():
        fake = FakeTwinkly(pixels=4)
        await fake.start()
        eng = await start_bridge(fake, on_loss="black", loss_timeout=0.5)
        try:
            eng.feed("artnet", 0, rgbw_universe(4, (9, 9, 9, 9)), "2.0.0.1")
            assert await wait_for(lambda: fake.pixel(0) == (9, 9, 9, 9))
            assert await wait_for(lambda: fake.pixel(0) == (0, 0, 0, 0), 3.0)
            assert fake.mode == "rt"
        finally:
            await eng.stop()
            await fake.stop()

    run(scenario())


def test_puts_device_back_in_rt_if_it_leaves(fast):
    async def scenario():
        fake = FakeTwinkly(pixels=4)
        await fake.start()
        eng = await start_bridge(fake)
        try:
            eng.feed("artnet", 0, rgbw_universe(4, (1, 1, 1, 1)), "2.0.0.1")
            assert await wait_for(lambda: fake.mode == "rt")
            fake.mode = "movie"  # someone pressed a button in the app
            assert await wait_for(lambda: fake.mode == "rt", 2.0)
        finally:
            await eng.stop()
            await fake.stop()

    run(scenario())


def test_logs_in_again_when_token_is_rejected(fast):
    async def scenario():
        fake = FakeTwinkly(pixels=4)
        await fake.start()
        eng = await start_bridge(fake)
        try:
            eng.feed("artnet", 0, rgbw_universe(4, (1, 1, 1, 1)), "2.0.0.1")
            assert await wait_for(lambda: fake.mode == "rt")
            fake.token_b64 = "invalid="  # device rebooted / token expired
            fake.mode = "movie"
            assert await wait_for(lambda: fake.logins >= 2 and fake.mode == "rt", 3.0)
            frames = fake.frames
            assert await wait_for(lambda: fake.frames > frames)
        finally:
            await eng.stop()
            await fake.stop()

    run(scenario())


def test_offline_device_recovers(fast):
    async def scenario():
        fake = FakeTwinkly(pixels=4)
        await fake.start()
        port, rt = fake.http_port, fake.rt_port
        await fake.stop()
        eng = await start_bridge(fake)
        try:
            dev = eng.devices[0]
            assert await wait_for(lambda: dev.state == "offline")
            assert dev.error
            fake2 = FakeTwinkly(pixels=4, http_port=port, rt_port=rt)
            await fake2.start()
            try:
                eng.feed("artnet", 0, rgbw_universe(4, (3, 3, 3, 3)), "2.0.0.1")
                assert await wait_for(lambda: fake2.mode == "rt" and fake2.frames > 0, 5.0)
            finally:
                await eng.stop()
                await fake2.stop()
        finally:
            await eng.stop()

    run(scenario())


def test_pixel_count_mismatch_is_reported(fast):
    async def scenario():
        fake = FakeTwinkly(pixels=100)
        await fake.start()
        eng = await start_bridge(fake, pixels=120)
        try:
            dev = eng.devices[0]
            assert await wait_for(lambda: dev.ready)
            assert any("device reports 100" in w for w in dev.warnings)
            eng.feed("artnet", 0, rgbw_universe(120, (1, 2, 3, 4)), "2.0.0.1")
            assert await wait_for(lambda: fake.frames > 0)
            assert len(fake.last_frame) == 400
        finally:
            await eng.stop()
            await fake.stop()

    run(scenario())


def test_rgb_strand_auto_order(fast):
    async def scenario():
        fake = FakeTwinkly(pixels=3, profile="RGB")
        await fake.start()
        eng = await start_bridge(fake)
        try:
            eng.feed("artnet", 0, rgbw_universe(3, (7, 8, 9, 99)), "2.0.0.1")
            assert await wait_for(lambda: fake.frames > 0 and fake.pixel(2) == (7, 8, 9))
            assert eng.devices[0].order == "RGB"
        finally:
            await eng.stop()
            await fake.stop()

    run(scenario())


def test_status_is_json_friendly():
    import json

    cfg = make_config([{"ip": "10.0.0.1", "universe": 0, "pixels": 10}])
    eng = Engine(cfg, sender=lambda p, a: True)
    json.dumps(eng.status())
    assert isinstance(eng.devices[0], DeviceRunner)


def test_streams_to_pro_v2_firmware(fast):
    async def scenario():
        fake = FakeTwinkly(pixels=200, api="v2", firmware="4.0.1", product="TWPROCTRLPLC21")
        await fake.start()
        eng = await start_bridge(fake)
        try:
            dev = eng.devices[0]
            assert await wait_for(lambda: dev.ready)
            assert dev.info.api == "v2" and dev.info.firmware == "4.0.1" and dev.protocol == 3
            assert dev.info.led_profile == "RGBW" and dev.info.group_mode == "none"
            assert not dev.warnings
            eng.feed("artnet", 0, rgbw_universe(128, (10, 20, 30, 40)), "2.0.0.1")
            eng.feed("artnet", 1, rgbw_universe(72, (1, 2, 3, 4)), "2.0.0.1")
            assert await wait_for(lambda: fake.mode == "rt" and fake.frames > 0 and fake.pixel(199) == (4, 1, 2, 3))
            assert fake.pixel(0) == (40, 10, 20, 30) and fake.bad_packets == 0
            assert fake.modes == ["rt"]  # via /xled/v2/application/mode
        finally:
            await eng.stop()
            await fake.stop()
        assert fake.mode == "movie" and fake.modes == ["rt", "movie"]

    run(scenario())


def test_group_slave_is_reported(fast):
    async def scenario():
        fake = FakeTwinkly(pixels=4, api="v2", firmware="4.0.1", group_slave=True)
        await fake.start()
        eng = await start_bridge(fake)
        try:
            dev = eng.devices[0]
            assert await wait_for(lambda: dev.ready)
            assert any("group slave" in w for w in dev.warnings)
            eng.feed("artnet", 0, rgbw_universe(4, (1, 1, 1, 1)), "2.0.0.1")
            assert await wait_for(lambda: dev.state == "offline", 3.0)
            assert "403" in dev.error and "group" in dev.error
            assert fake.mode == "movie" and fake.frames == 0
        finally:
            await eng.stop()
            await fake.stop()

    run(scenario())


def test_rejected_token_backs_off_instead_of_spinning(fast):
    async def scenario():
        fake = FakeTwinkly(pixels=4, api="v2", firmware="4.0.1")
        fake.reject_tokens = True  # every authenticated call is 401, even right after login
        await fake.start()
        eng = await start_bridge(fake)
        try:
            dev = eng.devices[0]
            assert await wait_for(lambda: dev.state == "offline" and "token rejected" in dev.error)
            await asyncio.sleep(0.6)
            assert fake.logins <= 10  # BACKOFF is 0.1 s here; a tight loop would log in hundreds of times
        finally:
            await eng.stop()
            await fake.stop()

    run(scenario())


# -- camera mapping scan ------------------------------------------------------


def test_mapping_scan_replaces_the_input_then_hands_back():
    from types import SimpleNamespace

    from twinklybridge import mapcode
    from twinklybridge.twinkly import DeviceInfo

    cfg = make_config([{"name": "T", "ip": "10.0.0.1", "universe": "0.0.0", "pixels": 8}])
    sent = []
    eng = Engine(cfg, sender=lambda p, a: sent.append(p) or True, hostname="box")
    dev = eng.devices[0]
    now = [100.0]
    dev.clock = lambda: now[0]
    with pytest.raises(engine_mod.ScanError):
        dev.start_scan(slot=0.4, brightness=50, color="white", repeat=1)  # not connected yet

    # as if connect() had run against an 8 pixel RGBW controller
    dev.order, dev.info, dev._ready, dev.state = "WRGB", DeviceInfo(led_profile="RGBW", number_of_led=8), True, "ready"
    dev.client = SimpleNamespace(token=b"\x01" * 8)
    job = dev.start_scan(slot=0.4, brightness=50, color="white", repeat=1)
    assert dev.scanning(now[0]) and dev.wants_rt(now[0]) and not dev.ever_had_data
    assert job.pixels == 8 and job.pass_seconds == len(mapcode.schedule(8)) * 0.4
    dev.rt_armed = True

    def pixels():
        return sent[-1][12:]  # protocol v3 header is 12 bytes

    dev.frame_step(now[0])  # a lead slot: everything off
    assert pixels() == bytes(8 * 4)
    now[0] = 100.0 + mapcode.LEAD * 0.4 + 0.1  # first mark: every pixel on, W chip only
    dev.frame_step(now[0])
    assert pixels()[0::4] == bytes([50] * 8) and pixels()[1::4] == bytes(8)
    first_data = mapcode.LEAD + mapcode.MARK + 1
    now[0] = 100.0 + first_data * 0.4 + 0.1  # bit 0, phase 1
    dev.frame_step(now[0])
    for i in range(8):
        assert (pixels()[4 * i] == 50) == bool(mapcode.codeword(i, 8) & 1)
    assert dev.status(now[0])["scan"]["pass"] == 1

    now[0] = 100.0 + job.total_seconds + 0.01
    dev.frame_step(now[0])
    assert dev.scan is None and not dev.wants_rt(now[0])  # never had input: hands back
    assert dev.status(now[0])["scan"] is None

    # input arriving during a scan is remembered and shown once the scan is over
    dev.start_scan(slot=0.4, brightness=50, color="rgb", repeat=1)
    eng.feed("artnet", 0, bytes([9, 8, 7, 6] * 8), "2.0.0.1", now[0])  # stamped with the test clock
    dev.frame_step(now[0])
    assert pixels() == bytes(8 * 4)  # lead slot of the new scan, not the Art-Net look
    dev.stop_scan()
    assert not dev.scanning(now[0]) and dev.wants_rt(now[0])
    dev.frame_step(now[0])
    assert dev.scan is None and pixels()[:4] == bytes([6, 9, 8, 7])  # WRGB of the console's RGBW
