"""Whole-frame coalescing: the output only ever carries a complete input frame.

The console sends no ArtSync, so the bridge works out the frame boundary itself.
It stages each universe and promotes a device's pixel buffer only once every one
of its universes has landed. Without this, the output fed by the last-arriving
universe (the far end of a controller) sends the previous frame's pixels while
the rest send the new ones, so that one output trails ~1 frame — frame tearing.
"""

from twinklybridge import engine as engine_mod
from twinklybridge.config import ARTNET, parse_config
from twinklybridge.engine import Engine


def _eng(devices, cidr="127.0.0.1/8"):
    cfg = parse_config({"input": {"cidr": cidr}, "devices": devices})
    return Engine(cfg, sender=lambda p, addr: True)


def _dev_300():
    # 300 RGBW pixels -> three universes (128 + 128 + 44), i.e. three segments.
    eng = _eng([{"name": "A", "ip": "127.0.0.2", "universe": "0.0.0", "pixels": 300}])
    dev = eng.devices[0]
    assert len(dev.cfg.segments) == 3 and dev.cfg.coalesce is True
    return eng, dev


def test_output_holds_until_the_whole_frame_has_arrived():
    eng, dev = _dev_300()
    blank = bytearray(300 * 4)

    # Frame 1 lands one universe at a time. The output stays blank until the
    # last universe arrives, then shows the whole frame at once.
    eng.feed(ARTNET, 0, bytes((10, 10, 10, 10)) * 128, "src")
    eng.feed(ARTNET, 1, bytes((10, 10, 10, 10)) * 128, "src")
    assert dev.buffer == blank  # two of three universes in: still not shown
    eng.feed(ARTNET, 2, bytes((10, 10, 10, 10)) * 44, "src")
    assert dev.buffer[:4] == bytes((10, 10, 10, 10))  # latched on the last universe
    assert dev.buffer[-4:] == bytes((10, 10, 10, 10))
    assert dev.dirty


def test_the_trailing_output_cannot_run_a_frame_ahead():
    eng, dev = _dev_300()
    for u, n in ((0, 128), (1, 128), (2, 44)):
        eng.feed(ARTNET, u, bytes((10, 10, 10, 10)) * n, "src")

    # Frame 2's first two universes arrive. The far output (universe 2) has not.
    # The whole buffer must still read frame 1 — the first output must not jump
    # ahead of the last, which is exactly the tear we are fixing.
    eng.feed(ARTNET, 0, bytes((20, 20, 20, 20)) * 128, "src")
    eng.feed(ARTNET, 1, bytes((20, 20, 20, 20)) * 128, "src")
    assert dev.buffer[:4] == bytes((10, 10, 10, 10))  # first output held at frame 1
    assert dev.buffer[-4:] == bytes((10, 10, 10, 10))  # last output too

    eng.feed(ARTNET, 2, bytes((20, 20, 20, 20)) * 44, "src")
    assert dev.buffer[:4] == bytes((20, 20, 20, 20))  # whole of frame 2, together
    assert dev.buffer[-4:] == bytes((20, 20, 20, 20))


def test_a_dropped_universe_is_flushed_and_holds_its_last_value(monkeypatch):
    monkeypatch.setattr(engine_mod, "FRAME_LATCH_TIMEOUT", 0.1)
    eng, dev = _dev_300()

    # Frame 1 complete at t=0.
    for u, n in ((0, 128), (1, 128), (2, 44)):
        eng.feed(ARTNET, u, bytes((5, 5, 5, 5)) * n, "src", now=0.0)
    assert dev.buffer[:4] == bytes((5, 5, 5, 5)) and dev.buffer[-4:] == bytes((5, 5, 5, 5))

    # Frame 2 drops universe 2. Only 0 and 1 arrive, so nothing is shown yet.
    eng.feed(ARTNET, 0, bytes((9, 9, 9, 9)) * 128, "src", now=0.03)
    eng.feed(ARTNET, 1, bytes((9, 9, 9, 9)) * 128, "src", now=0.03)
    assert dev.buffer[:4] == bytes((5, 5, 5, 5))  # incomplete: still frame 1

    # Frame 3's first universe comes well past the timeout: the partial frame is
    # flushed — universes 0/1 fresh, the dropped universe 2 holds its last value.
    eng.feed(ARTNET, 0, bytes((9, 9, 9, 9)) * 128, "src", now=0.2)
    assert dev.buffer[:4] == bytes((9, 9, 9, 9))  # flushed
    assert dev.buffer[-4:] == bytes((5, 5, 5, 5))  # dropped universe held, not torn


def test_coalesce_off_writes_each_universe_as_it_lands():
    eng = _eng([{"name": "A", "ip": "127.0.0.2", "universe": "0.0.0", "pixels": 300, "coalesce": False}])
    dev = eng.devices[0]
    assert dev.cfg.coalesce is False

    # With coalescing off the old behaviour is back: the first universe shows up
    # immediately while the last is still blank (this is the tear, made opt-in).
    eng.feed(ARTNET, 0, bytes((7, 7, 7, 7)) * 128, "src")
    assert dev.buffer[:4] == bytes((7, 7, 7, 7))
    assert dev.buffer[-4:] == bytearray(4)
    assert dev.dirty


def test_coalesce_can_be_set_in_defaults():
    cfg = parse_config(
        {
            "input": {"cidr": "127.0.0.1/8"},
            "defaults": {"coalesce": False},
            "devices": [{"name": "A", "ip": "127.0.0.2", "universe": "0.0.0", "pixels": 64}],
        }
    )
    assert cfg.devices[0].coalesce is False
