import pytest

from twinklybridge.config import ConfigError, build_segments, overlaps, parse_config, parse_yaml

BASE_INPUT = {"cidr": "2.0.0.20/8"}


def cfg(devices, **extra):
    raw = {"input": dict(BASE_INPUT), "devices": devices}
    raw.update(extra)
    return parse_config(raw)


def test_rgbw_1500_pixels_uses_12_universes():
    c = cfg([{"name": "Tree", "ip": "10.0.0.1", "universe": "0.0.0", "pixels": 1500}])
    d = c.devices[0]
    assert d.channels == 4 and d.per_universe == 128
    segs = d.segments
    assert len(segs) == 12
    assert segs[0].universe == 0 and segs[0].count == 128 and segs[0].channel == 0
    assert segs[-1].universe == 11 and segs[-1].first_pixel == 1408 and segs[-1].count == 92
    assert d.universe_label == "0.0.0 - 0.0.11"


def test_universes_roll_over_subnet():
    c = cfg([{"ip": "10.0.0.1", "universe": "0.0.14", "pixels": 300}])
    assert [u for u in c.devices[0].universes] == [14, 15, 16]
    assert c.devices[0].universe_label == "0.0.14 - 0.1.0"


def test_rgb_input_fits_170_per_universe():
    c = cfg([{"ip": "10.0.0.1", "universe": 0, "pixels": 340, "input": "rgb"}])
    segs = c.devices[0].segments
    assert [s.count for s in segs] == [170, 170]


def test_start_channel_shifts_only_first_universe():
    segs = build_segments(0, 200, 4, 128, start_channel=9)
    assert segs[0].channel == 8 and segs[0].count == 126  # (512 - 8) // 4
    assert segs[1].channel == 0 and segs[1].first_pixel == 126 and segs[1].count == 74


def test_defaults_apply_and_device_overrides():
    c = cfg(
        [
            {"name": "A", "ip": "10.0.0.1", "universe": 0, "pixels": 10},
            {"name": "B", "ip": "10.0.0.2", "universe": 1, "pixels": 10, "fps": 40, "order": "grbw"},
        ],
        defaults={"fps": 30, "on_loss": "black", "input": "GRBW"},
    )
    a, b = c.devices
    assert (a.fps, a.on_loss, a.input, a.order) == (30, "black", "GRBW", "auto")
    assert (b.fps, b.order) == (40, "GRBW")


def test_packet_gap_defaults_on_and_can_be_overridden():
    c = cfg(
        [
            {"name": "A", "ip": "10.0.0.1", "universe": 0, "pixels": 10},
            {"name": "B", "ip": "10.0.0.2", "universe": 1, "pixels": 10, "packet_gap_ms": 0},
        ]
    )
    assert [d.packet_gap_ms for d in c.devices] == [1.0, 0.0]
    c = cfg([{"name": "A", "ip": "10.0.0.1", "universe": 0, "pixels": 10}], defaults={"packet_gap_ms": 2.5})
    assert c.devices[0].packet_gap_ms == 2.5
    with pytest.raises(ConfigError, match="defaults.packet_gap_ms"):
        cfg([{"name": "A", "ip": "10.0.0.1", "universe": 0, "pixels": 10}], defaults={"packet_gap_ms": 50})


def test_sacn_device_needs_sacn_input():
    with pytest.raises(ConfigError, match="input.sacn is off"):
        cfg([{"ip": "10.0.0.1", "universe": 1, "pixels": 10, "protocol": "sacn"}])
    c = parse_config(
        {
            "input": {"ip": "2.0.0.20", "sacn": True},
            "devices": [{"ip": "10.0.0.1", "universe": 1, "pixels": 10, "protocol": "sacn"}],
        }
    )
    assert c.devices[0].universes == [1]


@pytest.mark.parametrize(
    "device, message",
    [
        ({"universe": 0, "pixels": 10}, "ip: is required"),
        ({"ip": "10.0.0.1", "pixels": 10}, "universe: is required"),
        ({"ip": "10.0.0.1", "universe": 0}, "pixels: is required"),
        ({"ip": "nope", "universe": 0, "pixels": 10}, "not an IPv4 address"),
        ({"ip": "10.0.0.1", "universe": 0, "pixels": 10, "order": "RGBX"}, "channel order"),
        ({"ip": "10.0.0.1", "universe": 0, "pixels": 10, "on_loss": "panic"}, "not one of"),
        ({"ip": "10.0.0.1", "universe": 0, "pixels": 10, "colour": "red"}, "unknown key 'colour'"),
        ({"ip": "10.0.0.1", "universe": "0.0.16", "pixels": 10}, "out of range"),
        ({"ip": "10.0.0.1", "universe": 32767, "pixels": 300}, "run past the last universe"),
    ],
)
def test_device_errors_name_the_key(device, message):
    with pytest.raises(ConfigError, match=message):
        cfg([device])


def test_duplicate_ip_and_name_rejected():
    with pytest.raises(ConfigError, match="already used"):
        cfg([{"ip": "10.0.0.1", "universe": 0, "pixels": 1}, {"ip": "10.0.0.1", "universe": 1, "pixels": 1}])
    with pytest.raises(ConfigError, match="used twice"):
        cfg(
            [
                {"name": "X", "ip": "10.0.0.1", "universe": 0, "pixels": 1},
                {"name": "X", "ip": "10.0.0.2", "universe": 1, "pixels": 1},
            ]
        )


def test_overlap_warning():
    c = cfg(
        [
            {"name": "A", "ip": "10.0.0.1", "universe": 0, "pixels": 200},
            {"name": "B", "ip": "10.0.0.2", "universe": 1, "pixels": 10},
        ]
    )
    assert overlaps(c) == ["artnet universe 0.0.1 feeds A, B"]


def test_input_broadcast_from_cidr():
    c = cfg([])
    assert (c.input.ip, c.input.broadcast) == ("2.0.0.20", "2.255.255.255")


def test_example_config_parses():
    import pathlib

    text = (pathlib.Path(__file__).parent.parent / "twinklybridge" / "example-config.yaml").read_text()
    c = parse_yaml(text)
    assert [d.name for d in c.devices] == ["Tree 1", "Tree 2"]
    assert c.devices[1].universes[0] == 12
