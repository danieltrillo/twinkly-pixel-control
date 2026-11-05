from twinklybridge.pixels import convert
from twinklybridge.twinkly import (
    FRAGMENT_BYTES,
    DeviceInfo,
    build_packets,
    pack_ip_reversed,
    parse_discovery_reply,
    protocol_for_firmware,
    valid_order,
)

TOKEN = bytes(range(1, 9))


def test_protocol_for_firmware():
    assert protocol_for_firmware("2.8.18") == 3
    assert protocol_for_firmware("2.4.14") == 3
    assert protocol_for_firmware("2.4.6") == 2
    assert protocol_for_firmware("1.99.24") == 1
    assert protocol_for_firmware("3.5.0") == 3


def test_v3_fragments_at_900_bytes():
    frame = bytes(range(256)) * 9  # 2304 bytes
    packets = build_packets(3, TOKEN, frame, 576)
    assert len(packets) == 3
    for i, p in enumerate(packets):
        assert p[0] == 3 and p[1:9] == TOKEN and p[9:11] == b"\x00\x00" and p[11] == i
    assert [len(p) - 12 for p in packets] == [FRAGMENT_BYTES, FRAGMENT_BYTES, 2304 - 2 * FRAGMENT_BYTES]
    assert b"".join(p[12:] for p in packets) == frame


def test_v1_and_v2_headers():
    frame = b"\x01\x02\x03" * 5
    v1 = build_packets(1, TOKEN, frame, 5)[0]
    assert v1[:10] == b"\x01" + TOKEN + b"\x05" and v1[10:] == frame
    v2 = build_packets(2, TOKEN, frame, 5)[0]
    assert v2[:10] == b"\x02" + TOKEN + b"\x00" and v2[10:] == frame


def test_convert_rgbw_console_to_wrgb_wire():
    src = bytes([10, 20, 30, 40, 11, 21, 31, 41])  # two pixels R,G,B,W
    assert bytes(convert(src, "RGBW", "WRGB", 2)) == bytes([40, 10, 20, 30, 41, 11, 21, 31])


def test_convert_rgb_console_to_rgbw_strand_leaves_white_off():
    src = bytes([1, 2, 3, 4, 5, 6])
    assert bytes(convert(src, "RGB", "WRGB", 2)) == bytes([0, 1, 2, 3, 0, 4, 5, 6])


def test_convert_rgbw_console_to_rgb_strand_drops_white():
    src = bytes([1, 2, 3, 9])
    assert bytes(convert(src, "RGBW", "GRB", 1)) == bytes([2, 1, 3])


def test_convert_is_fast_enough_for_1500_pixels():
    import time

    src = bytes(1500 * 4)
    t0 = time.perf_counter()
    for _ in range(100):
        convert(src, "RGBW", "WRGB", 1500)
    assert time.perf_counter() - t0 < 0.5  # well under 5 ms a frame


def test_valid_order():
    assert valid_order("wrgb") and valid_order("GRB") and valid_order("RGBW")
    assert not valid_order("RGGB") and not valid_order("RG") and not valid_order("RGBWX")


def test_discovery_reply():
    data = pack_ip_reversed("10.20.30.101") + b"OK" + b"Twinkly_ABC123\x00"
    d = parse_discovery_reply(data, "10.20.30.101")
    assert (d.ip, d.name) == ("10.20.30.101", "Twinkly_ABC123")
    assert parse_discovery_reply(b"\x01discover", "1.2.3.4") is None


def test_device_info_from_gestalt():
    info = DeviceInfo.from_api({"led_profile": "rgbw", "number_of_led": "1500", "product_code": "X"}, "2.8.18")
    assert info.bytes_per_led == 4 and info.number_of_led == 1500 and info.firmware == "2.8.18"
