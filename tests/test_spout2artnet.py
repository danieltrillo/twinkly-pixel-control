"""spout2artnet: map parsing, sampling, colour, Art-Net packets, and a loopback run."""

import os
import socket
import struct
import sys
import threading

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "spout2artnet"))
import spout2artnet as s  # noqa: E402

SLICE = (
    '<DmxSlice uniqueId="{uid}"><Params name="Common"><Param name="Name" T="STRING" default="Layer" value="{name}"/>'
    "{enabled}</Params>"
    '<Params name="Input"><ParamChoice name="Fixture" T="STRING" default="" value="f"/>'
    '<ParamRange name="Start Channel" T="DOUBLE" default="1" value="{start}"/></Params>'
    '<InputRect orientation="0">{v}</InputRect>'
    '<OutputRect orientation="0"><v x="-0.5" y="-0.5"/><v x="0.5" y="-0.5"/><v x="0.5" y="0.5"/><v x="-0.5" y="0.5"/></OutputRect>'
    '<FixtureInstance name="FixtureInstance"><Fixture name="Fixture" uuid="u" fixtureName="">'
    '<Params name="Params"><ParamFixturePixels storage="0" name="Pixels">'
    '<ParamRange name="Width" T="DOUBLE" default="1" value="{w}"/>'
    '<ParamRange name="Height" T="DOUBLE" default="1" value="{h}"/>'
    '<ParamChoice name="Color Format" T="STRING" default="rgb" value="{fmt}"/>{gamma}'
    "</ParamFixturePixels></Params></Fixture></FixtureInstance></DmxSlice>"
)


def screen(name, lumi, slices, target="TT_DISABLED", subnet=0, universe=0):
    return (
        f'<DmxScreen name="{name}" uniqueId="1{lumi}" LumiverseId="{lumi}">'
        f'<Params name="Params"><Param name="Name" T="STRING" default="" value="{name}"/>'
        '<Param name="Enabled" T="BOOL" default="1" value="1"/></Params><layers>'
        + "".join(slices)
        + '</layers><OutputDevice><OutputDeviceDmx name="Lumiverse" deviceId="Lumiverse"><Params name="Params"/>'
        f'<DmxOutputParams name="Params"><Param name="TargetIP" default="TT_DISABLED" value="{target}"/>'
        f'<ParamRange name="Subnet" T="DOUBLE" default="0" value="{subnet}"/>'
        f'<ParamRange name="Universe" T="DOUBLE" default="0" value="{universe}"/>'
        "</DmxOutputParams></OutputDeviceDmx></OutputDevice></DmxScreen>"
    )


def rect(x0, y0, x1, y1):
    """Axis-aligned rect, v0 top-left, v1 top-right, v2 bottom-right, v3 bottom-left."""
    return f'<v x="{x0}" y="{y0}"/><v x="{x1}" y="{y0}"/><v x="{x1}" y="{y1}"/><v x="{x0}" y="{y1}"/>'


def slice_(name, start, v, w=1, h=1, fmt="rgbw", gamma="", enabled=""):
    return SLICE.format(
        uid=abs(hash(name)) % 10**9, name=name, start=start, v=v, w=w, h=h, fmt=fmt, gamma=gamma, enabled=enabled
    )


def write_map(tmp_path, screens, w=100, h=50):
    p = tmp_path / "map.xml"
    p.write_text(
        '<?xml version="1.0" encoding="utf-8"?><XmlState name="t"><ScreenSetup name="ScreenSetup">'
        f'<CurrentCompositionTextureSize width="{w}" height="{h}"/><screens>'
        + "".join(screens)
        + "</screens></ScreenSetup></XmlState>"
    )
    return str(p)


@pytest.fixture
def two_lumis(tmp_path):
    # Lumiverse 1: a vertical 1x4 line at x=10 (y 0..40, pixel 1 at the top), then a
    # pixel that starts on universe 2. Lumiverse 2: one RGB pixel, its own target IP.
    l1 = [
        slice_(
            "line", 1, rect(5, 0, 15, 40), h=4, gamma='<ParamRange name="Gamma" T="DOUBLE" default="2.5" value="2.2"/>'
        ),
        slice_("far", 513, rect(60, 20, 70, 30)),
        slice_("off", 600, rect(0, 0, 1, 1), enabled='<Param name="Enabled" T="BOOL" default="1" value="0"/>'),
    ]
    l2 = [slice_("rgb", 1, rect(90, 40, 100, 50), fmt="rgb")]
    return write_map(
        tmp_path, [screen("Lumiverse 1", 0, l1), screen("Lumiverse 2", 1, l2, target="10.0.0.2", subnet=1, universe=2)]
    )


def test_load(two_lumis):
    rig = s.load_resolume(two_lumis)
    assert (rig.comp_w, rig.comp_h) == (100, 50)
    l1, l2 = rig.lumiverses
    assert [sl.name for sl in l1.slices] == ["line", "far"]  # disabled slice skipped
    assert l1.slices[0].height == 4 and l1.slices[0].gamma == 2.2
    assert (l2.subnet, l2.universe, l2.target, l2.base) == (1, 2, "10.0.0.2", 0x12)
    assert rig.pixels == 6


def test_cell_centres_order():
    sl = s.Slice("x", 1, np.array([[0, 0], [10, 0], [10, 40], [0, 40]], float), 1, 4, "rgbw", None)
    assert s.cell_centres(sl).tolist() == [[5, 5], [5, 15], [5, 25], [5, 35]]
    grid = s.Slice("g", 1, np.array([[0, 0], [20, 0], [20, 20], [0, 20]], float), 2, 2, "rgb", None)
    assert s.cell_centres(grid).tolist() == [[5, 5], [15, 5], [5, 15], [15, 15]]  # row by row


def test_targets(two_lumis):
    rig = s.load_resolume(two_lumis)
    with pytest.raises(SystemExit, match="lumiverse 1"):
        s.resolve_targets(rig, {}, {})
    sends = s.resolve_targets(rig, {"1": "10.0.0.1"}, {})
    assert [(lu.number, ip, net) for lu, ip, net in sends] == [(1, "10.0.0.1", 0), (2, "10.0.0.2", 0)]
    sends = s.resolve_targets(rig, {"all": "10.9.9.9", "2": "off"}, {"1": 1})
    assert [(lu.number, ip, net) for lu, ip, net in sends] == [(1, "10.9.9.9", 1)]


def unpack(pkt):
    head = s._ARTDMX.unpack(pkt[:18])
    assert head[0] == b"Art-Net\x00" and head[1] == 0x5000 and head[3] == 14
    pa = head[6] | (head[7] << 8)
    n = (head[8] << 8) | head[9]
    return pa, head[4], pkt[18 : 18 + n]


def test_render_and_packets(two_lumis):
    rig = s.load_resolume(two_lumis)
    sends = s.resolve_targets(rig, {"1": "10.0.0.1"}, {"2": 3})
    r = s.Renderer(rig, sends, gamma=1.0, brightness=1.0, white="none", area=1)
    img = np.zeros((50, 100, 4), np.uint8)
    img[0:10, :, 0] = 200  # pixel 1 of the line: red
    img[30:40, :, 2] = 100  # pixel 4: blue
    img[20:30, 60:70] = (1, 2, 3, 255)  # "far"
    img[40:50, 90:100] = (9, 8, 7, 255)
    r.render(img)
    pk = r.packets()
    got = {(addr[0], unpack(p)[0]): unpack(p) for p, addr in pk}
    assert sorted(got) == [("10.0.0.1", 0), ("10.0.0.1", 1), ("10.0.0.2", 0x312)]
    u0 = got[("10.0.0.1", 0)][2]
    assert len(u0) == 16 and u0[:4] == bytes([200, 0, 0, 0]) and u0[12:16] == bytes([0, 0, 100, 0])
    assert got[("10.0.0.1", 1)][2] == bytes([1, 2, 3, 0])
    assert got[("10.0.0.2", 0x312)][2] == bytes([9, 8, 7, 0])  # rgb, padded to even length
    assert got[("10.0.0.1", 0)][1] == 1
    assert unpack(r.packets()[0][0])[1] == 2  # sequence counts up


def test_gamma_from_xml_and_scaling(two_lumis):
    rig = s.load_resolume(two_lumis)
    sends = s.resolve_targets(rig, {"all": "127.0.0.1"}, {})
    r = s.Renderer(rig, sends, gamma=None, brightness=1.0, white="none", area=1)
    assert r.gamma == 2.2
    # a Spout sender at half the composition size still lands on the same spots
    img = np.zeros((25, 50, 4), np.uint8)
    img[0:5, :, 1] = 255
    r.render(img)
    assert r.outputs[0].buffer[:4].tolist() == [0, 255, 0, 0]
    assert r.size == (50, 25)


def test_white_modes():
    rgb = np.array([[200, 100, 50]], np.uint8)
    assert s.to_rgbw(rgb, "none").tolist() == [[200, 100, 50, 0]]
    assert s.to_rgbw(rgb, "min").tolist() == [[150, 50, 0, 50]]
    assert s.to_rgbw(rgb, "add").tolist() == [[200, 100, 50, 50]]
    lut = s.make_lut(2.0, 0.5)
    assert lut[0] == 0 and lut[255] == 128 and lut[128] == round(255 * (128 / 255) ** 2 * 0.5)


def test_area_average(two_lumis):
    rig = s.load_resolume(two_lumis)
    sends = s.resolve_targets(rig, {"all": "127.0.0.1"}, {})
    r = s.Renderer(rig, sends, gamma=1.0, brightness=1.0, white="none", area=3)
    img = np.zeros((50, 100, 3), np.uint8)
    img[4, 9:12, 0] = 90  # top row of the 3x3 patch around (10, 5)
    r.render(img)
    assert r.outputs[0].buffer[0] == 30


def test_png_snapshot(tmp_path, two_lumis):
    rig = s.load_resolume(two_lumis)
    r = s.Renderer(rig, s.resolve_targets(rig, {"all": "127.0.0.1"}, {}), gamma=1, brightness=1, white="none", area=1)
    img = np.zeros((50, 100, 3), np.uint8)
    r.render(img)
    out = tmp_path / "snap.png"
    s.snapshot(str(out), img, r.groups)
    data = out.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n" and struct.unpack(">II", data[16:24]) == (100, 50)


def test_run_loopback(two_lumis, capsys):
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        rx.bind(("127.0.0.1", s.ARTNET_PORT))
    except OSError:
        pytest.skip("Art-Net port busy")
    rx.settimeout(0.2)
    seen = []

    def listen():
        while True:
            try:
                seen.append(rx.recv(2048))
            except OSError:
                return

    t = threading.Thread(target=listen, daemon=True)
    t.start()
    try:
        s.main(
            [
                "run",
                "--map",
                two_lumis,
                "--target",
                "all=127.0.0.1",
                "--net",
                "2=1",
                "--source",
                "test:white",
                "--fps",
                "20",
                "--duration",
                "0.5",
                "--gamma",
                "1",
            ]
        )
    finally:
        t.join(1.0)
        rx.close()
    addrs = {unpack(p)[0] for p in seen}
    assert addrs == {0, 1, (1 << 8) | 0x12}
    assert unpack(seen[0])[2][:4] == bytes([255, 255, 255, 0])
    assert len(seen) >= 3 * 5
