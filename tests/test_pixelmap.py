import csv
import importlib.util
import json
import math
import pathlib
import sys
import xml.etree.ElementTree as ET

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from twinklybridge import mapcode  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("pixelmap", ROOT / "tools" / "pixelmap.py")
pixelmap = importlib.util.module_from_spec(_spec)
sys.modules["pixelmap"] = pixelmap
_spec.loader.exec_module(pixelmap)


def _render(points, lit, shape, sigma=1.5, level=200.0):
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.full(shape, 12.0, np.float32)
    for (x, y), on in zip(points, lit, strict=True):
        if on:
            img += level * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2))
    return img


def test_decode_slots_finds_every_pixel():
    n = 60
    rng = np.random.default_rng(0)
    pts = [(10 + 9 * (i % 10) + rng.normal(0, 0.5), 10 + 12 * (i // 10)) for i in range(n)]
    shape = (90, 110)
    words = [mapcode.codeword(i, n) for i in range(n)]
    slots = np.stack([_render(pts, mapcode.lit_mask(s, n, words), shape) for s in mapcode.schedule(n) if s[0] == "bit"])
    on = _render(pts, [True] * n, shape)
    off = _render(pts, [False] * n, shape)
    found, _, _ = pixelmap.decode_slots(slots, on, off, n)
    assert [f["pixel"] for f in found] == list(range(n))
    for f in found:
        x, y = pts[f["pixel"]]
        assert math.hypot(f["x"] - x, f["y"] - y) < 0.5


def test_find_passes_two_back_to_back():
    n, slot, fps = 100, 0.4, 30.0
    sched = mapcode.schedule(n)
    words = [mapcode.codeword(i, n) for i in range(n)]
    levels = [sum(mapcode.lit_mask(s, n, words)) / n for s in sched]
    t = np.arange(0, 2 * len(sched) * slot + 2, 1 / fps)
    k = ((t - 1.0) / (slot * 1.003)).astype(int)
    tr = np.array([levels[i % len(sched)] if 0 <= i < 2 * len(sched) else 0.0 for i in k])
    tr[(t > 30) & (t < 40)] *= 0.8  # exposure drift on the second pass
    passes = pixelmap.find_passes(t, tr, n, slot)
    assert len(passes) == 2
    for p in passes:
        assert abs(p.slot - slot * 1.003) < 0.01


def _project(xyz, rvec, tvec, k):
    pts, _ = cv2.projectPoints(xyz, rvec, tvec, k, None)
    return pts.reshape(-1, 2)


def test_solve3d_recovers_shape(tmp_path):
    rng = np.random.default_rng(2)
    n = 400
    t = np.linspace(0, 1, n)
    xyz = np.stack(
        [(1 - t) * 0.8 * np.cos(t * 40), 2.0 * t - 1.0, (1 - t) * 0.8 * np.sin(t * 40)], axis=1
    )  # a cone tree, metres, Y up
    cam = xyz * [1, -1, -1]  # camera frame: Y down, Z away from camera 1 (a rotation, not a mirror)
    w, h, fov = 1920, 1080, 70.0
    f = (w / 2) / math.tan(math.radians(fov) / 2)
    k = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
    views = []
    for yaw in (0.0, 0.7):
        # camera 4 m out, orbiting the tree, looking at its middle
        rvec = np.array([0.0, yaw, 0.0])
        r = cv2.Rodrigues(rvec)[0]
        c = np.array([4 * math.sin(yaw), 0.0, -4 * math.cos(yaw)])
        tvec = -r @ c
        views.append(_project(cam, rvec, tvec, k) + rng.normal(0, 0.3, (n, 2)))
    paths = []
    for i, v in enumerate(views):
        doc = {
            "pixels": n,
            "source": f"v{i}",
            "image": {"width": w, "height": h},
            "found": [{"pixel": p, "x": float(x), "y": float(y)} for p, (x, y) in enumerate(v)],
        }
        p = tmp_path / f"v{i}.json"
        p.write_text(json.dumps(doc))
        paths.append(str(p))
    out = tmp_path / "3d"
    pixelmap.main(
        [
            "solve3d",
            paths[0],
            paths[1],
            "--out",
            str(out),
            "--scale",
            "1",
            str(n),
            str(np.linalg.norm(xyz[0] - xyz[-1])),
        ]
    )
    got = json.loads((out / "map.json").read_text())
    assert got["dims"] == 3 and len(got["found"]) > 0.95 * n
    ids = [q["pixel"] for q in got["found"]]
    a = np.array([[q["x"], q["y"], q["z"]] for q in got["found"]])
    b = xyz[ids]
    # same shape up to rotation/translation (scale fixed by --scale)
    a0, b0 = a - a.mean(0), b - b.mean(0)
    u, _, vt = np.linalg.svd(a0.T @ b0)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    err = np.linalg.norm(a0 @ r - b0, axis=1)
    assert np.median(err) < 0.03  # 3 cm on a 2 m tree


def test_export_madrix_and_resolume(tmp_path):
    n = 300
    found = [{"pixel": i, "x": float(i % 20) * 10, "y": float(i // 20) * 10, "duplicates": []} for i in range(n)]
    m = tmp_path / "map.json"
    m.write_text(json.dumps({"pixels": n, "dims": 2, "found": found}))
    mad, res = tmp_path / "m.csv", tmp_path / "r.xml"
    pixelmap.main(["export", str(m), "--madrix", str(mad), "--resolume", str(res), "--universe", "0.0.3"])
    rows = list(csv.DictReader(mad.open()))
    assert len(rows) == n
    assert rows[0]["DMX Universe"] == "4" and rows[0]["DMX Channel"] == "1"  # Art-Net 0.0.3 -> MADRIX 4
    assert rows[128]["DMX Universe"] == "5" and rows[128]["DMX Channel"] == "1"  # 128 RGBW px per universe
    assert rows[1]["DMX Channel"] == "5"
    assert len({(r["Position X"], r["Position Y"]) for r in rows}) == n
    root = ET.parse(res).getroot()
    slices = root.findall(".//DmxSlice")
    assert len(slices) == n
    starts = [float(p.get("value")) for s in slices for p in s.iter("ParamRange") if p.get("name") == "Start Channel"]
    assert starts[0] == 1 and starts[1] == 5 and starts[128] == 513
    uni = [p.get("value") for p in root.iter("ParamRange") if p.get("name") == "Universe"]
    assert uni == ["3"]


def test_export_from_bridge_config(tmp_path):
    cfg = tmp_path / "b.yaml"
    cfg.write_text(
        "input: {cidr: 10.0.0.2/24}\ndevices:\n  - {name: Tree, ip: 10.0.0.5, universe: 0.0.12, pixels: 200}\n"
    )
    m = tmp_path / "map.json"
    m.write_text(json.dumps({"pixels": 200, "dims": 2, "found": [{"pixel": 199, "x": 1.0, "y": 2.0}]}))
    mad = tmp_path / "m.csv"
    pixelmap.main(["export", str(m), "--madrix", str(mad), "--bridge-config", str(cfg)])
    row = list(csv.DictReader(mad.open()))[0]
    assert row["DMX Universe"] == "14" and row["DMX Channel"] == str((199 - 128) * 4 + 1)


def test_synth_video_end_to_end(tmp_path):
    video = str(tmp_path / "s.mp4")
    pixelmap.main(["synth", video, "--pixels", "80", "--size", "320", "240", "--blur", "1.5", "--hidden", "0"])
    out = tmp_path / "out"
    pixelmap.main(["decode", video, "--pixels", "80", "--out", str(out)])
    assert pixelmap.main(["score", str(out / "map.json"), video + ".truth.json"]) == 0
