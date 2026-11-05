import csv
import json
import math
import xml.etree.ElementTree as ET

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from twinklybridge import mapcode  # noqa: E402
from twinklybridge import mapping as pixelmap  # noqa: E402


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


def _cone_tree(n=400):
    t = np.linspace(0, 1, n)
    ang = t * 40
    xyz = np.stack([(1 - t) * 0.8 * np.cos(ang), 2.0 * t - 1.0, (1 - t) * 0.8 * np.sin(ang)], axis=1)
    normals = np.stack([np.cos(ang), np.zeros(n), np.sin(ang)], axis=1)  # pointing out of the tree
    return xyz, normals  # metres, Y up


def _orbit_views(tmp_path, xyz, normals, yaws, fov=70.0, occlude=False, seed=2, outliers=0.0):
    """Decoded-view JSONs of the tree from cameras 4 m out, level, orbiting it."""
    rng = np.random.default_rng(seed)
    n = len(xyz)
    flip = np.array([1, -1, -1])  # world (Y up) -> camera-style frame (Y down); a rotation, not a mirror
    w, h = 1920, 1080
    f = (w / 2) / math.tan(math.radians(fov) / 2)
    k = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
    paths = []
    for i, yaw in enumerate(yaws):
        rvec = np.array([0.0, yaw, 0.0])
        r = cv2.Rodrigues(rvec)[0]
        c = np.array([4 * math.sin(yaw), 0.0, -4 * math.cos(yaw)])
        uv = _project(xyz * flip, rvec, -r @ c, k) + rng.normal(0, 0.3, (n, 2))
        bad = rng.random(n) < outliers  # spots that landed somewhere wrong (reflections, mirrors)
        uv[bad] += rng.normal(0, 80, (int(bad.sum()), 2))
        to_cam = c * flip - xyz
        keep = np.ones(n, bool)
        if occlude:  # the far side of the tree is hidden behind it
            keep = (normals * to_cam).sum(1) / np.linalg.norm(to_cam, axis=1) > -0.15
        doc = {
            "pixels": n,
            "source": f"v{i}",
            "image": {"width": w, "height": h},
            "found": [{"pixel": p, "x": float(x), "y": float(y)} for p, (x, y) in enumerate(uv) if keep[p]],
            "bad": [p for p in range(n) if keep[p] and bad[p]],
        }
        p = tmp_path / f"v{i}" / "map.json"
        p.parent.mkdir()
        p.write_text(json.dumps(doc))
        paths.append(str(p))
    return paths


def _shape_error(got, xyz):
    ids = [q["pixel"] for q in got["found"]]
    a = np.array([[q["x"], q["y"], q["z"]] for q in got["found"]])
    b = xyz[ids]
    a0, b0 = a - a.mean(0), b - b.mean(0)
    u, _, vt = np.linalg.svd(a0.T @ b0)
    d = np.diag([1, 1, np.sign(np.linalg.det(u @ vt))])
    r = u @ d @ vt
    err = np.linalg.norm(a0 @ r - b0, axis=1)
    up = np.corrcoef(a[:, 1], b[:, 1])[0, 1]  # our Y must be the real up
    return float(np.median(err)), float(up), ids


def test_solve3d_two_views(tmp_path):
    xyz, normals = _cone_tree()
    paths = _orbit_views(tmp_path, xyz, normals, (0.0, 0.7))
    out = tmp_path / "3d"
    scale = str(np.linalg.norm(xyz[0] - xyz[-1]))
    pixelmap.main(["solve3d", *paths, "--out", str(out), "--scale", "1", str(len(xyz)), scale])
    got = json.loads((out / "map.json").read_text())
    assert got["dims"] == 3 and len(got["found"]) > 0.95 * len(xyz)
    err, up, _ = _shape_error(got, xyz)
    assert err < 0.03  # 3 cm on a 2 m tree
    assert up > 0.99


def test_solve3d_many_views_fill_gaps_and_fix_fov(tmp_path):
    xyz, normals = _cone_tree()
    n = len(xyz)
    yaws = [i * 2 * math.pi / 6 for i in range(6)]  # all the way round, 60 degrees apart
    paths = _orbit_views(tmp_path, xyz, normals, yaws, fov=64.0, occlude=True, outliers=0.02)
    docs = [json.load(open(q)) for q in paths]
    good = [{f["pixel"] for f in d["found"]} - set(d["bad"]) for d in docs]
    seen_by_two = sum(1 for p in range(n) if sum(p in g for g in good) >= 2)
    out = tmp_path / "3d"
    scale = str(np.linalg.norm(xyz[0] - xyz[-1]))
    # --fov 70 is deliberately wrong (the "phone" is 64): the solver should find it
    pixelmap.main(["solve3d", *paths, "--fov", "70", "--out", str(out), "--scale", "1", str(n), scale])
    got = json.loads((out / "map.json").read_text())
    assert len(got["views_used"]) == 6
    assert len(got["found"]) >= 0.97 * seen_by_two
    assert abs(got["fov"] - 64.0) < 1.0
    err, up, _ = _shape_error(got, xyz)
    assert err < 0.02
    assert up > 0.99


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


def test_solve3d_rejects_the_same_view_twice(tmp_path):
    xyz, normals = _cone_tree(120)
    paths = _orbit_views(tmp_path, xyz, normals, (0.3, 0.3))
    with pytest.raises(SystemExit) as exc:
        pixelmap.main(["solve3d", *paths, "--out", str(tmp_path / "3d")])
    assert "different spots" in str(exc.value)
