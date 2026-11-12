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
    pixelmap.main(
        ["export", str(m), "--madrix", str(mad), "--resolume", str(res), "--universe", "0.0.3", "--line-tolerance", "0"]
    )
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


# -- whole rig: unwrap + one lumiverse per Art-Net net ----------------------------


def test_unwrap_cylinder_angle_and_height():
    ang = np.linspace(-math.pi + 0.05, math.pi - 0.05, 200)
    h = np.linspace(-1, 1, 200)
    r = 0.5 + 0.3 * (1 - h)  # a cone, wider at the bottom
    xyz = np.stack([r * np.sin(ang) + 0.2, h, r * np.cos(ang) - 0.1], axis=1)  # trunk off-centre
    u, v = pixelmap.unwrap_cylinder(xyz)
    assert np.allclose(u, (ang + math.pi) / (2 * math.pi), atol=0.02)  # front (+Z) lands mid-canvas
    assert v[np.argmax(h)] == 0 and abs(v[np.argmin(h)] - 1) < 1e-9  # top of the tree at the top


def _rig_manifest(path, devices):
    rig = {"kind": "twinkly-rig", "version": 1, "total": 0, "slot": 0.4, "devices": [], "skipped": []}
    off = 0
    for name, uni, n in devices:
        segs, first, u = [], 0, uni
        while first < n:
            c = min(128, n - first)
            segs.append({"universe": u, "channel": 0, "first_pixel": first, "count": c})
            first, u = first + c, u + 1
        rig["devices"].append(
            {"name": name, "ip": "10.0.0.9", "offset": off, "pixels": n, "channels": 4, "feed": "", "segments": segs}
        )
        off += n
    rig["total"] = off
    path.write_text(json.dumps(rig))
    return rig


def test_export_whole_rig_splits_lumiverses_by_net(tmp_path):
    xyz, _ = _cone_tree(400)
    # A: 250 px on 0.15.14-0.15.15; B: 150 px from 1.0.0 (the second lumiverse)
    rig = tmp_path / "rig.json"
    _rig_manifest(rig, [("Tree A", 254, 250), ("Tree B", 256, 150)])
    found = [{"pixel": i, "x": float(p[0]), "y": float(p[1]), "z": float(p[2])} for i, p in enumerate(xyz)]
    m = tmp_path / "map.json"
    m.write_text(json.dumps({"pixels": 400, "dims": 3, "found": found}))
    res, png = tmp_path / "r.xml", tmp_path / "canvas.png"
    pixelmap.main(
        ["export", str(m), "--rig", str(rig), "--resolume", str(res), "--preview", str(png), "--line-tolerance", "0"]
    )
    root = ET.parse(res).getroot()
    assert root.find("versionInfo").get("minorVersion") == "24"
    screens = root.findall(".//DmxScreen")
    assert [s.get("LumiverseId") for s in screens] == ["0", "1"]
    a, b = screens
    assert len(a.findall(".//DmxSlice")) == 250 and len(b.findall(".//DmxSlice")) == 150

    def params(screen, name):
        return [p.get("value") for p in screen.iter("ParamRange") if p.get("name") == name]

    assert params(a, "Subnet") == ["15"] and params(a, "Universe") == ["14"]  # starts at 0.15.14
    assert params(b, "Subnet") == ["0"] and params(b, "Universe") == ["0"]  # 1.0.0 -> its own output
    starts_a = [float(v) for v in params(a, "Start Channel")]
    assert starts_a[0] == 1 and starts_a[1] == 5 and starts_a[128] == 513
    assert float(params(b, "Start Channel")[0]) == 1
    names = [s.find(".//Param[@name='Name']").get("value") for s in b.findall(".//DmxSlice")]
    assert names[0] == "Tree B 1" and names[-1] == "Tree B 150"
    for s in root.iter("InputRect"):
        v = np.array([[float(q.get("x")), float(q.get("y"))] for q in s.iter("v")])
        c = v.mean(axis=0)  # every pixel's sample square centre sits on the composition
        assert 0 < c[0] < 1920 and 0 < c[1] < 1080
    assert {p.get("value") for p in root.iter("ParamChoice") if p.get("name") == "Color Format"} == {"rgbw"}
    assert {p.get("value") for p in root.iter("Param") if p.get("name") == "TargetIP"} == {"TT_DISABLED"}
    img = cv2.imread(str(png))
    assert img is not None and img.shape[1] == 1280


def test_export_rig_skips_pixels_no_universe_carries(tmp_path):
    rig = tmp_path / "rig.json"
    r = _rig_manifest(rig, [("Tree A", 0, 10)])
    r["devices"][0]["segments"][0]["count"] = 6  # patched for 6, the controller shows 10
    rig.write_text(json.dumps(r))
    found = [{"pixel": i, "x": float(i), "y": 0.0} for i in range(10)]
    m = tmp_path / "map.json"
    m.write_text(json.dumps({"pixels": 10, "dims": 2, "found": found}))
    res = tmp_path / "r.xml"
    pixelmap.main(["export", str(m), "--rig", str(rig), "--resolume", str(res)])
    sl = ET.parse(res).getroot().findall(".//DmxSlice")
    assert len(sl) == 1  # six pixels in a straight row: one 1x6 line slice
    assert sl[0].find(".//ParamRange[@name='Height']").get("value") == "6"
    assert sl[0].find(".//Param[@name='Name']").get("value") == "Tree A 1-6"


def test_solve3d_places_cameras_from_a_sample_then_triangulates_everything(tmp_path):
    xyz, normals = _cone_tree(600)
    n = len(xyz)
    yaws = [i * 2 * math.pi / 6 for i in range(6)]
    paths = _orbit_views(tmp_path, xyz, normals, yaws, fov=64.0, occlude=True)
    out = tmp_path / "3d"
    scale = str(np.linalg.norm(xyz[0] - xyz[-1]))
    pixelmap.main(
        ["solve3d", *paths, "--fov", "64", "--max-points", "150", "--out", str(out), "--scale", "1", str(n), scale]
    )
    got = json.loads((out / "map.json").read_text())
    docs = [json.load(open(q)) for q in paths]
    seen_by_two = sum(1 for p in range(n) if sum(p in {f["pixel"] for f in d["found"]} for d in docs) >= 2)
    assert len(got["found"]) >= 0.97 * seen_by_two  # all of them, not just the 150
    err, up, _ = _shape_error(got, xyz)
    assert err < 0.02 and up > 0.99


def test_run_rect_matches_a_preset_saved_by_arena():
    # Arena 7.24.3: a 1x10 fixture, 32 x 320 px, rotated 30 degrees about (960, 540)
    saved = [(1026.142, 393.435), (1053.855, 409.435), (893.858, 686.565), (866.145, 670.565)]
    ey = np.array([-math.sin(math.pi / 6), math.cos(math.pi / 6)])
    centres = np.array([960.0, 540.0]) + np.outer((np.arange(10) + 0.5) / 10 * 320 - 160, ey)
    corners, angle = pixelmap._run_rect(centres, 32.0)
    assert angle == pytest.approx(0.5235899, abs=1e-4)  # Arena stores a float32
    for got, want in zip(corners, saved, strict=True):
        assert got == pytest.approx(want, abs=0.01)


def test_line_slices_cover_every_pixel_once(tmp_path):
    xyz, _ = _cone_tree(400)
    rig = tmp_path / "rig.json"
    _rig_manifest(rig, [("Tree A", 0, 400)])
    found = [{"pixel": i, "x": float(p[0]), "y": float(p[1]), "z": float(p[2])} for i, p in enumerate(xyz)]
    m = tmp_path / "map.json"
    m.write_text(json.dumps({"pixels": 400, "dims": 3, "found": found}))
    res = tmp_path / "r.xml"
    pixelmap.main(["export", str(m), "--rig", str(rig), "--resolume", str(res), "--line-tolerance", "1.5"])
    slices = ET.parse(res).getroot().findall(".//DmxSlice")
    assert len(slices) < 200  # far fewer slices than pixels
    covered = []
    for sl in slices:
        start = float(sl.find(".//ParamRange[@name='Start Channel']").get("value"))
        n = int(sl.find(".//ParamRange[@name='Height']").get("value"))
        first = int((start - 1) // 512 * 128 + ((start - 1) % 512) // 4)
        covered += range(first, first + n)
    assert sorted(covered) == list(range(400))
    # where Resolume samples each pixel (cell centres along the rect) vs where the export put it
    xy, _, _ = pixelmap.canvas_layout(found, 3, (1920, 1080), "cylinder")
    for sl in slices:
        v = np.array([[float(q.get("x")), float(q.get("y"))] for q in sl.find("InputRect")])
        n = int(sl.find(".//ParamRange[@name='Height']").get("value"))
        start = float(sl.find(".//ParamRange[@name='Start Channel']").get("value"))
        first = int((start - 1) // 512 * 128 + ((start - 1) % 512) // 4)
        top, bottom = (v[0] + v[1]) / 2, (v[3] + v[2]) / 2
        cells = top + np.outer((np.arange(n) + 0.5) / n, bottom - top)
        assert np.abs(cells - xy[first : first + n]).max() < 1.6


def test_compact_slices_are_smaller(tmp_path):
    found = [{"pixel": i, "x": float(i % 7) * 13, "y": float(i // 7) * 11} for i in range(200)]
    m = tmp_path / "map.json"
    m.write_text(json.dumps({"pixels": 200, "dims": 2, "found": found}))
    full, small = tmp_path / "full.xml", tmp_path / "small.xml"
    pixelmap.main(["export", str(m), "--resolume", str(full), "--line-tolerance", "0", "--resolume-full"])
    pixelmap.main(["export", str(m), "--resolume", str(small), "--line-tolerance", "0", "--resolume-compact"])
    assert small.stat().st_size < 0.4 * full.stat().st_size
    root = ET.parse(small).getroot()
    assert len(root.findall(".//DmxSlice")) == 200
