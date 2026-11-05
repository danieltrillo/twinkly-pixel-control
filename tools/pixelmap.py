#!/usr/bin/env python3
"""
pixelmap.py - find where every Twinkly pixel is from a phone video, then export
a pixel map for MADRIX 5 or Resolume Arena.

    1. Stream the blink code and film it (phone on a tripod, exposure locked):
         python3 tools/twinkly_probe.py scan <controller-ip>
    2. Decode the video into pixel positions:
         python3 tools/pixelmap.py decode scan.mp4 --pixels 1440 --out map-front
    3. Optional 3D: film again from a second angle, decode it, then
         python3 tools/pixelmap.py solve3d map-front/map.json map-side/map.json --out map-3d
    4. Export:
         python3 tools/pixelmap.py export map-front/map.json --universe 0.0.0 \
             --madrix madrix.csv --resolume twinkly.xml

Needs numpy and OpenCV:  pip install numpy opencv-python-headless
Test without hardware:   python3 tools/pixelmap.py synth fake.mp4 --pixels 300
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import uuid
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from twinklybridge import mapcode  # noqa: E402

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover - message for humans
    sys.exit("pixelmap.py needs numpy and OpenCV: pip install numpy opencv-python-headless")

DMX_SLOTS = 512


# ==========================================================================
# Video reading
# ==========================================================================


def _gray(frame: np.ndarray) -> np.ndarray:
    # max of the channels: a saturated colour pixel still reads as bright
    return frame.max(axis=2).astype(np.float32)


def _resize(img: np.ndarray, max_width: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= max_width:
        return img
    scale = max_width / w
    return cv2.resize(img, (max_width, round(h * scale)), interpolation=cv2.INTER_AREA)


def read_trace(path: str, small_width: int = 160) -> tuple[np.ndarray, np.ndarray, float, tuple[int, int]]:
    """First pass: timestamps and a tiny copy of every frame."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"can't open video {path!r}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    times, smalls = [], []
    size = (0, 0)
    i = 0
    last_t = -1.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        size = (frame.shape[1], frame.shape[0])
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if t <= last_t or (i > 0 and t == 0):
            t = i / fps  # container without usable timestamps
        last_t = t
        times.append(t)
        smalls.append(_resize(_gray(frame), small_width))
        i += 1
    cap.release()
    if len(times) < 10:
        raise SystemExit(f"video {path!r} has only {len(times)} frames")
    return np.array(times), np.stack(smalls), fps, size


def activity_trace(smalls: np.ndarray) -> np.ndarray:
    """Brightness of the busiest spots (the pixels), scaled 0..1."""
    std = smalls.std(axis=0)
    cut = np.percentile(std, 98)
    mask = std >= max(cut, 1e-6)
    tr = smalls[:, mask].mean(axis=1)
    lo, hi = np.percentile(tr, 2), np.percentile(tr, 99.5)
    if hi - lo < 1e-6:
        raise SystemExit("nothing in the video blinks - is the scan in frame?")
    return np.clip((tr - lo) / (hi - lo), 0, 1.2)


# ==========================================================================
# Timing: find the marks
# ==========================================================================


@dataclass
class Pass:
    t_fall: float  # end of the first mark (crosses half level)
    t_rise: float  # start of the second mark
    slot: float  # measured seconds per slot
    on_window: tuple[float, float]  # middle of the first mark
    off_window: tuple[float, float]


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    out, start = [], None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        elif not f and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(flags) - 1))
    return out


def _crossing(times: np.ndarray, tr: np.ndarray, i: int, j: int, level: float) -> float:
    """Interpolated time where the trace crosses level between frames i and j."""
    a, b = tr[i], tr[j]
    if a == b:
        return float((times[i] + times[j]) / 2)
    f = (level - a) / (b - a)
    return float(times[i] + min(max(f, 0), 1) * (times[j] - times[i]))


def find_passes(times: np.ndarray, tr: np.ndarray, n_pixels: int, slot_hint: float) -> list[Pass]:
    k = mapcode.data_slots(n_pixels) + 2  # off + data + off between the marks
    plateaus = []
    # data slots come in bit/inverse pairs, so the busy spots average about half
    # brightness there; only a mark stays above 0.65 for more than two slots
    for s, e in _runs(tr > 0.65):
        dur = times[e] - times[s]
        if dur >= (mapcode.MARK - 1.5) * slot_hint:
            plateaus.append((s, e))
    passes = []
    for (s1, e1), (s2, e2) in zip(plateaus, plateaus[1:], strict=False):
        # half of each mark's own level: exposure may differ between the two marks
        half1 = 0.5 * float(np.median(tr[s1 : e1 + 1]))
        half2 = 0.5 * float(np.median(tr[s2 : e2 + 1]))
        i = e1
        while i + 1 < len(tr) and tr[i + 1] > half1:
            i += 1
        t_fall = _crossing(times, tr, i, min(i + 1, len(tr) - 1), half1)
        j = s2
        while j - 1 >= 0 and tr[j - 1] > half2:
            j -= 1
        t_rise = _crossing(times, tr, max(j - 1, 0), j, half2)
        slot = (t_rise - t_fall) / k
        if not (0.7 * slot_hint <= slot <= 1.3 * slot_hint):
            continue  # gap between two marks of different passes, or wrong --pixels
        mark_mid = (times[s1] + t_fall) / 2
        passes.append(
            Pass(
                t_fall=t_fall,
                t_rise=t_rise,
                slot=slot,
                on_window=(mark_mid - slot, mark_mid + slot),
                off_window=(t_fall + 0.25 * slot, t_fall + 0.75 * slot),
            )
        )
    return passes


# ==========================================================================
# Decode
# ==========================================================================


def accumulate(path: str, passes: list[Pass], n_pixels: int, max_width: int, guard: float = 0.25):
    """Second pass: average the frames in the middle of every slot."""
    nd = mapcode.data_slots(n_pixels)
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    sums = cnts = on_sum = off_sum = None
    on_n = off_n = 0
    i, last_t = 0, -1.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if t <= last_t or (i > 0 and t == 0):
            t = i / fps
        last_t = t
        i += 1
        img = None
        for p in passes:
            if p.on_window[0] <= t <= p.on_window[1] or p.off_window[0] <= t <= p.off_window[1]:
                img = _resize(_gray(frame), max_width)
                if on_sum is None:
                    on_sum, off_sum = np.zeros_like(img), np.zeros_like(img)
                if t <= p.on_window[1] and t >= p.on_window[0]:
                    on_sum += img
                    on_n += 1
                else:
                    off_sum += img
                    off_n += 1
            s = (t - p.t_fall) / p.slot - 1.0  # data slot coordinate
            if 0 <= s < nd:
                idx, frac = int(s), s - int(s)
                if guard <= frac <= 1 - guard:
                    if img is None:
                        img = _resize(_gray(frame), max_width)
                    if sums is None:
                        sums = np.zeros((nd,) + img.shape, np.float32)
                        cnts = np.zeros(nd, np.int32)
                    sums[idx] += img
                    cnts[idx] += 1
    cap.release()
    if sums is None or on_n == 0 or off_n == 0:
        raise SystemExit("found the marks but no frames inside them - video too short?")
    if (cnts == 0).any():
        raise SystemExit(f"slots {np.nonzero(cnts == 0)[0].tolist()} got no frames: frame rate too low for --slot")
    return sums / cnts[:, None, None], on_sum / on_n, off_sum / off_n, cnts


def decode_slots(slots, on, off, n_pixels, min_contrast=20.0, min_margin=0.3, min_area=3):
    """Per camera pixel: which LED is this? Then one centroid per LED."""
    nb = mapcode.word_bits(n_pixels)
    contrast = on - off
    a, b = slots[0::2], slots[1::2]  # bit, then its inverse
    d = a - b
    margin = np.abs(d).min(axis=0) / np.maximum(contrast, 1.0)
    floor = max(min_contrast, 0.08 * float(np.percentile(contrast, 99.9)))
    valid = (contrast > floor) & (margin > min_margin)
    word = np.zeros(on.shape, np.int64)
    for bit in range(nb):
        word |= (d[bit] > 0).astype(np.int64) << bit
    table = np.array(mapcode.decode_table(n_pixels), np.int64)
    ids = np.where(valid, table[word], -1)
    valid &= ids >= 0

    n_lab, labels = cv2.connectedComponents(valid.astype(np.uint8), connectivity=8)
    ys, xs = np.nonzero(valid)
    pid = ids[ys, xs]
    w = contrast[ys, xs] * margin[ys, xs]
    key = labels[ys, xs].astype(np.int64) * n_pixels + pid
    uk, inv = np.unique(key, return_inverse=True)
    sw = np.bincount(inv, w)
    sx = np.bincount(inv, w * xs)
    sy = np.bincount(inv, w * ys)
    area = np.bincount(inv)
    blobs: dict[int, list[dict]] = {}
    for k, kk in enumerate(uk):
        if area[k] < min_area:
            continue
        p = int(kk % n_pixels)
        blobs.setdefault(p, []).append(
            {"x": float(sx[k] / sw[k]), "y": float(sy[k] / sw[k]), "weight": float(sw[k]), "area": int(area[k])}
        )
    found = []
    for p in sorted(blobs):
        bl = sorted(blobs[p], key=lambda q: -q["weight"])
        main = bl[0]
        # a second strand on a splitter shows the same pixel somewhere else
        dups = [q for q in bl[1:] if q["weight"] >= 0.25 * main["weight"] and _dist(q, main) > 6]
        found.append(
            {
                "pixel": p,
                "x": round(main["x"], 2),
                "y": round(main["y"], 2),
                "weight": round(main["weight"], 1),
                "area": main["area"],
                "duplicates": [{"x": round(q["x"], 2), "y": round(q["y"], 2)} for q in dups],
            }
        )
    return found, ids, valid


def _dist(a, b) -> float:
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def preview(on: np.ndarray, found: list[dict], path: str, label_every: int = 10) -> None:
    img = np.clip(on / max(np.percentile(on, 99.5), 1) * 160, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    n = max((f["pixel"] for f in found), default=0) + 1
    for f in found:
        hue = int(170 * f["pixel"] / max(n, 1))
        col = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
        c = (round(f["x"]), round(f["y"]))
        cv2.circle(img, c, 3, col, -1)
        for dup in f["duplicates"]:
            cv2.drawMarker(img, (round(dup["x"]), round(dup["y"])), col, cv2.MARKER_TILTED_CROSS, 8, 1)
        if f["pixel"] % label_every == 0:
            cv2.putText(img, str(f["pixel"] + 1), (c[0] + 4, c[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)
    cv2.imwrite(path, img)


def cmd_decode(a) -> int:
    t0 = time.time()
    times, smalls, fps, size = read_trace(a.video)
    tr = activity_trace(smalls)
    passes = find_passes(times, tr, a.pixels, a.slot)
    if not passes:
        raise SystemExit(
            "couldn't find a complete scan in the video. Check --pixels and --slot match the scan,"
            " and that the recording covers both white flashes."
        )
    print(f"{len(times)} frames at {fps:.1f} fps, {size[0]}x{size[1]}; {len(passes)} pass(es) found")
    for p in passes:
        print(f"  pass {p.t_fall:6.2f}s - {p.t_rise:6.2f}s, slot {p.slot * 1000:.0f} ms")
    slots, on, off, cnts = accumulate(a.video, passes, a.pixels, a.max_width)
    found, _, _ = decode_slots(slots, on, off, a.pixels, a.min_contrast, a.min_margin)
    scale = size[0] / on.shape[1]
    for f in found:  # report in full video pixels
        f["x"], f["y"] = round(f["x"] * scale, 2), round(f["y"] * scale, 2)
        for dup in f["duplicates"]:
            dup["x"], dup["y"] = round(dup["x"] * scale, 2), round(dup["y"] * scale, 2)
    seen = {f["pixel"] for f in found}
    missing = [i for i in range(a.pixels) if i not in seen]
    os.makedirs(a.out, exist_ok=True)
    doc = {
        "kind": "twinkly-pixelmap",
        "version": 1,
        "dims": 2,
        "pixels": a.pixels,
        "source": os.path.basename(a.video),
        "image": {"width": size[0], "height": size[1]},
        "passes": len(passes),
        "frames_per_slot": float(cnts.mean()),
        "found": found,
        "missing": missing,
    }
    with open(os.path.join(a.out, "map.json"), "w") as fh:
        json.dump(doc, fh, indent=1)
    write_positions_csv(os.path.join(a.out, "map.csv"), found, dims=2)
    big_on = cv2.resize(on, size, interpolation=cv2.INTER_LINEAR)
    preview(big_on, found, os.path.join(a.out, "preview.png"))
    dups = sum(1 for f in found if f["duplicates"])
    print(
        f"found {len(found)}/{a.pixels} pixels ({len(missing)} missing, {dups} also seen on a second strand)"
        f" in {time.time() - t0:.1f} s -> {a.out}/map.json, map.csv, preview.png"
    )
    if missing:
        print("  missing (1-based): " + compress_ranges([m + 1 for m in missing]))
    return 0


def compress_ranges(nums: list[int]) -> str:
    out, start, prev = [], None, None
    for n in nums:
        if start is None:
            start = prev = n
        elif n == prev + 1:
            prev = n
        else:
            out.append(f"{start}" if start == prev else f"{start}-{prev}")
            start = prev = n
    if start is not None:
        out.append(f"{start}" if start == prev else f"{start}-{prev}")
    return ", ".join(out)


def write_positions_csv(path: str, found: list[dict], dims: int) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pixel", "x", "y"] + (["z"] if dims == 3 else []) + ["weight", "duplicates"])
        for f in found:
            row = [f["pixel"] + 1, f["x"], f["y"]] + ([f["z"]] if dims == 3 else [])
            w.writerow(row + [f.get("weight", ""), len(f.get("duplicates", []))])


# ==========================================================================
# 3D from two views
# ==========================================================================


def cmd_solve3d(a) -> int:
    m1, m2 = (json.load(open(p)) for p in (a.view1, a.view2))
    p1 = {f["pixel"]: (f["x"], f["y"]) for f in m1["found"]}
    p2 = {f["pixel"]: (f["x"], f["y"]) for f in m2["found"]}
    common = sorted(set(p1) & set(p2))
    if len(common) < 12:
        raise SystemExit(f"only {len(common)} pixels seen in both views; need at least 12 (aim for 100+)")
    k1, k2 = _intrinsics(m1, a.fov), _intrinsics(m2, a.fov)
    x1 = np.array([p1[i] for i in common], np.float64)
    x2 = np.array([p2[i] for i in common], np.float64)
    n1 = cv2.undistortPoints(x1.reshape(-1, 1, 2), k1, None).reshape(-1, 2)
    n2 = cv2.undistortPoints(x2.reshape(-1, 1, 2), k2, None).reshape(-1, 2)
    e, inl = cv2.findEssentialMat(n1, n2, np.eye(3), method=cv2.RANSAC, prob=0.999, threshold=a.ransac / k1[0, 0])
    if e is None:
        raise SystemExit("couldn't relate the two views - film from angles 30-60 degrees apart")
    _, r, t, inl2 = cv2.recoverPose(e[:3], n1, n2, np.eye(3), mask=inl)
    good = inl2.ravel() > 0
    pa = np.hstack([np.eye(3), np.zeros((3, 1))])
    pb = np.hstack([r, t])
    xh = cv2.triangulatePoints(pa, pb, n1.T, n2.T)
    xyz = (xh[:3] / xh[3]).T
    # reprojection error in pixels, view 1
    rep = np.linalg.norm((xyz[:, :2] / xyz[:, 2:3]) - n1, axis=1) * k1[0, 0]
    good &= (xyz[:, 2] > 0) & (rep < 3 * a.ransac)
    xyz = xyz[good]
    ids = [c for c, g in zip(common, good, strict=True) if g]
    if len(ids) < 8:
        raise SystemExit("too few consistent pixels after triangulation; try another --fov or better angles")
    xyz = _upright(xyz)
    unit = "relative"
    if a.scale:
        ia, ib, meters = int(a.scale[0]) - 1, int(a.scale[1]) - 1, float(a.scale[2])
        if ia not in ids or ib not in ids:
            raise SystemExit("--scale pixels weren't both triangulated; pick two others")
        dist = np.linalg.norm(xyz[ids.index(ia)] - xyz[ids.index(ib)])
        xyz *= meters / dist
        unit = "m"
    found = [
        {
            "pixel": p,
            "x": round(float(v[0]), 4),
            "y": round(float(v[1]), 4),
            "z": round(float(v[2]), 4),
            "duplicates": [],
        }
        for p, v in zip(ids, xyz, strict=True)
    ]
    n = m1["pixels"]
    seen = set(ids)
    doc = {
        "kind": "twinkly-pixelmap",
        "version": 1,
        "dims": 3,
        "unit": unit,
        "pixels": n,
        "source": [m1["source"], m2["source"]],
        "found": found,
        "missing": [i for i in range(n) if i not in seen],
    }
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "map.json"), "w") as fh:
        json.dump(doc, fh, indent=1)
    write_positions_csv(os.path.join(a.out, "map.csv"), found, dims=3)
    print(
        f"3D: {len(ids)}/{n} pixels (seen in both views and consistent), median reprojection"
        f" {np.median(rep[good]):.2f} px, units {unit} -> {a.out}/map.json"
    )
    return 0


def _intrinsics(m: dict, fov_deg: float) -> np.ndarray:
    w, h = m["image"]["width"], m["image"]["height"]
    f = (max(w, h) / 2) / math.tan(math.radians(fov_deg) / 2)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float64)


def _upright(xyz: np.ndarray) -> np.ndarray:
    """Camera 1 looks down +Z with Y down. Re-express as X right, Y up, Z toward camera 1, centred."""
    out = np.stack([xyz[:, 0], -xyz[:, 1], -xyz[:, 2]], axis=1)
    return out - np.median(out, axis=0)


# ==========================================================================
# Export
# ==========================================================================


def patch_for(a, n_pixels: int) -> list[tuple[int, int]]:
    """(universe, 1-based channel) for each pixel, as the bridge reads them."""
    from twinklybridge.config import build_segments, load_config

    if a.bridge_config:
        cfg = load_config(a.bridge_config)
        devs = [d for d in cfg.devices if not a.device or d.name == a.device]
        if not devs:
            raise SystemExit(f"no device named {a.device!r} in {a.bridge_config}")
        d = devs[0]
        segs, channels = d.segments, d.channels
    else:
        channels = len(a.input)
        per = a.pixels_per_universe or DMX_SLOTS // channels
        segs = build_segments(parse_universe(a.universe), n_pixels, channels, per, 1)
    out = []
    for s in segs:
        for k in range(s.count):
            out.append((s.universe, s.channel + k * channels + 1))
    if len(out) < n_pixels:
        raise SystemExit(f"the patch covers {len(out)} pixels but the map has {n_pixels}")
    return out


def parse_universe(text: str) -> int:
    parts = [int(p) for p in str(text).split(".")]
    if len(parts) == 3:
        return (parts[0] << 8) | (parts[1] << 4) | parts[2]
    if len(parts) == 1:
        return parts[0]
    raise SystemExit(f"universe {text!r}: use net.subnet.universe like 0.0.0, or a plain number")


def grid_positions(found: list[dict], width: int, dims: int) -> tuple[list[tuple[int, ...]], int]:
    """Snap positions to a voxel grid `width` cells wide; nudge collisions to the nearest free cell."""
    pts = np.array([[f["x"], f["y"]] + ([f["z"]] if dims == 3 else []) for f in found], np.float64)
    if dims == 3:
        pts[:, 1] *= -1  # MADRIX Y runs down; our 3D Y runs up
    lo = pts.min(axis=0)
    span = max(float((pts.max(axis=0) - lo)[0]), float((pts.max(axis=0) - lo)[1]), 1e-9)
    if width <= 0:  # auto: smallest power-of-two-ish width with few collisions
        width = 32
        while width < 1024 and _collisions(np.round((pts - lo) / span * (width - 1))) > 0.01 * len(pts):
            width *= 2
    cells = np.round((pts - lo) / span * (width - 1)).astype(int)
    taken: set[tuple[int, ...]] = set()
    out = []
    nudged = 0
    for c in cells:
        cell = tuple(int(v) for v in c)
        if cell in taken:
            cell = _nearest_free(cell, taken)
            nudged += 1
        taken.add(cell)
        out.append(cell)
    return out, nudged


def _collisions(cells: np.ndarray) -> int:
    return len(cells) - len({tuple(c) for c in cells.astype(int)})


def _nearest_free(cell, taken):
    for r in range(1, 64):
        best = None
        for off in np.ndindex(*([2 * r + 1] * len(cell))):
            cand = tuple(c + o - r for c, o in zip(cell, off, strict=True))
            if min(cand) < 0 or cand in taken:
                continue
            dist = sum((o - r) ** 2 for o in off)
            if best is None or dist < best[0]:
                best = (dist, cand)
        if best:
            return best[1]
    return cell


def write_madrix(path: str, found, cells, patch, a) -> None:
    """MADRIX 5 Patch Editor > Import Fixture List (CSV). Map the columns in the import dialog."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["Product", "Fixture ID", "Name", "Position X", "Position Y", "Position Z", "DMX Universe", "DMX Channel"]
        )
        for f, cell in zip(found, cells, strict=True):
            uni, ch = patch[f["pixel"]]
            z = cell[2] if len(cell) == 3 else 0
            w.writerow(
                [
                    a.product,
                    a.fixture_id_base + f["pixel"],
                    f"{a.name} {f['pixel'] + 1}",
                    cell[0] + a.madrix_origin,
                    cell[1] + a.madrix_origin,
                    z + a.madrix_origin,
                    uni + a.madrix_universe_offset,
                    ch,
                ]
            )


def write_resolume(path: str, found, patch, a, dims: int) -> None:
    """Resolume Arena 7 Advanced Output preset: one Lumiverse, one 1x1 fixture slice per pixel."""
    pts = np.array([[f["x"], f["y"]] for f in found], np.float64)
    if dims == 3:
        pts[:, 1] *= -1  # front view, Y down like the composition
    comp_w, comp_h = a.comp
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    margin = 0.04 * min(comp_w, comp_h)
    scale = min((comp_w - 2 * margin) / span[0], (comp_h - 2 * margin) / span[1])
    xy = (pts - lo) * scale
    xy += [(comp_w - span[0] * scale) / 2, (comp_h - span[1] * scale) / 2]
    # sample square: a bit under the typical spacing between neighbours
    if len(xy) > 1:
        from_each = [np.partition(np.linalg.norm(xy - p, axis=1), 1)[1] for p in xy[: min(len(xy), 400)]]
        size = float(np.clip(0.6 * np.median(from_each), 1.0, 40.0))
    else:
        size = 8.0
    first_uni = min(u for u, _ in patch)
    fixture_uuid = uuid.uuid4().hex
    color = a.resolume_color
    stamp = int(time.time() * 1000)
    slices = []
    for k, (f, (x, y)) in enumerate(zip(found, xy, strict=True)):
        uni, ch = patch[f["pixel"]]
        start = (uni - first_uni) * DMX_SLOTS + ch
        h = size / 2
        slices.append(
            _RES_SLICE.format(
                uid=stamp + k + 1,
                name=f"{a.name} {f['pixel'] + 1}",
                fixture_uuid=fixture_uuid,
                start=start,
                start_phase=(start - 1) / 131071,
                x0=x - h,
                y0=y - h,
                x1=x + h,
                y1=y + h,
                color=color,
            )
        )
    xml = _RES_HEAD.format(name=a.name, w=comp_w, h=comp_h, uid=stamp, lumi=a.name) + "".join(slices)
    xml += _RES_TAIL.format(subnet=(first_uni >> 4) & 0xF, universe=first_uni & 0xF, target=a.resolume_target)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(xml)


def _range(name, value, lo, hi, default=None, phase=None, t="DOUBLE"):
    default = value if default is None else default
    phase = 0 if phase is None else phase
    return (
        f'<ParamRange name="{name}" T="{t}" default="{default}" value="{value}">'
        f'<PhaseSourceStatic name="PhaseSourceStatic" phase="{phase}"/><BehaviourDouble name="BehaviourDouble"/>'
        f'<ValueRange name="defaultRange" min="{lo}" max="{hi}"/><ValueRange name="minMax" min="{lo}" max="{hi}"/>'
        f'<ValueRange name="startStop" min="{lo}" max="{hi}"/></ParamRange>'
    )


_RES_ADJ = "".join(_range(n, 0, -1, 1, phase=0.5) for n in ("Brightness", "Contrast", "Red", "Green", "Blue"))

# Element layout copied from Advanced Output presets saved by Arena 7.13 - 7.26.
_RES_HEAD = (
    '<?xml version="1.0" encoding="utf-8"?>\n'
    '<XmlState name="{name}">\n'
    '<versionInfo name="Resolume Arena" majorVersion="7" minorVersion="18" microVersion="2" revision="29742"/>\n'
    '<ScreenSetup name="ScreenSetup">\n<Params name="ScreenSetupParams"/>\n'
    '<CurrentCompositionTextureSize width="{w}" height="{h}"/>\n<screens>\n'
    '<DmxScreen name="{lumi}" uniqueId="{uid}" LumiverseId="0">\n'
    '<Params name="Params"><Param name="Name" T="STRING" default="" value="{lumi}"/>'
    '<Param name="Enabled" T="BOOL" default="1" value="1"/><Param name="Hidden" T="BOOL" default="0" value="0"/>'
    '<Param name="Auto Span" T="BOOL" default="1" value="1"/><Param name="Align Output" T="BOOL" default="1" value="1"/></Params>\n'
    '<Params name="Output">' + _range("Opacity", 1, 0, 1, phase=1) + _RES_ADJ + "</Params>\n<layers>\n"
)

_RES_SLICE = (
    '<DmxSlice uniqueId="{uid}">'
    '<Params name="Common"><Param name="Name" T="STRING" default="Layer" value="{name}"/>'
    '<Param name="Enabled" T="BOOL" default="1" value="1"/></Params>'
    '<Params name="Input"><ParamChoice name="Input Source" default="0:1" value="0:1" storeChoices="0"/>'
    '<Param name="Input Opacity" T="BOOL" default="1" value="1"/><Param name="Input Bypass/Solo" T="BOOL" default="1" value="1"/>'
    '<ParamChoice name="Fixture" T="STRING" default="" value="{fixture_uuid}" storeChoices="0"/>'
    '<ParamRange name="Start Channel" T="DOUBLE" default="1" value="{start}"><PhaseSourceStatic name="PhaseSourceStatic" phase="{start_phase}"/>'
    '<BehaviourDouble name="BehaviourDouble"/><ValueRange name="defaultRange" min="1" max="131072"/>'
    '<ValueRange name="minMax" min="1" max="131072"/><ValueRange name="startStop" min="1" max="131072"/></ParamRange>'
    '<ParamChoice name="Filter Mode" T="INT32" default="0" value="0" storeChoices="0"/></Params>'
    '<Params name="Output"><Param name="Flip" T="UINT8" default="0" value="0"/>' + _RES_ADJ + "</Params>"
    '<InputRect orientation="0"><v x="{x0}" y="{y0}"/><v x="{x1}" y="{y0}"/><v x="{x1}" y="{y1}"/><v x="{x0}" y="{y1}"/></InputRect>'
    '<OutputRect orientation="0"><v x="-0.5" y="-0.5"/><v x="0.5" y="-0.5"/><v x="0.5" y="0.5"/><v x="-0.5" y="0.5"/></OutputRect>'
    '<FixtureInstance name="FixtureInstance"><Fixture name="Fixture" uuid="{fixture_uuid}" fixtureName="Twinkly pixel">'
    '<Params name="Params"><ParamFixturePixels storage="0" name="Pixels">'
    + _range("Width", 1, 1, 512, default=1, phase=0)
    + _range("Height", 1, 1, 512, default=1, phase=0)
    + '<ParamChoice name="Color Format" T="STRING" default="rgb" value="{color}" storeChoices="0"/>'
    '<ParamChoice name="Distribution" T="INT32" default="170" value="170" storeChoices="0"/>'
    + _range("Gamma", 2.5, 1, 3, default=2.5, phase=0.75)
    + "</ParamFixturePixels></Params></Fixture></FixtureInstance></DmxSlice>\n"
)

_RES_TAIL = (
    "</layers>\n<OutputDevice>"
    '<OutputDeviceDmx name="Lumiverse" deviceId="Lumiverse"><Params name="Params">'
    + _range("Framerate", 40, 1, 40, default=30, phase=1)
    + _range("Delay", 0, 0, 150, default=40, phase=0)
    + '<ParamChoice name="Dmx Interface" T="INT32" default="0" value="0" storeChoices="0"/></Params>'
    '<DmxOutputParams name="Params"><Param name="TargetIP" default="TT_DISABLED" value="{target}"/>'
    + _range("Subnet", "{subnet}", 0, 15, default=0)
    + _range("Universe", "{universe}", 0, 15, default=0)
    + "</DmxOutputParams></OutputDeviceDmx></OutputDevice>\n</DmxScreen>\n</screens>\n</ScreenSetup>\n</XmlState>\n"
)


def cmd_export(a) -> int:
    m = json.load(open(a.map))
    found = sorted(m["found"], key=lambda f: f["pixel"])
    if not found:
        raise SystemExit("map has no pixels")
    dims = m.get("dims", 2)
    patch = patch_for(a, m["pixels"])
    if a.madrix:
        cells, nudged = grid_positions(found, a.grid, dims)
        write_madrix(a.madrix, found, cells, patch, a)
        ext = [max(c[i] for c in cells) + 1 for i in range(len(cells[0]))]
        print(
            f"MADRIX: {len(found)} fixtures on a {' x '.join(map(str, ext))} grid ({nudged} nudged off a shared cell)"
            f" -> {a.madrix}"
        )
    if a.resolume:
        write_resolume(a.resolume, found, patch, a, dims)
        print(f"Resolume: {len(found)} pixel slices on {a.comp[0]}x{a.comp[1]} -> {a.resolume}")
    if not (a.madrix or a.resolume):
        print("nothing to do: add --madrix FILE and/or --resolume FILE")
    return 0


# ==========================================================================
# Synthetic test video
# ==========================================================================


def cmd_synth(a) -> int:
    """Render a fake phone video of a scan: blurry LEDs, noise, jitter, an exposure wobble."""
    rng = np.random.default_rng(a.seed)
    n = a.pixels
    w, h = a.size
    # a strand zig-zagging over a tree-ish shape, plus a few mirrored on a splitter
    t = np.linspace(0, 1, n)
    xs = w / 2 + (0.1 + 0.3 * t) * w * np.sin(t * 2 * math.pi * a.turns)
    ys = h * 0.1 + 0.8 * h * t
    xs += rng.normal(0, 2, n)
    ys += rng.normal(0, 2, n)
    truth = [{"pixel": i, "x": float(xs[i]), "y": float(ys[i])} for i in range(n)]
    hidden = set(rng.choice(n, size=int(n * a.hidden), replace=False).tolist()) if a.hidden else set()
    slots = mapcode.schedule(n)
    words = [mapcode.codeword(i, n) for i in range(n)]
    masks = [np.array(mapcode.lit_mask(s, n, words)) for s in slots]
    # stream timeline: countdown black, passes back to back, as twinkly_probe.py scan does
    stream_slot = a.slot * a.clock
    pre = 2.0
    total = pre + a.repeat * len(slots) * stream_slot + 1.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    bg = (rng.random((h, w)) * 20).astype(np.float32) + 10
    out = cv2.VideoWriter(a.video, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (w, h))
    sigma = a.blur
    # one sprite per LED, accumulated per frame
    r = int(3 * sigma) + 1
    sp_y, sp_x = np.mgrid[-r : r + 1, -r : r + 1]
    sprite = np.exp(-(sp_x**2 + sp_y**2) / (2 * sigma**2)).astype(np.float32) * 230
    send_jitter = 0.0
    frames = int(total * a.fps)
    for fi in range(frames):
        tt = fi / a.fps
        img = bg.copy()
        k = (tt - pre) / stream_slot
        if k >= 0 and k < a.repeat * len(slots):
            send_jitter = rng.normal(0, 0.01)
            si = int(k + send_jitter / stream_slot) % len(slots)
            lit = masks[si]
            gain = 1.0 + 0.15 * math.sin(tt * 0.7)  # auto exposure drifting
            for i in np.nonzero(lit)[0]:
                if i in hidden:
                    continue
                for px, py in [(xs[i], ys[i])] + ([(w - xs[i], ys[i])] if i % 97 == 5 else []):
                    cx, cy = int(round(px)), int(round(py))
                    x0, y0 = cx - r, cy - r
                    if x0 < 0 or y0 < 0 or x0 + 2 * r + 1 > w or y0 + 2 * r + 1 > h:
                        continue
                    img[y0 : y0 + 2 * r + 1, x0 : x0 + 2 * r + 1] += sprite * gain
        img += rng.normal(0, a.noise, (h, w)).astype(np.float32)
        img = np.clip(img, 0, 255).astype(np.uint8)
        out.write(cv2.merge([img, img, img]))
    out.release()
    with open(a.video + ".truth.json", "w") as fh:
        json.dump({"pixels": n, "found": truth, "hidden": sorted(hidden)}, fh)
    print(f"wrote {a.video} ({frames} frames, {w}x{h}) and {a.video}.truth.json")
    return 0


def cmd_score(a) -> int:
    m = json.load(open(a.map))
    truth = {f["pixel"]: f for f in json.load(open(a.truth))["found"]}
    hidden = set(json.load(open(a.truth)).get("hidden", []))
    # pixels sitting on top of another one can't be told apart from one angle
    pts = np.array([[truth[i]["x"], truth[i]["y"]] for i in sorted(truth)])
    sep = set()
    for i in sorted(truth):
        dd = np.linalg.norm(pts - pts[i], axis=1)
        dd[i] = 1e9
        if i not in hidden and dd.min() > a.min_sep:
            sep.add(i)
    separable = len(sep)
    errs, wrong = [], 0
    for f in m["found"]:
        t = truth[f["pixel"]]
        spots = [f] + f.get("duplicates", [])  # a mirrored strand may win the main spot
        e = min(math.hypot(q["x"] - t["x"], q["y"] - t["y"]) for q in spots)
        errs.append(e)
        wrong += e > 5 and f["pixel"] in sep
    got = len({f["pixel"] for f in m["found"]} & sep)
    print(
        f"found {got}/{separable} visible and separable ({len(m['found'])} in all), median error {np.median(errs):.2f} px,"
        f" p99 {np.percentile(errs, 99):.2f} px, {wrong} off by > 5 px"
    )
    return 0 if wrong == 0 and got >= 0.98 * separable else 1


# ==========================================================================
# CLI
# ==========================================================================


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("decode", help="video -> pixel positions")
    d.add_argument("video")
    d.add_argument("--pixels", type=int, required=True, help="pixel count the scan ran with (twinkly_probe prints it)")
    d.add_argument("--slot", type=float, default=mapcode.DEFAULT_SLOT, help="seconds per slot the scan ran with")
    d.add_argument("--out", default="pixelmap-out", help="output folder")
    d.add_argument("--max-width", type=int, default=1280, help="decode at this width (default 1280)")
    d.add_argument("--min-contrast", type=float, default=20.0, help="minimum on-minus-off brightness (0-255)")
    d.add_argument("--min-margin", type=float, default=0.3, help="how clear every bit must be (0-1)")

    s3 = sub.add_parser("solve3d", help="two decoded views -> 3D positions")
    s3.add_argument("view1")
    s3.add_argument("view2")
    s3.add_argument("--out", default="pixelmap-3d")
    s3.add_argument("--fov", type=float, default=70.0, help="camera field of view across the long side, degrees")
    s3.add_argument("--ransac", type=float, default=2.0, help="outlier threshold in pixels")
    s3.add_argument(
        "--scale", nargs=3, metavar=("PIXEL_A", "PIXEL_B", "METERS"), help="known distance between two pixels"
    )

    e = sub.add_parser("export", help="pixel positions -> MADRIX CSV / Resolume XML")
    e.add_argument("map")
    e.add_argument("--madrix", help="write a MADRIX 5 fixture list CSV here")
    e.add_argument("--resolume", help="write a Resolume Arena Advanced Output preset XML here")
    e.add_argument("--bridge-config", help="read the patch from the bridge YAML (with --device)")
    e.add_argument("--device", help="device name in --bridge-config")
    e.add_argument("--universe", default="0.0.0", help="first Art-Net universe (net.subnet.universe)")
    e.add_argument("--input", default="RGBW", help="channels per pixel as patched: RGBW or RGB")
    e.add_argument("--pixels-per-universe", type=int, default=0)
    e.add_argument("--name", default="Twinkly", help="fixture / lumiverse name")
    e.add_argument("--grid", type=int, default=0, help="MADRIX grid width in voxels (0 = auto)")
    e.add_argument("--product", default="Twinkly RGBW pixel", help="MADRIX product column text")
    e.add_argument("--fixture-id-base", type=int, default=1)
    e.add_argument("--madrix-origin", type=int, default=0, help="add to MADRIX positions (0 or 1)")
    e.add_argument("--madrix-universe-offset", type=int, default=1, help="MADRIX universe = Art-Net universe + this")
    e.add_argument(
        "--comp", type=int, nargs=2, default=(1920, 1080), metavar=("W", "H"), help="Resolume composition size"
    )
    e.add_argument("--resolume-color", default="rgbw", help="Resolume fixture Color Format value (rgb, rgbw)")
    e.add_argument("--resolume-target", default="TT_BROADCAST", help="Lumiverse TargetIP value")

    sy = sub.add_parser("synth", help="render a fake scan video for testing")
    sy.add_argument("video")
    sy.add_argument("--pixels", type=int, default=300)
    sy.add_argument("--slot", type=float, default=mapcode.DEFAULT_SLOT)
    sy.add_argument("--fps", type=float, default=30.0)
    sy.add_argument("--size", type=int, nargs=2, default=(960, 540))
    sy.add_argument("--repeat", type=int, default=2)
    sy.add_argument("--clock", type=float, default=1.004, help="streamer clock vs camera clock")
    sy.add_argument("--blur", type=float, default=2.5)
    sy.add_argument("--noise", type=float, default=6.0)
    sy.add_argument("--hidden", type=float, default=0.03, help="fraction of pixels out of view")
    sy.add_argument("--turns", type=float, default=6.0)
    sy.add_argument("--seed", type=int, default=1)

    sc = sub.add_parser("score", help="compare a decoded map with a synth truth file")
    sc.add_argument("map")
    sc.add_argument("truth")
    sc.add_argument("--min-sep", type=float, default=4.0, help="pixels closer than this to another don't count")

    a = ap.parse_args(argv)
    return {"decode": cmd_decode, "solve3d": cmd_solve3d, "export": cmd_export, "synth": cmd_synth, "score": cmd_score}[
        a.cmd
    ](a)


if __name__ == "__main__":
    sys.exit(main())
