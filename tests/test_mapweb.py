"""The phone mapping page's API: upload a clip, decode it in the background, export, delete."""

import asyncio
import csv
import io
import os

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")
pytest.importorskip("scipy")

import aiohttp  # noqa: E402
from aiohttp.test_utils import TestClient, TestServer  # noqa: E402

from twinklybridge import mapping  # noqa: E402
from twinklybridge.app import Supervisor  # noqa: E402
from twinklybridge.config import load_config  # noqa: E402
from twinklybridge.mapstore import MapStore, slug  # noqa: E402
from twinklybridge.web import make_app  # noqa: E402

CONFIG = """
input: {cidr: 10.0.0.2/24}
devices:
  - {name: Tree 1, ip: 10.0.0.5, universe: 0.0.0, pixels: 80}
mapping: {min_free_mb: 0}
"""


async def _state(client):
    r = await client.get("/api/map/state")
    assert r.status == 200
    return await r.json()


async def _wait_job(client, job_id, timeout=240.0):
    for _ in range(int(timeout / 0.2)):
        s = await _state(client)
        job = next((j for j in s["jobs"] if j["id"] == job_id), None)
        if job is None or job["status"] in ("done", "failed"):
            return job
        await asyncio.sleep(0.2)
    raise AssertionError("job did not finish")


def test_upload_decode_export_delete(tmp_path):
    video = str(tmp_path / "IMG_0001.MOV")
    mapping.main(
        ["synth", video, "--pixels", "80", "--size", "320", "240", "--blur", "1.5", "--hidden", "0", "--repeat", "1"]
    )
    cfg_path = tmp_path / "bridge.yaml"
    cfg_path.write_text(CONFIG)

    async def main():
        sup = Supervisor(str(cfg_path))
        sup.config = load_config(str(cfg_path))
        sup.mapstore = MapStore(sup.config.mapping, str(cfg_path), root=str(tmp_path / "data"))
        await sup.mapstore.start()
        async with TestClient(TestServer(make_app(sup))) as client:
            s = await _state(client)
            assert s["devices"][0]["name"] == "Tree 1" and s["devices"][0]["pixels"] == 80
            assert s["missing"] == "" and s["jobs"] == []

            with open(video, "rb") as fh:
                fd = aiohttp.FormData()
                fd.add_field("device", "Tree 1")
                fd.add_field("label", "front")
                fd.add_field("clip", fh, filename="IMG_0001.MOV", content_type="video/quicktime")
                r = await client.post("/api/map/upload", data=fd)
            j = await r.json()
            assert r.status == 200 and j["ok"], j
            job = j["job"]
            assert job["pixels"] == 80 and job["status"] == "queued" and job["slug"] == slug("Tree 1")
            assert job["clip"]["name"] == "IMG_0001.MOV" and job["clip"]["bytes"] == os.path.getsize(video)

            done = await _wait_job(client, job["id"])
            assert done["status"] == "done", done
            assert done["result"]["found"] >= 78 and done["result"]["passes"] == 1
            assert not done["clip"]["kept"]  # the clip is gone, the results stay
            jdir = os.path.join(sup.mapstore.root, done["slug"], done["id"])
            assert not os.path.exists(os.path.join(jdir, "clip.mov"))
            assert sorted(done["files"]) == ["log.txt", "map.csv", "map.json", "preview.png"]

            base = f"/api/map/jobs/{done['slug']}/{done['id']}"
            r = await client.get(base + "/preview.png")
            assert r.status == 200 and r.content_type == "image/png"
            r = await client.get(base + "/job.json")  # not on the menu
            assert r.status == 404
            r = await client.get(f"/api/map/jobs/{done['slug']}/..%2F..%2Fx/preview.png")
            assert r.status != 200

            exp = f"/api/map/export/{done['slug']}/{done['id']}"
            r = await client.get(exp + "/madrix.csv")
            assert r.status == 200, await r.text()
            rows = list(csv.DictReader(io.StringIO(await r.text())))
            assert len(rows) == done["result"]["found"]
            assert rows[0]["DMX Universe"] == "1" and rows[0]["DMX Channel"] == "1"  # Art-Net 0.0.0 -> MADRIX 1
            assert rows[0]["Name"].startswith("Tree 1 ")
            r = await client.get(exp + "/resolume.xml")
            assert r.status == 200 and "<DmxSlice" in await r.text()
            r = await client.get(exp + "/map.csv")
            assert r.status == 200

            r = await client.post("/api/map/solve", json={"device": "Tree 1"})
            assert r.status == 409  # one clip is not enough for 3D
            r = await client.post("/api/map/scan", json={"device": "Tree 1"})
            assert r.status == 404  # the bridge engine is not running in this test

            # the free-space guard refuses an upload before writing anything
            sup.mapstore.cfg.min_free_mb = 10**9
            fd = aiohttp.FormData()
            fd.add_field("device", "Tree 1")
            fd.add_field("clip", b"not really a video", filename="x.mp4", content_type="video/mp4")
            r = await client.post("/api/map/upload", data=fd)
            assert r.status == 507 and "disk" in (await r.json())["error"]
            assert len(os.listdir(os.path.join(sup.mapstore.root, done["slug"]))) == 2  # job + export
            sup.mapstore.cfg.min_free_mb = 0

            r = await client.delete(base)
            assert r.status == 200
            assert not os.path.exists(jdir)
            assert (await _state(client))["jobs"] == []
        await sup.mapstore.stop()

    asyncio.run(main())


def test_failed_decode_keeps_the_clip_and_reports(tmp_path):
    cfg_path = tmp_path / "bridge.yaml"
    cfg_path.write_text(CONFIG)
    bogus = tmp_path / "bogus.mp4"
    bogus.write_bytes(b"\x00" * 5000)

    async def main():
        sup = Supervisor(str(cfg_path))
        sup.config = load_config(str(cfg_path))
        sup.mapstore = MapStore(sup.config.mapping, str(cfg_path), root=str(tmp_path / "data"))
        await sup.mapstore.start()
        async with TestClient(TestServer(make_app(sup))) as client:
            with open(bogus, "rb") as fh:
                fd = aiohttp.FormData()
                fd.add_field("device", "Tree 1")
                fd.add_field("clip", fh, filename="bogus.mp4", content_type="video/mp4")
                r = await client.post("/api/map/upload", data=fd)
            job = (await r.json())["job"]
            done = await _wait_job(client, job["id"], timeout=60)
            assert done["status"] == "failed" and done["error"]
            assert done["clip"]["kept"] and "log.txt" in done["files"]
            r = await client.get(f"/api/map/export/{done['slug']}/{done['id']}/madrix.csv")
            assert r.status == 409
        await sup.mapstore.stop()

    asyncio.run(main())


def test_whole_rig_clip_decodes_and_exports_two_lumiverses(tmp_path):
    import xml.etree.ElementTree as ET

    from twinklybridge.mapstore import RIG, RIG_SLUG

    video = str(tmp_path / "tree.mp4")
    mapping.main(
        ["synth", video, "--pixels", "80", "--size", "320", "240", "--blur", "1.5", "--hidden", "0", "--repeat", "1"]
    )
    cfg_path = tmp_path / "bridge.yaml"
    cfg_path.write_text(CONFIG)

    def seg(u, n):
        return [{"universe": u, "channel": 0, "first_pixel": 0, "count": n}]

    rig = {
        "kind": "twinkly-rig",
        "version": 1,
        "created": 1.0,
        "total": 80,
        "slot": 0.4,
        "pass_seconds": 20.0,
        "devices": [
            {
                "name": "Tree A",
                "ip": "10.0.0.5",
                "offset": 0,
                "pixels": 50,
                "channels": 4,
                "feed": "",
                "segments": seg(0, 50),
            },
            {
                "name": "Tree B",
                "ip": "10.0.0.6",
                "offset": 50,
                "pixels": 30,
                "channels": 4,
                "feed": "",
                "segments": seg(256, 30),
            },
        ],
        "skipped": [],
    }

    async def upload(client):
        with open(video, "rb") as fh:
            fd = aiohttp.FormData()
            fd.add_field("device", RIG)
            fd.add_field("label", "front")
            fd.add_field("clip", fh, filename="tree.mp4", content_type="video/mp4")
            r = await client.post("/api/map/upload", data=fd)
        return r.status, await r.json()

    async def main():
        sup = Supervisor(str(cfg_path))
        sup.config = load_config(str(cfg_path))
        sup.mapstore = MapStore(sup.config.mapping, str(cfg_path), root=str(tmp_path / "data"))
        await sup.mapstore.start()
        async with TestClient(TestServer(make_app(sup))) as client:
            status, j = await upload(client)
            assert status == 400 and "whole-rig scan first" in j["error"]  # nothing to decode against yet

            sup.mapstore.note_rig_scan(rig)
            status, j = await upload(client)
            assert status == 200 and j["ok"], j
            job = j["job"]
            assert job["slug"] == RIG_SLUG and job["pixels"] == 80 and job["rig"]["controllers"] == 2
            done = await _wait_job(client, job["id"])
            assert done["status"] == "done", done
            assert done["result"]["found"] >= 78
            assert "rig.json" in done["files"]

            exp = f"/api/map/export/{RIG_SLUG}/{done['id']}"
            r = await client.get(exp + "/resolume.xml")
            assert r.status == 200, await r.text()
            root = ET.fromstring(await r.text())
            screens = root.findall(".//DmxScreen")
            assert len(screens) == 2
            names = [s.find(".//DmxSlice/Params/Param[@name='Name']").get("value") for s in screens]
            assert names[0].startswith("Tree A ") and names[1].startswith("Tree B ")
            r = await client.get(exp + "/canvas.png")
            assert r.status == 200 and r.content_type == "image/png"
            assert r.headers["Content-Disposition"].startswith("inline")

            r = await client.post("/api/map/solve", json={"device": RIG})
            assert r.status == 409  # one clip is not enough for 3D
        await sup.mapstore.stop()

    asyncio.run(main())
