"""Status page, the phone mapping page and the JSON API (aiohttp)."""

from __future__ import annotations

import pathlib
import time
from typing import TYPE_CHECKING

from aiohttp import web

from . import mapping
from .config import MAP_COLORS
from .engine import DeviceRunner, ScanError
from .mapstore import StoreError

if TYPE_CHECKING:
    from .app import Supervisor

PAGE = pathlib.Path(__file__).parent / "status.html"
MAP_PAGE = pathlib.Path(__file__).parent / "mapping.html"
NO_CACHE = {"Cache-Control": "no-cache"}


async def index(request: web.Request) -> web.FileResponse:
    return web.FileResponse(PAGE, headers=NO_CACHE)


async def map_page(request: web.Request) -> web.FileResponse:
    return web.FileResponse(MAP_PAGE, headers=NO_CACHE)


async def api_status(request: web.Request) -> web.Response:
    return web.json_response(request.app["supervisor"].status())


async def api_log(request: web.Request) -> web.Response:
    return web.json_response({"lines": list(request.app["supervisor"].ring.records)})


async def api_config(request: web.Request) -> web.Response:
    sup: Supervisor = request.app["supervisor"]
    return web.json_response({"path": sup.config_path, "yaml": sup.read_text(), "error": sup.config_error})


async def api_reload(request: web.Request) -> web.Response:
    sup: Supervisor = request.app["supervisor"]
    await sup.reload()
    return web.json_response({"ok": not sup.config_error, "error": sup.config_error})


# --------------------------------------------------------------------------
# Camera mapping
# --------------------------------------------------------------------------


def _fail(message: str, status: int = 400) -> web.Response:
    return web.json_response({"ok": False, "error": message}, status=status)


def _store(request: web.Request):
    store = request.app["supervisor"].mapstore
    if store is None:
        raise web.HTTPServiceUnavailable(text="the mapping store is not running")
    return store


def _device(request: web.Request, name: str) -> DeviceRunner | None:
    engine = request.app["supervisor"].engine
    if engine is None:
        return None
    for dev in engine.devices:
        if dev.name == name:
            return dev
    return None


async def api_map_state(request: web.Request) -> web.Response:
    sup: Supervisor = request.app["supervisor"]
    store = _store(request)
    now = time.monotonic()
    devices = []
    if sup.engine is not None:
        for d in sup.engine.devices:
            devices.append(
                {
                    "name": d.name,
                    "ip": d.cfg.ip,
                    "state": d.state,
                    "ready": d.ready,
                    "pixels": d.out_pixels,
                    "scan": d.scan.status(now) if d.scanning(now) else None,
                    "warnings": list(d.warnings),
                    "error": d.error,
                }
            )
    elif sup.config is not None:
        for c in sup.config.enabled_devices:
            devices.append(
                {
                    "name": c.name,
                    "ip": c.ip,
                    "state": "bridge not running",
                    "ready": False,
                    "pixels": c.pixels,
                    "scan": None,
                    "warnings": [],
                    "error": sup.config_error,
                }
            )
    cfg = store.cfg
    return web.json_response(
        {
            "devices": devices,
            "jobs": store.jobs(),
            "current": store.current,
            "pending": list(store.pending),
            "disk": store.disk(),
            "missing": mapping.missing_dependency(),
            "install_hint": mapping.INSTALL_HINT,
            "settings": {
                "slot": cfg.slot,
                "brightness": cfg.brightness,
                "color": cfg.color,
                "passes": cfg.passes,
                "keep_clips": cfg.keep_clips,
                "max_upload_mb": cfg.max_upload_mb,
            },
            "config_path": sup.config_path,
        }
    )


async def api_map_scan(request: web.Request) -> web.Response:
    store = _store(request)
    try:
        body = await request.json()
    except ValueError:
        return _fail("expected JSON")
    name = str(body.get("device") or "")
    dev = _device(request, name)
    if dev is None:
        return _fail(f"no running device named {name!r}", 404)
    cfg = store.cfg
    try:
        slot = float(body.get("slot") or cfg.slot)
        brightness = int(body.get("brightness") or cfg.brightness)
        passes = int(body.get("passes") or cfg.passes)
        color = str(body.get("color") or cfg.color)
    except (TypeError, ValueError):
        return _fail("slot, brightness and passes must be numbers")
    if not 0.1 <= slot <= 2.0 or not 1 <= brightness <= 255 or not 1 <= passes <= 100 or color not in MAP_COLORS:
        return _fail("slot 0.1-2 s, brightness 1-255, passes 1-100, colour white/rgb/green")
    try:
        job = dev.start_scan(slot=slot, brightness=brightness, color=color, repeat=passes)
    except ScanError as exc:
        return _fail(str(exc), 409)
    store.note_scan(dev.name, job.pixels, job.slot)
    return web.json_response({"ok": True, "scan": job.status(time.monotonic())})


async def api_map_scan_stop(request: web.Request) -> web.Response:
    try:
        body = await request.json()
    except ValueError:
        return _fail("expected JSON")
    dev = _device(request, str(body.get("device") or ""))
    if dev is None:
        return _fail("no such running device", 404)
    dev.stop_scan()
    return web.json_response({"ok": True})


async def api_map_upload(request: web.Request) -> web.Response:
    sup: Supervisor = request.app["supervisor"]
    store = _store(request)
    missing = mapping.missing_dependency()
    if missing:
        return _fail(f"decoding needs {missing} on the bridge box: {mapping.INSTALL_HINT}", 503)
    try:
        reader = await request.multipart()
    except (ValueError, AssertionError):
        return _fail("expected a multipart upload")
    device = ""
    label = ""
    job = None
    async for part in reader:
        if part.name == "device":
            device = (await part.text()).strip()
        elif part.name == "label":
            label = (await part.text()).strip()
        elif part.name == "clip":
            if not device:
                return _fail("send the device field before the clip")
            dev = _device(request, device)
            if dev is not None:
                fallback = dev.out_pixels
            else:
                known = {c.name: c for c in sup.config.devices} if sup.config else {}
                if device not in known:
                    return _fail(f"no device named {device!r} in the config", 404)
                fallback = known[device].pixels
            settings = store.settings_for(device, fallback)
            try:
                job = await store.save_upload(
                    device, label, part.filename or "", part, settings["pixels"], settings["slot"]
                )
            except StoreError as exc:
                return _fail(str(exc), 507 if "disk" in str(exc) else 400)
    if job is None:
        return _fail("no clip in the upload")
    return web.json_response({"ok": True, "job": job})


async def api_map_file(request: web.Request) -> web.Response:
    store = _store(request)
    try:
        path = store.result_file(request.match_info["slug"], request.match_info["job"], request.match_info["name"])
    except StoreError as exc:
        return _fail(str(exc), 404)
    return web.FileResponse(path, headers=NO_CACHE)


async def api_map_delete(request: web.Request) -> web.Response:
    store = _store(request)
    try:
        store.delete_job(request.match_info["slug"], request.match_info["job"])
    except StoreError as exc:
        return _fail(str(exc), 404)
    return web.json_response({"ok": True})


async def api_map_solve(request: web.Request) -> web.Response:
    store = _store(request)
    try:
        body = await request.json()
    except ValueError:
        return _fail("expected JSON")
    fov = body.get("fov")
    try:
        fov = float(fov) if fov not in (None, "", 0) else None
    except (TypeError, ValueError):
        return _fail("fov must be a number of degrees")
    if fov is not None and not 20 <= fov <= 150:
        return _fail("fov should be between 20 and 150 degrees")
    try:
        job = store.solve(str(body.get("device") or ""), fov)
    except StoreError as exc:
        return _fail(str(exc), 409)
    return web.json_response({"ok": True, "job": job})


async def api_map_export(request: web.Request) -> web.Response:
    store = _store(request)
    slug, source, kind = request.match_info["slug"], request.match_info["source"], request.match_info["kind"]
    try:
        path = await store.export(slug, source, kind)
    except StoreError as exc:
        return _fail(str(exc), 409)
    filename = f"{slug}-{source}-{kind}" if kind not in ("map.json", "map.csv") else f"{slug}-{source}-{kind}"
    return web.FileResponse(path, headers={**NO_CACHE, "Content-Disposition": f'attachment; filename="{filename}"'})


def make_app(supervisor: Supervisor) -> web.Application:
    app = web.Application()
    app["supervisor"] = supervisor
    app.router.add_get("/", index)
    app.router.add_get("/map", map_page)
    app.router.add_get("/api/status", api_status)
    app.router.add_get("/api/log", api_log)
    app.router.add_get("/api/config", api_config)
    app.router.add_post("/api/reload", api_reload)
    app.router.add_get("/api/map/state", api_map_state)
    app.router.add_post("/api/map/scan", api_map_scan)
    app.router.add_post("/api/map/scan/stop", api_map_scan_stop)
    app.router.add_post("/api/map/upload", api_map_upload)
    app.router.add_post("/api/map/solve", api_map_solve)
    app.router.add_get("/api/map/jobs/{slug}/{job}/{name}", api_map_file)
    app.router.add_delete("/api/map/jobs/{slug}/{job}", api_map_delete)
    app.router.add_get("/api/map/export/{slug}/{source}/{kind}", api_map_export)
    return app


async def make_runner(supervisor: Supervisor, bind: str, port: int) -> web.AppRunner:
    runner = web.AppRunner(make_app(supervisor), access_log=None)
    await runner.setup()
    await web.TCPSite(runner, bind, port, reuse_address=True).start()
    return runner
