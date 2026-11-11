"""Status page, the setup page, the phone mapping page and the JSON API (aiohttp)."""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import os
import pathlib
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from aiohttp import web

from . import configstore, mapping
from .config import MAP_COLORS, ConfigError, parse_config
from .engine import DeviceRunner, ScanError
from .mapstore import StoreError
from .twinkly import HTTP_PORT, Discovered, NotFound, TwinklyClient, TwinklyError, discover, local_networks

if TYPE_CHECKING:
    from .app import Supervisor

PAGE = pathlib.Path(__file__).parent / "status.html"
MAP_PAGE = pathlib.Path(__file__).parent / "mapping.html"
SETUP_PAGE = pathlib.Path(__file__).parent / "setup.html"
NO_CACHE = {"Cache-Control": "no-cache"}
log = logging.getLogger(__name__)


async def index(request: web.Request) -> web.FileResponse:
    return web.FileResponse(PAGE, headers=NO_CACHE)


async def map_page(request: web.Request) -> web.FileResponse:
    return web.FileResponse(MAP_PAGE, headers=NO_CACHE)


async def setup_page(request: web.Request) -> web.FileResponse:
    return web.FileResponse(SETUP_PAGE, headers=NO_CACHE)


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


# --------------------------------------------------------------------------
# Setup page: the config file and discovery
# --------------------------------------------------------------------------


async def _json_body(request: web.Request) -> dict | None:
    """The request's JSON object; {} for an empty body; None when it is not an object."""
    if not request.can_read_body:
        return {}
    try:
        body = await request.json()
    except ValueError:
        return None
    return body if isinstance(body, dict) else None


def _running(dev: DeviceRunner) -> dict:
    return {
        "state": dev.state,
        "ip": dev.cfg.ip,
        "artnet": dev.info.artnet_enabled if dev.info is not None else None,
    }


def _setup_state(sup: Supervisor) -> dict:
    try:
        raw = configstore.load_raw(sup.config_path)
        file_error = ""
    except ConfigError as exc:
        raw, file_error = {}, str(exc)
    state = configstore.describe(raw, sup.config_path)
    if file_error:
        state["error"] = file_error
    state["reload_error"] = sup.config_error
    state["running"] = {d.name: _running(d) for d in sup.engine.devices} if sup.engine is not None else {}
    state["networks"] = local_networks()
    return state


async def api_setup_state(request: web.Request) -> web.Response:
    return web.json_response(_setup_state(request.app["supervisor"]))


async def _save(request: web.Request, change: Callable[[dict], Any], text: str | None = None) -> web.Response:
    """Load the file, apply change(raw) (which validates), write it with a backup, reload the bridge.

    With text, that exact text is written instead of a regenerated file (the YAML editor).
    """
    sup: Supervisor = request.app["supervisor"]
    async with request.app["config_lock"]:
        try:
            raw = configstore.load_raw(sup.config_path) if text is None else configstore.raw_from_text(text)
            result = change(raw)
            output = configstore.dump(raw) if text is None else text
        except ConfigError as exc:
            return _fail(str(exc))
        try:
            backup = configstore.write_text(sup.config_path, output)
        except OSError as exc:
            return _fail(f"cannot write {sup.config_path}: {exc.strerror or exc}", 500)
        await sup.reload()
    return web.json_response(
        {
            "ok": True,
            "result": result,
            "backup": os.path.basename(backup) if backup else "",
            "reload_error": sup.config_error,
            "state": _setup_state(sup),
        }
    )


async def api_setup_devices_add(request: web.Request) -> web.Response:
    body = await _json_body(request)
    if body is None:
        return _fail("expected JSON")
    specs = body.get("devices")
    if specs is None and isinstance(body.get("device"), dict):
        specs = [body["device"]]
    if not isinstance(specs, list) or not specs or not all(isinstance(x, dict) for x in specs):
        return _fail("send device: {...} or devices: [{...}, ...]")
    return await _save(request, lambda raw: configstore.add_devices(raw, specs))


async def api_setup_device_replace(request: web.Request) -> web.Response:
    body = await _json_body(request)
    if body is None or not isinstance(body.get("device"), dict):
        return _fail("send device: {...}")
    name = request.match_info["name"]
    return await _save(request, lambda raw: configstore.replace_device(raw, name, body["device"]))


async def api_setup_device_patch(request: web.Request) -> web.Response:
    body = await _json_body(request)
    if body is None or not body:
        return _fail("send the keys to change")
    name = request.match_info["name"]
    return await _save(request, lambda raw: configstore.patch_device(raw, name, body))


async def api_setup_device_delete(request: web.Request) -> web.Response:
    name = request.match_info["name"]
    return await _save(request, lambda raw: configstore.remove_device(raw, name))


async def api_setup_settings(request: web.Request) -> web.Response:
    body = await _json_body(request)
    if body is None or not body:
        return _fail("send the sections to change: input, defaults, web, mapping, log_level")
    return await _save(request, lambda raw: configstore.set_sections(raw, body))


async def api_setup_yaml(request: web.Request) -> web.Response:
    body = await _json_body(request)
    if body is None or not isinstance(body.get("yaml"), str):
        return _fail("send yaml: <the whole file>")
    text = body["yaml"]
    if not text.endswith("\n"):
        text += "\n"

    def validate(raw: dict) -> None:
        parse_config(raw)

    return await _save(request, validate, text=text)


def _discovery_targets(sup: Supervisor, extra: str) -> list[str]:
    targets = [extra] if extra else []
    targets += [net["broadcast"] for net in local_networks()]
    if sup.config is not None and sup.config.input.broadcast != "255.255.255.255":
        targets.append(sup.config.input.broadcast)
    targets.append("255.255.255.255")
    return list(dict.fromkeys(targets))


async def _probe(session: Any, found: Discovered) -> dict:
    """Ask a controller what it is. No login, so a streaming device keeps its session."""
    row = {"ip": found.ip, "name": found.name, "info": None, "error": ""}
    try:
        info = await TwinklyClient(session, found.ip, port=found.http_port, timeout=3.0).info()
    except TwinklyError as exc:
        text = str(exc)
        if "Cannot connect" in text or "no answer" in text:
            text = f"no answer from {found.ip}:{found.http_port} (off, unreachable, or not a Twinkly)"
        row["error"] = text
        return row
    row["info"] = info.to_dict()
    row["name"] = found.name or info.device_name
    return row


async def api_setup_discover(request: web.Request) -> web.Response:
    import aiohttp

    sup: Supervisor = request.app["supervisor"]
    body = await _json_body(request)
    if body is None:
        return _fail("expected JSON")
    ip = str(body.get("ip") or "").strip()
    extra = str(body.get("broadcast") or "").strip()
    for value in (ip, extra):
        if value:
            try:
                ipaddress.IPv4Address(value)
            except ValueError:
                return _fail(f"{value!r} is not an IPv4 address")
    try:
        port = int(body.get("port") or HTTP_PORT)
        timeout = min(10.0, max(0.5, float(body.get("timeout") or 2.0)))
    except (TypeError, ValueError):
        return _fail("port and timeout must be numbers")
    if ip:
        targets: list[str] = []
        found = [Discovered(ip=ip, name="", http_port=port)]
    else:
        targets = _discovery_targets(sup, extra)
        found = await asyncio.to_thread(discover, targets, timeout)
    configured = {d.ip: d.name for d in sup.config.devices} if sup.config is not None else {}
    async with aiohttp.ClientSession() as session:
        rows = await asyncio.gather(*(_probe(session, d) for d in found))
    for row in rows:
        row["configured_as"] = configured.get(row["ip"])
    return web.json_response({"ok": True, "targets": targets, "devices": list(rows)})


async def api_setup_artnet(request: web.Request) -> web.Response:
    """Switch a controller's built-in Art-Net mode off (or on). A persistent change on the controller."""
    import aiohttp

    sup: Supervisor = request.app["supervisor"]
    body = await _json_body(request)
    if body is None:
        return _fail("expected JSON")
    ip = str(body.get("ip") or "").strip()
    try:
        ipaddress.IPv4Address(ip)
        port = int(body.get("port") or HTTP_PORT)
    except ValueError:
        return _fail("send ip (IPv4) and optionally port")
    enable = body.get("enable")
    if not isinstance(enable, bool):
        return _fail("enable must be true or false")
    runner = None
    if sup.engine is not None:
        runner = next((d for d in sup.engine.devices if d.cfg.ip == ip), None)
    client = getattr(runner, "client", None)
    if client is not None and client.token:
        session = None  # the bridge's own session and token: no second login on a streaming controller
    else:
        session = aiohttp.ClientSession()
        client = TwinklyClient(session, ip, port=port, timeout=5.0)
    try:
        if session is not None:
            await client.login()
        after = await client.set_artnet(enable)
    except NotFound:
        return _fail(f"{ip}: this firmware has no built-in Art-Net setting (consumer device?)", 400)
    except TwinklyError as exc:
        return _fail(f"{ip}: {exc}", 502)
    finally:
        if session is not None:
            await session.close()
    if runner is not None and runner.info is not None:
        runner.info.artnet_enabled = bool(after.get("enable"))
    log.info("built-in Art-Net on %s switched %s from the setup page", ip, "on" if enable else "off")
    return web.json_response({"ok": True, "ip": ip, "artnet": after})


def make_app(supervisor: Supervisor) -> web.Application:
    app = web.Application()
    app["supervisor"] = supervisor
    app["config_lock"] = asyncio.Lock()
    app.router.add_get("/", index)
    app.router.add_get("/map", map_page)
    app.router.add_get("/setup", setup_page)
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
    app.router.add_get("/api/setup/state", api_setup_state)
    app.router.add_post("/api/setup/devices", api_setup_devices_add)
    app.router.add_put("/api/setup/devices/{name}", api_setup_device_replace)
    app.router.add_route("PATCH", "/api/setup/devices/{name}", api_setup_device_patch)
    app.router.add_delete("/api/setup/devices/{name}", api_setup_device_delete)
    app.router.add_post("/api/setup/settings", api_setup_settings)
    app.router.add_post("/api/setup/yaml", api_setup_yaml)
    app.router.add_post("/api/setup/discover", api_setup_discover)
    app.router.add_post("/api/setup/artnet", api_setup_artnet)
    return app


async def make_runner(supervisor: Supervisor, bind: str, port: int) -> web.AppRunner:
    runner = web.AppRunner(make_app(supervisor), access_log=None)
    await runner.setup()
    await web.TCPSite(runner, bind, port, reuse_address=True).start()
    return runner
