"""Status page and JSON API (aiohttp)."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

from aiohttp import web

if TYPE_CHECKING:
    from .app import Supervisor

PAGE = pathlib.Path(__file__).parent / "status.html"


async def index(request: web.Request) -> web.FileResponse:
    return web.FileResponse(PAGE, headers={"Cache-Control": "no-cache"})


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


def make_app(supervisor: Supervisor) -> web.Application:
    app = web.Application()
    app["supervisor"] = supervisor
    app.router.add_get("/", index)
    app.router.add_get("/api/status", api_status)
    app.router.add_get("/api/log", api_log)
    app.router.add_get("/api/config", api_config)
    app.router.add_post("/api/reload", api_reload)
    return app


async def make_runner(supervisor: Supervisor, bind: str, port: int) -> web.AppRunner:
    runner = web.AppRunner(make_app(supervisor), access_log=None)
    await runner.setup()
    await web.TCPSite(runner, bind, port, reuse_address=True).start()
    return runner
