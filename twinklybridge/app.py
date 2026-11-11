"""Process supervisor: config file, engine lifetime, reloads and the status page."""

from __future__ import annotations

import asyncio
import collections
import logging
import signal
import time

from . import __version__
from .config import Config, ConfigError, MappingConfig, listen_feeds, load_config, overlaps
from .engine import Engine
from .mapstore import MapStore

log = logging.getLogger(__name__)


class RingLogHandler(logging.Handler):
    """Keeps the last few hundred log lines for the web page."""

    def __init__(self, size: int = 400) -> None:
        super().__init__()
        self.records: collections.deque[dict] = collections.deque(maxlen=size)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.records.append(
                {
                    "time": time.strftime("%H:%M:%S", time.localtime(record.created)),
                    "level": record.levelname.lower(),
                    "message": self.format(record),
                }
            )
        except Exception:
            pass


class Supervisor:
    def __init__(self, config_path: str, *, web_override: tuple[str, int] | None = None) -> None:
        self.config_path = config_path
        self.web_override = web_override
        self.config: Config | None = None
        self.config_error = ""
        self.engine: Engine | None = None
        self.mapstore: MapStore | None = None
        self.ring = RingLogHandler()
        self.ring.setFormatter(logging.Formatter("%(message)s"))
        self._stop = asyncio.Event()
        self._lock = asyncio.Lock()
        self._web_runner = None

    def read_text(self) -> str:
        try:
            with open(self.config_path, encoding="utf-8") as fh:
                return fh.read()
        except OSError:
            return ""

    async def reload(self) -> None:
        try:
            config = load_config(self.config_path)
        except ConfigError as exc:
            self.config_error = str(exc)
            log.error("config reload failed, keeping the running config: %s", exc)
            return
        async with self._lock:
            # A change that leaves the listen sockets and protocols the same
            # (enabling/disabling/editing a controller, or a defaults/mapping/
            # log tweak) is applied in place, so the unchanged controllers keep
            # streaming. An input change, or a change to the set of Art-Net feed
            # IPs (which adds or drops a listen socket), restarts the whole engine.
            if (
                self.engine is not None
                and self.engine.running
                and self.config is not None
                and config.input == self.config.input
                and set(listen_feeds(config)) == set(listen_feeds(self.config))
            ):
                self._apply_shared(config)
                await self.engine.apply_config(config)
                self.config = config
                self.config_error = ""
            else:
                await self._swap_engine(config)

    def _apply_shared(self, config: Config) -> None:
        if self.mapstore is not None:
            if config.mapping.dir and self.mapstore.root != config.mapping.dir:
                log.warning("mapping.dir changed; restart the bridge to use %s", config.mapping.dir)
            self.mapstore.cfg = config.mapping
        logging.getLogger("twinklybridge").setLevel(config.log_level.upper())
        for warning in overlaps(config):
            log.warning("note: %s", warning)

    async def _swap_engine(self, config: Config) -> None:
        old = self.engine
        if old is not None:
            await old.stop()
        self.config = config
        self.config_error = ""
        self._apply_shared(config)
        engine = Engine(config)
        try:
            await engine.start()
        except OSError as exc:
            self.config_error = f"cannot start: {exc}"
            log.error("bridge failed to start: %s", exc)
            await engine.stop()
            self.engine = None
            return
        self.engine = engine

    def request_stop(self) -> None:
        self._stop.set()

    async def run(self) -> int:
        logging.getLogger().addHandler(self.ring)
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self.request_stop)
            except (NotImplementedError, RuntimeError):
                pass
        try:
            loop.add_signal_handler(signal.SIGHUP, lambda: asyncio.ensure_future(self.reload()))
        except (NotImplementedError, RuntimeError, AttributeError):
            pass

        log.info("Twinkly Bridge %s starting, config %s", __version__, self.config_path)
        try:
            self.config = load_config(self.config_path)
        except ConfigError as exc:
            self.config_error = str(exc)
            log.error("config error: %s", exc)

        self.mapstore = MapStore(self.config.mapping if self.config else MappingConfig(), self.config_path)
        try:
            await self.mapstore.start()
        except OSError as exc:
            log.error("camera mapping is off: cannot use %s: %s", self.mapstore.root, exc)
            self.mapstore = None

        if self.web_override is not None:
            await self._start_web(*self.web_override)
        elif self.config is None:
            await self._start_web("0.0.0.0", 8080)
        elif self.config.web.enabled:
            await self._start_web(self.config.web.bind, self.config.web.port)

        if self.config is not None:
            await self._swap_engine(self.config)

        await self._stop.wait()
        log.info("shutting down, handing devices back to their own effects")
        if self.engine is not None:
            await self.engine.stop()
        if self._web_runner is not None:
            await self._web_runner.cleanup()
        if self.mapstore is not None:
            await self.mapstore.stop()
        return 0

    async def _start_web(self, bind: str, port: int) -> None:
        from .web import make_runner

        try:
            self._web_runner = await make_runner(self, bind, port)
            log.info("status page on http://%s:%d", bind, port)
        except OSError as exc:
            log.error("status page could not bind %s:%d: %s", bind, port, exc)

    def status(self) -> dict:
        if self.engine is not None:
            data = self.engine.status()
        else:
            data = {"version": __version__, "running": False, "devices": [], "unmapped": [], "input": None}
        data["config_path"] = self.config_path
        data["config_error"] = self.config_error
        return data
