"""Camera mapping jobs: uploaded clips, decoding in a subprocess, 3D solves and exports.

Everything lives under one directory (config `mapping.dir`):

    <device>/<job-id>/job.json      what it is and how it went
    <device>/<job-id>/clip.<ext>    the upload; deleted after a good decode unless keep_clips
    <device>/<job-id>/map.json      the decoder's results (plus map.csv, preview.png, log.txt)
    <device>/solve/                 the 3D solve over every decoded clip of that device
    <device>/export/                MADRIX / Resolume files, rebuilt on every download
    _rig/rig.json                   the last whole-rig scan: rig-wide pixel numbers per controller
    _rig/<job-id>/rig.json          the scan a clip was filmed against (each rig job keeps its own copy)

A whole-rig scan (device "*") blinks every controller at once with one pixel
numbering across the rig, so a clip of the whole tree decodes every controller
in one go and the export patches each pixel from the manifest.

Decoding runs `python -m twinklybridge.mapping` in a niced subprocess, one job at
a time, so a long decode never stalls the frames the bridge is streaming.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import secrets
import shutil
import sys
import time

from .config import MappingConfig

log = logging.getLogger(__name__)

MB = 1024 * 1024
SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi", ".3gp", ".mts", ".hevc"}
RESULT_FILES = ("map.json", "map.csv", "preview.png", "log.txt", "rig.json")
EXPORT_KINDS = {"madrix.csv": "--madrix", "resolume.xml": "--resolume", "canvas.png": "--preview"}
SOLVE_ID = "solve"
RIG = "*"  # the device name for a whole-rig scan
RIG_SLUG = "_rig"  # slug() strips leading underscores, so no controller can land here
RIG_FILE = "rig.json"
RIG_DECODE_WIDTH = 1920  # a whole tree packs many pixels into the frame: decode with more detail
CHUNK = 1 << 20


class StoreError(Exception):
    """Something the person can fix. The message is user facing."""


def slug(name: str) -> str:
    if name == RIG:
        return RIG_SLUG
    return SAFE_NAME.sub("_", str(name)).strip("._")[:48] or "device"


def resolve_dir(configured: str) -> str:
    if configured:
        return os.path.abspath(os.path.expanduser(configured))
    state = os.environ.get("STATE_DIRECTORY")
    if state:
        return os.path.join(state.split(":")[0], "mapping")
    return os.path.abspath("mapping-data")


def _read_json(path: str) -> dict | None:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def rig_key(rig: dict | None) -> str:
    """Two scans with the same key number the pixels the same way, so their clips solve together."""
    if not rig:
        return ""
    return json.dumps([[d["name"], d["offset"], d["pixels"]] for d in rig.get("devices", [])])


def _write_json(path: str, data: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1)
    os.replace(tmp, path)


def _nice() -> None:  # pragma: no cover - child process
    try:
        os.nice(10)
    except (AttributeError, OSError):
        pass


class MapStore:
    def __init__(self, cfg: MappingConfig, config_path: str, *, root: str | None = None) -> None:
        self.cfg = cfg
        self.config_path = config_path
        self.root = root or resolve_dir(cfg.dir)
        self.scan_settings: dict[str, dict] = {}  # device name -> {pixels, slot} of the last scan
        self.current: str | None = None  # "<slug>/<job id>" being processed
        self.pending: list[str] = []
        self._queue: asyncio.Queue[tuple[str, str]] | None = None
        self._worker: asyncio.Task | None = None
        self._procs: dict[str, asyncio.subprocess.Process] = {}

    # -- paths -----------------------------------------------------------------

    def device_dir(self, device_slug: str) -> str:
        return os.path.join(self.root, device_slug)

    def job_dir(self, device_slug: str, job_id: str) -> str:
        if not ID_RE.match(device_slug) or not ID_RE.match(job_id):
            raise StoreError("bad job reference")
        return os.path.join(self.root, device_slug, job_id)

    def result_file(self, device_slug: str, job_id: str, name: str) -> str:
        if name not in RESULT_FILES:
            raise StoreError(f"no such file: {name}")
        path = os.path.join(self.job_dir(device_slug, job_id), name)
        if not os.path.isfile(path):
            raise StoreError(f"{name} is not there (yet)")
        return path

    # -- lifetime ----------------------------------------------------------------

    async def start(self) -> None:
        os.makedirs(self.root, exist_ok=True)
        self._queue = asyncio.Queue()
        for job in self.jobs():
            if job["status"] in ("queued", "running"):
                if job["type"] == "decode" and self._clip_path(job):
                    self._enqueue(job)
                else:
                    self._update(job, status="failed", error="interrupted by a restart")
        self._worker = asyncio.get_running_loop().create_task(self._work())
        log.info("mapping store at %s", self.root)

    async def stop(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            await asyncio.gather(self._worker, return_exceptions=True)
            self._worker = None
        for proc in list(self._procs.values()):
            try:
                proc.terminate()
            except ProcessLookupError:
                pass
        self._procs.clear()

    # -- scans -----------------------------------------------------------------

    def note_scan(self, device: str, pixels: int, slot: float) -> None:
        self.scan_settings[device] = {"pixels": int(pixels), "slot": float(slot), "at": time.time()}

    def note_rig_scan(self, manifest: dict) -> None:
        os.makedirs(self.device_dir(RIG_SLUG), exist_ok=True)
        _write_json(os.path.join(self.device_dir(RIG_SLUG), RIG_FILE), manifest)
        self.note_scan(RIG, manifest["total"], manifest["slot"])

    def rig_manifest(self) -> dict | None:
        """The last whole-rig scan (kept on disk, so it survives a restart)."""
        return _read_json(os.path.join(self.device_dir(RIG_SLUG), RIG_FILE))

    def settings_for(self, device: str, fallback_pixels: int) -> dict:
        if device == RIG:
            rig = self.rig_manifest()
            if not rig:
                raise StoreError("run a whole-rig scan first: the clip is decoded against that scan's numbering")
            return {"pixels": int(rig["total"]), "slot": float(rig["slot"])}
        s = self.scan_settings.get(device)
        if s:
            return {"pixels": s["pixels"], "slot": s["slot"]}
        return {"pixels": int(fallback_pixels), "slot": float(self.cfg.slot)}

    # -- jobs ------------------------------------------------------------------

    def jobs(self, device: str | None = None) -> list[dict]:
        out = []
        try:
            device_dirs = sorted(os.listdir(self.root))
        except OSError:
            return out
        for d in device_dirs:
            ddir = os.path.join(self.root, d)
            if not os.path.isdir(ddir):
                continue
            for j in sorted(os.listdir(ddir)):
                job = _read_json(os.path.join(ddir, j, "job.json"))
                if not job or (device is not None and job.get("device") != device):
                    continue
                job["slug"] = d
                job["files"] = [f for f in RESULT_FILES if os.path.isfile(os.path.join(ddir, j, f))]
                out.append(job)
        out.sort(key=lambda j: (j.get("created", 0), j["id"]))
        return out

    def _job_path(self, job: dict) -> str:
        return os.path.join(self.root, job["slug"], job["id"], "job.json")

    def _update(self, job: dict, **fields) -> dict:
        job.update(fields)
        job["updated"] = time.time()
        stored = {k: v for k, v in job.items() if k not in ("slug", "files")}
        _write_json(self._job_path(job), stored)
        return job

    def _clip_path(self, job: dict) -> str | None:
        clip = job.get("clip") or {}
        name = clip.get("file")
        if not name:
            return None
        path = os.path.join(self.root, job["slug"], job["id"], name)
        return path if os.path.isfile(path) else None

    def _enqueue(self, job: dict) -> None:
        assert self._queue is not None
        key = f"{job['slug']}/{job['id']}"
        if key not in self.pending and key != self.current:
            self.pending.append(key)
            self._queue.put_nowait((job["slug"], job["id"]))

    # -- disk --------------------------------------------------------------------

    def disk(self) -> dict:
        try:
            usage = shutil.disk_usage(self.root)
            free = usage.free
        except OSError:
            free = 0
        used = 0
        for base, _dirs, files in os.walk(self.root):
            for f in files:
                try:
                    used += os.path.getsize(os.path.join(base, f))
                except OSError:
                    pass
        return {
            "dir": self.root,
            "free_mb": free // MB,
            "used_mb": round(used / MB, 1),
            "min_free_mb": self.cfg.min_free_mb,
        }

    def _check_free(self, extra: int = 0) -> None:
        free = shutil.disk_usage(self.root).free - extra
        if free < self.cfg.min_free_mb * MB:
            raise StoreError(
                f"not enough disk space: {max(free, 0) // MB} MB free, the limit is {self.cfg.min_free_mb} MB"
                " (delete old clips or lower mapping.min_free_mb)"
            )

    # -- uploads -----------------------------------------------------------------

    async def save_upload(self, device: str, label: str, filename: str, part, pixels: int, slot: float) -> dict:
        """Stream one uploaded clip to disk and queue it for decoding.

        `part` is anything with `await part.read_chunk(size) -> bytes` (an aiohttp multipart part).
        """
        if self._queue is None:
            raise StoreError("the mapping store is not running")
        os.makedirs(self.root, exist_ok=True)
        self._check_free()
        ext = os.path.splitext(filename or "")[1].lower()
        if ext not in VIDEO_EXTS:
            ext = ".mp4"
        device_slug = slug(device)
        job_id = time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)
        jdir = self.job_dir(device_slug, job_id)
        os.makedirs(jdir, exist_ok=True)
        job = {
            "id": job_id,
            "slug": device_slug,
            "device": device,
            "type": "decode",
            "label": (label or "").strip()[:40],
            "status": "uploading",
            "created": time.time(),
            "clip": {"name": os.path.basename(filename or "clip"), "file": "clip" + ext, "bytes": 0, "kept": True},
            "pixels": int(pixels),
            "slot": float(slot),
            "result": None,
            "error": "",
        }
        if device == RIG:
            rig = self.rig_manifest()
            if not rig:
                os.rmdir(jdir)
                raise StoreError("run a whole-rig scan first: the clip is decoded against that scan's numbering")
            _write_json(os.path.join(jdir, RIG_FILE), rig)
            job["rig"] = {"controllers": len(rig["devices"]), "key": rig_key(rig), "scan": rig.get("created")}
        self._update(job)
        limit = self.cfg.max_upload_mb * MB
        size = 0
        try:
            with open(os.path.join(jdir, "clip" + ext), "wb") as fh:
                while True:
                    chunk = await part.read_chunk(CHUNK)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > limit:
                        raise StoreError(f"the clip is bigger than mapping.max_upload_mb ({self.cfg.max_upload_mb} MB)")
                    if size % (16 * MB) < CHUNK:
                        self._check_free()
                    fh.write(chunk)
        except BaseException:
            shutil.rmtree(jdir, ignore_errors=True)
            raise
        if size == 0:
            shutil.rmtree(jdir, ignore_errors=True)
            raise StoreError("the upload was empty")
        job["clip"]["bytes"] = size
        self._update(job, status="queued")
        self._enqueue(job)
        log.info("%s: clip %s (%s, %.1f MB) queued for decoding", device, job_id, job["clip"]["name"], size / MB)
        return job

    def delete_job(self, device_slug: str, job_id: str) -> None:
        jdir = self.job_dir(device_slug, job_id)
        key = f"{device_slug}/{job_id}"
        proc = self._procs.get(key)
        if proc is not None:
            try:
                proc.terminate()
            except ProcessLookupError:
                pass
        if key in self.pending:
            self.pending.remove(key)
        shutil.rmtree(jdir, ignore_errors=True)
        log.info("mapping job %s deleted", key)

    # -- 3D ----------------------------------------------------------------------

    def solve(self, device: str, fov: float | None = None) -> dict:
        if self._queue is None:
            raise StoreError("the mapping store is not running")
        clips = [j for j in self.jobs(device) if j["type"] == "decode" and j["status"] == "done"]
        if device == RIG and clips:
            # only clips filmed against the same numbering (the newest clip's) solve together
            key = (clips[-1].get("rig") or {}).get("key")
            clips = [j for j in clips if (j.get("rig") or {}).get("key") == key]
        if len(clips) < 2:
            raise StoreError("3D needs at least two decoded clips from different angles (6-8 is better)")
        device_slug = slug(device)
        sdir = self.job_dir(device_slug, SOLVE_ID)
        os.makedirs(sdir, exist_ok=True)
        for f in RESULT_FILES:
            try:
                os.remove(os.path.join(sdir, f))
            except OSError:
                pass
        if device == RIG:
            shutil.copyfile(
                os.path.join(self.job_dir(device_slug, clips[-1]["id"]), RIG_FILE), os.path.join(sdir, RIG_FILE)
            )
        job = {
            "id": SOLVE_ID,
            "slug": device_slug,
            "device": device,
            "type": "solve",
            "label": "3D solve",
            "status": "queued",
            "created": time.time(),
            "clips": [{"id": j["id"], "label": j["label"]} for j in clips],
            "fov": fov,
            "result": None,
            "error": "",
        }
        self._update(job)
        self._enqueue(job)
        return job

    # -- exports -----------------------------------------------------------------

    async def export(self, device_slug: str, source: str, kind: str) -> str:
        """Build (or find) one export file and return its path."""
        src_dir = self.job_dir(device_slug, source)
        job = _read_json(os.path.join(src_dir, "job.json"))
        if not job or job.get("status") != "done":
            raise StoreError("that clip has not been decoded yet")
        src = os.path.join(src_dir, "map.json")
        if kind in ("map.json", "map.csv"):
            return self.result_file(device_slug, source, kind)
        flag = EXPORT_KINDS.get(kind)
        if flag is None:
            raise StoreError(f"unknown export {kind}")
        out_dir = os.path.join(self.device_dir(device_slug), "export")
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, f"{source}-{kind}")
        if job["device"] == RIG:
            rig = os.path.join(src_dir, RIG_FILE)
            if not os.path.isfile(rig):
                raise StoreError("this result has no rig manifest; decode a clip from a whole-rig scan")
            cmd = ["export", src, "--rig", rig, "--name", "Twinkly rig", flag, out]
        else:
            cmd = [
                "export",
                src,
                "--bridge-config",
                self.config_path,
                "--device",
                job["device"],
                "--name",
                job["device"],
                flag,
                out,
            ]
        if kind != "madrix.csv":
            cmd += ["--comp", str(self.cfg.comp_width), str(self.cfg.comp_height)]
        rc, text, _ = await self._run_mapping(None, cmd)
        if rc != 0 or not os.path.isfile(out):
            raise StoreError(f"export failed: {_last_line(text)}")
        return out

    # -- the worker ----------------------------------------------------------------

    async def _work(self) -> None:
        assert self._queue is not None
        while True:
            device_slug, job_id = await self._queue.get()
            key = f"{device_slug}/{job_id}"
            if key in self.pending:
                self.pending.remove(key)
            job = _read_json(os.path.join(self.root, device_slug, job_id, "job.json"))
            if not job:
                continue  # deleted while it waited
            job["slug"] = device_slug
            self.current = key
            try:
                if job["type"] == "solve":
                    await self._run_solve(job)
                else:
                    await self._run_decode(job)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # never let one job kill the worker
                log.exception("mapping job %s failed", key)
                if os.path.isdir(os.path.dirname(self._job_path(job))):
                    self._update(job, status="failed", error=f"{exc.__class__.__name__}: {exc}")
            finally:
                self.current = None

    async def _run_decode(self, job: dict) -> None:
        clip = self._clip_path(job)
        jdir = os.path.join(self.root, job["slug"], job["id"])
        if clip is None:
            self._update(job, status="failed", error="the clip file is gone")
            return
        self._update(job, status="running", started=time.time())
        cmd = [
            "decode",
            clip,
            "--pixels",
            str(job["pixels"]),
            "--slot",
            str(job["slot"]),
            "--out",
            jdir,
            "--max-width",
            str(max(self.cfg.decode_width, RIG_DECODE_WIDTH) if job.get("device") == RIG else self.cfg.decode_width),
        ]
        rc, text, seconds = await self._run_mapping(job, cmd)
        if not os.path.isdir(jdir):
            return  # deleted meanwhile
        result = _read_json(os.path.join(jdir, "map.json")) if rc == 0 else None
        if result:
            summary = {
                "found": len(result.get("found", [])),
                "missing": len(result.get("missing", [])),
                "mirrored": sum(1 for f in result.get("found", []) if f.get("duplicates")),
                "passes": result.get("passes"),
                "frames_per_slot": round(float(result.get("frames_per_slot", 0)), 1),
                "image": result.get("image"),
            }
            kept = bool(self.cfg.keep_clips)
            if not kept:
                try:
                    os.remove(clip)
                except OSError:
                    kept = True
            job["clip"]["kept"] = kept
            self._update(job, status="done", result=summary, seconds=round(seconds, 1), error="")
            log.info(
                "%s: clip %s decoded, %d/%d pixels found in %.0f s",
                job["device"],
                job["id"],
                summary["found"],
                job["pixels"],
                seconds,
            )
        else:
            self._update(job, status="failed", seconds=round(seconds, 1), error=_last_line(text) or f"exit code {rc}")
            log.warning("%s: clip %s failed to decode: %s", job["device"], job["id"], job["error"])

    async def _run_solve(self, job: dict) -> None:
        sdir = os.path.join(self.root, job["slug"], job["id"])
        maps = []
        for c in job.get("clips", []):
            path = os.path.join(self.root, job["slug"], c["id"], "map.json")
            if os.path.isfile(path):
                maps.append(path)
        if len(maps) < 2:
            self._update(job, status="failed", error="fewer than two decoded clips are left")
            return
        self._update(job, status="running", started=time.time())
        cmd = ["solve3d", *maps, "--out", sdir]
        if job.get("fov"):
            cmd += ["--fov", str(job["fov"])]
        rc, text, seconds = await self._run_mapping(job, cmd)
        if not os.path.isdir(sdir):
            return
        result = _read_json(os.path.join(sdir, "map.json")) if rc == 0 else None
        if result:
            errs = [f.get("error_px", 0.0) for f in result.get("found", [])]
            errs.sort()
            summary = {
                "found": len(result.get("found", [])),
                "missing": len(result.get("missing", [])),
                "views": len(result.get("views_used", [])),
                "clips": len(maps),
                "fov": result.get("fov"),
                "median_error_px": errs[len(errs) // 2] if errs else None,
                "unit": result.get("unit"),
            }
            self._update(job, status="done", result=summary, seconds=round(seconds, 1), error="")
            log.info("%s: 3D solve done, %d pixels from %d clips", job["device"], summary["found"], summary["views"])
        else:
            self._update(job, status="failed", seconds=round(seconds, 1), error=_last_line(text) or f"exit code {rc}")
            log.warning("%s: 3D solve failed: %s", job["device"], job["error"])

    async def _run_mapping(self, job: dict | None, args: list[str]) -> tuple[int, str, float]:
        """Run `python -m twinklybridge.mapping <args>` and keep its output in log.txt."""
        env = dict(os.environ)
        pkg_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env["PYTHONPATH"] = pkg_parent + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env["PYTHONUNBUFFERED"] = "1"
        cmd = [sys.executable, "-m", "twinklybridge.mapping", *args]
        key = f"{job['slug']}/{job['id']}" if job else None
        t0 = time.monotonic()
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
            preexec_fn=_nice if hasattr(os, "nice") else None,
        )
        if key:
            self._procs[key] = proc
        try:
            out, _ = await proc.communicate()
        finally:
            if key:
                self._procs.pop(key, None)
        text = out.decode("utf-8", "replace")
        if job:
            jdir = os.path.join(self.root, job["slug"], job["id"])
            if os.path.isdir(jdir):
                try:
                    with open(os.path.join(jdir, "log.txt"), "w", encoding="utf-8") as fh:
                        fh.write("$ " + " ".join(cmd) + "\n" + text[-20000:])
                except OSError:
                    pass
        return proc.returncode or 0, text, time.monotonic() - t0


def _last_line(text: str) -> str:
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return lines[-1][:300] if lines else ""
