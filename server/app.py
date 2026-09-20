"""Browser control, bounded generation sessions, skeleton preview and optional simulation."""

import asyncio
import io
import json
import logging
import socket
import sys
import tempfile
import threading
import time
import uuid
import zipfile
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from light_deploy import __version__
from light_deploy.protocol import GenerateRequest
from server.config import ControlConfig
from server.generation_store import GenerationStore
from server.humanoid_preview import decode_human_action_for_web
from server.inference_client import InferenceClient
from server.process import ManagedProcess
from server.schemas import SonicRequest, StartRequest
from server.sonic import check_runtime, run_sonic_simulation, write_sonic_plan

LOGGER = logging.getLogger(__name__)


class ControlSession:
    def __init__(self, config, inference):
        self.config = config
        self.inference = inference
        self.process = ManagedProcess(config.temp_root)
        self.connected = bool(config.inference_url)
        self.start_time = None
        self.model_path = str(config.model_path or "")
        self.generations = GenerationStore()
        self.generation_task = None
        self.sonic_task = None
        self.sonic_cancel = threading.Event()
        self.sonic_jobs = OrderedDict()
        self.lock = asyncio.Lock()
        self.sonic_ready = False
        self.sonic_reason = "Sonic checkpoints are not configured"

    async def status(self):
        result = {"state": "stopped", "model_path": self.model_path, "ready": None, "error": None}
        if not self.connected:
            return result
        result["state"] = "starting"
        if not self.config.inference_url and not self.process.running:
            result["state"] = "error"
            result["error"] = {"message": "Inference process exited; inspect its terminal log before restarting"}
            return result
        try:
            health = await self.inference.health()
            if health["ready"]:
                result.update(state="ready", ready=health, model_path=health["model"])
        except (RuntimeError, KeyError) as error:
            if self.config.inference_url or self.start_time is None or time.monotonic() - self.start_time > 300:
                result["state"] = "error"
                result["error"] = {"message": str(error)}
        return result

    async def start(self, request):
        async with self.lock:
            if self.config.inference_url:
                if request.model_fields_set:
                    raise HTTPException(409, "Remote model lifecycle belongs to the GPU API host")
                self.connected = True
                return await self.status()
            if self.process.running:
                raise HTTPException(409, "Inference is already running")
            values = request.model_dump(exclude_none=True)
            model = values.pop("model_path", self.model_path)
            if not model or not Path(model).expanduser().is_absolute():
                raise HTTPException(422, "An absolute checkpoint directory is required")
            with socket.socket() as probe:
                try:
                    probe.bind(("127.0.0.1", self.config.inference_port))
                except OSError as error:
                    raise HTTPException(409, "Inference port is occupied; choose another --inference-port") from error
            command = [
                sys.executable,
                "-m",
                "light_deploy.api",
                "--model",
                model,
                "--port",
                str(self.config.inference_port),
            ]
            for field in ("device", "max_model_len", "kv_cache_memory_bytes"):
                command.extend(["--" + field.replace("_", "-"), str(values.get(field, getattr(self.config, field)))])
            if values.get("deterministic", self.config.deterministic):
                command.append("--deterministic")
            await asyncio.to_thread(self.process.start, command)
            LOGGER.info("Inference log: %s/inference.log", self.process.directory.name)
            self.model_path = model
            self.connected = True
            self.start_time = time.monotonic()
            return {"state": "starting", "model_path": model, "ready": None, "error": None}

    async def stop(self):
        async with self.lock:
            self.connected = False
            self.sonic_cancel.set()
            if self.generation_task is not None:
                self.generation_task.cancel()
                await asyncio.gather(self.generation_task, return_exceptions=True)
            if self.sonic_task is not None:
                await self.sonic_task
            await asyncio.to_thread(self.process.stop)
            self.generations.clear()
            for job in self.sonic_jobs.values():
                job["directory"].cleanup()
            self.sonic_jobs.clear()
            return {"state": "stopped"}

    async def generate(self, job, request):
        try:
            result = await self.inference.generate(request.model_dump())
            self.generations.complete(job.generation_id, result)
        except asyncio.CancelledError:
            job.state = "cancelled"
            raise
        except Exception as error:
            LOGGER.exception("Generation %s failed", job.generation_id)
            job.state = "error"
            job.error = (
                str(error) if isinstance(error, RuntimeError) else "Invalid inference response; check control logs"
            )


def create_app(config: ControlConfig | None = None, *, inference_client=None):
    resolved = config or ControlConfig.from_env()
    inference = inference_client or InferenceClient(resolved.api_url)
    session = ControlSession(resolved, inference)

    @asynccontextmanager
    async def lifespan(application):
        if resolved.sonic_checkpoint is not None:
            session.sonic_ready, session.sonic_reason = await asyncio.to_thread(
                check_runtime, resolved.sonic_python, resolved.sonic_checkpoint
            )
        if resolved.model_path is not None:
            await session.start(StartRequest())
        try:
            yield
        finally:
            await session.stop()
            await inference.close()

    application = FastAPI(title="Light Deploy Compact Action", version=__version__, lifespan=lifespan)
    application.state.session = session
    application.state.config = resolved

    def generation(generation_id, *, completed=False):
        try:
            item = session.generations.get(generation_id)
        except KeyError as error:
            raise HTTPException(404, "Generation expired or not found") from error
        if completed and item.state != "done":
            raise HTTPException(409, "Generation has not completed")
        return item

    @application.get("/api/defaults")
    async def defaults():
        return {
            "ok": True,
            "defaults": {
                "mode": "remote" if resolved.inference_url else "local",
                "inference_url": resolved.api_url,
                "model_path": str(resolved.model_path or ""),
                "device": resolved.device,
                "max_model_len": resolved.max_model_len,
                "kv_cache_memory_bytes": resolved.kv_cache_memory_bytes,
                "deterministic": resolved.deterministic,
                "sonic_unavailable_reason": session.sonic_reason if not session.sonic_ready else None,
                "generation": GenerateRequest(prompt="defaults").model_dump(exclude={"prompt", "images"}),
                "capabilities": {
                    "inference": True,
                    "thinking": True,
                    "vision": True,
                    "sonic": session.sonic_ready,
                },
            },
        }

    @application.get("/api/health")
    async def health():
        status = await session.status()
        return {
            "ok": True,
            "control": {"ready": True, "version": __version__},
            "inference": {"ready": status["state"] == "ready", "state": status["state"]},
        }

    @application.get("/api/service/status")
    async def service_status():
        return await session.status()

    @application.post("/api/service/start", status_code=202)
    async def start(request: StartRequest):
        return await session.start(request)

    @application.post("/api/service/stop")
    async def stop():
        return await session.stop()

    @application.post("/api/generations", status_code=202)
    async def generate(request: GenerateRequest):
        async with session.lock:
            if (await session.status())["state"] != "ready":
                raise HTTPException(503, "Inference service is not ready")
            if session.generation_task is not None and not session.generation_task.done():
                raise HTTPException(409, "Generation already running")
            job = session.generations.create(request.model_dump())
            session.generation_task = asyncio.create_task(session.generate(job, request))
            return job.public()

    @application.get("/api/generations/{generation_id}")
    async def generation_status(generation_id: str):
        return generation(generation_id).public()

    @application.get("/api/generations/{generation_id}/download")
    async def download_generation(generation_id: str):
        item = generation(generation_id, completed=True)
        action_bytes = io.BytesIO()
        np.save(action_bytes, item.action, allow_pickle=False)
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            bundle.writestr("action.npy", action_bytes.getvalue())
            bundle.writestr("reasoning.txt", item.reasoning)
            bundle.writestr(
                "generation.json",
                json.dumps(
                    {
                        **item.public(),
                        "schema_version": 1,
                        "settings": item.request,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            )
        return Response(
            archive.getvalue(),
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="action-{item.generation_id}.zip"',
                "Cache-Control": "no-store",
            },
        )

    @application.get("/api/generations/{generation_id}/action")
    async def action(generation_id: str):
        item = generation(generation_id, completed=True)
        return await asyncio.to_thread(decode_human_action_for_web, item.action)

    @application.post("/api/generations/{generation_id}/sonic/sim", status_code=202)
    async def sonic(generation_id: str, request: SonicRequest):
        async with session.lock:
            item = generation(generation_id, completed=True)
            if not session.sonic_ready:
                raise HTTPException(503, "Configure Sonic policy checkpoints and a simulation Python environment")
            if session.sonic_task is not None and not session.sonic_task.done():
                raise HTTPException(409, "Sonic simulation already running")
            if len(session.sonic_jobs) >= 8:
                _, oldest = session.sonic_jobs.popitem(last=False)
                oldest["directory"].cleanup()
            directory = tempfile.TemporaryDirectory(prefix="light-sonic-", dir=resolved.temp_root)
            job_id = uuid.uuid4().hex
            job = {
                "job_id": job_id,
                "generation_id": generation_id,
                "action_sha256": item.action_sha256,
                "state": "running",
                "directory": directory,
                "result": None,
                "error": None,
            }
            session.sonic_jobs[job_id] = job
            session.sonic_cancel = threading.Event()
            cancel_event = session.sonic_cancel

            async def run():
                try:
                    root = Path(directory.name)
                    plan = await asyncio.to_thread(
                        write_sonic_plan,
                        item.action,
                        root / "plan.npz",
                        provenance={"generation_id": generation_id, "action_sha256": item.action_sha256},
                    )
                    result = await asyncio.to_thread(
                        run_sonic_simulation,
                        plan,
                        root / "result.json",
                        sim_python=resolved.sonic_python,
                        sonic_checkpoint=resolved.sonic_checkpoint,
                        mp4_path=root / "result.mp4",
                        cancel_event=cancel_event,
                        **request.model_dump(),
                    )
                    if cancel_event.is_set():
                        job["state"] = "cancelled"
                        return
                    job["state"] = "done"
                    job["result"] = {key: value for key, value in result.items() if "path" not in key}
                except Exception:
                    LOGGER.exception("Sonic job %s failed", job_id)
                    job["state"] = "cancelled" if cancel_event.is_set() else "error"
                    job["error"] = {"message": "Sonic simulation failed; check control logs and policy configuration"}

            session.sonic_task = asyncio.create_task(run())
            return {key: value for key, value in job.items() if key != "directory"}

    def sonic_job(job_id):
        try:
            return session.sonic_jobs[job_id]
        except KeyError as error:
            raise HTTPException(404, "Sonic job expired or not found") from error

    @application.get("/api/sonic/jobs/{job_id}")
    async def sonic_status(job_id: str):
        return {key: value for key, value in sonic_job(job_id).items() if key != "directory"}

    @application.post("/api/sonic/jobs/{job_id}/cancel")
    async def cancel_sonic(job_id: str):
        job = sonic_job(job_id)
        if job["state"] == "running":
            session.sonic_cancel.set()
            await session.sonic_task
        return {"state": job["state"]}

    @application.get("/api/sonic/jobs/{job_id}/video")
    async def video(job_id: str):
        job = sonic_job(job_id)
        if job["state"] != "done":
            raise HTTPException(409, "Sonic video is not ready")
        path = Path(job["directory"].name) / "result.mp4"
        if path.is_symlink() or not path.is_file():
            raise HTTPException(404, "Sonic video not found")
        return FileResponse(path, media_type="video/mp4")

    if (resolved.webui_dist / "index.html").is_file():
        application.mount("/", StaticFiles(directory=resolved.webui_dist, html=True), name="webui")
    else:

        @application.get("/")
        async def missing_webui():
            return JSONResponse(
                status_code=503, content={"error": "WebUI build missing; run cd webui && npm run build"}
            )

    return application
