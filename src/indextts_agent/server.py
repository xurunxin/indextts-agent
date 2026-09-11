import asyncio
import hashlib
import hmac
import os
import re
import threading
import time
import traceback
import uuid
import wave
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from .common import Failure, InstanceLock
from .engine import Cancelled, IndexEngine, TestEngine
from .schema import JobRequest, capabilities
from .store import Store


class Worker:
    def __init__(self, config, store):
        self.config, self.store = config, store
        self.stop = threading.Event()
        self.wake = threading.Event()
        self.state, self.error, self.active = "loading", None, None
        self.thread = threading.Thread(target=self.run, name="model-worker", daemon=True)

    def run(self):
        try:
            self.engine = (TestEngine if self.config["engine"] == "test" else IndexEngine)(self.config)
            self.state = "ready"
        except Exception as e:  # noqa: BLE001 - isolate arbitrary model import/load failures from the HTTP service
            traceback.print_exc()
            self.state, self.error = "failed", {"code": "MODEL_LOAD_FAILED", "message": str(e)}
            return
        while not self.stop.is_set():
            job = self.store.claim()
            if job is None:
                self.wake.wait(0.5)
                self.wake.clear()
                continue
            jid = job["id"]
            self.active = jid
            output_dir = self.store.directory / "outputs"
            output_dir.mkdir(exist_ok=True)
            pending = output_dir / (jid + ".part.wav")
            output = output_dir / (jid + ".wav")
            began = time.perf_counter()
            try:
                def progress(value, desc="", job_id=jid):
                    if self.store.get(job_id)["cancel_requested"]:
                        raise Cancelled()
                    self.store.progress(job_id, float(value), desc)

                req = job["request"]
                speaker = self.store.directory / "assets" / self.store.asset(req["speaker"])["file"]
                emotion = self.store.directory / "assets" / self.store.asset(req["emotion_audio"])["file"] if req["emotion_audio"] else None
                details = self.engine.generate(req, speaker, emotion, pending, progress)
                progress(0.99, "validating WAV")
                with wave.open(str(pending), "rb") as wav:
                    duration = wav.getnframes() / wav.getframerate()
                    sr, channels = wav.getframerate(), wav.getnchannels()
                    if duration <= 0:
                        raise ValueError("模型未生成有效音频")
                elapsed = time.perf_counter() - began
                os.replace(pending, output)
                details.update({"audio_url": f"/v1/jobs/{jid}/audio", "duration_seconds": round(duration, 4),
                                "generation_seconds": round(elapsed, 3), "rtf": round(elapsed / duration, 3),
                                "sample_rate": sr, "channels": channels, "bytes": output.stat().st_size,
                                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "backend": self.config["engine"]})
                self.store.finish(jid, "succeeded", result=details)
            except Cancelled:
                self.store.finish(jid, "cancelled")
            except Exception as e:  # noqa: BLE001 - persist arbitrary model failures as a terminal task result
                traceback.print_exc()
                self.store.finish(jid, "failed", error={"code": "GENERATION_FAILED", "message": str(e)})
                # Return unused allocator blocks after OOM; no automatic duplicate retry.
                if "torch" in __import__("sys").modules:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            finally:
                pending.unlink(missing_ok=True)
                self.active = None
        self.state = "stopped"


class VoiceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(pattern=r"^[\w.-]{1,80}$")
    asset: str = Field(pattern=r"^[0-9a-f]{64}$")


def create_app(config, shutdown=None):
    directory = Path(config["data_dir"])
    store = Store(directory)
    worker = Worker(config, store)
    instance = uuid.uuid4().hex
    token = config["api_key"]
    stopping = False

    @asynccontextmanager
    async def lifespan(app):
        with InstanceLock(directory / "service.lock"):
            store.recover()
            worker.thread.start()
            yield
            worker.stop.set()
            worker.wake.set()
            # Running GPU work drains before exit. Queue stays durable for next start.
            await asyncio.to_thread(worker.thread.join)

    app = FastAPI(title="IndexTTS Agent API", version="1.0", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.worker, app.state.store = worker, store

    @app.exception_handler(Failure)
    async def failed(request, exc):
        status = 404 if exc.code.endswith("NOT_FOUND") else 409 if exc.code in {"IDEMPOTENCY_CONFLICT", "VOICE_EXISTS", "SERVICE_BUSY"} else 422
        return JSONResponse(status_code=status, content={"ok": False, "error": {"code": exc.code, "message": str(exc)}})

    @app.exception_handler(RequestValidationError)
    async def invalid(request, exc):
        # Do not echo user scripts or authentication material in errors.
        message = "; ".join(".".join(map(str, e["loc"])) + ": " + e["msg"] for e in exc.errors())
        return JSONResponse(status_code=422, content={"ok": False, "error": {"code": "INVALID_REQUEST", "message": message}})

    async def auth(authorization: str | None = Header(default=None)):
        from fastapi import HTTPException
        if not authorization or not hmac.compare_digest(authorization, "Bearer " + token):
            raise HTTPException(status_code=401, detail="API key required")

    deps = [Depends(auth)]

    @app.get("/health")
    def health():
        return {"service": "indextts-agent", "state": "stopping" if stopping else worker.state}

    @app.get("/v1/status", dependencies=deps)
    def status():
        return {"service": "indextts-agent", "instance": instance, "state": "stopping" if stopping else worker.state,
                "active_job": worker.active, "error": worker.error,
                "backend": config["engine"], "text_emotion_enabled": config["qwen_emotion"]}

    @app.get("/v1/capabilities", dependencies=deps)
    def caps():
        return capabilities(config["qwen_emotion"], config["engine"])

    @app.get("/v1/openapi.json", dependencies=deps)
    def openapi():
        return app.openapi()

    @app.post("/v1/assets", dependencies=deps)
    async def upload(request: Request):
        import soundfile as sf
        asset_dir = directory / "assets"
        asset_dir.mkdir(exist_ok=True)
        pending = asset_dir / ("upload-" + uuid.uuid4().hex)
        digest, size = hashlib.sha256(), 0
        try:
            with pending.open("wb") as f:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > 40 * 1024 * 1024:
                        raise Failure("ASSET_TOO_LARGE", "参考音频上限 40 MiB", 2)
                    digest.update(chunk)
                    f.write(chunk)
            try:
                info = sf.info(str(pending))
            except (RuntimeError, ValueError):
                raise Failure("INVALID_AUDIO", "无法解码音频；请提供 WAV 或 FLAC", 2)
            if info.format not in {"WAV", "WAVEX", "FLAC"} or not 0.25 <= info.duration <= 60 or info.channels not in {1, 2}:
                raise Failure("INVALID_AUDIO", "仅接受 0.25..60 秒、单/双声道 WAV/FLAC；建议干净的 3..15 秒人声", 2)
            aid = digest.hexdigest()
            filename = aid + (".flac" if info.format == "FLAC" else ".wav")
            target = asset_dir / filename
            if not target.exists():
                os.replace(pending, target)
            metadata = {"id": aid, "file": filename, "bytes": size, "duration_seconds": round(info.duration, 4),
                        "sample_rate": info.samplerate, "channels": info.channels,
                        "warnings": ["上游仅使用参考音频前 15 秒"] if info.duration > 15 else []}
            store.add_asset(metadata)
            return metadata
        finally:
            pending.unlink(missing_ok=True)

    @app.get("/v1/voices", dependencies=deps)
    def voices():
        return {"voices": store.voices()}

    @app.post("/v1/voices", dependencies=deps)
    def add_voice(body: VoiceRequest):
        return store.add_voice(body.name, body.asset)

    @app.post("/v1/jobs", dependencies=deps)
    def submit(body: JobRequest, idempotency_key: str | None = Header(default=None)):
        if stopping or worker.state != "ready":
            raise Failure("SERVICE_BUSY", f"模型状态为 {worker.state}，就绪后再提交")
        if idempotency_key and (len(idempotency_key) > 128 or not re.fullmatch(r"[\w.:/-]+", idempotency_key)):
            raise Failure("INVALID_IDEMPOTENCY_KEY", "幂等键仅接受字母数字及 . : / - _，最多128字符", 2)
        store.asset(body.speaker)
        if body.emotion_audio:
            store.asset(body.emotion_audio)
        if (body.emotion_auto or body.emotion_text is not None) and not config["qwen_emotion"]:
            raise Failure("TEXT_EMOTION_DISABLED", "请以 --qwen-emotion 启动服务，或使用八维情绪向量", 2)
        job, replay = store.submit(body.model_dump(), idempotency_key)
        worker.wake.set()
        return {"job": job, "replayed": replay}

    @app.get("/v1/jobs", dependencies=deps)
    def jobs(limit: int = Query(default=50, ge=1, le=200)):
        return {"jobs": store.list(limit)}

    @app.get("/v1/jobs/{jid}", dependencies=deps)
    def job(jid: str):
        return store.get(jid)

    @app.post("/v1/jobs/{jid}/cancel", dependencies=deps)
    def cancel(jid: str):
        return store.cancel(jid)

    @app.get("/v1/jobs/{jid}/audio", dependencies=deps)
    def audio(jid: str):
        job = store.get(jid)
        if job["state"] != "succeeded":
            raise Failure("RESULT_NOT_READY", f"任务状态: {job['state']}", 2)
        path = directory / "outputs" / (job["id"] + ".wav")
        if not path.is_file():
            raise Failure("RESULT_NOT_FOUND", "结果文件不存在", 2)
        return FileResponse(path, media_type="audio/wav", filename=job["id"] + ".wav")

    @app.post("/v1/service/stop", dependencies=deps)
    def stop_service():
        nonlocal stopping
        was_stopping = stopping
        stopping = True
        worker.stop.set()
        worker.wake.set()
        if shutdown and not was_stopping:
            def drain_and_exit():
                worker.thread.join()
                shutdown()
            threading.Thread(target=drain_and_exit, name="shutdown-drain", daemon=True).start()
        return {"state": "stopping", "active_job": worker.active, "policy": "drain active job; preserve queue"}

    return app


def serve(config):
    import uvicorn
    server = None
    def shutdown():
        server.should_exit = True
    app = create_app(config, shutdown)
    server = uvicorn.Server(uvicorn.Config(app, host=config["host"], port=config["port"], access_log=False, log_level="warning"))
    server.run()
