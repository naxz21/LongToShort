"""Servidor local de LongToShort (FastAPI).

Uso:  python -m app.main [--open-browser]
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import re  # noqa: E402
import shutil  # noqa: E402
import socket  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import urllib.request  # noqa: E402
import webbrowser  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from . import pipeline  # noqa: E402
from .config import (DOWNLOADS_DIR, OUTPUT_DIR, STATIC_DIR, TEMP_DIR, TEMPLATES_DIR,  # noqa: E402
                     ensure_dirs, load_config)
from .downloader import parse_video_id  # noqa: E402
from .errors import InvalidURLError  # noqa: E402
from .jobs import jobs  # noqa: E402
from .logging_setup import get_logger, setup_logging  # noqa: E402

setup_logging()
ensure_dirs()
log = get_logger("main")
_VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")
PID_FILE = TEMP_DIR / "server.pid"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    PID_FILE.write_text(str(os.getpid()), encoding="utf-8")
    if not shutil.which("ffmpeg"):
        log.error("FFmpeg no está instalado o no está en el PATH")
    log.info("LongToShort iniciado (pid %s)", os.getpid())
    yield
    PID_FILE.unlink(missing_ok=True)


app = FastAPI(title="LongToShort", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")


class AnalyzeRequest(BaseModel):
    url: str


class GenerateRequest(BaseModel):
    video_id: str
    candidate_ids: list[int]


@app.get("/")
def index():
    return FileResponse(TEMPLATES_DIR / "index.html")


@app.get("/api/health")
def health():
    gpu = False
    try:
        import ctranslate2
        gpu = ctranslate2.get_cuda_device_count() > 0
    except Exception:  # noqa: BLE001
        pass
    return {"ok": True, "ffmpeg": bool(shutil.which("ffmpeg")), "gpu": gpu}


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    try:
        video_id = parse_video_id(req.url)
    except InvalidURLError as exc:
        raise HTTPException(400, exc.user_message) from exc
    if not shutil.which("ffmpeg"):
        raise HTTPException(500, "No se encontró FFmpeg. Instalalo (ver README) y reiniciá LongToShort.")
    url = f"https://www.youtube.com/watch?v={video_id}"
    job = jobs.submit("analyze", lambda j: pipeline.run_analysis(j, url), final_status="ready")
    return {"job_id": job.id, "video_id": video_id}


@app.get("/api/job/{job_id}")
def job_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Ese proceso ya no existe (¿se reinició el servidor?).")
    return job.to_dict()


@app.post("/api/generate")
def generate(req: GenerateRequest):
    if not _VIDEO_ID.match(req.video_id):
        raise HTTPException(400, "ID de video inválido.")
    if not req.candidate_ids:
        raise HTTPException(400, "Elegí al menos un momento para generar.")
    job = jobs.submit("generate", lambda j: pipeline.run_generation(j, req.video_id, req.candidate_ids),
                      final_status="completed")
    return {"job_id": job.id}


@app.get("/media/{video_id}/original.mp4")
def media(video_id: str):
    path = DOWNLOADS_DIR / video_id / "original.mp4"
    if not _VIDEO_ID.match(video_id) or not path.exists():
        raise HTTPException(404, "Video no encontrado.")
    return FileResponse(path, media_type="video/mp4")


def _open_browser_when_ready(url: str) -> None:
    for _ in range(60):
        try:
            urllib.request.urlopen(url + "/api/health", timeout=1)
            webbrowser.open(url)
            return
        except Exception:  # noqa: BLE001
            time.sleep(0.5)


def run() -> None:
    import uvicorn
    cfg = load_config()
    host, port = cfg["server"]["host"], int(cfg["server"]["port"])
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe_sock:
        if probe_sock.connect_ex((host, port)) == 0:
            print(f"\n  El puerto {port} ya está en uso: LongToShort ya está corriendo (o usá otro puerto en config.json).\n")
            if "--open-browser" in sys.argv:
                webbrowser.open(f"http://{host}:{port}")
            return
    (TEMP_DIR / "server.port").write_text(str(port), encoding="utf-8")
    if "--open-browser" in sys.argv:
        threading.Thread(target=_open_browser_when_ready, args=(f"http://{host}:{port}",), daemon=True).start()
    print(f"\n  LongToShort corriendo en http://{host}:{port}  (Ctrl+C para detener)\n")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    run()
