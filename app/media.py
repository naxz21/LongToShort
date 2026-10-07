"""Utilidades de FFmpeg/FFprobe compartidas por los distintos módulos."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Callable

from .errors import FFmpegMissingError
from .logging_setup import get_logger

log = get_logger("media")

# Evita que se abra una consola por cada proceso en Windows.
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def ffmpeg_path() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise FFmpegMissingError("ffmpeg no está en el PATH")
    return path


def ffprobe_path() -> str:
    path = shutil.which("ffprobe")
    if not path:
        raise FFmpegMissingError("ffprobe no está en el PATH")
    return path


def probe(path: Path) -> dict:
    """Devuelve duración, resolución y streams de un archivo, o {} si no es válido."""
    cmd = [ffprobe_path(), "-v", "error", "-print_format", "json",
           "-show_format", "-show_streams", str(path)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60,
                             creationflags=_NO_WINDOW, encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        return {}
    if out.returncode != 0:
        log.warning("ffprobe falló con %s: %s", path, out.stderr.strip()[:500])
        return {}
    data = json.loads(out.stdout or "{}")
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    info = {
        "duration": float(data.get("format", {}).get("duration") or 0),
        "has_video": video is not None,
        "audio_streams": len(audio),
        "audio_languages": [(s.get("tags") or {}).get("language") for s in audio],
    }
    if video:
        info.update(width=int(video.get("width") or 0), height=int(video.get("height") or 0),
                    vcodec=video.get("codec_name"))
    return info


def run_ffmpeg(args: list[str], duration: float | None = None,
               on_progress: Callable[[float], None] | None = None) -> None:
    """Ejecuta ffmpeg informando el progreso (0-1) a partir de -progress."""
    cmd = [ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y",
           "-progress", "pipe:1", "-nostats", *args]
    log.debug("ffmpeg: %s", subprocess.list2cmdline(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, creationflags=_NO_WINDOW,
                            encoding="utf-8", errors="replace")
    assert proc.stdout is not None
    for line in proc.stdout:
        if on_progress and duration and line.startswith("out_time_us="):
            value = line.split("=", 1)[1].strip()
            if value.isdigit():
                on_progress(min(1.0, int(value) / 1e6 / duration))
    stderr = proc.stderr.read() if proc.stderr else ""
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg terminó con código {proc.returncode}: {stderr.strip()[-2000:]}")
