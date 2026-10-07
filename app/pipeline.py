"""Orquestación de las etapas. Cada función recibe el Job para informar progreso.

Análisis:   URL → descarga → transcripción → análisis → candidatos  (estado final: ready)
Generación: candidatos elegidos → clips 9:16                        (estado final: completed)

Etapas futuras (subtítulos, zooms, B-roll, música...) se agregarían como pasos
adicionales de run_generation sobre cada clip, sin tocar la detección.
"""
from __future__ import annotations

import json
import time

from . import analyzer, clip_detector, clip_generator, downloader, transcription
from .config import DOWNLOADS_DIR, load_config
from .errors import ClipGenerationError
from .jobs import Job


def public_video_info(meta: dict) -> dict:
    audio = meta.get("audio_format") or {}
    vfmt = meta.get("file") or meta.get("video_format") or {}
    lang = (audio.get("language") or meta.get("original_language") or "?").split("-")[0]
    is_original = audio.get("language_preference") == 10 or lang == meta.get("original_language")
    return {
        "id": meta.get("id"), "title": meta.get("title"), "channel": meta.get("channel"),
        "duration": meta.get("duration"), "thumbnail": meta.get("thumbnail"),
        "url": meta.get("webpage_url"), "original_language": meta.get("original_language"),
        "audio_track": f"{lang} · original" if is_original else lang,
        "audio_tracks": len(meta.get("audio_tracks") or []),
        "resolution": f"{vfmt.get('width')}x{vfmt.get('height')}" if vfmt.get("width") else None,
    }


def run_analysis(job: Job, url: str) -> dict:
    cfg = load_config()
    job.update(status="downloading", progress=0, message="Obteniendo información del video")
    asset = downloader.download(
        url, cfg,
        on_info=lambda meta: job.data.update(video=public_video_info(meta)),
        on_progress=lambda p, msg: job.update(progress=p * 100, message=msg),
    )
    job.data["video"] = public_video_info(asset.metadata)

    job.update(status="transcribing", progress=0, message="Transcribiendo")
    transcript = transcription.transcribe(
        asset, cfg, on_progress=lambda p, msg: job.update(progress=p * 100, message=msg), warn=job.warn)
    job.data["transcription"] = {"language": transcript.get("language"), "model": transcript.get("model"),
                                 "segments": len(transcript["segments"])}

    job.update(status="analyzing", progress=0, message="Analizando el contenido")
    energy = transcription.load_energy(asset)
    analysis = analyzer.analyze(transcript, energy, cfg,
                                on_progress=lambda p, msg: job.update(progress=p * 50, message=msg))
    candidates = clip_detector.detect(analysis, cfg,
                                      on_progress=lambda p, msg: job.update(progress=50 + p * 50, message=msg))

    result = {
        "video": job.data["video"],
        "video_url": f"/media/{asset.video_id}/original.mp4",
        "transcription": job.data["transcription"],
        "candidates": candidates,
        "sentences": len(analysis.sentences),
    }
    (asset.folder / "analysis.json").write_text(json.dumps(
        {**result, "created_at": time.time(), "config": {"clips": cfg["clips"], "scoring": cfg["scoring"]}},
        ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def run_generation(job: Job, video_id: str, candidate_ids: list[int]) -> dict:
    cfg = load_config()
    folder = DOWNLOADS_DIR / video_id
    analysis_path = folder / "analysis.json"
    if not analysis_path.exists():
        raise ClipGenerationError("No se encontró el análisis de este video. Volvé a analizarlo.",
                                  f"falta {analysis_path}")
    saved = json.loads(analysis_path.read_text(encoding="utf-8"))
    by_id = {c["id"]: c for c in saved["candidates"]}
    chosen = [by_id[i] for i in sorted(set(candidate_ids)) if i in by_id]
    if not chosen:
        raise ClipGenerationError("Elegí al menos un momento para generar.", f"ids={candidate_ids}")

    job.data["video"] = saved.get("video")
    job.update(status="generating", progress=0, message="Generando Shorts")
    shorts = clip_generator.generate(
        video_id, folder / "original.mp4", chosen, cfg,
        on_progress=lambda p, msg: job.update(progress=p * 100, message=msg), warn=job.warn)
    return {"shorts": shorts, "output_dir": str((clip_generator.OUTPUT_DIR / video_id).resolve())}
