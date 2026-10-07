"""Generación de los Shorts: corte preciso + recorte 9:16 + MP4.

Por cada candidato elegido:
    inicio/fin → plan de recorte (vertical_converter) → FFmpeg → output/VIDEO_ID/short_XX.mp4
Además se mantiene output/VIDEO_ID/shorts.json con qué momento es cada archivo.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Callable

from .config import OUTPUT_DIR
from .errors import ClipGenerationError
from .logging_setup import get_logger
from .media import probe, run_ffmpeg
from .vertical_converter import crop_filter, plan_crop

log = get_logger("clip_generator")


def _next_index(out_dir: Path) -> int:
    used = [int(m.group(1)) for p in out_dir.glob("short_*.mp4") if (m := re.match(r"short_(\d+)\.mp4$", p.name))]
    return max(used, default=0) + 1


def _encode(src: Path, dst: Path, start: float, end: float, vf: str, cfg: dict,
            on_progress: Callable[[float], None]) -> None:
    o = cfg["output"]
    dur = end - start
    fade = float(o.get("audio_fade") or 0)
    af = f"afade=t=in:d={fade},afade=t=out:st={max(0.0, dur - fade):.3f}:d={fade}" if fade > 0 else "anull"
    args = [
        "-ss", f"{start:.3f}", "-i", str(src), "-t", f"{dur:.3f}",
        "-map", "0:v:0", "-map", "0:a:0",
        "-vf", vf, "-af", af,
        "-c:v", o["video_codec"], "-preset", str(o["preset"]), "-crf", str(o["crf"]),
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", str(o["audio_bitrate"]), "-ar", "48000",
        "-movflags", "+faststart", str(dst),
    ]
    run_ffmpeg(args, duration=dur, on_progress=on_progress)


def generate(video_id: str, video_path: Path, candidates: list[dict], cfg: dict,
             on_progress: Callable[[float, str], None] | None = None,
             warn: Callable[[str], None] | None = None) -> list[dict]:
    progress = on_progress or (lambda p, m: None)
    info = probe(video_path)
    if not info.get("has_video"):
        raise ClipGenerationError("No se encontró el video original. Volvé a analizarlo.", f"probe={info}")
    out_dir = OUTPUT_DIR / video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "shorts.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else []
    out_w, out_h = int(cfg["output"]["width"]), int(cfg["output"]["height"])

    results, failures = [], []
    total = len(candidates)
    for n, cand in enumerate(candidates):
        base = n / total
        label = f"Short {n + 1}/{total}"
        try:
            progress(base, f"{label}: encuadrando al sujeto")
            plan = plan_crop(video_path, cand["start"], cand["end"], info["width"], info["height"], cfg)
            vf = crop_filter(plan, out_w, out_h)
            dst = out_dir / f"short_{_next_index(out_dir):02d}.mp4"
            t0 = time.time()
            _encode(video_path, dst, cand["start"], cand["end"], vf, cfg,
                    lambda p: progress(base + (0.1 + 0.9 * p) / total, f"{label}: generando video"))
            check = probe(dst)
            if not check.get("has_video") or check.get("width") != out_w or check.get("height") != out_h:
                raise RuntimeError(f"salida inválida: {check}")
            log.info("%s listo en %.1fs (%s, %d tramos de crop)", dst.name, time.time() - t0,
                     plan.method, len(plan.segments))
            entry = {"file": dst.name, "url": f"/output/{video_id}/{dst.name}", "candidate": cand["id"],
                     "score": cand["score"], "start": cand["start"], "end": cand["end"],
                     "duration": round(check["duration"], 2), "crop": plan.to_dict(), "hook": cand["hook"]}
            results.append(entry)
            manifest.append(entry)
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001 - un clip fallido no debe frenar al resto
            log.error("Falló el candidato #%s: %s", cand.get("id"), exc, exc_info=True)
            failures.append(cand.get("id"))

    if failures and warn:
        warn(f"No se pudieron generar los momentos {', '.join(f'#{f}' for f in failures)}. "
             "Revisá logs/longtoshort.log.")
    if not results:
        raise ClipGenerationError("No se pudo generar ningún Short. Revisá logs/longtoshort.log.",
                                  f"fallaron: {failures}")
    return results
