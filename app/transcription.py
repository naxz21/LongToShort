"""Transcripción local con faster-whisper, con timestamps por palabra.

- El idioma se DETECTA del audio (nunca se fuerza). Si no coincide con el
  idioma original informado por YouTube, el problema es la pista de audio:
  se intenta descargar la pista correcta y se vuelve a verificar.
- Usa GPU NVIDIA si está disponible (con las DLL de requirements-gpu.txt);
  si falla, cae a CPU automáticamente.
- El resultado se guarda en downloads/VIDEO_ID/transcript.json y se reutiliza.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable

import numpy as np

from .config import MODELS_DIR
from .downloader import VideoAsset, replace_audio_track
from .errors import TranscriptionError
from .logging_setup import get_logger
from .media import ffmpeg_path

log = get_logger("transcription")

SAMPLE_RATE = 16000
ENERGY_FPS = 20          # resolución del perfil de energía de audio (frames por segundo)
TRANSCRIPT_VERSION = 2   # subir si cambia el formato o la forma de transcribir (invalida el caché)

# Whisper imita el estilo del texto previo: un prompt bien puntuado en el idioma
# del audio hace que la transcripción tenga puntos, comas y signos de pregunta,
# que son la base para detectar frases completas. No cambia el idioma detectado.
STYLE_PROMPTS = {
    "es": "Hola, ¿cómo estás? Muy bien, gracias. Hoy vamos a hablar de algo importante.",
    "en": "Hello, how are you? I'm fine, thanks. Today we're going to talk about something important.",
    "pt": "Olá, tudo bem? Tudo ótimo, obrigado. Hoje vamos falar de algo importante.",
    "fr": "Bonjour, comment ça va ? Très bien, merci. Aujourd'hui, nous allons parler de quelque chose d'important.",
    "it": "Ciao, come stai? Molto bene, grazie. Oggi parliamo di qualcosa di importante.",
    "de": "Hallo, wie geht's? Sehr gut, danke. Heute sprechen wir über etwas Wichtiges.",
}

_model_lock = threading.Lock()
_model_cache: dict[str, object] = {}


def _register_cuda_dlls() -> None:
    """Hace visibles las DLL de los paquetes nvidia-* instalados en el .venv."""
    if os.name != "nt":
        return
    base = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    if not base.exists():
        return
    for bin_dir in base.glob("*/bin"):
        os.add_dll_directory(str(bin_dir))
        os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")


def _pick_compute_type(device: str, requested: str) -> str:
    if requested != "auto":
        return requested
    import ctranslate2
    supported = ctranslate2.get_supported_compute_types(device)
    for ct in (("int8_float16", "int8", "float16", "float32") if device == "cuda" else ("int8", "float32")):
        if ct in supported:
            return ct
    return "float32"


def _load(model_name: str, device: str, compute_type: str):
    from faster_whisper import WhisperModel
    key = f"{model_name}|{device}|{compute_type}"
    if key not in _model_cache:
        log.info("Cargando modelo Whisper %s en %s (%s)", model_name, device, compute_type)
        model = WhisperModel(model_name, device=device, compute_type=compute_type,
                             download_root=str(MODELS_DIR / "whisper"), cpu_threads=max(1, (os.cpu_count() or 4) - 2))
        # Las DLL de CUDA se cargan recién al ejecutar el encoder: probar acá evita fallar a mitad del video.
        model.detect_language(np.zeros(SAMPLE_RATE * 2, dtype=np.float32))
        _model_cache.clear()
        _model_cache[key] = model
    return _model_cache[key]


def get_model(cfg: dict) -> tuple[object, str]:
    """Devuelve (modelo, descripción) usando GPU si se puede y CPU si no."""
    tcfg = cfg["transcription"]
    with _model_lock:
        want = tcfg["device"]
        if want in ("auto", "cuda"):
            _register_cuda_dlls()
            try:
                import ctranslate2
                if ctranslate2.get_cuda_device_count() > 0:
                    ct = _pick_compute_type("cuda", tcfg["compute_type"])
                    return _load(tcfg["model"], "cuda", ct), f"{tcfg['model']} · GPU ({ct})"
            except Exception as exc:  # noqa: BLE001 - cualquier fallo de CUDA => CPU
                log.warning("No se pudo usar la GPU (%s). Se usará la CPU.", exc)
                _model_cache.clear()
        ct = _pick_compute_type("cpu", tcfg["compute_type"])
        name = tcfg.get("cpu_model") or tcfg["model"]
        return _load(name, "cpu", ct), f"{name} · CPU ({ct})"


def load_audio(video_path: Path) -> np.ndarray:
    """Audio mono 16 kHz float32, decodificado con FFmpeg."""
    cmd = [ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-i", str(video_path),
           "-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1"]
    proc = subprocess.run(cmd, capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if proc.returncode != 0:
        raise TranscriptionError("No se pudo leer el audio del video.",
                                 proc.stderr.decode("utf-8", "replace")[-1500:])
    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    if audio.size < SAMPLE_RATE:
        raise TranscriptionError("El video no tiene audio suficiente para transcribir.", f"samples={audio.size}")
    return audio


def compute_energy(audio: np.ndarray) -> np.ndarray:
    """Loudness (dB) en ventanas de 1/ENERGY_FPS s: sirve para detectar énfasis y silencios."""
    hop = SAMPLE_RATE // ENERGY_FPS
    n = audio.size // hop
    frames = audio[: n * hop].reshape(n, hop).astype(np.float32)
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-10)
    return (20 * np.log10(rms)).astype(np.float32)


def _detect_language(model, audio: np.ndarray) -> tuple[str, float]:
    lang, prob, _ = model.detect_language(audio, vad_filter=True, language_detection_segments=4,
                                          language_detection_threshold=0.6)
    return lang, float(prob)


def transcribe(asset: VideoAsset, cfg: dict, on_progress: Callable[[float, str], None] | None = None,
               warn: Callable[[str], None] | None = None) -> dict:
    out_path = asset.folder / "transcript.json"
    energy_path = asset.folder / "audio_energy.npy"
    if out_path.exists() and energy_path.exists():
        try:
            data = json.loads(out_path.read_text(encoding="utf-8"))
            if data.get("segments") and data.get("version") == TRANSCRIPT_VERSION:
                log.info("Reutilizando transcripción de %s", asset.video_id)
                return data
        except json.JSONDecodeError:
            pass

    tcfg = cfg["transcription"]
    progress = on_progress or (lambda p, m: None)
    progress(0.01, "Preparando el modelo de transcripción")
    try:
        model, model_desc = get_model(cfg)
    except Exception as exc:  # noqa: BLE001
        raise TranscriptionError("No se pudo cargar el modelo de transcripción.", str(exc)) from exc

    progress(0.03, "Leyendo el audio")
    audio = load_audio(asset.video_path)
    np.save(energy_path, compute_energy(audio))

    progress(0.05, "Verificando el idioma del audio")
    language, lang_prob = _detect_language(model, audio)
    expected = asset.metadata.get("original_language")
    log.info("Idioma detectado: %s (%.2f) | idioma original esperado: %s", language, lang_prob, expected)
    if cfg["audio"]["verify_language"] and expected and language != expected and lang_prob > 0.5:
        current = ((asset.metadata.get("audio_format") or {}).get("language") or "").split("-")[0]
        log.warning("El audio descargado (%s, pista %s) no coincide con el idioma original (%s): "
                    "la pista de audio elegida es incorrecta", language, current or "?", expected)
        progress(0.06, "Corrigiendo la pista de audio")
        if replace_audio_track(asset, cfg, expected):
            audio = load_audio(asset.video_path)
            np.save(energy_path, compute_energy(audio))
            language, lang_prob = _detect_language(model, audio)
            log.info("Tras corregir la pista: idioma detectado %s (%.2f)", language, lang_prob)
        if language != expected and warn:
            warn(f"El idioma del audio ({language}) no coincide con el idioma original indicado por YouTube "
                 f"({expected}). Puede que YouTube no ofrezca la pista original; se transcribe el audio tal cual.")

    duration = audio.size / SAMPLE_RATE
    progress(0.08, "Transcribiendo")
    try:
        segments_iter, info = model.transcribe(
            audio,
            language=language,  # el idioma detectado en el propio audio, no uno impuesto
            beam_size=int(tcfg["beam_size"]),
            word_timestamps=True,
            vad_filter=bool(tcfg["vad_filter"]),
            vad_parameters={"min_silence_duration_ms": 500},
            condition_on_previous_text=bool(tcfg["condition_on_previous_text"]),
            initial_prompt=STYLE_PROMPTS.get(language) if tcfg.get("punctuation_prompt", True) else None,
            hallucination_silence_threshold=2.0,
        )
        segments = []
        for seg in segments_iter:
            if seg.compression_ratio > 2.6 and seg.avg_logprob < -0.8:
                log.debug("Segmento descartado por repetición: %.1f-%.1f %r", seg.start, seg.end, seg.text)
                continue
            words = [{"start": round(w.start, 3), "end": round(w.end, 3), "word": w.word,
                      "p": round(w.probability, 3)} for w in (seg.words or [])]
            if not words:
                continue
            segments.append({"start": round(seg.start, 3), "end": round(seg.end, 3),
                             "text": seg.text.strip(), "words": words})
            progress(0.08 + 0.92 * min(1.0, seg.end / duration), "Transcribiendo")
    except Exception as exc:  # noqa: BLE001
        raise TranscriptionError("Falló la transcripción del audio.", str(exc)) from exc

    if not segments:
        raise TranscriptionError("No se detectó habla en el video, así que no hay momentos para analizar.",
                                 "transcripción vacía")
    data = {
        "version": TRANSCRIPT_VERSION,
        "language": language,
        "language_probability": round(lang_prob, 3),
        "expected_language": expected,
        "model": model_desc,
        "duration": round(duration, 3),
        "segments": segments,
    }
    out_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    log.info("Transcripción lista: %d segmentos, idioma %s, modelo %s", len(segments), language, model_desc)
    return data


def load_energy(asset: VideoAsset) -> np.ndarray:
    path = asset.folder / "audio_energy.npy"
    if path.exists():
        return np.load(path)
    energy = compute_energy(load_audio(asset.video_path))
    np.save(path, energy)
    return energy
