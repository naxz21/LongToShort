"""ConfiguraciÃ³n central: rutas del proyecto y lectura de config.json.

Todos los parÃ¡metros ajustables viven en config.json. Los valores de
DEFAULTS solo se usan si falta una clave, para que la app arranque igual.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOWNLOADS_DIR = ROOT / "downloads"
OUTPUT_DIR = ROOT / "output"
TEMP_DIR = ROOT / "temp"
LOGS_DIR = ROOT / "logs"
MODELS_DIR = ROOT / "models"
STATIC_DIR = ROOT / "static"
TEMPLATES_DIR = ROOT / "templates"
CONFIG_PATH = ROOT / "config.json"

DEFAULTS: dict = {
    "server": {"host": "127.0.0.1", "port": 8765},
    "video": {"preferred_height": 1080, "fallback_height": 720,
              "preferred_codecs": ["avc1", "vp9", "av01"]},
    "audio": {"prefer_original": True, "verify_language": True},
    "youtube": {"cookies_file": "cookies.txt", "cookies_from_browser": ""},
    "transcription": {"model": "large-v3-turbo", "cpu_model": "small", "device": "auto",
                      "compute_type": "auto", "beam_size": 5, "vad_filter": True,
                      "condition_on_previous_text": True, "punctuation_prompt": True},
    "analysis": {"embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 "sentence_max_pause": 0.9, "sentence_max_duration": 25},
    "clips": {"min_duration": 20, "max_duration": 60, "duration_flex": 15,
              "max_candidates": 15, "min_candidates": 5, "min_score": 50,
              "relative_cutoff": 0.82, "max_overlap": 0.2,
              "start_padding": 0.15, "end_padding": 0.4},
    "scoring": {"hook": 0.20, "impact": 0.20, "value": 0.15, "storytelling": 0.15,
                "autonomy": 0.15, "retention": 0.10, "boundaries": 0.05},
    "crop": {"mode": "smart", "sample_fps": 3, "face_min_confidence": 0.75,
             "shot_change_threshold": 0.55, "min_shot_duration": 0.8},
    "output": {"width": 1080, "height": 1920, "video_codec": "libx264", "preset": "medium",
               "crf": 20, "audio_bitrate": "192k", "audio_fade": 0.04},
}


def _merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config() -> dict:
    """Lee config.json en cada llamada para que los cambios apliquen sin reiniciar."""
    user_cfg = {}
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, encoding="utf-8") as fh:
            user_cfg = json.load(fh)
    return _merge(DEFAULTS, user_cfg)


def ensure_dirs() -> None:
    for d in (DOWNLOADS_DIR, OUTPUT_DIR, TEMP_DIR, LOGS_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
