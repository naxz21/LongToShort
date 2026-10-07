"""Descarga de videos de YouTube con yt-dlp.

Puntos clave:
- El video y el audio se eligen explícitamente (no con "bestvideo+bestaudio")
  para controlar la resolución y, sobre todo, la pista de audio ORIGINAL.
  YouTube puede ofrecer doblajes (a veces automáticos) y la primera pista no
  siempre es la original.
- Todo se guarda en downloads/VIDEO_ID/ y se reutiliza si ya es válido.
"""
from __future__ import annotations

import http.cookiejar
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import yt_dlp

from .config import DOWNLOADS_DIR, ROOT, TEMP_DIR
from .errors import DownloadError, InvalidURLError
from .logging_setup import get_logger
from .media import ffmpeg_path, probe, run_ffmpeg

log = get_logger("downloader")

_ID_RE = re.compile(r"(?:v=|youtu\.be/|/shorts/|/live/|/embed/|/v/)([A-Za-z0-9_-]{11})(?![A-Za-z0-9_-])")
_HOST_RE = re.compile(r"^(?:https?://)?(?:www\.|m\.|music\.)?(?:youtube\.com|youtu\.be|youtube-nocookie\.com)/", re.I)

ORIGINAL_LANG_PREF = 10   # valores que asigna yt-dlp a language_preference
DEFAULT_LANG_PREF = 5


@dataclass
class VideoAsset:
    video_id: str
    folder: Path
    video_path: Path
    metadata: dict


def parse_video_id(url: str) -> str:
    url = (url or "").strip()
    if not _HOST_RE.match(url):
        raise InvalidURLError("La URL no parece ser de YouTube. Pegá un enlace como https://www.youtube.com/watch?v=...")
    match = _ID_RE.search(url)
    if not match:
        raise InvalidURLError("No se encontró el ID del video en la URL. Revisá que el enlace esté completo.")
    return match.group(1)


def _base_lang(code: str | None) -> str | None:
    return code.split("-")[0].lower() if code else None


def _ydl_opts(**extra) -> dict:
    opts = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "noprogress": True,
        # YouTube necesita un runtime JS para resolver los formatos; se usa el que esté instalado.
        "js_runtimes": {"deno": {}, "node": {}},
        "ffmpeg_location": str(Path(ffmpeg_path()).parent),
        "logger": _YtdlpLogger(),
    }
    opts.update(extra)
    return opts


class _YtdlpLogger:
    def debug(self, msg):
        if not msg.startswith("[debug] "):
            log.debug("yt-dlp: %s", msg)

    def info(self, msg):
        log.debug("yt-dlp: %s", msg)

    def warning(self, msg):
        log.warning("yt-dlp: %s", msg)

    def error(self, msg):
        log.error("yt-dlp: %s", msg)


_LOGIN_NEEDLES = ("confirm your age", "age-restricted", "inappropriate for some users", "not a bot")


def _needs_login(exc: Exception) -> bool:
    low = str(exc).lower()
    return any(n in low for n in _LOGIN_NEEDLES)


def _auth_opts(cfg: dict) -> dict:
    """Cookies de una sesión de YouTube, si están configuradas. Solo se usan
    cuando YouTube exige iniciar sesión (restricción de edad, chequeo anti-bot)."""
    yt = cfg.get("youtube") or {}
    if yt.get("cookies_file"):
        path = Path(yt["cookies_file"])
        if not path.is_absolute():
            path = ROOT / path
        if path.is_file():
            return {"cookiefile": str(path)}
    if yt.get("cookies_from_browser"):
        return {"cookiesfrombrowser": (yt["cookies_from_browser"],)}
    return {}


def _friendly_download_error(exc: Exception, auth: dict | None = None) -> DownloadError:
    text = str(exc)
    low = text.lower()
    if auth:
        age_msg = ("El video tiene restricción de edad y YouTube rechazó las cookies configuradas: "
                   "probablemente vencieron o la cuenta no tiene la edad verificada. Exportá cookies.txt de nuevo.")
    else:
        age_msg = ("El video tiene restricción de edad. Para descargarlo, exportá las cookies de tu sesión de "
                   "YouTube a cookies.txt en la carpeta del proyecto (ver README) y volvé a intentar.")
    rules = [
        (("private video",), "El video es privado. Solo se pueden analizar videos públicos o no listados."),
        (("members-only", "join this channel"), "El video es exclusivo para miembros del canal."),
        (("confirm your age", "age-restricted", "inappropriate for some users"), age_msg),
        (("not a bot",), "YouTube bloqueó temporalmente la descarga. Esperá unos minutos y volvé a intentar."),
        (("netscape",), "El archivo cookies.txt no es válido. Exportalo de nuevo en formato Netscape."),
        (("removed", "no longer available", "video unavailable", "terminated", "does not exist"),
         "El video no está disponible: puede haber sido eliminado o bloqueado en tu país."),
        (("getaddrinfo", "timed out", "connection", "network is unreachable"),
         "No se pudo conectar con YouTube. Revisá tu conexión a internet."),
        (("no space left", "errno 28"), "No hay espacio suficiente en el disco para descargar el video."),
    ]
    for needles, message in rules:
        if any(n in low for n in needles):
            return DownloadError(message, text)
    return DownloadError("No se pudo descargar el video. Revisá que la URL sea válida y que el video sea público.", text)


def _extract(url: str, auth: dict) -> dict:
    try:
        with yt_dlp.YoutubeDL(_ydl_opts(skip_download=True, **auth)) as ydl:
            return ydl.extract_info(url, download=False)
    except (yt_dlp.utils.DownloadError, http.cookiejar.LoadError) as exc:
        raise _friendly_download_error(exc, auth) from exc


def fetch_info(url: str, cfg: dict) -> tuple[dict, dict]:
    """Devuelve la info del video y las opciones de sesión con las que se obtuvo
    (hay que reusarlas al descargar: las URLs de los formatos dependen de la sesión)."""
    auth: dict = {}
    try:
        info = _extract(url, auth)
    except DownloadError as exc:
        auth = _auth_opts(cfg)
        if not auth or not _needs_login(exc.detail):
            raise
        log.info("YouTube pide iniciar sesión; reintentando con cookies (%s)", next(iter(auth)))
        # Con sesión, los clientes por defecto de YouTube suelen exigir PO tokens para los
        # formatos HD de videos con restricción de edad; web_safari ofrece HLS hasta 1080p sin ellos.
        auth["extractor_args"] = {"youtube": {"player_client": ["default", "web_safari"]}}
        info = _extract(url, auth)
    if not info:
        raise DownloadError("No se pudo leer la información del video.", "extract_info devolvió None")
    if info.get("is_live") or info.get("live_status") in ("is_live", "is_upcoming"):
        raise DownloadError("El video es una transmisión en vivo o programada. Probá cuando haya terminado.",
                            f"live_status={info.get('live_status')}")
    return info, auth


# --------------------------------------------------------------------------
# Selección de la pista de audio original
# --------------------------------------------------------------------------

def _audio_formats(info: dict) -> list[dict]:
    return [f for f in info.get("formats", [])
            if f.get("acodec") not in (None, "none") and f.get("vcodec") in (None, "none")
            and f.get("protocol", "https").startswith("http")]


def _is_drc(f: dict) -> bool:
    return "drc" in str(f.get("format_id", "")).lower() or "drc" in str(f.get("format_note", "")).lower()


def detect_original_language(info: dict) -> tuple[str | None, str]:
    """Idioma del audio original y de dónde se obtuvo (para los logs/UI)."""
    afmts = _audio_formats(info)
    for f in afmts:
        if f.get("language_preference") == ORIGINAL_LANG_PREF and f.get("language"):
            return _base_lang(f["language"]), "pista marcada como original por YouTube"
    # Los subtítulos automáticos "xx-orig" se generan sobre el audio original.
    for key in (info.get("automatic_captions") or {}):
        if key.endswith("-orig"):
            return _base_lang(key[:-5]), "subtítulos automáticos del audio original"
    if info.get("language"):
        return _base_lang(info["language"]), "idioma declarado del video"
    for f in afmts:
        if f.get("language_preference") == DEFAULT_LANG_PREF and f.get("language"):
            return _base_lang(f["language"]), "pista por defecto"
    return None, "sin información de idioma"


def list_audio_tracks(info: dict) -> list[dict]:
    tracks: dict[str, dict] = {}
    for f in _audio_formats(info):
        lang = _base_lang(f.get("language")) or "und"
        entry = tracks.setdefault(lang, {"language": lang, "note": f.get("format_note") or "",
                                         "preference": f.get("language_preference")})
        if (f.get("language_preference") or -100) > (entry["preference"] or -100):
            entry.update(note=f.get("format_note") or "", preference=f.get("language_preference"))
    return sorted(tracks.values(), key=lambda t: -(t["preference"] or -100))


def select_audio_format(info: dict, language: str | None) -> dict:
    afmts = _audio_formats(info)
    if not afmts:
        raise DownloadError("El video no tiene una pista de audio descargable.", "sin formatos de audio")
    pool = [f for f in afmts if language and _base_lang(f.get("language")) == language]
    if not pool:
        # Sin coincidencia de idioma: la pista con mayor preferencia de YouTube (original > default).
        best_pref = max((f.get("language_preference") or -100) for f in afmts)
        pool = [f for f in afmts if (f.get("language_preference") or -100) == best_pref]
    pool = [f for f in pool if "desc" not in str(f.get("language") or "")] or pool
    pool = [f for f in pool if not _is_drc(f)] or pool
    # m4a/AAC primero: se combina sin problemas en MP4 y se reproduce en cualquier navegador.
    pool.sort(key=lambda f: (f.get("ext") == "m4a", f.get("abr") or f.get("tbr") or 0), reverse=True)
    return pool[0]


# --------------------------------------------------------------------------
# Selección del video
# --------------------------------------------------------------------------

def select_video_format(info: dict, cfg: dict) -> dict:
    preferred = int(cfg["video"]["preferred_height"])
    codecs = [c.lower() for c in cfg["video"].get("preferred_codecs", [])]
    vfmts = [f for f in info.get("formats", [])
             if f.get("vcodec") not in (None, "none") and f.get("acodec") in (None, "none")
             and f.get("height") and f.get("protocol", "https").startswith("http")]
    if not vfmts:
        raise DownloadError("No se encontró un formato de video descargable.", "sin formatos de video")

    def codec_rank(f):
        vc = str(f.get("vcodec", "")).lower()
        for i, c in enumerate(codecs):
            if vc.startswith(c):
                return len(codecs) - i
        return 0

    within = [f for f in vfmts if f["height"] <= preferred] or vfmts
    best_height = max(f["height"] for f in within)
    pool = [f for f in within if f["height"] == best_height]
    pool.sort(key=lambda f: (codec_rank(f), min(f.get("fps") or 30, 60), f.get("tbr") or 0), reverse=True)
    return pool[0]


def select_muxed_format(info: dict, cfg: dict) -> dict:
    """Formato con video y audio juntos (p. ej. HLS de web_safari). Se usa cuando
    YouTube no entrega streams separados, típico con videos con restricción de edad."""
    preferred = int(cfg["video"]["preferred_height"])
    codecs = [c.lower() for c in cfg["video"].get("preferred_codecs", [])]
    muxed = [f for f in info.get("formats", [])
             if f.get("vcodec") not in (None, "none") and f.get("acodec") not in (None, "none")
             and f.get("height") and str(f.get("protocol", "https")).split("+")[0] in ("http", "https", "m3u8", "m3u8_native")]
    if not muxed:
        raise DownloadError("No se encontró un formato de video descargable.", "sin formatos de video ni combinados")

    def codec_rank(f):
        vc = str(f.get("vcodec", "")).lower()
        return next((len(codecs) - i for i, c in enumerate(codecs) if vc.startswith(c)), 0)

    within = [f for f in muxed if f["height"] <= preferred] or muxed
    best_height = max(f["height"] for f in within)
    pool = [f for f in within if f["height"] == best_height]
    pool.sort(key=lambda f: (codec_rank(f), f.get("tbr") or 0), reverse=True)
    return pool[0]


def _size(f: dict, duration: float) -> float:
    size = f.get("filesize") or f.get("filesize_approx")
    if not size and f.get("tbr") and duration:
        size = f["tbr"] * 1000 / 8 * duration
    return float(size or 0)


# --------------------------------------------------------------------------
# Descarga
# --------------------------------------------------------------------------

def _cached_asset(video_id: str) -> VideoAsset | None:
    folder = DOWNLOADS_DIR / video_id
    video_path = folder / "original.mp4"
    meta_path = folder / "metadata.json"
    if not (video_path.exists() and meta_path.exists()):
        return None
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    info = probe(video_path)
    expected = float(metadata.get("duration") or 0)
    if not info.get("has_video") or not info.get("audio_streams"):
        return None
    if expected and abs(info["duration"] - expected) > max(3.0, expected * 0.02):
        log.warning("Archivo en caché incompleto para %s (%.1fs vs %.1fs)", video_id, info["duration"], expected)
        return None
    return VideoAsset(video_id, folder, video_path, metadata)


def build_metadata(info: dict, vfmt: dict, afmt: dict, orig_lang: str | None, lang_source: str) -> dict:
    return {
        "id": info["id"],
        "title": info.get("title"),
        "channel": info.get("channel") or info.get("uploader"),
        "duration": info.get("duration"),
        "thumbnail": info.get("thumbnail"),
        "upload_date": info.get("upload_date"),
        "webpage_url": info.get("webpage_url"),
        "original_language": orig_lang,
        "original_language_source": lang_source,
        "audio_tracks": list_audio_tracks(info),
        "video_format": {k: vfmt.get(k) for k in ("format_id", "height", "width", "vcodec", "fps")},
        "audio_format": {k: afmt.get(k) for k in ("format_id", "language", "format_note", "acodec", "abr",
                                                   "language_preference")},
    }


def download(url: str, cfg: dict, on_info: Callable[[dict], None] | None = None,
             on_progress: Callable[[float, str], None] | None = None) -> VideoAsset:
    video_id = parse_video_id(url)
    cached = _cached_asset(video_id)
    if cached:
        log.info("Reutilizando descarga existente de %s", video_id)
        if on_info:
            on_info(cached.metadata)
        return cached

    info, auth = fetch_info(f"https://www.youtube.com/watch?v={video_id}", cfg)
    duration = float(info.get("duration") or 0)
    orig_lang, lang_source = detect_original_language(info)
    try:
        vfmt = select_video_format(info, cfg)
        afmt = select_audio_format(info, orig_lang if cfg["audio"]["prefer_original"] else None)
        format_spec = f"{vfmt['format_id']}+{afmt['format_id']}"
    except DownloadError:
        vfmt = afmt = select_muxed_format(info, cfg)
        format_spec = vfmt["format_id"]
        log.info("Sin streams separados de video/audio: se usa el formato combinado %s", format_spec)
    metadata = build_metadata(info, vfmt, afmt, orig_lang, lang_source)
    log.info("Video %s: %sp %s | audio %s (%s) | idioma original=%s (%s) | pistas=%s",
             video_id, vfmt.get("height"), vfmt.get("vcodec"), afmt.get("format_id"),
             afmt.get("language"), orig_lang, lang_source,
             [t["language"] for t in metadata["audio_tracks"]])
    if on_info:
        on_info(metadata)

    if int(vfmt["height"]) < int(cfg["video"]["fallback_height"]):
        log.warning("La mejor calidad disponible es %sp (menor al fallback configurado)", vfmt["height"])

    folder = DOWNLOADS_DIR / video_id
    folder.mkdir(parents=True, exist_ok=True)
    sizes = {f["format_id"]: _size(f, duration) for f in (vfmt, afmt)}
    needed = sum(sizes.values()) * 2.5
    free = shutil.disk_usage(folder).free
    if needed and free < needed:
        raise DownloadError("No hay espacio suficiente en el disco para descargar el video.",
                            f"libre={free} necesario~{needed}")

    done: dict[str, float] = {}

    def hook(d):
        fid = (d.get("info_dict") or {}).get("format_id")
        if d.get("status") == "downloading" and fid:
            done[fid] = float(d.get("downloaded_bytes") or 0)
            if not sizes.get(fid):
                sizes[fid] = float(d.get("total_bytes") or d.get("total_bytes_estimate") or 0)
        elif d.get("status") == "finished" and fid:
            done[fid] = sizes.get(fid) or float(d.get("downloaded_bytes") or 0)
        total = sum(sizes.values())
        if on_progress and total:
            on_progress(min(0.99, sum(done.values()) / total), "Descargando video")

    def pp_hook(d):
        if on_progress and d.get("status") == "started" and d.get("postprocessor") == "Merger":
            on_progress(0.99, "Uniendo video y audio")

    opts = _ydl_opts(
        **auth,
        format=format_spec,
        outtmpl={"default": str(folder / "original.%(ext)s")},
        merge_output_format="mp4",
        progress_hooks=[hook],
        postprocessor_hooks=[pp_hook],
        overwrites=True,
        continuedl=True,
    )
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.process_ie_result(info, download=True)
    except yt_dlp.utils.DownloadError as exc:
        raise _friendly_download_error(exc, auth) from exc
    except OSError as exc:
        if getattr(exc, "errno", None) == 28:
            raise DownloadError("No hay espacio suficiente en el disco para descargar el video.", str(exc)) from exc
        raise

    video_path = folder / "original.mp4"
    check = probe(video_path) if video_path.exists() else {}
    if not check.get("has_video"):
        raise DownloadError("La descarga terminó pero el archivo de video no es válido.", f"probe={check}")
    if not check.get("audio_streams"):
        raise DownloadError("El video descargado no tiene audio, así que no se puede transcribir.", f"probe={check}")
    metadata["file"] = {"width": check.get("width"), "height": check.get("height"),
                        "duration": check.get("duration"), "vcodec": check.get("vcodec")}
    (folder / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return VideoAsset(video_id, folder, video_path, metadata)


def replace_audio_track(asset: VideoAsset, cfg: dict, language: str) -> bool:
    """Corrige una pista de audio equivocada: descarga la pista `language` y la
    vuelve a combinar con el video. Devuelve False si esa pista no existe."""
    info, auth = fetch_info(f"https://www.youtube.com/watch?v={asset.video_id}", cfg)
    try:
        afmt = select_audio_format(info, language)
    except DownloadError:
        return False  # solo hay formatos combinados: no hay otra pista de audio para elegir
    if _base_lang(afmt.get("language")) != language:
        return False
    if afmt.get("format_id") == (asset.metadata.get("audio_format") or {}).get("format_id"):
        return False
    tmp_audio = TEMP_DIR / f"{asset.video_id}_audio"
    opts = _ydl_opts(**auth, format=afmt["format_id"], outtmpl={"default": str(tmp_audio) + ".%(ext)s"},
                     overwrites=True)
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.process_ie_result(info, download=True)
    except yt_dlp.utils.DownloadError as exc:
        raise _friendly_download_error(exc, auth) from exc
    audio_file = next(TEMP_DIR.glob(f"{asset.video_id}_audio.*"))
    fixed = asset.folder / "original.fixed.mp4"
    run_ffmpeg(["-i", str(asset.video_path), "-i", str(audio_file), "-map", "0:v:0", "-map", "1:a:0",
                "-c", "copy", "-movflags", "+faststart", str(fixed)])
    audio_file.unlink(missing_ok=True)
    fixed.replace(asset.video_path)
    asset.metadata["audio_format"] = {k: afmt.get(k) for k in ("format_id", "language", "format_note", "acodec",
                                                                 "abr", "language_preference")}
    (asset.folder / "metadata.json").write_text(json.dumps(asset.metadata, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
    return True
