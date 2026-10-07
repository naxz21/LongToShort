"""Conversión a 9:16 por recorte (nunca se estira la imagen).

Estrategia actual ("smart"):
  1. Muestrea frames del clip (crop.sample_fps por segundo).
  2. Detecta cambios de plano (histograma de color) → divide el clip en planos.
  3. En cada plano detecta rostros (YuNet de OpenCV) y elige el sujeto
     principal (el más grande y persistente).
  4. Ventana vertical centrada en ese sujeto; sin detección confiable → centro.

El resultado es un CropPlan (lista de tramos con su posición de recorte) que
se traduce a un filtro de FFmpeg. Para agregar en el futuro face tracking,
seguimiento de personas u objetos, o un crop dinámico/suavizado, basta con
otra función que produzca un CropPlan con más tramos (o posiciones
interpoladas); el resto del pipeline no cambia.
"""
from __future__ import annotations

import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import MODELS_DIR
from .logging_setup import get_logger

log = get_logger("vertical_converter")

YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
             "face_detection_yunet_2023mar.onnx")
YUNET_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
DETECT_WIDTH = 640


@dataclass
class CropSegment:
    t0: float      # segundos relativos al inicio del clip
    t1: float
    x: int         # esquina superior izquierda del recorte en el video fuente
    y: int
    source: str    # "face" | "center"


@dataclass
class CropPlan:
    src_w: int
    src_h: int
    crop_w: int
    crop_h: int
    segments: list[CropSegment] = field(default_factory=list)
    method: str = "center"

    def to_dict(self) -> dict:
        return {"method": self.method, "crop": [self.crop_w, self.crop_h],
                "segments": [{"t0": round(s.t0, 2), "t1": round(s.t1, 2), "x": s.x, "y": s.y, "source": s.source}
                             for s in self.segments]}


def _even(v: float) -> int:
    return int(v) // 2 * 2


def crop_size(src_w: int, src_h: int, out_w: int, out_h: int) -> tuple[int, int]:
    """Mayor rectángulo con la relación de aspecto de salida que entra en el video."""
    target = out_w / out_h
    if src_w / src_h > target:
        return _even(src_h * target), _even(src_h)
    return _even(src_w), _even(src_w / target)


# --------------------------------------------------------------------------
# Detección de rostros
# --------------------------------------------------------------------------

class _FaceDetector:
    def __init__(self, min_conf: float):
        import cv2
        self.cv2 = cv2
        self.min_conf = min_conf
        self.yunet = None
        self.haar = None
        try:
            if not YUNET_PATH.exists():
                YUNET_PATH.parent.mkdir(parents=True, exist_ok=True)
                log.info("Descargando detector de rostros YuNet")
                urllib.request.urlretrieve(YUNET_URL, YUNET_PATH)
            self.yunet = cv2.FaceDetectorYN.create(str(YUNET_PATH), "", (320, 320), min_conf, 0.3, 50)
        except Exception as exc:  # noqa: BLE001 - sin YuNet se intenta Haar
            log.warning("YuNet no disponible (%s); se usa Haar cascade", exc)
            data = getattr(cv2, "data", None)
            cascade = Path(data.haarcascades) / "haarcascade_frontalface_default.xml" if data else None
            if cascade and cascade.exists():
                self.haar = cv2.CascadeClassifier(str(cascade))

    @property
    def available(self) -> bool:
        return self.yunet is not None or self.haar is not None

    def detect(self, frame: np.ndarray) -> list[tuple[float, float, float, float]]:
        """Devuelve [(cx, cy, ancho, score)] normalizados a 0-1."""
        h, w = frame.shape[:2]
        if self.yunet is not None:
            self.yunet.setInputSize((w, h))
            _, faces = self.yunet.detect(frame)
            if faces is None:
                return []
            return [((f[0] + f[2] / 2) / w, (f[1] + f[3] / 2) / h, f[2] / w, float(f[-1]))
                    for f in faces if f[-1] >= self.min_conf]
        if self.haar is not None:
            gray = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2GRAY)
            boxes = self.haar.detectMultiScale(gray, 1.1, 6, minSize=(w // 25, w // 25))
            return [((x + bw / 2) / w, (y + bh / 2) / h, bw / w, 0.8) for x, y, bw, bh in boxes]
        return []


_detector: _FaceDetector | None = None


def _get_detector(cfg: dict) -> _FaceDetector:
    global _detector
    if _detector is None:
        _detector = _FaceDetector(float(cfg["crop"]["face_min_confidence"]))
    return _detector


# --------------------------------------------------------------------------
# Plan de recorte
# --------------------------------------------------------------------------

def _sample(video_path: Path, start: float, end: float, cfg: dict, detector: _FaceDetector):
    """Recorre el clip. Los cortes de plano se detectan en CADA frame (histograma
    de color, barato) para que el cambio de encuadre caiga en el frame exacto;
    los rostros se buscan solo sample_fps veces por segundo.
    Devuelve (tiempos de corte, [(t, rostros)]) relativos al inicio del clip."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(fps / float(cfg["crop"]["sample_fps"])))
    threshold = float(cfg["crop"]["shot_change_threshold"])
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    cuts, samples, frame_no, prev_hist = [], [], 0, None
    try:
        while cap.grab():
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if t > end:
                break
            if t < start - 0.02:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                break
            tiny = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
            hist = cv2.calcHist([cv2.cvtColor(tiny, cv2.COLOR_BGR2HSV)], [0, 1], None, [32, 32], [0, 180, 0, 256])
            cv2.normalize(hist, hist)
            is_cut = prev_hist is not None and cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL) < threshold
            if is_cut:
                cuts.append(t - start)
            prev_hist = hist
            if frame_no % step == 0 or is_cut:   # tras un corte, mirar el plano nuevo enseguida
                scale = DETECT_WIDTH / frame.shape[1]
                small = cv2.resize(frame, (DETECT_WIDTH, int(frame.shape[0] * scale)))
                samples.append((t - start, [fc for fc in detector.detect(small) if fc[2] >= 0.03]))
            frame_no += 1
    finally:
        cap.release()
    return cuts, samples


def _shots(cuts: list[float], samples: list, duration: float) -> list[tuple[float, float, list]]:
    bounds = [0.0, *cuts, duration]
    shots = []
    for t0, t1 in zip(bounds[:-1], bounds[1:]):
        shots.append((t0, t1, [faces for t, faces in samples if t0 - 1e-3 <= t < t1 - 1e-3]))
    return shots


def _subject_x(face_samples: list[list]) -> float | None:
    """Centro horizontal del sujeto principal de un plano, o None si no es confiable."""
    if not face_samples:
        return None
    with_faces = [fs for fs in face_samples if fs]
    if len(with_faces) / len(face_samples) < 0.3:
        return None
    faces = [fc for fs in with_faces for fc in fs]
    # Agrupar por posición horizontal: el sujeto principal es el grupo con más
    # "peso" (tamaño x confianza x persistencia), no simplemente el primero.
    clusters: list[list] = []
    for fc in sorted(faces, key=lambda f: f[0]):
        if clusters and abs(fc[0] - np.mean([c[0] for c in clusters[-1]])) < 0.08:
            clusters[-1].append(fc)
        else:
            clusters.append([fc])
    best = max(clusters, key=lambda c: sum(f[2] * f[3] for f in c))
    xs = np.array([f[0] for f in best])
    ws = np.array([f[2] * f[3] for f in best])
    order = np.argsort(xs)
    cum = np.cumsum(ws[order])
    return float(xs[order][np.searchsorted(cum, cum[-1] / 2)])   # mediana ponderada


def plan_crop(video_path: Path, start: float, end: float, src_w: int, src_h: int, cfg: dict) -> CropPlan:
    out_w, out_h = int(cfg["output"]["width"]), int(cfg["output"]["height"])
    cw, ch = crop_size(src_w, src_h, out_w, out_h)
    duration = end - start
    center = CropSegment(0.0, duration, _even((src_w - cw) / 2), _even((src_h - ch) / 2), "center")
    plan = CropPlan(src_w, src_h, cw, ch, [center], "center")
    if cfg["crop"]["mode"] != "smart" or cw >= src_w:
        return plan

    detector = _get_detector(cfg)
    if not detector.available:
        return plan
    cuts, samples = _sample(video_path, start, end, cfg, detector)
    if not samples:
        log.warning("No se pudieron leer frames para el crop inteligente; se usa crop centrado")
        return plan

    segments: list[CropSegment] = []
    for t0, t1, face_samples in _shots(cuts, samples, duration):
        cx = _subject_x(face_samples)
        if cx is None:
            segments.append(CropSegment(t0, t1, center.x, center.y, "center"))
        else:
            x = _even(min(max(cx * src_w - cw / 2, 0), src_w - cw))
            segments.append(CropSegment(t0, t1, x, center.y, "face"))

    # Planos muy cortos (falsos cortes, flashes) se absorben en el anterior y
    # tramos contiguos con casi la misma posición se unen: evita saltos.
    min_len = float(cfg["crop"]["min_shot_duration"])
    merged: list[CropSegment] = []
    for seg in segments:
        if merged and (seg.t1 - seg.t0 < min_len or abs(seg.x - merged[-1].x) < src_w * 0.04):
            merged[-1].t1 = seg.t1
        else:
            merged.append(seg)
    if len(merged) > 1 and merged[0].t1 - merged[0].t0 < min_len:
        merged[1].t0 = 0.0
        merged.pop(0)

    plan.segments = merged
    plan.method = "smart" if any(s.source == "face" for s in merged) else "center"
    return plan


def crop_filter(plan: CropPlan, out_w: int, out_h: int) -> str:
    """Filtro de FFmpeg: crop (posición por tramo) + escalado a la resolución final."""
    if len(plan.segments) == 1:
        x_expr = str(plan.segments[0].x)
    else:
        # if(lt(t,T1),X1,if(lt(t,T2),X2,...)) — crop evalúa x en cada frame.
        x_expr = str(plan.segments[-1].x)
        for seg in reversed(plan.segments[:-1]):
            x_expr = f"if(lt(t\\,{seg.t1 - 0.004:.3f})\\,{seg.x}\\,{x_expr})"
    y = plan.segments[0].y
    return (f"crop={plan.crop_w}:{plan.crop_h}:{x_expr}:{y},"
            f"scale={out_w}:{out_h}:flags=lanczos,setsar=1,format=yuv420p")
