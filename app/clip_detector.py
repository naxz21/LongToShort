"""Detección de los mejores momentos.

1. Genera TODAS las ventanas que empiezan y terminan en límites de frase y
   duran ~min-max segundos (con tolerancia si la narrativa lo justifica).
2. Puntúa cada ventana (clip_scorer).
3. Deduplica con supresión de no-máximos: si dos candidatos se superponen
   demasiado, queda el mejor.
4. Se queda con los mejores (pocos y buenos, no muchos mediocres).
"""
from __future__ import annotations

from .analyzer import Analysis
from .clip_scorer import DIMENSIONS, WindowScorer, reasons
from .logging_setup import get_logger

log = get_logger("clip_detector")


def _overlap_ratio(a: dict, b: dict) -> float:
    inter = min(a["t_end"], b["t_end"]) - max(a["t_start"], b["t_start"])
    if inter <= 0:
        return 0.0
    return inter / min(a["t_end"] - a["t_start"], b["t_end"] - b["t_start"])


def _clip_times(a: Analysis, i: int, j: int, cfg: dict) -> tuple[float, float]:
    """Tiempos de corte con un pequeño margen, sin invadir la frase vecina."""
    s = a.sentences
    pad_s, pad_e = float(cfg["clips"]["start_padding"]), float(cfg["clips"]["end_padding"])
    lower = s[i].lead_end if s[i].lead_end is not None else (s[i - 1].end if i > 0 else 0.0)
    start = max(lower + 0.03, s[i].start - pad_s, 0.0)
    upper = s[j + 1].raw_start - 0.05 if j + 1 < len(s) else a.duration
    end = min(s[j].end + pad_e, max(upper, s[j].end))
    return round(start, 2), round(end, 2)


def generate_windows(a: Analysis, cfg: dict):
    c = cfg["clips"]
    lo = float(c["min_duration"]) - float(c["duration_flex"]) * 0.5
    hi = float(c["max_duration"]) + float(c["duration_flex"])
    s = a.sentences
    for i in range(len(s)):
        for j in range(i + 1, len(s)):
            d = s[j].end - s[i].start
            if d > hi:
                break
            if d >= lo:
                yield i, j


def detect(a: Analysis, cfg: dict, on_progress=None) -> list[dict]:
    progress = on_progress or (lambda p, m: None)
    c = cfg["clips"]
    min_d, max_d = float(c["min_duration"]), float(c["max_duration"])
    scorer = WindowScorer(a, cfg)
    windows = list(generate_windows(a, cfg))
    log.info("Evaluando %d ventanas sobre %d frases", len(windows), len(a.sentences))

    # Si la transcripción tiene puntuación razonable, exigir que el clip empiece y
    # termine en frases completas. Si es muy pobre, se deja como penalización.
    sents = a.sentences
    clean_ratio = sum(s.starts_clean for s in sents) / len(sents)
    strict = clean_ratio >= 0.35
    log.info("Inicios de frase limpios: %.0f%% (%s)", clean_ratio * 100, "estricto" if strict else "flexible")

    scored = []
    for n, (i, j) in enumerate(windows):
        if strict and not (sents[i].starts_clean and sents[j].ends_clean):
            continue
        r = scorer.score(i, j)
        # Fuera del rango de duración solo si la estructura narrativa lo justifica.
        if not (min_d <= r["duration"] <= max_d) and (r["dims"]["storytelling"] < 0.75 or r["dims"]["hook"] < 0.55):
            continue
        r["t_start"], r["t_end"] = _clip_times(a, i, j, cfg)
        scored.append(r)
        if n % 2000 == 0:
            progress(n / max(1, len(windows)), "Evaluando momentos")
    scored.sort(key=lambda r: r["total"], reverse=True)

    # Deduplicación por superposición (NMS).
    max_overlap = float(c["max_overlap"])
    picked: list[dict] = []
    for r in scored:
        if all(_overlap_ratio(r, p) <= max_overlap for p in picked):
            picked.append(r)
        if len(picked) >= int(c["max_candidates"]):
            break

    # Pocos y buenos: cortar los que quedan muy por debajo del mejor.
    if picked:
        threshold = max(float(c["min_score"]), picked[0]["total"] * float(c["relative_cutoff"]))
        keep = [p for p in picked if p["total"] >= threshold]
        if len(keep) < int(c["min_candidates"]):
            keep = picked[: int(c["min_candidates"])]
        picked = keep

    candidates = []
    for rank, r in enumerate(picked, start=1):
        sents = a.sentences[r["i"]: r["j"] + 1]
        candidates.append({
            "id": rank,
            "start": float(r["t_start"]),
            "end": float(r["t_end"]),
            "duration": round(float(r["t_end"] - r["t_start"]), 2),
            "score": int(round(float(r["total"]))),
            "scores": {d: round(float(r["dims"][d]) * 10, 1) for d in DIMENSIONS},
            "hook": sents[0].text,
            "text": " ".join(s.text for s in sents),
            "reasons": reasons(r),
            "sentences": [int(r["i"]), int(r["j"])],
        })
    log.info("Candidatos finales: %s", [(c_["id"], c_["score"], c_["start"]) for c_ in candidates])
    return candidates
