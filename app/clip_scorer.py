"""Puntuación de una ventana de frases como candidato a Short.

Cada dimensión se calcula en 0-1 combinando varias señales (semánticas,
estructurales, prosódicas y léxicas) y se muestra en 0-10. El puntaje total
0-100 usa los pesos de config.json → "scoring" (único lugar donde se definen).
"""
from __future__ import annotations

import numpy as np

from .analyzer import Analysis

DIMENSIONS = ("hook", "impact", "value", "storytelling", "autonomy", "retention", "boundaries")


def _ramp(x: float, lo: float, hi: float) -> float:
    return float(min(1.0, max(0.0, (x - lo) / (hi - lo))))


def _tri(x: float, lo: float, best_lo: float, best_hi: float, hi: float) -> float:
    """1 dentro de [best_lo, best_hi], baja linealmente a 0 en lo / hi."""
    if best_lo <= x <= best_hi:
        return 1.0
    if x < best_lo:
        return _ramp(x, lo, best_lo)
    return 1.0 - _ramp(x, best_hi, hi)


class WindowScorer:
    def __init__(self, analysis: Analysis, cfg: dict):
        self.a = analysis
        self.cfg = cfg
        weights = cfg["scoring"]
        self.weights = {d: float(weights.get(d, 0)) for d in DIMENSIONS}
        self.weight_sum = sum(self.weights.values()) or 1.0
        c = cfg["clips"]
        self.min_d, self.max_d, self.flex = float(c["min_duration"]), float(c["max_duration"]), float(c["duration_flex"])

        f = analysis.feats
        self.f = f
        s = analysis.sentences
        self.start = np.array([x.start for x in s])
        self.end = np.array([x.end for x in s])
        self.pause_before = np.array([x.pause_before for x in s])
        self.pause_after = np.array([x.pause_after for x in s])
        self.n = len(s)
        # Sumas acumuladas para promedios O(1) por ventana.
        self._cum = {k: np.concatenate([[0.0], np.cumsum(f[k])]) for k in (
            "n_words", "fillers", "value_lex", "numbers", "impact_lex", "laugh", "opinion", "story_lex",
            "value_sem", "story_sem", "impact_sem", "backref", "low_conf", "question", "payoff_lex")}
        long_pause = np.clip(self.pause_before - 0.6, 0, None)
        self._cum["long_pause"] = np.concatenate([[0.0], np.cumsum(long_pause)])
        self._cum_emb = np.vstack([np.zeros((1, analysis.emb.shape[1])), np.cumsum(analysis.emb, axis=0)])

    # -- helpers ----------------------------------------------------------
    def _sum(self, key: str, i: int, j: int) -> float:
        return float(self._cum[key][j + 1] - self._cum[key][i])

    def _mean(self, key: str, i: int, j: int) -> float:
        return self._sum(key, i, j) / (j - i + 1)

    def _vec(self, i: int, j: int) -> np.ndarray:
        return self._cum_emb[j + 1] - self._cum_emb[i]

    def _cos(self, u: np.ndarray, v: np.ndarray) -> float:
        return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))

    def duration_fit(self, d: float) -> float:
        """1 en la zona ideal (p. ej. 25-52 s para 20-60), 0.8 en los extremos del rango
        y bajando a 0 fuera de él (hasta duration_flex segundos de tolerancia)."""
        lo, hi = self.min_d, self.max_d
        ideal_lo, ideal_hi = lo + 0.125 * (hi - lo), hi - 0.2 * (hi - lo)
        if d < lo:
            return 0.8 * _ramp(d, lo - self.flex * 0.5, lo)
        if d > hi:
            return 0.8 * (1 - _ramp(d, hi, hi + self.flex))
        if d < ideal_lo:
            return 0.8 + 0.2 * _ramp(d, lo, ideal_lo)
        if d > ideal_hi:
            return 1.0 - 0.2 * _ramp(d, ideal_hi, hi)
        return 1.0

    # -- scoring ----------------------------------------------------------
    def score(self, i: int, j: int) -> dict:
        f, k = self.f, j - i + 1
        dur = float(self.end[j] - self.start[i])

        # Frases que forman la apertura (primeros ~8 s) y el cierre.
        open_j = i
        while open_j + 1 <= j and self.start[open_j + 1] - self.start[i] < 8.0 and open_j - i < 2:
            open_j += 1
        close_i = max(i, j - 1)

        # ---------------- HOOK: ¿la primera frase atrapa? ----------------
        h_sem = max(f["hook_sem"][i], *(0.8 * f["hook_sem"][x] for x in range(i + 1, open_j + 1))) if open_j > i \
            else f["hook_sem"][i]
        h_lex = max(f["hook_lex"][i], 0.7 * f["hook_lex"][open_j])
        h_q = 1.0 if f["question"][i] else (0.7 if f["question"][i:open_j + 1].any() else 0.0)
        h_strong = max(f["opinion"][i], f["impact_sem"][i] * 0.9, f["impact_lex"][i])
        nw = f["n_words"][i]
        h_punchy = _tri(nw, 2, 5, 18, 40)
        h_energy = _ramp(f["energy_z"][i], -0.5, 1.2)
        # Arranque "in medias res": un hecho personal vívido también atrapa.
        h_story = max(f["story_sem"][i], f["impact_sem"][i]) * (1.0 if nw >= 5 else 0.5)
        hook = (0.26 * h_sem + 0.10 * h_lex + 0.14 * h_q + 0.14 * h_strong + 0.12 * h_story
                + 0.12 * h_punchy + 0.12 * h_energy)
        if f["dep_start"][i]:
            hook -= 0.25
        hook -= 0.35 * f["mid_start"][i]   # arranca a mitad de una frase (proporcional)
        hook -= 0.25 * f["dependent_sem"][i]
        if nw < 3 and not f["question"][i]:
            hook -= 0.15
        hook = min(1.0, max(0.0, hook) * 1.15)

        # ---------------- IMPACTO ----------------
        imp_vals = np.sort(f["impact_sem"][i:j + 1])
        i_sem = float(imp_vals[-2:].mean()) if k > 1 else float(imp_vals[-1])
        i_lex = min(1.0, self._sum("impact_lex", i, j) / 2.0)
        i_op = min(1.0, self._sum("opinion", i, j) / 3.0)
        ez = f["energy_z"][i:j + 1]
        i_peak = _ramp(float(ez.max()), 0.3, 2.0)
        i_dyn = _ramp(float(ez.std()) if k > 2 else 0.0, 0.3, 1.1)
        i_laugh = min(1.0, self._sum("laugh", i, j))
        i_excl = min(1.0, float(f["exclaim"][i:j + 1].sum()) / 2)
        impact = (0.34 * i_sem + 0.14 * i_lex + 0.14 * i_op + 0.13 * i_peak + 0.08 * i_dyn
                  + 0.10 * i_laugh + 0.07 * i_excl)
        impact = min(1.0, impact * 1.2)

        # ---------------- VALOR ----------------
        words = max(1.0, self._sum("n_words", i, j))
        v_sem = self._mean("value_sem", i, j)
        v_peak = float(np.sort(f["value_sem"][i:j + 1])[-2:].mean())
        v_lex = min(1.0, self._sum("value_lex", i, j) / max(1.5, k * 0.35))
        v_num = min(1.0, self._sum("numbers", i, j) / 1.5)
        filler_ratio = self._sum("fillers", i, j) / words
        v_clean = 1.0 - _ramp(filler_ratio, 0.01, 0.08)
        s_story = max(self._mean("story_sem", i, j), float(np.sort(f["story_sem"][i:j + 1])[-2:].mean()) * 0.8)
        value = 0.30 * v_sem + 0.20 * v_peak + 0.18 * v_lex + 0.12 * v_num + 0.10 * v_clean + 0.10 * s_story
        value = min(1.0, value * 1.15)

        # ---------------- STORYTELLING: hook → desarrollo → payoff ----------------
        emb_sum = self._vec(i, j)
        coherence = float(np.linalg.norm(emb_sum)) / k if k > 1 else 0.5
        coh = _ramp(coherence, 0.38, 0.68)
        # El remate cuenta si está al FINAL: la penúltima frase solo pesa completo si la
        # última es una muletilla corta ("Exacto.", "Sí, sí.").
        prev_w = 0.85 if f["n_words"][j] <= 3 else 0.55
        p_sem = max(f["payoff_sem"][j], prev_w * f["payoff_sem"][close_i])
        p_lex = max(f["payoff_lex"][j], prev_w * f["payoff_lex"][close_i])
        payoff = min(1.0, 0.65 * p_sem + 0.35 * p_lex + 0.15 * f["clean_end"][j])
        if f["question"][j]:
            payoff *= 0.3   # termina en una pregunta: no hay remate
        arc = _ramp(self._cos(self._vec(i, open_j), self._vec(close_i, j)), 0.25, 0.65) if k >= 3 else 0.0
        early = [x for x in range(i, j + 1) if self.start[x] - self.start[i] <= 0.25 * dur]
        late = [x for x in range(i + 1, j + 1) if self.start[x] >= self.end[j] - 0.25 * dur]
        qa = 1.0 if (f["question"][early].any() and not f["question"][late].any()) else 0.0
        # Al final se abre una pregunta nueva o se cambia de tema: el clip queda colgado.
        open_loop = 1.0 if (f["question"][late].any() or f["transition"][max(i, j - 1):j + 1].any()) else 0.0
        narrative = max(s_story, min(1.0, self._sum("story_lex", i, j) / 2.0))
        develop = _tri(k, 2, 4, 14, 24)
        storytelling = (0.26 * payoff + 0.18 * coh + 0.14 * arc + 0.16 * narrative + 0.08 * qa
                        + 0.10 * hook + 0.08 * develop)
        storytelling = min(1.0, max(0.0, storytelling * 1.1 - 0.3 * open_loop))

        # ---------------- AUTONOMÍA: ¿se entiende solo? ----------------
        a_start = 1.0 - max(f["dep_start"][i] * 0.8, f["dependent_sem"][i], f["mid_start"][i])
        a_shift = float(self.a.topic_shift[i])
        a_backref = 1.0 - min(1.0, self._sum("backref", i, j) / 1.5)
        a_end = float(self.a.topic_shift[j + 1]) if j + 1 < self.n else 1.0
        # Si la frase anterior es una pregunta, empezar acá es empezar la respuesta (natural).
        # Si no, arrancar sin ninguna pausa suele ser seguir el mismo hilo sin su contexto.
        prev_q = i > 0 and bool(f["question"][i - 1])
        a_flow = 1.0 if prev_q else _ramp(self.pause_before[i], 0.05, 0.6)
        autonomy = (0.32 * a_start + 0.22 * a_shift + 0.18 * coh + 0.08 * a_backref + 0.08 * a_end
                    + 0.12 * a_flow)

        # ---------------- RETENCIÓN ----------------
        rate = words / dur
        r_rate = _tri(rate, 1.4, 2.4, 4.2, 5.5)
        silence = self._sum("long_pause", i + 1, j) / dur if k > 1 else 0.0
        r_silence = 1.0 - _ramp(silence, 0.02, 0.15)
        r_energy = _ramp(float(np.mean(ez)), -0.8, 0.8)
        avg_len = words / k
        r_len = _tri(avg_len, 3, 6, 24, 45)
        r_dur = self.duration_fit(dur)
        retention = (0.24 * r_rate + 0.20 * r_silence + 0.12 * v_clean + 0.12 * r_energy
                     + 0.10 * r_len + 0.22 * r_dur)

        # ---------------- INICIO / FINAL ----------------
        b_start = (0.30 * _ramp(self.pause_before[i], 0.1, 0.7) + 0.25 * (1 - f["dep_start"][i])
                   + 0.30 * (1 - f["mid_start"][i]) + 0.15 * a_shift)
        b_end = (0.40 * f["clean_end"][j] + 0.20 * (1 - f["dangling"][j])
                 + 0.25 * _ramp(self.pause_after[j], 0.1, 0.8) + 0.15 * a_end)
        b_end -= 0.4 * open_loop
        boundaries = max(0.0, 0.5 * b_start + 0.5 * b_end)

        dims = {"hook": hook, "impact": impact, "value": value, "storytelling": storytelling,
                "autonomy": autonomy, "retention": retention, "boundaries": boundaries}
        total = sum(self.weights[d] * dims[d] for d in DIMENSIONS) / self.weight_sum
        # Transcripción poco fiable en la ventana => texto (y análisis) dudoso.
        total *= 1.0 - 0.4 * self._mean("low_conf", i, j)
        # Penalizaciones directas: cortar una frase o dejar el final colgado arruina un Short
        # aunque el contenido sea bueno.
        total *= 1.0 - 0.15 * f["mid_start"][i]
        if not f["clean_end"][j]:
            total *= 0.9
        if open_loop:
            total *= 0.9
        if not (self.min_d <= dur <= self.max_d):
            total *= 0.93
        if a_flow < 0.2:
            total *= 0.95
        # Logística del programa (pausas, saludos, autopromoción) dentro del clip.
        meta = float(max(f["meta_lex"][i:j + 1].max(), _ramp(float(f["meta_sem"][i:j + 1].max()), 0.3, 0.7)))
        total *= 1.0 - 0.4 * meta

        return {
            "i": i, "j": j, "duration": dur, "total": total * 100, "dims": dims,
            "signals": {"question_open": bool(f["question"][i]), "dep_start": bool(f["dep_start"][i]),
                        "payoff": payoff, "narrative": narrative, "laugh": i_laugh, "coherence": coherence,
                        "qa": bool(qa), "value_peak": v_peak, "impact_peak": i_sem,
                        "mid_start": bool(f["mid_start"][i] > 0.5), "open_loop": bool(open_loop),
                        "meta": meta},
        }


def reasons(scored: dict) -> list[str]:
    """Explicación breve (para la UI) de por qué el momento es bueno o flojo."""
    d, s = scored["dims"], scored["signals"]
    out: list[tuple[float, str]] = []
    if s["question_open"] and d["hook"] >= 0.5:
        out.append((d["hook"] + 0.1, "Abre con una pregunta"))
    elif d["hook"] >= 0.6:
        out.append((d["hook"], "Arranca con un gancho fuerte"))
    if s["qa"]:
        out.append((0.7, "Plantea una pregunta y la responde"))
    if s["payoff"] >= 0.5:
        out.append((s["payoff"], "Cierra con una conclusión"))
    if s["narrative"] >= 0.55:
        out.append((s["narrative"], "Cuenta una historia"))
    if s["value_peak"] >= 0.6 and d["value"] >= 0.55:
        out.append((d["value"], "Aporta consejos o datos"))
    if s["laugh"]:
        out.append((0.75, "Tiene humor"))
    if d["impact"] >= 0.6:
        out.append((d["impact"], "Momento intenso o emotivo"))
    if d["autonomy"] >= 0.7:
        out.append((d["autonomy"] - 0.1, "Se entiende sin contexto"))
    picked = [txt for _, txt in sorted(out, reverse=True)[:3]]
    if s["mid_start"]:
        picked.append("⚠ Empieza a mitad de una frase")
    elif s["dep_start"]:
        picked.append("⚠ Empieza retomando algo anterior")
    if s["open_loop"]:
        picked.append("⚠ El final abre otro tema")
    if s["meta"] >= 0.5:
        picked.append("⚠ Incluye logística del programa")
    return picked
