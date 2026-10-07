"""Análisis del contenido: convierte la transcripción en frases con rasgos.

Salida (Analysis):
- frases con límites naturales (puntuación + pausas), sin muletillas iniciales;
- rasgos por frase: léxicos, prosódicos (energía, ritmo) y semánticos
  (similitud con prototipos de hook/payoff/historia/valor/impacto/dependencia);
- embeddings por frase y "fuerza de cambio de tema" en cada límite entre
  frases (estilo TextTiling), para saber dónde empieza y termina una idea.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field

import numpy as np

from . import lexicon as lx
from .config import MODELS_DIR
from .errors import AnalysisError
from .logging_setup import get_logger
from .transcription import ENERGY_FPS

log = get_logger("analyzer")

_TERMINAL = re.compile(r"[.?!…]+[\"»”')\]]*$")
_embed_lock = threading.Lock()
_embedder = None
_proto_cache: dict[str, np.ndarray] = {}


@dataclass
class Sentence:
    idx: int
    start: float          # inicio del contenido (sin muletillas iniciales)
    end: float
    raw_start: float      # inicio incluyendo muletillas
    text: str
    words: list[dict] = field(repr=False)
    pause_before: float = 0.0
    pause_after: float = 0.0
    lead_end: float | None = None   # fin de la última muletilla recortada, si hubo
    start_quality: float = 1.0      # 1 = empieza una frase; 0.7 = tras pausa larga; 0 = a mitad de frase
    starts_clean: bool = True       # start_quality > 0
    ends_clean: bool = True         # termina una frase completa

    @property
    def duration(self) -> float:
        return max(0.01, self.end - self.start)


@dataclass
class Analysis:
    sentences: list[Sentence]
    emb: np.ndarray                 # (n, d) normalizados
    feats: dict[str, np.ndarray]    # rasgos por frase, todos de largo n
    topic_shift: np.ndarray         # (n,) fuerza del cambio de tema ANTES de cada frase, 0-1
    duration: float
    language: str | None


# --------------------------------------------------------------------------
# Frases
# --------------------------------------------------------------------------

def _norm_token(word: str) -> str:
    return re.sub(r"[^\wáéíóúüñ]", "", word.lower())


def _split_long(words: list[dict], max_dur: float) -> list[list[dict]]:
    """Parte frases larguísimas (sin puntos) en el mejor punto cercano al medio."""
    if len(words) < 6 or words[-1]["end"] - words[0]["start"] <= max_dur:
        return [words]
    best, best_score = None, -1.0
    n = len(words)
    for k in range(int(n * 0.25), int(n * 0.75)):
        gap = words[k + 1]["start"] - words[k]["end"]
        comma = words[k]["word"].strip().endswith((",", ";", ":"))
        score = gap + (0.5 if comma else 0) - abs(k / n - 0.5) * 0.3
        if score > best_score:
            best, best_score = k, score
    return _split_long(words[: best + 1], max_dur) + _split_long(words[best + 1:], max_dur)


def _starts_upper(word: str) -> bool:
    w = word.strip().lstrip("\"'«“(-")
    return bool(w) and (w[0] in "¿¡" or w[0].isupper() or w[0].isdigit())


def build_sentences(transcript: dict, cfg: dict) -> list[Sentence]:
    """Agrupa palabras en frases. Corta en puntuación final y, como Whisper no
    siempre puntúa, también en pausas + mayúscula o en pausas muy largas. Una pausa
    seguida de minúscula suele ser alguien pensando a mitad de frase: no se corta."""
    acfg = cfg["analysis"]
    max_pause = float(acfg["sentence_max_pause"])
    max_dur = float(acfg["sentence_max_duration"])
    words, seg_last = [], set()
    for seg in transcript["segments"]:
        ws = [w for w in seg["words"] if w["word"].strip()]
        words.extend(ws)
        if ws:
            seg_last.add(len(words) - 1)
    groups: list[list[dict]] = []
    current: list[dict] = []
    for i, w in enumerate(words):
        current.append(w)
        nxt = words[i + 1] if i + 1 < len(words) else None
        if nxt is None:
            break
        gap = nxt["start"] - w["end"]
        upper = _starts_upper(nxt["word"])
        if (_TERMINAL.search(w["word"].strip())
                or (upper and gap >= max_pause)
                or (upper and i in seg_last and gap >= 0.3)
                or gap >= max_pause * 1.8):
            groups.append(current)
            current = []
    if current:
        groups.append(current)

    pieces = [p for g in groups for p in _split_long(g, max_dur)]
    sentences: list[Sentence] = []
    for p in pieces:
        k = 0  # recortar muletillas iniciales ("Eh, bueno, ...")
        while k < len(p) - 2 and _norm_token(p[k]["word"]) in lx.LEADING_TRIM and (
                p[k]["word"].strip().endswith((",", ".", "...")) or _norm_token(p[k]["word"]) in lx.FILLER_WORDS):
            k += 1
        content = p[k:]
        text = "".join(w["word"] for w in content).strip()
        if not re.search(r"\w", text):
            continue
        sentences.append(Sentence(idx=len(sentences), start=content[0]["start"], end=p[-1]["end"],
                                  raw_start=p[0]["start"], text=text, words=content,
                                  lead_end=p[k - 1]["end"] if k else None))
    for i, s in enumerate(sentences):
        prev = sentences[i - 1] if i > 0 else None
        s.pause_before = s.raw_start - prev.end if prev else 2.0
        s.pause_after = sentences[i + 1].raw_start - s.end if i + 1 < len(sentences) else 2.0
        # "..." al final = la frase quedó trunca.
        s.ends_clean = bool(_TERMINAL.search(s.text)) and not s.text.endswith(("...", "…"))
        # Inicio: la frase anterior terminó, o esta arranca con mayúscula tras una pausa.
        # Una pausa larga sin conector también marca un límite de cláusula en el habla
        # real ("…en la facultad, [2 s] me quedé muda…"), aunque Whisper no puntúe.
        if prev is None or prev.ends_clean or (_starts_upper(s.words[0]["word"]) and s.pause_before >= 0.25):
            s.start_quality = 1.0
        elif s.pause_before >= 1.5 and not lx.DEPENDENT_START.search(s.text.lower()[:40]):
            s.start_quality = 0.7
        else:
            s.start_quality = 0.0
        s.starts_clean = s.start_quality > 0
    for i, s in enumerate(sentences):
        nxt = sentences[i + 1] if i + 1 < len(sentences) else None
        # Fin limpio: puntuación final, o la siguiente frase arranca limpia tras una pausa.
        s.ends_clean = s.ends_clean or nxt is None or (not s.text.endswith(("...", "…")) and
            _starts_upper(nxt.words[0]["word"]) and s.pause_after >= 0.4 and not lx.DANGLING_END.search(s.text))
    return sentences


# --------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------

def _get_embedder(cfg: dict):
    global _embedder
    with _embed_lock:
        if _embedder is None:
            from fastembed import TextEmbedding
            name = cfg["analysis"]["embedding_model"]
            log.info("Cargando modelo de embeddings %s", name)
            _embedder = TextEmbedding(name, cache_dir=str(MODELS_DIR / "embeddings"))
        return _embedder


def embed(texts: list[str], cfg: dict) -> np.ndarray:
    model = _get_embedder(cfg)
    vecs = np.array(list(model.embed(texts, batch_size=64)), dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
    return vecs


def _prototype_matrix(cfg: dict) -> dict[str, np.ndarray]:
    if not _proto_cache:
        for cat, examples in lx.PROTOTYPES.items():
            _proto_cache[cat] = embed(examples, cfg)
    return _proto_cache


def _semantic_scale(sim: np.ndarray) -> np.ndarray:
    """Lleva similitudes coseno a 0-1 combinando escala absoluta y relativa al video
    (lo relativo destaca lo mejor de ESTE video; lo absoluto evita inflar videos flojos)."""
    absolute = np.clip((sim - 0.22) / (0.62 - 0.22), 0, 1)
    med, top = np.median(sim), np.percentile(sim, 97)
    relative = np.clip((sim - med) / max(top - med, 1e-3), 0, 1)
    return 0.45 * absolute + 0.55 * relative


def _topic_shift(emb: np.ndarray, window: int = 3) -> np.ndarray:
    """Profundidad de valle de similitud entre bloques de frases (TextTiling)."""
    n = len(emb)
    sims = np.ones(n)
    for i in range(1, n):
        left = emb[max(0, i - window):i].mean(axis=0)
        right = emb[i:min(n, i + window)].mean(axis=0)
        sims[i] = float(left @ right / (np.linalg.norm(left) * np.linalg.norm(right) + 1e-9))
    depth = np.zeros(n)
    for i in range(1, n):
        lpeak = sims[max(1, i - 4):i + 1].max()
        rpeak = sims[i:min(n, i + 5)].max()
        depth[i] = (lpeak - sims[i]) + (rpeak - sims[i])
    depth[0] = depth.max() if n > 1 else 1.0
    scale = np.percentile(depth, 95) if n > 5 else (depth.max() or 1.0)
    return np.clip(depth / max(scale, 1e-6), 0, 1)


# --------------------------------------------------------------------------
# Rasgos
# --------------------------------------------------------------------------

def _count(rx: re.Pattern, text: str) -> int:
    return sum(1 for _ in rx.finditer(text))


def analyze(transcript: dict, energy: np.ndarray, cfg: dict, on_progress=None) -> Analysis:
    progress = on_progress or (lambda p, m: None)
    sentences = build_sentences(transcript, cfg)
    if len(sentences) < 5:
        raise AnalysisError("El video tiene muy poco diálogo para encontrar momentos.", f"{len(sentences)} frases")
    n = len(sentences)
    progress(0.1, "Analizando frases")

    f: dict[str, np.ndarray] = {k: np.zeros(n) for k in (
        "n_words", "question", "exclaim", "terminal", "hook_lex", "dep_start", "backref", "payoff_lex",
        "value_lex", "story_lex", "impact_lex", "laugh", "opinion", "numbers", "fillers", "dangling",
        "energy", "rate", "low_conf", "mid_start", "clean_end", "transition", "meta_lex")}
    for i, s in enumerate(sentences):
        t = s.text
        low = t.lower().lstrip("¿¡\"'«- ")
        tokens = [_norm_token(w["word"]) for w in s.words]
        f["n_words"][i] = len(tokens)
        f["question"][i] = 1.0 if ("?" in t or t.startswith("¿")) else 0.0
        f["exclaim"][i] = 1.0 if ("!" in t or "¡" in t) else 0.0
        f["terminal"][i] = 1.0 if _TERMINAL.search(t) else 0.0
        f["hook_lex"][i] = min(1.0, _count(lx.HOOK, t) / 1.5)
        f["dep_start"][i] = 1.0 if lx.DEPENDENT_START.search(low[:60]) else 0.0
        f["backref"][i] = 1.0 if lx.BACK_REFERENCE.search(t) else 0.0
        f["payoff_lex"][i] = min(1.0, _count(lx.PAYOFF, t) / 1.5)
        f["value_lex"][i] = min(1.0, _count(lx.VALUE, t) / 2.0)
        f["story_lex"][i] = min(1.0, _count(lx.STORY, t) / 1.5)
        f["impact_lex"][i] = min(1.0, _count(lx.IMPACT, t) / 1.5)
        f["laugh"][i] = 1.0 if lx.LAUGH.search(t) else 0.0
        f["opinion"][i] = min(1.0, _count(lx.OPINION, t) / 2.0)
        f["numbers"][i] = min(1.0, len(lx.NUMBER.findall(t)) / 2.0)
        f["fillers"][i] = sum(1 for tok in tokens if tok in lx.FILLER_WORDS) + _count(lx.FILLER_PHRASES, t)
        f["dangling"][i] = 1.0 if lx.DANGLING_END.search(t) else 0.0
        f["mid_start"][i] = 1.0 - s.start_quality
        f["clean_end"][i] = 1.0 if s.ends_clean else 0.0
        f["transition"][i] = 1.0 if lx.TRANSITION.search(t) else 0.0
        f["meta_lex"][i] = 1.0 if lx.META.search(t) else 0.0
        f["rate"][i] = len(tokens) / s.duration
        f["low_conf"][i] = float(np.mean([w.get("p", 1.0) < 0.4 for w in s.words]))
        a, b = int(s.start * ENERGY_FPS), max(int(s.start * ENERGY_FPS) + 1, int(s.end * ENERGY_FPS))
        seg = energy[a:b] if a < len(energy) else energy[-1:]
        f["energy"][i] = float(np.percentile(seg, 75)) if seg.size else -60.0

    # Energía relativa al propio video (cada canal/micrófono tiene otro nivel).
    e = f["energy"]
    f["energy_z"] = (e - np.median(e)) / (np.std(e) + 1e-6)

    progress(0.3, "Calculando embeddings semánticos")
    emb =embed([s.text for s in sentences], cfg)
    progress(0.7, "Comparando con patrones de buenos momentos")
    protos = _prototype_matrix(cfg)
    for cat, pm in protos.items():
        sims = emb @ pm.T                              # (n, k)
        top2 = np.sort(sims, axis=1)[:, -2:].mean(axis=1)
        if cat == "meta":
            # Escala solo absoluta: la mayoría de los videos casi no tienen logística,
            # y una escala relativa marcaría frases normales como "meta".
            f["meta_sem"] = np.clip((top2 - 0.5) / (0.72 - 0.5), 0, 1)
        else:
            f[f"{cat}_sem"] = _semantic_scale(top2)
    # Frases muy cortas tienen embeddings poco fiables: atenuar su señal semántica.
    reliability = np.clip(f["n_words"] / 6.0, 0.3, 1.0)
    for cat in protos:
        f[f"{cat}_sem"] *= reliability

    progress(0.85, "Detectando cambios de tema")
    shift = _topic_shift(emb)
    # Una pausa larga también suele marcar un cambio de idea.
    pauses = np.array([s.pause_before for s in sentences])
    shift = np.clip(0.75 * shift + 0.25 * np.clip(pauses / 1.5, 0, 1), 0, 1)

    return Analysis(sentences=sentences, emb=emb, feats=f, topic_shift=shift,
                    duration=float(transcript.get("duration") or sentences[-1].end),
                    language=transcript.get("language"))
