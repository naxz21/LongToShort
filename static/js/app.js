"use strict";

const $ = (sel) => document.querySelector(sel);
const STAGES = ["downloading", "transcribing", "analyzing"];
const DIM_LABELS = {
  hook: "Hook", impact: "Impacto", value: "Valor", storytelling: "Storytelling",
  autonomy: "Autonomía", retention: "Retención", boundaries: "Inicio/final",
};
const STATUS_TEXT = {
  queued: "En cola", downloading: "Descargando…", transcribing: "Transcribiendo…",
  analyzing: "Analizando momentos…", ready: "Listo", generating: "Generando Shorts…",
  completed: "Shorts generados", error: "Error",
};

const state = { videoId: null, candidates: [], selected: new Set(), segmentEnd: null };

function fmtTime(sec) {
  sec = Math.max(0, Math.round(sec));
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
  const mm = String(m).padStart(2, "0"), ss = String(s).padStart(2, "0");
  return h ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
}

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" }, ...options,
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail || "No se pudo comunicar con LongToShort.");
  return body;
}

function poll(jobId, onUpdate) {
  return new Promise((resolve, reject) => {
    const tick = async () => {
      try {
        const job = await api(`/api/job/${jobId}`);
        onUpdate(job);
        if (job.status === "error") return reject(new Error(job.error));
        if (job.status === "ready" || job.status === "completed") return resolve(job);
        setTimeout(tick, 800);
      } catch (err) {
        reject(err);
      }
    };
    tick();
  });
}

// ---------------------------------------------------------------- análisis

function resetStages() {
  document.querySelectorAll(".stage").forEach((el) => {
    el.classList.remove("active", "done");
    el.querySelector(".fill").style.width = "0%";
    el.querySelector(".pct").textContent = "0%";
  });
}

function renderVideo(video) {
  if (!video) return;
  $("#thumb").src = video.thumbnail || "";
  $("#v-title").textContent = video.title || video.id;
  $("#v-channel").textContent = video.channel || "";
  const parts = [];
  if (video.duration) parts.push(`Duración ${fmtTime(video.duration)}`);
  if (video.resolution) parts.push(video.resolution);
  if (video.audio_track) parts.push(`Audio: ${video.audio_track}${video.audio_tracks > 1 ? ` (de ${video.audio_tracks} pistas)` : ""}`);
  $("#v-details").textContent = parts.join(" · ");
}

function renderJob(job) {
  renderVideo(job.data && job.data.video);
  $("#status-text").textContent = job.status === "error" ? "Error" : (job.message || STATUS_TEXT[job.status]);
  const current = STAGES.indexOf(job.status);
  const finished = job.status === "ready";
  STAGES.forEach((stage, i) => {
    const el = document.querySelector(`.stage[data-stage="${stage}"]`);
    let pct = 0;
    if (finished || (current >= 0 && i < current)) pct = 100;
    else if (i === current) pct = job.progress;
    if (job.status === "error" && !el.classList.contains("done") && !el.classList.contains("active")) return;
    el.classList.toggle("done", pct >= 100);
    el.classList.toggle("active", i === current && pct < 100);
    el.querySelector(".fill").style.width = `${pct}%`;
    el.querySelector(".pct").textContent = `${Math.round(pct)}%`;
  });
  renderWarnings(job.warnings);
}

function renderWarnings(warnings) {
  const box = $("#warnings");
  box.hidden = !warnings || !warnings.length;
  box.innerHTML = "";
  (warnings || []).forEach((w) => {
    const p = document.createElement("p");
    p.style.margin = "2px 0";
    p.textContent = `⚠ ${w}`;
    box.appendChild(p);
  });
}

function showError(message) {
  const box = $("#error");
  box.hidden = false;
  box.textContent = message;
}

async function analyze(event) {
  event.preventDefault();
  const url = $("#url").value.trim();
  if (!url) return;
  $("#analyze-btn").disabled = true;
  $("#error").hidden = true;
  $("#moments-section").hidden = true;
  $("#shorts-section").hidden = true;
  $("#video-section").hidden = false;
  $("#thumb").removeAttribute("src");
  $("#v-title").textContent = "Obteniendo información…";
  $("#v-channel").textContent = "";
  $("#v-details").textContent = "";
  resetStages();
  try {
    const { job_id, video_id } = await api("/api/analyze", { method: "POST", body: JSON.stringify({ url }) });
    state.videoId = video_id;
    const job = await poll(job_id, renderJob);
    renderMoments(job.result);
  } catch (err) {
    $("#status-text").textContent = "Error";
    showError(err.message);
  } finally {
    $("#analyze-btn").disabled = false;
  }
}

// ---------------------------------------------------------------- momentos

function scoreClass(score) {
  return score >= 70 ? "high" : score >= 55 ? "mid" : "low";
}

function renderMoments(result) {
  state.candidates = result.candidates;
  state.selected.clear();
  const list = $("#moments");
  list.innerHTML = "";
  const player = $("#player");
  player.src = result.video_url;

  const tpl = $("#moment-tpl");
  result.candidates.forEach((c) => {
    const li = tpl.content.firstElementChild.cloneNode(true);
    li.dataset.id = c.id;
    li.querySelector(".rank").textContent = `#${c.id}`;
    const score = li.querySelector(".score");
    score.textContent = `${c.score}/100`;
    score.classList.add(scoreClass(c.score));
    li.querySelector(".time").textContent = `${fmtTime(c.start)} → ${fmtTime(c.end)} · ${Math.round(c.duration)} s`;
    li.querySelector(".hook").textContent = c.hook;
    li.querySelector(".full-text").textContent = c.text;
    const reasons = li.querySelector(".reasons");
    c.reasons.forEach((r) => {
      const item = document.createElement("li");
      item.textContent = r;
      if (r.startsWith("⚠")) item.classList.add("neg");
      reasons.appendChild(item);
    });
    const dims = li.querySelector(".dims");
    Object.entries(c.scores).forEach(([key, value]) => {
      const d = document.createElement("div");
      d.className = "dim";
      d.innerHTML = `<span></span><div class="bar"><div class="fill"></div></div>`;
      d.querySelector("span").textContent = `${DIM_LABELS[key] || key}: ${value}/10`;
      d.querySelector(".fill").style.width = `${value * 10}%`;
      dims.appendChild(d);
    });
    const checkbox = li.querySelector('input[type="checkbox"]');
    checkbox.addEventListener("change", () => {
      if (checkbox.checked) state.selected.add(c.id); else state.selected.delete(c.id);
      li.classList.toggle("selected", checkbox.checked);
      updateGenerateButton();
    });
    li.querySelector(".play").addEventListener("click", () => playSegment(c, li));
    list.appendChild(li);
  });

  $("#moments-count").textContent = `${result.candidates.length} candidatos · idioma ${result.transcription.language} · ${result.transcription.model}`;
  $("#select-all").checked = false;
  $("#moments-section").hidden = false;
  $("#gen-status").hidden = true;
  updateGenerateButton();
}

function playSegment(c, li) {
  const player = $("#player");
  document.querySelectorAll(".moment.playing").forEach((el) => el.classList.remove("playing"));
  li.classList.add("playing");
  state.segmentEnd = c.end;
  $("#preview-label").textContent = `#${c.id} · ${fmtTime(c.start)} → ${fmtTime(c.end)}`;
  player.currentTime = c.start;
  player.play();
}

function updateGenerateButton() {
  const n = state.selected.size;
  const btn = $("#generate-btn");
  btn.disabled = n === 0;
  btn.textContent = n ? `Generar Shorts (${n})` : "Generar Shorts";
}

function toggleAll(event) {
  document.querySelectorAll("#moments .moment").forEach((li) => {
    const cb = li.querySelector('input[type="checkbox"]');
    if (cb.checked !== event.target.checked) {
      cb.checked = event.target.checked;
      cb.dispatchEvent(new Event("change"));
    }
  });
}

// ---------------------------------------------------------------- generación

async function generate() {
  const ids = [...state.selected];
  if (!ids.length) return;
  $("#generate-btn").disabled = true;
  $("#gen-status").hidden = false;
  $("#error").hidden = true;
  try {
    const { job_id } = await api("/api/generate", {
      method: "POST", body: JSON.stringify({ video_id: state.videoId, candidate_ids: ids }),
    });
    const job = await poll(job_id, (j) => {
      $("#gen-text").textContent = j.message || STATUS_TEXT[j.status];
      $("#gen-pct").textContent = `${Math.round(j.progress)}%`;
      $("#gen-fill").style.width = `${j.progress}%`;
      renderWarnings(j.warnings);
    });
    renderShorts(job.result);
  } catch (err) {
    $("#gen-text").textContent = "Error";
    $("#video-section").hidden = false;
    showError(err.message);
  } finally {
    updateGenerateButton();
  }
}

function renderShorts(result) {
  const box = $("#shorts");
  result.shorts.forEach((s) => {
    const card = document.createElement("div");
    card.className = "short";
    card.innerHTML = `<video controls preload="metadata"></video><p></p><a download>Descargar</a>`;
    card.querySelector("video").src = s.url;
    card.querySelector("p").textContent = `${s.file} · momento #${s.candidate} · ${Math.round(s.duration)} s`;
    const link = card.querySelector("a");
    link.href = s.url;
    link.download = s.file;
    box.prepend(card);
  });
  $("#output-dir").textContent = `Guardados en ${result.output_dir}`;
  $("#shorts-section").hidden = false;
  $("#shorts-section").scrollIntoView({ behavior: "smooth" });
}

// ---------------------------------------------------------------- init

async function checkHealth() {
  try {
    const h = await api("/api/health");
    const box = $("#health");
    if (!h.ffmpeg) {
      box.hidden = false;
      box.textContent = "⚠ No se encontró FFmpeg: instalalo para poder analizar y generar clips (ver README).";
    }
  } catch (_) { /* el servidor responde con error en el primer análisis */ }
}

$("#analyze-form").addEventListener("submit", analyze);
$("#generate-btn").addEventListener("click", generate);
$("#select-all").addEventListener("change", toggleAll);
$("#player").addEventListener("timeupdate", (e) => {
  if (state.segmentEnd !== null && e.target.currentTime >= state.segmentEnd) {
    e.target.pause();
    state.segmentEnd = null;
  }
});
checkHealth();

// /?url=... precarga la URL y la analiza (útil para volver a abrir un análisis).
const preset = new URLSearchParams(location.search).get("url");
if (preset) {
  $("#url").value = preset;
  $("#analyze-form").requestSubmit();
}
