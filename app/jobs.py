"""Sistema de jobs en segundo plano.

Un único worker procesa los jobs en orden: descargar, transcribir y generar
clips usan mucha CPU/GPU, así que ejecutarlos en paralelo solo los volvería
más lentos. Los jobs viven en memoria; los resultados pesados (transcripción,
análisis, clips) se guardan en disco por cada pipeline.
"""
from __future__ import annotations

import queue
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from .errors import UserFacingError
from .logging_setup import get_logger

log = get_logger("jobs")

STATUSES = ("queued", "downloading", "transcribing", "analyzing", "ready",
            "generating", "completed", "error")


@dataclass
class Job:
    id: str
    kind: str
    status: str = "queued"
    progress: float = 0.0          # progreso del estado actual, 0-100
    message: str = "En cola"
    error: str | None = None
    result: Any = None
    data: dict = field(default_factory=dict)   # info parcial (p. ej. metadata del video)
    warnings: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def update(self, status: str | None = None, progress: float | None = None,
               message: str | None = None) -> None:
        if status is not None:
            assert status in STATUSES, status
            self.status = status
        if progress is not None:
            self.progress = max(0.0, min(100.0, float(progress)))
        if message is not None:
            self.message = message
        self.updated_at = time.time()

    def warn(self, text: str) -> None:
        if text not in self.warnings:
            self.warnings.append(text)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "kind": self.kind, "status": self.status,
            "progress": round(self.progress, 1), "message": self.message,
            "error": self.error, "result": self.result, "data": self.data,
            "warnings": self.warnings,
        }


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._queue: queue.Queue[tuple[Job, Callable[[Job], Any], str]] = queue.Queue()
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._run, name="job-worker", daemon=True)
        self._worker.start()

    def submit(self, kind: str, func: Callable[[Job], Any], final_status: str) -> Job:
        """func(job) hace el trabajo y devuelve el resultado; al terminar el job
        pasa a final_status ('ready' para análisis, 'completed' para generación)."""
        job = Job(id=uuid.uuid4().hex[:12], kind=kind)
        with self._lock:
            self._jobs[job.id] = job
        self._queue.put((job, func, final_status))
        log.info("Job %s (%s) en cola", job.id, kind)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def _run(self) -> None:
        while True:
            job, func, final_status = self._queue.get()
            started = time.time()
            try:
                job.result = func(job)
                job.update(status=final_status, progress=100, message="Listo")
                log.info("Job %s terminado en %.1fs", job.id, time.time() - started)
            except UserFacingError as exc:
                job.error = exc.user_message
                job.update(status="error", message=exc.user_message)
                log.error("Job %s falló: %s\n%s", job.id, exc.detail, traceback.format_exc())
            except Exception as exc:  # noqa: BLE001 - cualquier fallo debe llegar a la UI
                job.error = "Ocurrió un error inesperado. Revisá logs/longtoshort.log para más detalles."
                job.update(status="error", message=job.error)
                log.error("Job %s falló inesperadamente: %s\n%s", job.id, exc, traceback.format_exc())
            finally:
                self._queue.task_done()


jobs = JobManager()
