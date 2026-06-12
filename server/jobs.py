"""Очередь задач: JSON на диске + subprocess (переживает reload gunicorn)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
_LOCK_NAME = ".store.lock"


@contextmanager
def _store_lock(lock_path: Path):
    """Межпроцессная блокировка (gunicorn workers + subprocess)."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
    try:
        if os.name != "nt":
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX)
        else:
            import msvcrt

            msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
        yield
    finally:
        if os.name != "nt":
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pid_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, 0, pid)
        if not handle:
            return False
        code = ctypes.c_ulong()
        ok = ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(handle)
        return bool(ok and code.value == 259)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    done = "done"
    error = "error"


@dataclass
class Job:
    id: str
    status: JobStatus
    project_name: str
    created_at: str
    video_path: str
    source_language: str
    target_language: str
    started_at: str | None = None
    finished_at: str | None = None
    result_path: str | None = None
    error: str | None = None
    pid: int | None = None
    options: dict[str, Any] = field(default_factory=dict)


class JobStore:
    def __init__(self, jobs_dir: Path) -> None:
        self.jobs_dir = jobs_dir
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._load_all()

    def _job_path(self, job_id: str) -> Path:
        return self.jobs_dir / f"{job_id}.json"

    def _load_all(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._merge_disk()

    def _merge_disk(self) -> None:
        for p in self.jobs_dir.glob("*.json"):
            if p.name.endswith(".result.json"):
                continue
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                data["status"] = JobStatus(data["status"])
                job = Job(**data)
                if job.status == JobStatus.running and not _pid_alive(job.pid):
                    job.status = JobStatus.error
                    job.error = (job.error or "") + "\nworker process died"
                    job.finished_at = job.finished_at or _utc_now()
                    self._jobs[job.id] = job
                    self._persist(job)
                else:
                    self._jobs[job.id] = job
            except (json.JSONDecodeError, KeyError, ValueError):
                continue

    def _persist(self, job: Job) -> None:
        data = asdict(job)
        data["status"] = job.status.value
        self._job_path(job.id).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8",
        )

    def _has_active_locked(self) -> bool:
        return any(
            j.status in (JobStatus.queued, JobStatus.running) for j in self._jobs.values()
        )

    def enqueue(
        self,
        project_name: str,
        video_path: Path,
        source_language: str,
        target_language: str,
        options: dict | None = None,
    ) -> Job | None:
        """Атомарно: одна задача queued/running на кластер."""
        lock_path = self.jobs_dir / _LOCK_NAME
        with _store_lock(lock_path):
            with self._lock:
                self._merge_disk()
                if self._has_active_locked():
                    return None
                job = Job(
                    id=uuid.uuid4().hex[:12],
                    status=JobStatus.queued,
                    project_name=project_name,
                    created_at=_utc_now(),
                    video_path=str(video_path.resolve()),
                    source_language=source_language,
                    target_language=target_language,
                    options=options or {},
                )
                self._jobs[job.id] = job
                self._persist(job)
                return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            self._merge_disk()
            return self._jobs.get(job_id)

    def list_jobs(self, limit: int = 50) -> list[Job]:
        with self._lock:
            self._merge_disk()
            jobs = list(self._jobs.values())
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs[:limit]

    def update(self, job_id: str, **fields) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            for k, v in fields.items():
                if k == "status" and isinstance(v, str):
                    v = JobStatus(v)
                setattr(job, k, v)
            self._persist(job)

    def to_dict(self, job: Job) -> dict:
        d = asdict(job)
        d["status"] = job.status.value
        return d

    def active_job_id(self) -> str | None:
        with self._lock:
            self._merge_disk()
            for j in sorted(self._jobs.values(), key=lambda x: x.created_at, reverse=True):
                if j.status in (JobStatus.queued, JobStatus.running):
                    return j.id
        return None


def start_job(job: Job, projects_root: Path, logs_dir: Path) -> subprocess.Popen:
    """Запуск пайплайна в отдельном процессе (не daemon-thread)."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / f"job_{job.id}.log"
    env = os.environ.copy()
    env["SPEECHLAB_PROJECTS_ROOT"] = str(projects_root.resolve())
    venv_bin = str((ROOT / ".venv" / "bin").resolve())
    env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")

    cmd = [sys.executable, "-m", "server.run_job", job.id]
    kwargs: dict = {
        "cwd": str(ROOT),
        "env": env,
        "stdout": log_file.open("a", encoding="utf-8"),
        "stderr": subprocess.STDOUT,
    }
    if os.name != "nt":
        kwargs["start_new_session"] = True

    proc = subprocess.Popen(cmd, **kwargs)
    return proc
