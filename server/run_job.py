"""Воркер GPU-пайплайна: python -m server.run_job <job_id>

Запускается как subprocess из routes.create_dub() / jobs.start_job().
Обновляет jobs/<id>.json: running → done | error.
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env", override=False)
load_dotenv(ROOT / "config" / ".env", override=True)

from server.config import ServerConfig
from server.jobs import JobStatus, JobStore, _utc_now


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python -m server.run_job <job_id>", file=sys.stderr)
        return 2

    job_id = sys.argv[1].strip()
    cfg = ServerConfig()
    cfg.ensure_dirs()
    store = JobStore(cfg.jobs_dir)
    job = store.get(job_id)
    if not job:
        print(f"job not found: {job_id}", file=sys.stderr)
        return 1

    # Помечаем задачу running; pid нужен JobStore для детекта «зомби» после crash
    store.update(job_id, status=JobStatus.running, started_at=_utc_now(), pid=os.getpid())

    try:
        import main as pipeline

        opts = job.options
        # Полный пайплайн PRD: demucs → ASR → LLM → TTS → mux MP4
        out = pipeline.run(
            job.project_name,
            job.video_path,
            job.source_language,
            job.target_language,
            projects_root=cfg.projects_root,
            dub_volume_percent=opts.get("dub_volume_percent"),
            original_audio_ratio=opts.get("original_audio_ratio"),
            voice_design_template=opts.get("voice_design_template"),
            voice_design_by_key=opts.get("voice_design_by_key"),
            voice_gender=opts.get("voice_gender"),
            voice_age=opts.get("voice_age"),
            voice_design_temperature=opts.get("voice_design_temperature"),
            voice_clone_samples=opts.get("voice_clone_samples"),
            cast_voice=opts.get("cast_voice"),
            cast_mode=opts.get("cast_mode"),
        )
        store.update(
            job_id,
            status=JobStatus.done,
            finished_at=_utc_now(),
            result_path=str(out.resolve()),
            error=None,
        )
        print(f"OK: {out}")
        return 0
    except Exception as exc:
        # traceback обрезается — JSON job не раздувается
        store.update(
            job_id,
            status=JobStatus.error,
            finished_at=_utc_now(),
            error=f"{exc}\n{traceback.format_exc()[-4000:]}",
        )
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
