"""Sync Labs lipsync для second_seg (PRD: только full dubbing).

Документация: https://sync.so/docs/quickstart
SDK: syncsdk — generations.create_with_files + poll.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import httpx

SYNC_MODEL = os.environ.get("SPEECHLAB_SYNC_MODEL", "lipsync-2")
POLL_SEC = float(os.environ.get("SPEECHLAB_SYNC_POLL_SEC", "10"))


def get_sync_api_key() -> str:
    from config.env_config import get_sync_api_key as _get
    return _get()


def _client():
    from sync import Sync
    return Sync(api_key=get_sync_api_key())


def lipsync_files(video_path: Path, audio_path: Path, out_mp4: Path) -> Path:
    """Локальные video+audio → lipsynced MP4 (create_with_files + poll)."""
    from sync.common import GenerationOptions
    from sync.core.api_error import ApiError

    video_path = Path(video_path)
    audio_path = Path(audio_path)
    out_mp4 = Path(out_mp4)
    if not video_path.is_file() or not audio_path.is_file():
        raise FileNotFoundError(f"lipsync input: {video_path} / {audio_path}")

    client = _client()
    try:
        with video_path.open("rb") as vf, audio_path.open("rb") as af:
            job = client.generations.create_with_files(
                model=SYNC_MODEL,  # type: ignore[arg-type]
                video=(video_path.name, vf, "video/mp4"),
                audio=(audio_path.name, af, "audio/wav"),
                options=GenerationOptions(sync_mode="cut_off"),
            )
    except ApiError as e:
        raise RuntimeError(f"Sync create failed: {e.status_code} {e.body}") from e

    job_id = job.id
    status = job.status
    while status not in ("COMPLETED", "FAILED", "REJECTED"):
        time.sleep(POLL_SEC)
        job = client.generations.get(job_id)
        status = job.status

    if status != "COMPLETED" or not job.output_url:
        err = getattr(job, "error", None) or status
        raise RuntimeError(f"Sync job {job_id} failed: {err}")

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    with httpx.stream("GET", job.output_url, follow_redirects=True, timeout=600.0) as r:
        r.raise_for_status()
        with out_mp4.open("wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)
    return out_mp4
