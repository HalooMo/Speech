"""REST API дубляжа."""
from __future__ import annotations

import json
import re
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request, send_file
from werkzeug.utils import secure_filename

from server.jobs import JobStatus, start_job
from server.security import allowed_result_path, allowed_video_path

bp = Blueprint("api", __name__)

_PROJECT_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_ALLOWED_VIDEO = {".mp4", ".mkv", ".mov", ".avi", ".wav", ".mp3", ".m4a", ".webm"}


def _cfg():
    return current_app.extensions["server_config"]


def _store():
    return current_app.extensions["job_store"]


def _json_body() -> dict:
    if not request.is_json:
        return {}
    return request.get_json(silent=True) or {}


def _opt_float(key: str):
    v = request.form.get(key)
    if v is None:
        v = _json_body().get(key)
    if v is None or v == "":
        return None
    return float(v)


def _opt_str(key: str):
    v = request.form.get(key)
    if v is None:
        v = _json_body().get(key)
    return (v or "").strip() or None


def _opt_str_any(*keys: str) -> str | None:
    for key in keys:
        v = _opt_str(key)
        if v:
            return v
    return None


def _parse_voice_design_by_key() -> dict | None:
    raw = _json_body().get("voice_design_by_key")
    if isinstance(raw, dict) and raw:
        return raw
    form_raw = request.form.get("voice_design_by_key")
    if form_raw:
        try:
            parsed = json.loads(form_raw)
            if isinstance(parsed, dict) and parsed:
                return parsed
        except json.JSONDecodeError:
            pass
    return None


def _parse_options() -> dict:
    """Опции дубляжа: голос, громкость, промпт VoiceDesign."""
    voice_prompt = _opt_str_any(
        "voice_prompt",
        "voice_design_prompt",
        "voice_design_template",
    )
    opts = {
        "dub_volume_percent": _opt_float("dub_volume_percent"),
        "original_audio_ratio": _opt_float("original_audio_ratio"),
        "voice_design_template": voice_prompt,
        "voice_gender": _opt_str_any("voice_gender", "gender"),
        "voice_age": _opt_float("voice_age"),
        "voice_design_temperature": _opt_float("voice_design_temperature"),
    }
    by_key = _parse_voice_design_by_key()
    if by_key:
        opts["voice_design_by_key"] = by_key
    return {k: v for k, v in opts.items() if v is not None}


@bp.get("/health")
def health():
    cfg = _cfg()
    store = _store()
    return jsonify({
        "status": "ok",
        "service": "speechlab",
        "env": cfg.env,
        "active_job": store.active_job_id(),
    })


@bp.post("/api/v1/dub")
def create_dub():
    """Запуск дубляжа: multipart (video) или JSON {video_path, project_name, ...}.

    Промпт VoiceDesign (опционально):
      voice_prompt / voice_design_template — шаблон с плейсхолдерами
        {lang}, {gender_hint}, {age_hint}
      voice_gender — male | female (для всех реплик)
      voice_age — возраст в годах (число)
      voice_design_temperature — 0..1
      voice_design_by_key — JSON, напр. {"male_mature": "deep baritone narrator"}
    """
    cfg = _cfg()
    store = _store()

    if request.is_json:
        data = request.get_json(silent=True) or {}
        project_name = (data.get("project_name") or "").strip()
        source_lang = (data.get("source_language") or data.get("source_lang") or "").strip()
        target_lang = (data.get("target_language") or data.get("target_lang") or "").strip()
        video_raw = (data.get("video_path") or "").strip()
        if not all([project_name, source_lang, target_lang, video_raw]):
            return jsonify({
                "error": "Нужны project_name, source_language, target_language, video_path",
            }), 400
        video_path = Path(video_raw).resolve()
        options = _parse_options()
    else:
        project_name = (request.form.get("project_name") or "").strip()
        source_lang = (request.form.get("source_language") or request.form.get("source_lang") or "").strip()
        target_lang = (request.form.get("target_language") or request.form.get("target_lang") or "").strip()
        if not all([project_name, source_lang, target_lang]):
            return jsonify({"error": "Нужны project_name, source_language, target_language"}), 400
        if "video" not in request.files:
            return jsonify({"error": "Загрузите файл video (multipart) или передайте video_path в JSON"}), 400
        f = request.files["video"]
        if not f.filename:
            return jsonify({"error": "Пустое имя файла video"}), 400
        ext = Path(f.filename).suffix.lower()
        if ext not in _ALLOWED_VIDEO:
            return jsonify({"error": f"Неподдерживаемый формат: {ext}"}), 400
        safe = secure_filename(f.filename) or f"upload{ext}"
        video_path = cfg.upload_dir / f"{project_name}_{safe}"
        f.save(video_path)
        options = _parse_options()

    if not _PROJECT_RE.match(project_name):
        return jsonify({"error": "project_name: только буквы, цифры, _ и - (до 64)"}), 400
    if not allowed_video_path(video_path, cfg.video_roots):
        return jsonify({"error": "video_path вне разрешённых каталогов upload/projects"}), 403

    job = store.enqueue(project_name, video_path, source_lang, target_lang, options)
    if not job:
        return jsonify({
            "error": "Пайплайн уже выполняется",
            "active_job_id": store.active_job_id(),
        }), 503

    proc = start_job(job, cfg.projects_root, cfg.logs_dir)
    store.update(job.id, pid=proc.pid)
    return jsonify(store.to_dict(store.get(job.id))), 202


@bp.get("/api/v1/jobs")
def list_jobs():
    return jsonify([_store().to_dict(j) for j in _store().list_jobs()])


@bp.get("/api/v1/jobs/<job_id>")
def get_job(job_id: str):
    job = _store().get(job_id)
    if not job:
        return jsonify({"error": "job не найден"}), 404
    return jsonify(_store().to_dict(job))


@bp.get("/api/v1/jobs/<job_id>/download")
def download_job(job_id: str):
    cfg = _cfg()
    job = _store().get(job_id)
    if not job:
        return jsonify({"error": "job не найден"}), 404
    if job.status != JobStatus.done or not job.result_path:
        return jsonify({"error": "Результат ещё не готов", "status": job.status.value}), 409
    path = Path(job.result_path)
    if not allowed_result_path(path, cfg.projects_root):
        return jsonify({"error": "Недопустимый путь результата"}), 403
    return send_file(path, as_attachment=True, download_name=path.name)
