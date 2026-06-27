"""REST API дубляжа.

Эндпоинты:
  GET  /health              — проверка сервиса
  POST /api/v1/dub          — создать задачу (multipart video или JSON video_path)
  GET  /api/v1/jobs         — список задач
  GET  /api/v1/jobs/<id>    — статус
  GET  /api/v1/jobs/<id>/download — скачать MP4 при status=done
"""
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
_ALLOWED_VOICE_SAMPLE = {".mp3", ".wav"}
_VOICE_AGE_GROUPS = frozenset({"child", "teenager", "mature", "elderly"})
_MAX_VOICE_SAMPLE_BYTES = 10 * 1024 * 1024


# --- Доступ к конфигу и JobStore из Flask current_app ---
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


def _opt_bool(key: str) -> bool | None:
    v = request.form.get(key)
    if v is None:
        v = _json_body().get(key)
    if v is None or v == "":
        return None
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{key}: ожидается true/false, получено {v!r}")


def _opt_str(key: str):
    v = request.form.get(key)
    if v is None:
        v = _json_body().get(key)
    return (v or "").strip() or None


# --- Парсинг полей формы / JSON (multipart и application/json) ---
def _opt_str_any(*keys: str) -> str | None:
    for key in keys:
        v = _opt_str(key)
        if v:
            return v
    return None


# --- VoiceDesign: шаблоны промптов по ключу (male_mature и т.д.) ---
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


# --- Валидация возрастных групп для клонирования голоса ---
def _parse_age_groups(raw: str | list | None) -> list[str] | None:
    """None → все 4 возрастные группы."""
    if raw is None or raw == "":
        return None
    if isinstance(raw, list):
        items = [str(x).strip().lower() for x in raw if str(x).strip()]
    else:
        text = str(raw).strip()
        if text.startswith("["):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    items = [str(x).strip().lower() for x in parsed if str(x).strip()]
                else:
                    return None
            except json.JSONDecodeError:
                return None
        else:
            items = [p.strip().lower() for p in re.split(r"[,;\s]+", text) if p.strip()]
    bad = [x for x in items if x not in _VOICE_AGE_GROUPS]
    if bad:
        raise ValueError(
            f"age_groups: недопустимо {bad!r}, ожидается: {sorted(_VOICE_AGE_GROUPS)}"
        )
    return items or None


def _norm_clone_gender(raw: str | None) -> str | None:
    if not raw:
        return None
    g = raw.strip().lower()
    if g in ("m", "male", "man"):
        return "male"
    if g in ("f", "female", "woman"):
        return "female"
    raise ValueError(f"gender: ожидается male/female, получено {raw!r}")


# --- Сохранение загруженного voice sample (mp3/wav до 10 МБ) ---
def _save_voice_sample_file(cfg, project_name: str, gender: str, f) -> Path:
    if not f or not f.filename:
        raise ValueError(f"voice_sample_{gender}: пустой файл")
    ext = Path(f.filename).suffix.lower()
    if ext not in _ALLOWED_VOICE_SAMPLE:
        raise ValueError(
            f"voice_sample_{gender}: формат {ext!r}, нужен .mp3 или .wav"
        )
    dest_dir = cfg.upload_dir / "voice_samples"
    dest_dir.mkdir(parents=True, exist_ok=True)
    safe = secure_filename(f.filename) or f"sample{ext}"
    dest = dest_dir / f"{project_name}_{gender}_{safe}"
    f.save(dest)
    if dest.stat().st_size > _MAX_VOICE_SAMPLE_BYTES:
        dest.unlink(missing_ok=True)
        raise ValueError(
            f"voice_sample_{gender}: файл больше {_MAX_VOICE_SAMPLE_BYTES // (1024 * 1024)} МБ"
        )
    return dest.resolve()


def _append_clone_sample(
    samples: list[dict],
    *,
    gender: str,
    sample_path: Path,
    age_groups: list[str] | None,
    ref_text: str | None,
) -> None:
    samples.append({
        "gender": gender,
        "path": str(sample_path),
        "age_groups": age_groups,
        "ref_text": ref_text,
    })


# --- Клонирование голоса: JSON-массив, form JSON или voice_sample_male/female ---
def _parse_voice_clone_samples(cfg, project_name: str) -> list[dict] | None:
    """Сэмплы клонирования: multipart-файлы, JSON-массив или form JSON."""
    samples: list[dict] = []
    errors: list[str] = []

    raw_json = _json_body().get("voice_clone_samples")
    if isinstance(raw_json, list):
        for i, item in enumerate(raw_json):
            if not isinstance(item, dict):
                errors.append(f"voice_clone_samples[{i}]: ожидается объект")
                continue
            try:
                gender = _norm_clone_gender(item.get("gender"))
                if not gender:
                    errors.append(f"voice_clone_samples[{i}]: нужен gender")
                    continue
                path_raw = (item.get("sample_path") or item.get("path") or "").strip()
                if not path_raw:
                    errors.append(f"voice_clone_samples[{i}]: нужен sample_path")
                    continue
                path = Path(path_raw).resolve()
                if not allowed_video_path(path, cfg.video_roots):
                    errors.append(f"voice_clone_samples[{i}]: путь вне upload/projects")
                    continue
                ages = _parse_age_groups(item.get("age_groups"))
                ref_text = (item.get("ref_text") or "").strip() or None
                _append_clone_sample(
                    samples, gender=gender, sample_path=path,
                    age_groups=ages, ref_text=ref_text,
                )
            except ValueError as exc:
                errors.append(str(exc))
        if errors:
            raise ValueError("; ".join(errors))
        return samples or None

    form_json = request.form.get("voice_clone_samples")
    if form_json:
        try:
            parsed = json.loads(form_json)
        except json.JSONDecodeError:
            raise ValueError("voice_clone_samples: невалидный JSON") from None
        if isinstance(parsed, list):
            for i, item in enumerate(parsed):
                if not isinstance(item, dict):
                    errors.append(f"voice_clone_samples[{i}]: ожидается объект")
                    continue
                gender = _norm_clone_gender(item.get("gender"))
                if not gender:
                    errors.append(f"voice_clone_samples[{i}]: нужен gender")
                    continue
                file_key = item.get("file_field") or f"voice_sample_{gender}"
                ref_text = (item.get("ref_text") or "").strip() or None
                try:
                    ages = _parse_age_groups(item.get("age_groups"))
                except ValueError as exc:
                    errors.append(str(exc))
                    continue
                if file_key in request.files and request.files[file_key].filename:
                    path = _save_voice_sample_file(cfg, project_name, gender, request.files[file_key])
                    _append_clone_sample(
                        samples, gender=gender, sample_path=path,
                        age_groups=ages, ref_text=ref_text,
                    )
                elif item.get("sample_path") or item.get("path"):
                    path = Path((item.get("sample_path") or item.get("path")).strip()).resolve()
                    if not allowed_video_path(path, cfg.video_roots):
                        errors.append(f"voice_clone_samples[{i}]: путь вне upload/projects")
                        continue
                    _append_clone_sample(
                        samples, gender=gender, sample_path=path,
                        age_groups=ages, ref_text=ref_text,
                    )
                else:
                    errors.append(f"voice_clone_samples[{i}]: нет файла или sample_path")

    for gender in ("male", "female"):
        file_key = f"voice_sample_{gender}"
        if file_key not in request.files or not request.files[file_key].filename:
            continue
        try:
            path = _save_voice_sample_file(cfg, project_name, gender, request.files[file_key])
            ages = _parse_age_groups(request.form.get(f"voice_sample_{gender}_ages"))
            ref_text = _opt_str(f"voice_sample_{gender}_ref_text")
            _append_clone_sample(
                samples, gender=gender, sample_path=path,
                age_groups=ages, ref_text=ref_text,
            )
        except ValueError as exc:
            errors.append(str(exc))

    if errors:
        raise ValueError("; ".join(errors))
    return samples or None


# --- Silero TTS (только target_language ru) ---
def _validate_silero_voices(raw: list) -> list[dict]:
    """Проверка silero_voices: speaker, gender?, age_groups?."""
    from tools import silero_tts

    out: list[dict] = []
    errors: list[str] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            errors.append(f"silero_voices[{i}]: ожидается объект")
            continue
        try:
            spk = silero_tts.normalize_speaker(item.get("speaker") or "")
        except ValueError as exc:
            errors.append(f"silero_voices[{i}]: {exc}")
            continue
        gender = item.get("gender")
        if gender is not None:
            try:
                gender = _norm_clone_gender(gender)
            except ValueError as exc:
                errors.append(f"silero_voices[{i}]: {exc}")
                continue
        try:
            ages = _parse_age_groups(item.get("age_groups"))
        except ValueError as exc:
            errors.append(f"silero_voices[{i}]: {exc}")
            continue
        entry: dict = {"speaker": spk}
        if gender:
            entry["gender"] = gender
        if ages is not None:
            entry["age_groups"] = ages
        out.append(entry)
    if errors:
        raise ValueError("; ".join(errors))
    return out


def _parse_silero_voices() -> list[dict] | None:
    raw = _json_body().get("silero_voices")
    if isinstance(raw, list) and raw:
        return _validate_silero_voices(raw)
    form_raw = request.form.get("silero_voices")
    if form_raw:
        try:
            parsed = json.loads(form_raw)
        except json.JSONDecodeError:
            raise ValueError("silero_voices: невалидный JSON") from None
        if isinstance(parsed, list) and parsed:
            return _validate_silero_voices(parsed)
    return None


def _parse_silero_options() -> dict:
    """Silero v5_ru: speaker, all_replicas, age_groups, silero_voices."""
    opts: dict = {}
    speaker = _opt_str_any("silero_speaker", "silero_voice")
    if speaker:
        opts["silero_speaker"] = speaker
    all_rep = _opt_bool("silero_all_replicas")
    if all_rep is not None:
        opts["silero_all_replicas"] = all_rep
    raw_ages = request.form.get("silero_age_groups")
    if raw_ages is None:
        raw_ages = _json_body().get("silero_age_groups")
    ages = _parse_age_groups(raw_ages)
    if ages is not None:
        opts["silero_age_groups"] = ages
    voices = _parse_silero_voices()
    if voices:
        opts["silero_voices"] = voices
    return opts


# --- Опции дубляжа: громкость, VoiceDesign, температура ---
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


# --- Сборка всех опций для передачи в job.options → main.run() ---
def _merge_options(project_name: str) -> dict:
    opts = _parse_options()
    clone_samples = _parse_voice_clone_samples(_cfg(), project_name)
    if clone_samples:
        opts["voice_clone_samples"] = clone_samples
    try:
        silero = _parse_silero_options()
    except ValueError:
        raise
    opts.update(silero)
    return opts


# --- GET /health — без API key, для nginx/monitoring ---
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


# --- POST /api/v1/dub — создание задачи и запуск subprocess ---
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

    Клонирование голоса из аудио-сэмпла (опционально, mp3/wav до 10 МБ):
      voice_sample_male / voice_sample_female — файлы
      voice_sample_male_ages / voice_sample_female_ages — child,teenager,mature,elderly
        (пусто = все 4 группы)
      voice_sample_male_ref_text / voice_sample_female_ref_text — транскрипт сэмпла
      voice_clone_samples — JSON-массив (расширенный формат)

    Silero TTS (только target_language ru):
      silero_speaker — aidar | baya | eugene | kseniya | xenia
      silero_all_replicas — true: все реплики одним спикером
      silero_age_groups — child,teenager,mature,elderly (пусто = все)
      silero_voices — JSON [{speaker, gender?, age_groups?}, ...]
    """
    cfg = _cfg()
    store = _store()

    # Ветка 1: JSON с video_path (файл уже на сервере)
    # Ветка 2: multipart — сохраняем upload в server/uploads/
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
        try:
            options = _merge_options(project_name)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
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
        try:
            options = _merge_options(project_name)
        except ValueError as exc:
            video_path.unlink(missing_ok=True)
            return jsonify({"error": str(exc)}), 400

    # Валидация имени проекта и пути к видео (security.allowed_video_path)
    if not _PROJECT_RE.match(project_name):
        return jsonify({"error": "project_name: только буквы, цифры, _ и - (до 64)"}), 400
    if not allowed_video_path(video_path, cfg.video_roots):
        return jsonify({"error": "video_path вне разрешённых каталогов upload/projects"}), 403

    # Очередь: одна GPU-задача; при занятости — 503
    job = store.enqueue(project_name, video_path, source_lang, target_lang, options)
    if not job:
        return jsonify({
            "error": "Пайплайн уже выполняется",
            "active_job_id": store.active_job_id(),
        }), 503

    proc = start_job(job, cfg.projects_root, cfg.logs_dir)
    store.update(job.id, pid=proc.pid)
    return jsonify(store.to_dict(store.get(job.id))), 202


# --- GET /api/v1/jobs — последние 50 задач ---
@bp.get("/api/v1/jobs")
def list_jobs():
    return jsonify([_store().to_dict(j) for j in _store().list_jobs()])


# --- GET /api/v1/jobs/<id> — опрос статуса (queued → running → done) ---
@bp.get("/api/v1/jobs/<job_id>")
def get_job(job_id: str):
    job = _store().get(job_id)
    if not job:
        return jsonify({"error": "job не найден"}), 404
    return jsonify(_store().to_dict(job))


# --- GET /api/v1/jobs/<id>/download — MP4 только при status=done ---
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
