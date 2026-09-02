"""SpeechLab — один пайплайн дубляжа (PRD.md).

Точки входа:
  CLI:  python main.py <project> <video> <src> <tgt>
  API:  server/run_job.py → run(...)

Поток (сверху вниз в этом файле):
  video → 16k → demucs → first_seg(40–90s) → ASR →
  перевод(+emo tags) → casting → Fish TTS(clone, 44.1k) →
  fit/timeline/full_dub/mux на OUT_SR (44.1k)

Resume (SPEECHLAB_RESUME=1):
  A — demucs/first_seg пропускаются, если vocals+segment хеши в pipeline_state совпали
  B — ASR reuse при том же segment.wav + source_language;
      перевод — ещё и тот же target_language;
      casting/TTS — ещё и тот же voice_fingerprint (cast/сэмплы/микс)

Отдельно: config/, server/, tools/ (TTS/LLM), prompt.py, asr.py.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import prompt
import soundfile as sf

from asr import init_asr_models, run_segment_pipeline, unload_asr_models
from config.env_config import get_hf_token
from tools import llm

ROOT = Path(__file__).resolve().parent
SR = 16000  # extract / demucs / ASR / casting
# Пост-TTS (Fish уже 44.1k): таймлайн, full_dub, микс с оригиналом
OUT_SR = int(os.environ.get("SPEECHLAB_OUT_SR", "44100"))

SPEECH_TXT = re.compile(r"^speech_(\d+)\.txt$", re.I)
TIME_LINE = re.compile(r"^\s*([\d.]+)\s*-\s*([\d.]+)\s*$")
TTS_LANGUAGE = {
    "ru": "Russian", "russian": "Russian",
    "en": "English", "english": "English",
    "de": "German", "german": "German",
    "es": "Spanish", "spanish": "Spanish",
    "fr": "French", "french": "French",
    "auto": "Auto",
}
_AUDIO_ONLY = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".aac"}

MIN_PRIMARY_SEC = float(os.environ.get("SPEECHLAB_MIN_PRIMARY_SEC", "40"))
MAX_PRIMARY_SEC = float(os.environ.get("SPEECHLAB_MAX_PRIMARY_SEC", "90"))
ORIGINAL_AUDIO_RATIO = float(os.environ.get("SPEECHLAB_ORIGINAL_AUDIO_RATIO", "0.3"))
DUB_VOLUME_PERCENT = float(os.environ.get("SPEECHLAB_DUB_VOLUME_PERCENT", "100"))
TRANSLATE_BATCH_SIZE = int(os.environ.get("SPEECHLAB_TRANSLATE_BATCH_SIZE", "12"))
RESUME = os.environ.get("SPEECHLAB_RESUME", "1").lower() not in ("0", "false", "no")
CAST_PER_SPEAKER = os.environ.get("SPEECHLAB_CAST_PER_SPEAKER", "1").lower() not in ("0", "false", "no")
SKIP_DEMUCS = os.environ.get("SPEECHLAB_SKIP_DEMUCS", "0").lower() in ("1", "true", "yes")
STATE_FILE = "pipeline_state.json"


# =============================================================================
# Утилиты: ffmpeg, WAV, файлы реплик, resume
# =============================================================================
def _run(cmd, *, check=True):
    print("$", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        if r.stderr:
            print(r.stderr[-4000:])
        if check:
            raise subprocess.CalledProcessError(r.returncode, cmd, r.stdout, r.stderr)
    return r


def _tts_lang(code):
    return TTS_LANGUAGE.get(code.strip().lower(), code)


def _duration(path):
    info = sf.info(path)
    return info.frames / info.samplerate


def _has_video(path):
    if path.suffix.lower() in _AUDIO_ONLY:
        return False
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    return r.returncode == 0 and "video" in (r.stdout or "").lower()


def _resample_mono(data, sr, target=SR):
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr == target:
        return data.astype(np.float32)
    import torch
    import torchaudio
    t = torch.from_numpy(data.astype(np.float32)).unsqueeze(0)
    return torchaudio.functional.resample(t, sr, target).squeeze(0).numpy()


def _projects_root(root=None):
    if root:
        return Path(root).resolve()
    env = os.environ.get("SPEECHLAB_PROJECTS_ROOT", "").strip()
    return Path(env).resolve() if env else ROOT


def _sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _list_txt(d):
    files = [p for p in d.glob("speech_*.txt") if SPEECH_TXT.match(p.name)]
    return sorted(files, key=lambda p: int(SPEECH_TXT.match(p.name).group(1)))


def _parse_txt(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        return None
    m = TIME_LINE.match(lines[0])
    if not m:
        return None
    return {
        "start": float(m.group(1)), "end": float(m.group(2)),
        "speaker": lines[1].strip(), "text": "\n".join(lines[2:]).strip(),
        "file": path.name,
    }


def _idx(name):
    return int(SPEECH_TXT.match(name).group(1))


def _load_state(project_dir: Path) -> dict:
    p = project_dir / STATE_FILE
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _resume_ok(project_dir, video_path):
    """True, если SPEECHLAB_RESUME=1 и входное видео не менялось (SHA256)."""
    if not RESUME:
        return False
    prev = _load_state(project_dir)
    if not prev:
        return False
    if prev.get("input_sha256") != _sha256(video_path):
        print("  resume: входной файл изменился")
        return False
    return True


def _voice_fingerprint(
    *,
    voice_gender=None,
    voice_age=None,
    voice_clone_samples=None,
    cast_voice=None,
    cast_mode=None,
    dub_volume_percent=None,
    original_audio_ratio=None,
) -> str:
    """Отпечаток опций голоса/микса — смена → нельзя resume casting/TTS."""
    clone_meta = []
    for s in voice_clone_samples or []:
        clone_meta.append({
            "gender": s.get("gender"),
            "path": str(s.get("path") or ""),
            "age_groups": s.get("age_groups"),
            "ref_text": s.get("ref_text"),
        })
    payload = {
        "gender": voice_gender,
        "age": voice_age,
        "clone": clone_meta,
        "cast_voice": (cast_voice or "").strip().lower() or None,
        "cast_mode": (cast_mode or "").strip().lower() or None,
        "dub_vol": dub_volume_percent,
        "orig_ratio": original_audio_ratio,
        "tts": "fish",
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _save_state(project_dir, video_path, **extra):
    """pipeline_state.json: хеш видео + vocals + каждого segment.wav (для A+B resume)."""
    data = {
        "input_sha256": _sha256(video_path),
        "video_path": str(video_path.resolve()),
        **extra,
    }
    (project_dir / STATE_FILE).write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8",
    )


def _segment_hashes(first_seg_dir: Path, manifest: list[dict]) -> dict[str, str]:
    """folder → sha256(segment.wav) — вход ASR; смена хеша → нельзя reuse ASR."""
    out = {}
    for item in manifest:
        folder = item["folder"]
        seg = first_seg_dir / folder / "segment.wav"
        if seg.is_file():
            out[folder] = _sha256(seg)
    return out


def _load_manifest(first_seg_dir: Path) -> list[dict] | None:
    path = first_seg_dir / "manifest.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, list) and data else None


def _can_skip_demucs_and_seg(
    project_dir: Path,
    project_vocals: Path,
    stems_dir: Path,
    first_seg_dir: Path,
    state: dict,
) -> tuple[bool, list[dict] | None, Path | None]:
    """Вариант A: пропуск шагов 1–2, если vocals/music/manifest/сегменты на месте
    и хеши совпадают с pipeline_state (вариант B).
    """
    music_ok = list(stems_dir.rglob("no_vocals.wav")) if stems_dir.is_dir() else []
    if not project_vocals.is_file() or not music_ok:
        return False, None, None
    manifest = _load_manifest(first_seg_dir)
    if not manifest:
        return False, None, None
    for item in manifest:
        if not (first_seg_dir / item["folder"] / "segment.wav").is_file():
            return False, None, None
    # Старый state без хешей сегментов — не доверяем, пересоберём 1–2
    saved_vocals = state.get("vocals_sha256")
    saved_segs = state.get("segment_sha256") or {}
    if not saved_vocals or not saved_segs:
        return False, None, None
    if saved_vocals != _sha256(project_vocals):
        print("  resume: vocals.wav изменился — demucs/нарезка заново")
        return False, None, None
    cur_segs = _segment_hashes(first_seg_dir, manifest)
    if cur_segs != saved_segs:
        print("  resume: segment.wav изменились — demucs/нарезка заново")
        return False, None, None
    return True, manifest, music_ok[0]


def _step_done(second_seg, step):
    if step == "casting":
        return (second_seg / "casting.json").is_file()
    if step == "asr":
        td, ad = second_seg / "output_text_segments", second_seg / "output_audio_segments"
        if not td.is_dir() or not ad.is_dir():
            return False
        txts = _list_txt(td)
        return bool(txts) and all(list(ad.glob(f"speech_{_idx(t.name):03d}_*.wav")) for t in txts)
    if step == "translate":
        tgt, src = second_seg / "target_text", second_seg / "output_text_segments"
        if not tgt.is_dir() or not src.is_dir():
            return False
        items = [t for t in _list_txt(src) if _parse_txt(t)]
        return bool(items) and all((tgt / t.name).is_file() for t in items)
    if step == "dub":
        tgt, fin = second_seg / "target_text", second_seg / "final_audio"
        if not tgt.is_dir() or not fin.is_dir():
            return False
        txts = [t for t in _list_txt(tgt) if _parse_txt(t)]
        return bool(txts) and all(list(fin.glob(f"speech_{_idx(t.name):03d}_*_dub.wav")) for t in txts)
    raise ValueError(step)


# =============================================================================
# Шаг 1: аудио 16 kHz + demucs
# =============================================================================
def extract_audio(video_path, out_wav, *, ar=SR):
    """Вытащить mono PCM WAV с заданной частотой (ASR=16k, финальный микс=OUT_SR)."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    _run(["ffmpeg", "-y", "-i", str(video_path),
          "-vn", "-acodec", "pcm_s16le", "-ar", str(ar), "-ac", "1", str(out_wav)])


def extract_audio_16k(video_path, out_wav):
    """Совместимость: вход demucs/ASR всегда 16 kHz."""
    return extract_audio(video_path, out_wav, ar=SR)


def separate_stems(audio_wav, stems_dir):
    stems_dir.mkdir(parents=True, exist_ok=True)
    _run([sys.executable, "-m", "demucs", "--two-stems=vocals", "-o", str(stems_dir), str(audio_wav)])
    vocals = next(stems_dir.rglob("vocals.wav"), None)
    if not vocals:
        raise FileNotFoundError(f"demucs не создал vocals.wav в {stems_dir}")
    no_vocals = vocals.parent / "no_vocals.wav"
    if not no_vocals.is_file():
        raise FileNotFoundError(f"Нет no_vocals.wav рядом с {vocals}")
    return vocals, no_vocals


# =============================================================================
# Шаг 2: первичная нарезка 40–90 с по тишине
# =============================================================================
def _split_long(start, end, max_len):
    out, s = [], start
    while end - s > max_len:
        out.append((s, s + max_len))
        s += max_len
    if end > s + 0.05:
        out.append((s, end))
    return out


def _adjust_bounds(bounds, min_len, max_len):
    if len(bounds) < 2:
        return bounds
    segs = []
    for i in range(len(bounds) - 1):
        segs.extend(_split_long(bounds[i], bounds[i + 1], max_len))
    changed = True
    while changed:
        changed, merged, i = False, [], 0
        while i < len(segs):
            s, e = segs[i]
            while (e - s) < min_len and i + 1 < len(segs):
                _, ne = segs[i + 1]
                if ne - s <= max_len:
                    e, i, changed = ne, i + 1, True
                else:
                    break
            merged.append((s, e))
            i += 1
        segs = merged
    out = [segs[0][0]]
    for _, e in segs:
        if e > out[-1] + 0.05:
            out.append(e)
    return out


def split_primary_segments(vocals_wav, first_seg_dir):
    first_seg_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["ffmpeg", "-i", str(vocals_wav), "-af", "silencedetect=n=-30dB:d=0.5", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    duration = _duration(vocals_wav)
    points = [0.0]
    for line in proc.stderr.splitlines():
        if "silence_end:" in line:
            try:
                t = float(line.split("silence_end:")[1].strip().split()[0])
                if 0 < t < duration:
                    points.append(t)
            except ValueError:
                pass
    if points[-1] < duration - 0.01:
        points.append(duration)
    bounds = _adjust_bounds(sorted(set(points)), MIN_PRIMARY_SEC, MAX_PRIMARY_SEC)

    manifest = []
    for i in range(len(bounds) - 1):
        start, end = bounds[i], bounds[i + 1]
        if end - start < 0.3:
            continue
        folder = first_seg_dir / f"{i + 1:03d}_{start:.2f}-{end:.2f}"
        folder.mkdir(parents=True, exist_ok=True)
        seg = folder / "segment.wav"
        _run(["ffmpeg", "-y", "-i", str(vocals_wav), "-ss", str(start), "-to", str(end),
              "-acodec", "pcm_s16le", "-ar", str(SR), "-ac", "1", str(seg)])
        manifest.append({"index": i + 1, "start": start, "end": end, "folder": folder.name})
    (first_seg_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Первичных сегментов: {len(manifest)}")
    return manifest


# =============================================================================
# Шаг 4: перевод LLM
# =============================================================================
def _parse_batch(raw):
    s = (raw or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.I)
        s = re.sub(r"\s*```\s*$", "", s).strip()
    i, j = s.find("["), s.rfind("]")
    if i != -1 and j > i:
        s = s[i:j + 1]
    data = json.loads(s)
    if not isinstance(data, list):
        raise ValueError("Ожидался JSON-массив")
    return data


def translate_segments(second_seg, source_lang, target_lang):
    text_dir = second_seg / "output_text_segments"
    target_dir = second_seg / "target_text"
    target_dir.mkdir(parents=True, exist_ok=True)
    entries = [(t, m) for t in _list_txt(text_dir) if (m := _parse_txt(t)) and m["text"]]

    for i in range(0, len(entries), TRANSLATE_BATCH_SIZE):
        batch = entries[i:i + TRANSLATE_BATCH_SIZE]
        lines = [{
            "id": txt.name, "text": meta["text"],
            "source_chars": len(meta["text"]), "source_words": len(meta["text"].split()),
            "slot_sec": round(float(meta["end"]) - float(meta["start"]), 2),
        } for txt, meta in batch]
        by_id = {}
        try:
            raw = llm.llm_response_retry(
                prompt.get_prompt(3, {"source_lang": source_lang, "target_lang": target_lang, "lines": lines}),
                json_only=False, batch_translate=True, retries=3,
            )
            for item in _parse_batch(raw):
                rid, text = str(item.get("id", "")).strip(), str(item.get("text", "")).strip()
                if rid and text:
                    by_id[rid] = text
        except Exception as exc:
            print(f"  batch перевод не удался ({exc}), по одной…")

        for txt, meta in batch:
            translated = (by_id.get(txt.name) or "").strip()
            if not translated:
                # соседние реплики батча — контекст для эмо-тегов
                neighbors = [
                    f"{m['speaker']}: {m['text']}"
                    for _, m in batch if m["text"]
                ]
                try:
                    translated = (llm.llm_response_retry(prompt.get_prompt(2, {
                        "text": meta["text"], "source_lang": source_lang, "target_lang": target_lang,
                        "source_chars": len(meta["text"]), "source_words": len(meta["text"].split()),
                        "slot_sec": float(meta["end"]) - float(meta["start"]),
                        "context": "\n".join(neighbors),
                    }), retries=3) or "").strip()
                except Exception as exc:
                    print(f"  перевод {txt.name} ошибка ({exc})")
                    translated = ""
            # Пустой ответ LLM → не пишем пустой target (TTS упадёт); берём исходник
            if not translated:
                print(f"  перевод {txt.name}: пусто → исходный текст")
                translated = meta["text"]
            # убрать случайные кавычки/markdown вокруг ответа
            if translated.startswith("```"):
                translated = re.sub(r"^```(?:\w+)?\s*", "", translated)
                translated = re.sub(r"\s*```\s*$", "", translated).strip()
            if len(translated) >= 2 and translated[0] == translated[-1] and translated[0] in "\"'":
                translated = translated[1:-1].strip()
            (target_dir / txt.name).write_text(
                f"{meta['start']:.2f} - {meta['end']:.2f}\n{meta['speaker']}\n{translated}",
                encoding="utf-8")
        print(f"  перевод batch {i // TRANSLATE_BATCH_SIZE + 1} ({len(batch)} реплик)")
    return target_dir


# =============================================================================
# Шаг 5: casting (пол/возраст)
# =============================================================================
def build_casting(second_seg, *, unload=True):
    from tools import dubbing
    from tools.cast_voices import assign_cast_to_speakers
    from tools.get_param import profile_from_wav, unload_model

    audio_dir = second_seg / "output_audio_segments"
    casting = {"segments": {}}
    entries = []
    for txt in _list_txt(second_seg / "output_text_segments"):
        meta = _parse_txt(txt)
        if not meta:
            continue
        wavs = list(audio_dir.glob(f"speech_{_idx(txt.name):03d}_*.wav"))
        if not wavs:
            continue
        entries.append({
            "txt": txt, "meta": meta, "wav": wavs[0], "speaker": meta["speaker"],
            "dur": float(meta["end"]) - float(meta["start"]),
        })

    def entry(e, profile):
        return {
            "file": e["txt"].name, "speaker": e["speaker"], "wav": e["wav"].name,
            "start": e["meta"]["start"], "end": e["meta"]["end"], "profile": profile,
        }

    # Режим speakers: каждому спикеру — cast_id из data/Cast (Локи / Том Харди / Тор)
    cast_map = {}
    if dubbing.cast_mode() == "speakers" and dubbing.cast_ids():
        speakers = sorted({e["speaker"] for e in entries})
        cast_map = assign_cast_to_speakers(speakers)
        print(f"  cast speakers: {cast_map}")

    if CAST_PER_SPEAKER and entries:
        by_spk = {}
        for e in entries:
            by_spk.setdefault(e["speaker"], []).append(e)
        profiles = {}
        for spk, items in by_spk.items():
            p = dubbing.apply_voice_override(
                profile_from_wav(max(items, key=lambda x: x["dur"])["wav"]))
            if spk in cast_map:
                p["cast_id"] = cast_map[spk]
            profiles[spk] = p
        for e in entries:
            casting["segments"][e["txt"].name] = entry(e, profiles[e["speaker"]])
    else:
        for e in entries:
            p = dubbing.apply_voice_override(profile_from_wav(e["wav"]))
            if e["speaker"] in cast_map:
                p["cast_id"] = cast_map[e["speaker"]]
            casting["segments"][e["txt"].name] = entry(e, p)

    (second_seg / "casting.json").write_text(
        json.dumps(casting, ensure_ascii=False, indent=2), encoding="utf-8")
    if unload:
        unload_model()
    return casting


# =============================================================================
# Шаг 6: TTS + нормализация громкости дубляжа по оригинальной реплике
# =============================================================================
def dub_segments(second_seg, target_lang, *, unload=True, bank_ready=False):
    from tools import dubbing
    from tools.dubbing import dub_from_profile, ensure_voice_bank, unload_model
    from tools.fit_audio import match_loudness

    lang = _tts_lang(target_lang)
    if not bank_ready:
        print("  TTS: Fish voice bank…")
        ensure_voice_bank()
        bank_ready = True

    casting = json.loads((second_seg / "casting.json").read_text(encoding="utf-8"))
    seg_map = casting.get("segments", {})
    target_dir = second_seg / "target_text"
    final_dir = second_seg / "final_audio"
    final_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = second_seg / "output_audio_segments"

    for txt in _list_txt(target_dir):
        meta = _parse_txt(txt)
        if not meta:
            continue
        if not (meta.get("text") or "").strip():
            print(f"  пропуск {txt.name}: пустой текст (нет озвучки)")
            continue
        profile = dubbing.apply_voice_override(seg_map.get(txt.name, {}).get("profile", {}))
        wavs = list(audio_dir.glob(f"speech_{_idx(txt.name):03d}_*.wav"))
        ref = wavs[0].stem if wavs else txt.stem
        out = final_dir / f"{ref}_dub.wav"
        dub_from_profile(meta["text"], lang, profile, out, bank_ready=bank_ready)
        # Громкость дубляжа → как у исходной speech_*.wav (чтобы не тонул в шуме)
        if wavs:
            gain = match_loudness(out, wavs[0])
            print(f"  озвучка {out.name} (loudness×{gain:.2f})")
        else:
            print(f"  озвучка {out.name}")
    if unload:
        unload_model()
    return final_dir

# =============================================================================
# Шаги 7–9: склейка, микс, mux (всё на OUT_SR — качество Fish TTS)
# =============================================================================
def _timeline(placements, min_dur):
    """Накладывает *_dub.wav на таймлайн первичного сегмента (OUT_SR)."""
    max_end = float(min_dur)
    for start, path in placements:
        max_end = max(max_end, start + _duration(path))
    out = np.zeros(int(round(max_end * OUT_SR)), dtype=np.float32)
    for start, path in placements:
        raw, sr = sf.read(path, dtype="float32")
        data = _resample_mono(raw, sr, target=OUT_SR)
        pos = int(round(start * OUT_SR))
        end = pos + len(data)
        if end > len(out):
            out = np.pad(out, (0, end - len(out)))
        out[pos:end] += data[: end - pos]
    peak = float(np.max(np.abs(out))) or 1.0
    return out / peak * 0.98 if peak > 1.0 else out


def restore_primary_segment(primary_dir):
    from tools.fit_audio import schedule_placements

    second = primary_dir / "second_seg"
    seg_wav = primary_dir / "segment.wav"
    slots = []
    if second.is_dir():
        fin, txt_dir = second / "final_audio", second / "output_text_segments"
        if fin.is_dir() and txt_dir.is_dir():
            for txt in _list_txt(txt_dir):
                meta = _parse_txt(txt)
                if not meta:
                    continue
                dub = list(fin.glob(f"speech_{_idx(txt.name):03d}_*_dub.wav"))
                if dub:
                    slots.append((meta["start"], meta["end"], dub[0], meta["speaker"]))

    out = primary_dir / "restored.wav"
    if not slots:
        # segment.wav = 16k → апсемпл до OUT_SR
        data, sr = sf.read(seg_wav, dtype="float32")
        sf.write(out, _resample_mono(data, sr, target=OUT_SR), OUT_SR)
        return out
    mixed = _timeline(schedule_placements(slots), _duration(seg_wav))
    sf.write(out, mixed, OUT_SR)
    print(f"  таймлайн: {len(slots)} реплик @ {OUT_SR} Hz")
    return out


def restore_full_vocals(project_dir, manifest, music_stem):
    """Собрать full_dub: restored (OUT_SR) + music stem (16k → OUT_SR)."""
    pieces, max_end = [], 0.0
    for item in manifest:
        folder = project_dir / "first_seg" / item["folder"]
        wav = folder / "restored.wav"
        if not wav.is_file():
            wav = folder / "segment.wav"
        data, sr = sf.read(wav, dtype="float32")
        pieces.append((float(item["start"]), _resample_mono(data, sr, target=OUT_SR)))
        max_end = max(max_end, float(item["end"]))

    n = int(round(max_end * OUT_SR))
    vocals = np.zeros(n, dtype=np.float32)
    for start, data in pieces:
        pos = int(round(start * OUT_SR))
        end = min(n, pos + len(data))
        vocals[pos:end] = data[: end - pos]

    music, sr_m = sf.read(music_stem, dtype="float32")
    music = _resample_mono(music, sr_m, target=OUT_SR)
    music = np.pad(music, (0, max(0, n - len(music))))[:n]
    mixed = vocals + music * 0.85
    peak = np.max(np.abs(mixed)) or 1.0
    if peak > 1.0:
        mixed = mixed / peak * 0.98
    out = project_dir / "full_dub.wav"
    sf.write(out, mixed, OUT_SR)
    return out


def mix_dub_with_original(video_path, dub_wav, out_wav, *, original_ratio=None, dub_volume_percent=None):
    """Микс full_dub + оригинал на OUT_SR (не даунсэмплим Fish)."""
    ratio = ORIGINAL_AUDIO_RATIO if original_ratio is None else original_ratio
    pct = DUB_VOLUME_PERCENT if dub_volume_percent is None else dub_volume_percent
    if pct <= 0:
        raise ValueError("Громкость дубляжа (%) должна быть > 0")
    gain = pct / 100.0

    if not _has_video(video_path):
        shutil.copy2(dub_wav, out_wav)
        return out_wav

    orig_tmp = out_wav.parent / "_original_mix.wav"
    extract_audio(video_path, orig_tmp, ar=OUT_SR)
    dub = _resample_mono(*sf.read(dub_wav, dtype="float32"), target=OUT_SR)
    orig = _resample_mono(*sf.read(orig_tmp, dtype="float32"), target=OUT_SR)
    n = max(len(dub), len(orig))
    dub = np.pad(dub, (0, max(0, n - len(dub))))[:n]
    orig = np.pad(orig, (0, max(0, n - len(orig))))[:n]
    mixed = dub * gain + orig * ratio
    peak = float(np.max(np.abs(mixed))) or 1.0
    if peak > 1.0:
        mixed = mixed / peak * 0.98
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_wav, mixed.astype(np.float32), OUT_SR)
    print(f"  микс @{OUT_SR} Hz: дубляж × {gain * 100:.0f}% + оригинал × {ratio:.2f}")
    return out_wav


def mux_video(video_path, audio_path, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not _has_video(video_path):
        dur = _duration(audio_path)
        _run(["ffmpeg", "-y", "-f", "lavfi",
              "-i", f"color=c=black:s=1280x720:r=24:d={dur:.3f}",
              "-i", str(audio_path),
              "-c:v", "libx264", "-pix_fmt", "yuv420p",
              "-c:a", "aac", "-b:a", "192k", "-shortest", str(out_path)])
        return out_path
    _run(["ffmpeg", "-y", "-i", str(video_path), "-i", str(audio_path),
          "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
          "-map", "0:v:0", "-map", "1:a:0", "-shortest", str(out_path)])
    return out_path


# =============================================================================
# Точка входа: run() — настройка голоса + 9 шагов подряд
# =============================================================================
def run(
    project_name,
    video_path,
    source_language,
    target_language,
    *,
    hf_token=None,
    dub_volume_percent=None,
    original_audio_ratio=None,
    voice_gender=None,
    voice_age=None,
    voice_clone_samples=None,
    cast_voice=None,
    cast_mode=None,
    projects_root=None,
):
    """Единая точка входа пайплайна (CLI и Flask worker)."""
    from tools import dubbing
    from tools.cast_voices import list_cast_voices, resolve_cast_voice, to_clone_sample
    from tools.dubbing import ensure_voice_bank, unload_model as unload_tts
    from tools.get_param import unload_model as unload_casting

    video_path = Path(video_path).resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    # Защита от path traversal в имени проекта (как в API)
    if not re.match(r"^[a-zA-Z0-9_-]{1,64}$", str(project_name or "")):
        raise ValueError("project_name: только буквы, цифры, _ и - (до 64)")

    project_dir = _projects_root(projects_root) / project_name
    project_dir.mkdir(parents=True, exist_ok=True)

    # --- встроенные cast-голоса (data/Cast): один пресет или раздача по спикерам ---
    clone_samples = list(voice_clone_samples or [])
    cast_list = None
    mode = (cast_mode or "").strip().lower() or None
    if (cast_voice or "").strip():
        one = resolve_cast_voice(cast_voice)
        # Cast первым: пользовательский voice_sample_* поверх перезапишет те же слоты
        clone_samples = [to_clone_sample(one, age_groups=None)] + clone_samples
        if not voice_gender:
            voice_gender = one["gender"]
        print(f"  cast_voice: {one['name']} ({one['id']})")
    elif mode == "speakers":
        cast_list = []
        for info in list_cast_voices():
            if not info["available"]:
                raise FileNotFoundError(f"cast sample missing: {info['sample_path']}")
            cast_list.append({
                "id": info["id"],
                "path": info["sample_path"],
                "ref_text": info["ref_text"],
            })
        print(f"  cast_mode=speakers: {[c['id'] for c in cast_list]}")
    elif mode:
        raise ValueError(f"cast_mode: ожидается speakers, получено {cast_mode!r}")

    # --- голос на этот прогон (сброс в finally); без clone/cast — ошибка ---
    use_bank = bool(clone_samples or cast_list)
    dubbing.set_voice_prompts(
        cache_dir=project_dir / "voice_bank" if use_bank else None,
        gender=voice_gender, age=voice_age,
    )
    dubbing.set_voice_clone_samples(clone_samples or None)
    dubbing.set_cast_voices(cast_list, mode=mode)
    dubbing.require_clone_sources()

    voice_fp = _voice_fingerprint(
        voice_gender=voice_gender,
        voice_age=voice_age,
        voice_clone_samples=clone_samples or None,
        cast_voice=cast_voice,
        cast_mode=mode,
        dub_volume_percent=dub_volume_percent,
        original_audio_ratio=original_audio_ratio,
    )

    token = hf_token or get_hf_token()
    try:
        first_seg_dir = project_dir / "first_seg"
        project_vocals = project_dir / "vocals.wav"
        stems_dir = project_dir / "demucs_stems"
        prev_state = _load_state(project_dir)
        resume_video = _resume_ok(project_dir, video_path)

        # Resume downstream: языки и голос должны совпадать с прошлым прогоном
        same_source = (
            (prev_state.get("source_language") or "").strip().lower()
            == (source_language or "").strip().lower()
        )
        same_target = (
            (prev_state.get("target_language") or "").strip().lower()
            == (target_language or "").strip().lower()
        )
        same_voice = prev_state.get("voice_fingerprint") == voice_fp
        if resume_video and not same_source:
            print("  resume: source_language изменился — ASR заново")
        if resume_video and same_source and not same_target:
            print("  resume: target_language изменился — перевод/TTS заново")
        if resume_video and same_source and same_target and not same_voice:
            print("  resume: опции голоса/микса изменились — casting/TTS заново")

        # --- 1–2: A — пропуск demucs/нарезки при совпадении хешей; иначе пересборка ---
        print("=== 1. Аудио 16 kHz + demucs ===")
        skip_12, manifest, music_stem = False, None, None
        if resume_video:
            skip_12, manifest, music_stem = _can_skip_demucs_and_seg(
                project_dir, project_vocals, stems_dir, first_seg_dir, prev_state,
            )

        if skip_12:
            print("  demucs + first_seg: пропуск (resume, хеши vocals/segment совпали)")
        else:
            music_ok = list(stems_dir.rglob("no_vocals.wav")) if stems_dir.is_dir() else []
            if SKIP_DEMUCS and project_vocals.is_file() and music_ok:
                music_stem = music_ok[0]
                print("  demucs: пропуск (SPEECHLAB_SKIP_DEMUCS)")
            else:
                audio_16k = project_dir / "audio_16k.wav"
                extract_audio_16k(video_path, audio_16k)
                vocals_wav, music_stem = separate_stems(audio_16k, stems_dir)
                project_vocals.write_bytes(vocals_wav.read_bytes())
                print(f"vocals: {project_vocals}")

            print("=== 2. Первичная нарезка ===")
            manifest = split_primary_segments(project_vocals, first_seg_dir)

        # Актуальные хеши входов ASR (вариант B)
        vocals_sha = _sha256(project_vocals) if project_vocals.is_file() else ""
        seg_hashes = _segment_hashes(first_seg_dir, manifest)
        saved_segs = prev_state.get("segment_sha256") or {}

        _save_state(
            project_dir, video_path,
            source_language=source_language,
            target_language=target_language,
            voice_fingerprint=voice_fp,
            vocals_sha256=vocals_sha,
            segment_sha256=seg_hashes,
        )

        # --- 3 ASR: reuse только если segment.wav и source_language те же ---
        print("=== 3. ASR ===")
        jobs = []
        # folder → можно ли reuse ASR (и все зависимые шаги 4–6)
        reuse_asr_for: dict[str, bool] = {}
        n_replicas = 0
        init_asr_models(source_language, token)
        try:
            for item in manifest:
                primary = first_seg_dir / item["folder"]
                second = primary / "second_seg"
                second.mkdir(parents=True, exist_ok=True)
                jobs.append((primary, second))
                folder = item["folder"]
                seg_same = (
                    resume_video
                    and bool(seg_hashes.get(folder))
                    and saved_segs.get(folder) == seg_hashes.get(folder)
                )
                can_reuse = seg_same and same_source and _step_done(second, "asr")
                reuse_asr_for[folder] = can_reuse
                print(f"--- {folder} ---")
                if can_reuse:
                    print("  ASR: resume (segment.wav не менялся)")
                    n_replicas += len(_list_txt(second / "output_text_segments"))
                else:
                    if resume_video and _step_done(second, "asr") and not seg_same:
                        print("  ASR: segment.wav изменился — пересчёт")
                    elif resume_video and _step_done(second, "asr") and not same_source:
                        print("  ASR: source_language изменился — пересчёт")
                    n_replicas += len(run_segment_pipeline(
                        primary / "segment.wav",
                        second / "output_audio_segments",
                        second / "output_text_segments",
                        source_language, hf_token=token, reuse_asr=True,
                    ))
        finally:
            unload_asr_models()

        if n_replicas == 0:
            raise ValueError("ASR не нашёл реплик. Проверьте source_language и vocals.")
        print(f"  реплик: {n_replicas}")

        # --- 4–6: downstream resume только если ASR reuse + те же язык/голос ---
        print("=== 4. Перевод ===")
        for primary, second in jobs:
            folder = primary.name
            if reuse_asr_for.get(folder) and same_target and _step_done(second, "translate"):
                print(f"  перевод {folder}: resume")
                continue
            translate_segments(second, source_language, target_language)

        print("=== 5. casting ===")
        for primary, second in jobs:
            folder = primary.name
            if (
                reuse_asr_for.get(folder)
                and same_target
                and same_voice
                and _step_done(second, "casting")
            ):
                print(f"  casting {folder}: resume")
                continue
            build_casting(second, unload=False)
        unload_casting()

        print("=== 6. TTS ===")
        ensure_voice_bank()
        bank_ready = True
        for primary, second in jobs:
            folder = primary.name
            if (
                reuse_asr_for.get(folder)
                and same_target
                and same_voice
                and _step_done(second, "translate")
                and _step_done(second, "casting")
                and _step_done(second, "dub")
            ):
                print(f"  TTS {folder}: resume")
                continue
            dub_segments(second, target_language, unload=False, bank_ready=bank_ready)
        unload_tts()

        # --- 7–9 ---
        print(f"=== 7. Склейка реплик @ {OUT_SR} Hz ===")
        for primary, _ in jobs:
            restore_primary_segment(primary)

        print(f"=== 8. full_dub.wav @ {OUT_SR} Hz ===")
        full = restore_full_vocals(project_dir, manifest, music_stem)

        print("=== 9. mux MP4 ===")
        mux_audio = project_dir / "final_mux_audio.wav"
        mix_dub_with_original(
            video_path, full, mux_audio,
            original_ratio=original_audio_ratio, dub_volume_percent=dub_volume_percent,
        )
        out = project_dir / f"{project_name}_dubbed.mp4"
        mux_video(video_path, mux_audio, out)
        (project_dir / "dub_output_path.txt").write_text(str(out.resolve()), encoding="utf-8")
        print(f"Готово: {out}")
        return out
    finally:
        dubbing.clear_voice_prompts()


def main():
    if len(sys.argv) < 5:
        print("Использование: python main.py <project> <video> <src_lang> <tgt_lang>")
        sys.exit(1)
    run(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])


if __name__ == "__main__":
    main()
