"""SpeechLab — один пайплайн дубляжа (PRD.md).

Точки входа:
  CLI:  python main.py <project> <video> <src> <tgt>
  API:  server/run_job.py → run(...)

Поток (сверху вниз в этом файле):
  video → 16k → demucs → first_seg(40–90s) → ASR(test/asr) →
  перевод → casting → TTS → fit → full_dub → mux MP4

Отдельно: config/, server/, tools/ (TTS/LLM), prompt.py, test.py (ASR).
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

from config.env_config import get_hf_token
from test import init_asr_models, run_segment_pipeline, unload_asr_models
from tools import llm

ROOT = Path(__file__).resolve().parent
SR = 16000

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


def _resume_ok(project_dir, video_path):
    if not RESUME:
        return False
    p = project_dir / STATE_FILE
    if not p.is_file():
        return False
    try:
        prev = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if prev.get("input_sha256") != _sha256(video_path):
        print("  resume: входной файл изменился")
        return False
    return True


def _save_state(project_dir, video_path, **extra):
    (project_dir / STATE_FILE).write_text(json.dumps({
        "input_sha256": _sha256(video_path),
        "video_path": str(video_path.resolve()),
        **extra,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


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
def extract_audio_16k(video_path, out_wav):
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    _run(["ffmpeg", "-y", "-i", str(video_path),
          "-vn", "-acodec", "pcm_s16le", "-ar", str(SR), "-ac", "1", str(out_wav)])


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
                json_only=False, batch_translate=True, retries=2,
            )
            for item in _parse_batch(raw):
                rid, text = str(item.get("id", "")).strip(), str(item.get("text", "")).strip()
                if rid and text:
                    by_id[rid] = text
        except Exception as exc:
            print(f"  batch перевод не удался ({exc}), по одной…")

        for txt, meta in batch:
            translated = by_id.get(txt.name)
            if not translated:
                orig = meta["text"]
                translated = (llm.llm_response(prompt.get_prompt(2, {
                    "text": orig, "source_lang": source_lang, "target_lang": target_lang,
                    "source_chars": len(orig), "source_words": len(orig.split()),
                    "slot_sec": float(meta["end"]) - float(meta["start"]),
                })) or "").strip()
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

    if CAST_PER_SPEAKER and entries:
        by_spk = {}
        for e in entries:
            by_spk.setdefault(e["speaker"], []).append(e)
        profiles = {
            spk: dubbing.apply_voice_override(
                profile_from_wav(max(items, key=lambda x: x["dur"])["wav"]))
            for spk, items in by_spk.items()
        }
        for e in entries:
            casting["segments"][e["txt"].name] = entry(e, profiles[e["speaker"]])
    else:
        for e in entries:
            casting["segments"][e["txt"].name] = entry(
                e, dubbing.apply_voice_override(profile_from_wav(e["wav"])))

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
    from tools.dubbing import dub_from_profile, ensure_voice_bank, needs_qwen_bank, unload_model
    from tools.fit_audio import match_loudness

    lang = _tts_lang(target_lang)
    if not bank_ready and needs_qwen_bank(lang):
        print("  TTS: банк 8 голосов…")
        ensure_voice_bank(lang)
        bank_ready = True
    elif not bank_ready and dubbing.silero_enabled(lang):
        print("  TTS: Silero (ru)")

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
# Шаги 7–9: склейка, микс, mux
# =============================================================================
def _timeline(placements, min_dur):
    max_end = float(min_dur)
    for start, path in placements:
        max_end = max(max_end, start + _duration(path))
    out = np.zeros(int(round(max_end * SR)), dtype=np.float32)
    for start, path in placements:
        raw, sr = sf.read(path, dtype="float32")
        data = _resample_mono(raw, sr)
        pos = int(round(start * SR))
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
        data, sr = sf.read(seg_wav, dtype="float32")
        sf.write(out, _resample_mono(data, sr), SR)
        return out
    mixed = _timeline(schedule_placements(slots), _duration(seg_wav))
    sf.write(out, mixed, SR)
    print(f"  таймлайн: {len(slots)} реплик")
    return out


def restore_full_vocals(project_dir, manifest, music_stem):
    pieces, max_end = [], 0.0
    for item in manifest:
        folder = project_dir / "first_seg" / item["folder"]
        wav = folder / "restored.wav"
        if not wav.is_file():
            wav = folder / "segment.wav"
        data, sr = sf.read(wav, dtype="float32")
        pieces.append((float(item["start"]), _resample_mono(data, sr)))
        max_end = max(max_end, float(item["end"]))

    n = int(round(max_end * SR))
    vocals = np.zeros(n, dtype=np.float32)
    for start, data in pieces:
        pos = int(round(start * SR))
        end = min(n, pos + len(data))
        vocals[pos:end] = data[: end - pos]

    music, sr_m = sf.read(music_stem, dtype="float32")
    music = _resample_mono(music, sr_m)
    music = np.pad(music, (0, max(0, n - len(music))))[:n]
    mixed = vocals + music * 0.85
    peak = np.max(np.abs(mixed)) or 1.0
    if peak > 1.0:
        mixed = mixed / peak * 0.98
    out = project_dir / "full_dub.wav"
    sf.write(out, mixed, SR)
    return out


def mix_dub_with_original(video_path, dub_wav, out_wav, *, original_ratio=None, dub_volume_percent=None):
    ratio = ORIGINAL_AUDIO_RATIO if original_ratio is None else original_ratio
    pct = DUB_VOLUME_PERCENT if dub_volume_percent is None else dub_volume_percent
    if pct <= 0:
        raise ValueError("Громкость дубляжа (%) должна быть > 0")
    gain = pct / 100.0

    if not _has_video(video_path):
        shutil.copy2(dub_wav, out_wav)
        return out_wav

    orig_tmp = out_wav.parent / "_original_16k.wav"
    extract_audio_16k(video_path, orig_tmp)
    dub = _resample_mono(*sf.read(dub_wav, dtype="float32"))
    orig = _resample_mono(*sf.read(orig_tmp, dtype="float32"))
    n = max(len(dub), len(orig))
    dub = np.pad(dub, (0, max(0, n - len(dub))))[:n]
    orig = np.pad(orig, (0, max(0, n - len(orig))))[:n]
    mixed = dub * gain + orig * ratio
    peak = float(np.max(np.abs(mixed))) or 1.0
    if peak > 1.0:
        mixed = mixed / peak * 0.98
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_wav, mixed.astype(np.float32), SR)
    print(f"  микс: дубляж × {gain * 100:.0f}% + оригинал × {ratio:.2f}")
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
    voice_design_template=None,
    voice_design_by_key=None,
    voice_gender=None,
    voice_age=None,
    voice_design_temperature=None,
    voice_clone_samples=None,
    silero_speaker=None,
    silero_all_replicas=False,
    silero_age_groups=None,
    silero_voices=None,
    projects_root=None,
):
    """Единая точка входа пайплайна (CLI и Flask worker)."""
    from tools import dubbing
    from tools.dubbing import ensure_voice_bank, needs_qwen_bank, unload_model as unload_tts
    from tools.get_param import unload_model as unload_casting

    video_path = Path(video_path).resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)

    project_dir = _projects_root(projects_root) / project_name
    project_dir.mkdir(parents=True, exist_ok=True)

    # --- голос на этот прогон (сброс в finally) ---
    use_bank = bool(
        (voice_design_template or "").strip() or voice_design_by_key
        or voice_design_temperature is not None or voice_clone_samples
    )
    dubbing.set_voice_prompts(
        template=voice_design_template, by_key=voice_design_by_key,
        cache_dir=project_dir / "voice_bank" if use_bank else None,
        gender=voice_gender, age=voice_age, design_temperature=voice_design_temperature,
    )
    dubbing.set_voice_clone_samples(voice_clone_samples)
    if (silero_speaker or "").strip() or silero_voices:
        from tools import silero_tts
        if not silero_tts.is_russian_target(target_language):
            raise ValueError("Silero TTS только при target_language=ru")
        dubbing.set_silero_options(
            speaker=silero_speaker, all_replicas=bool(silero_all_replicas),
            age_groups=silero_age_groups, voices=silero_voices,
        )

    token = hf_token or get_hf_token()
    try:
        # --- 1 demucs ---
        print("=== 1. Аудио 16 kHz + demucs ===")
        project_vocals = project_dir / "vocals.wav"
        stems_dir = project_dir / "demucs_stems"
        music_ok = list(stems_dir.rglob("no_vocals.wav")) if stems_dir.is_dir() else []
        if SKIP_DEMUCS and project_vocals.is_file() and music_ok:
            music_stem = music_ok[0]
            print("  demucs: пропуск")
        else:
            audio_16k = project_dir / "audio_16k.wav"
            extract_audio_16k(video_path, audio_16k)
            vocals_wav, music_stem = separate_stems(audio_16k, stems_dir)
            project_vocals.write_bytes(vocals_wav.read_bytes())
            print(f"vocals: {project_vocals}")

        # --- 2 first_seg ---
        print("=== 2. Первичная нарезка ===")
        first_seg_dir = project_dir / "first_seg"
        manifest = split_primary_segments(project_vocals, first_seg_dir)
        resume = _resume_ok(project_dir, video_path)
        _save_state(project_dir, video_path, source_language=source_language, target_language=target_language)

        # --- 3 ASR ---
        print("=== 3. ASR ===")
        jobs = []
        n_replicas = 0
        init_asr_models(source_language, token)
        try:
            for item in manifest:
                primary = first_seg_dir / item["folder"]
                second = primary / "second_seg"
                second.mkdir(parents=True, exist_ok=True)
                jobs.append((primary, second))
                print(f"--- {item['folder']} ---")
                if resume and _step_done(second, "asr"):
                    print("  ASR: resume")
                    n_replicas += len(_list_txt(second / "output_text_segments"))
                else:
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

        # --- 4 translate ---
        print("=== 4. Перевод ===")
        for _, second in jobs:
            if resume and _step_done(second, "translate"):
                print("  перевод: resume")
                continue
            translate_segments(second, source_language, target_language)

        # --- 5 casting ---
        print("=== 5. casting ===")
        for _, second in jobs:
            if resume and _step_done(second, "casting"):
                print("  casting: resume")
                continue
            build_casting(second, unload=False)
        unload_casting()

        # --- 6 TTS ---
        print("=== 6. TTS ===")
        lang = _tts_lang(target_language)
        if needs_qwen_bank(lang):
            ensure_voice_bank(lang)
            bank_ready = True
        elif dubbing.silero_enabled(lang):
            print("  Silero, Qwen пропущен")
            bank_ready = False
        else:
            ensure_voice_bank(lang)
            bank_ready = True
        for _, second in jobs:
            if resume and _step_done(second, "dub"):
                print("  TTS: resume")
                continue
            dub_segments(second, target_language, unload=False, bank_ready=bank_ready)
        unload_tts()

        # --- 7–9 ---
        print("=== 7. Склейка реплик ===")
        for primary, _ in jobs:
            restore_primary_segment(primary)

        print("=== 8. full_dub.wav ===")
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
