"""Собирает читаемый диалог из output_text_segments/ → cur_diolog.txt."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEGMENTS_DIR = ROOT / "output_text_segments"
AUDIO_DIR = ROOT / "output_audio_segments"
OUTPUT_FILE = ROOT / "cur_diolog.txt"

TIME_LINE = re.compile(r"^\s*([\d.]+)\s*-\s*([\d.]+)\s*$")
SPEAKERS_LINE = re.compile(r"^Speakers:\s*(.+)\s*$", re.IGNORECASE)
NAME_SPEECH = re.compile(r"^speech_(\d+)\.txt$", re.IGNORECASE)
NAME_OVERLAP = re.compile(r"^overlap_(\d+)\.txt$", re.IGNORECASE)


@dataclass
class Segment:
    kind: str  # "speech" | "overlap"
    start: float
    end: float
    speakers: list[str]
    text: str
    source: str
    audio_files: list[str] = field(default_factory=list)


def _posix_relpath(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def find_audio_for_txt(txt_path: Path) -> list[str]:
    """speech_003.txt -> speech_003_*.wav"""
    if not AUDIO_DIR.is_dir():
        return []
    stem = txt_path.stem
    matches: list[Path] = []
    for pat in (f"{stem}_*.wav", f"{stem}_*.WAV"):
        matches.extend(AUDIO_DIR.glob(pat))
    seen: set[str] = set()
    out: list[str] = []
    for p in sorted(matches, key=lambda x: x.name.lower()):
        key = p.resolve().as_posix()
        if key not in seen:
            seen.add(key)
            out.append(_posix_relpath(p))
    return out


def _fmt_time(sec: float) -> str:
    m = int(sec // 60)
    s = sec - m * 60
    if m:
        return f"{m:d}:{s:05.2f}"
    return f"{s:.2f}s"


def _parse_file(path: Path) -> Segment | None:
    raw = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    lines = [ln.rstrip() for ln in raw.split("\n")]
    while lines and lines[0] == "":
        lines.pop(0)
    if len(lines) < 2:
        return None

    m = TIME_LINE.match(lines[0])
    if not m:
        return None
    start, end = float(m.group(1)), float(m.group(2))

    kind = "speech"
    if NAME_OVERLAP.match(path.name):
        kind = "overlap"
    elif NAME_SPEECH.match(path.name):
        kind = "speech"

    body_start = 2
    sm = SPEAKERS_LINE.match(lines[1])
    if sm:
        kind = "overlap"
        speakers = [s.strip() for s in sm.group(1).split(",") if s.strip()]
    else:
        speakers = [lines[1].strip()] if lines[1].strip() else ["?"]

    text = "\n".join(lines[body_start:]).strip()
    return Segment(
        kind=kind,
        start=start,
        end=end,
        speakers=speakers,
        text=text,
        source=path.name,
    )


def collect_segments(directory: Path) -> list[Segment]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Нет папки: {directory}")
    out: list[Segment] = []
    for path in sorted(directory.iterdir(), key=lambda p: p.name.lower()):
        if not path.is_file() or path.suffix.lower() != ".txt":
            continue
        seg = _parse_file(path)
        if seg:
            seg.audio_files = find_audio_for_txt(path)
            out.append(seg)
    out.sort(key=lambda s: (s.start, s.end, s.source))
    return out


def render_dialog(segments: list[Segment]) -> str:
    lines: list[str] = []
    w = 72
    lines.append("═" * w)
    lines.append(" Диалог по сегментам (output_text_segments)".center(w))
    lines.append("═" * w)
    lines.append("")

    for i, seg in enumerate(segments):
        t0, t1 = _fmt_time(seg.start), _fmt_time(seg.end)
        span = f"[{t0} — {t1}]"
        if seg.kind == "overlap":
            spk = ", ".join(seg.speakers)
            header = f"{span} ⟪ одновременно ⟫ спикеры: {spk}"
        else:
            spk = seg.speakers[0] if seg.speakers else "?"
            header = f"{span} {spk}"
        lines.append(header)
        lines.append("─" * min(len(header), w))

        if seg.audio_files:
            lines.append(f" аудио: {' | '.join(seg.audio_files)}")
        else:
            lines.append(f" аудио: (нет файла в output_audio_segments для {seg.source})")
        lines.append(f" текст: {seg.source}")
        if seg.text:
            for para in seg.text.split("\n"):
                lines.append(para.strip() if para.strip() else "")
        else:
            lines.append("…")
        lines.append("")

        if i < len(segments) - 1:
            gap = seg.end
            nxt = segments[i + 1].start
            if nxt > gap + 0.05:
                pause = nxt - gap
                lines.append(f"  ··· пауза ~{pause:.2f} с ···")
                lines.append("")

    lines.append("─" * w)
    lines.append(f"Всего сегментов: {len(segments)}")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    segs = collect_segments(SEGMENTS_DIR)
    text = render_dialog(segs)
    OUTPUT_FILE.write_text(text, encoding="utf-8")
    print(f"OK: {len(segs)} segments -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
