"""Сверка голосов в output_audio_segments, унификация спикеров A/B/C."""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path

from comp_voice import comp_voice

ROOT = Path(__file__).resolve().parent
AUDIO_DIR = ROOT / "output_audio_segments"
TEXT_DIR = ROOT / "output_text_segments"

SPEECH_WAV = re.compile(
    r"^speech_(\d{3})_([^_]+)_([\d.]+)-([\d.]+)s\.wav$",
    re.IGNORECASE,
)
OVERLAP_TXT_SPEAKERS = re.compile(r"^Speakers:\s*(.+)\s*$", re.IGNORECASE)


def _label_for_cluster(rank: int) -> str:
    if rank < 26:
        return chr(ord("A") + rank)
    return f"S{rank + 1}"


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _read_txt_speaker(txt_path: Path) -> str | None:
    lines = txt_path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        return None
    line = lines[1].strip()
    if OVERLAP_TXT_SPEAKERS.match(line):
        return None
    return line


def _map_overlap_speakers(line: str, label_map: dict[str, str]) -> str:
    m = OVERLAP_TXT_SPEAKERS.match(line.strip())
    if not m:
        return line
    names = [s.strip() for s in m.group(1).split(",") if s.strip()]
    mapped = [label_map.get(n, n) for n in names]
    return "Speakers: " + ", ".join(mapped)


def align_speakers(
    audio_dir: Path = AUDIO_DIR,
    text_dir: Path = TEXT_DIR,
) -> dict[str, str]:
    wavs = []
    for p in audio_dir.glob("speech_*.wav"):
        if SPEECH_WAV.match(p.name):
            wavs.append(p)
    wavs.sort(key=lambda p: int(SPEECH_WAV.match(p.name).group(1)))  # type: ignore[union-attr]

    if not wavs:
        print("Нет speech_*.wav в", audio_dir)
        return {}

    entries = []
    for p in wavs:
        m = SPEECH_WAV.match(p.name)
        if not m:
            continue
        idx, old_spk, t0, t1 = m.group(1), m.group(2), m.group(3), m.group(4)
        txt = text_dir / f"speech_{idx}.txt"
        txt_spk = _read_txt_speaker(txt) if txt.is_file() else None
        entries.append({
            "path": p,
            "idx": idx,
            "old_spk": txt_spk or old_spk,
            "t0": t0,
            "t1": t1,
        })

    n = len(entries)
    uf = _UnionFind(n)
    print(f"Сравнение {n} файлов ({n * (n - 1) // 2} пар)...")

    for i in range(n):
        for j in range(i + 1, n):
            if comp_voice(str(entries[i]["path"]), str(entries[j]["path"])):
                uf.union(i, j)
                print(f" один голос: speech_{entries[i]['idx']} ~ speech_{entries[j]['idx']}")

    clusters: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        clusters[uf.find(i)].append(i)

    ordered_roots = sorted(clusters.keys(), key=lambda r: min(clusters[r]))
    root_to_label = {_root: _label_for_cluster(k) for k, _root in enumerate(ordered_roots)}

    index_new: dict[int, str] = {}
    remap_pairs: list[tuple[str, str]] = []
    renames: list[tuple[Path, Path]] = []

    for i, ent in enumerate(entries):
        new_spk = root_to_label[uf.find(i)]
        index_new[i] = new_spk
        remap_pairs.append((ent["old_spk"], new_spk))
        new_name = f"speech_{ent['idx']}_{new_spk}_{ent['t0']}-{ent['t1']}s.wav"
        new_path = ent["path"].with_name(new_name)
        if new_path != ent["path"]:
            renames.append((ent["path"], new_path))

    label_map: dict[str, str] = {}
    votes: dict[str, Counter] = defaultdict(Counter)
    for old, new in remap_pairs:
        votes[old][new] += 1
    for old, ctr in votes.items():
        label_map[old] = ctr.most_common(1)[0][0]

    for src, dst in renames:
        if dst.exists() and dst != src:
            dst.unlink()
        src.rename(dst)
        print(f" audio: {src.name} -> {dst.name}")

    for i, ent in enumerate(entries):
        new_spk = index_new[i]
        txt = text_dir / f"speech_{ent['idx']}.txt"
        if not txt.is_file():
            continue
        lines = txt.read_text(encoding="utf-8").splitlines()
        if len(lines) >= 2 and not OVERLAP_TXT_SPEAKERS.match(lines[1]):
            old_line = lines[1]
            if old_line != new_spk:
                lines[1] = new_spk
                txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
                print(f" text: speech_{ent['idx']}.txt: {old_line} -> {new_spk}")

    for txt in sorted(text_dir.glob("overlap_*.txt")):
        lines = txt.read_text(encoding="utf-8").splitlines()
        if len(lines) < 2:
            continue
        new_line = _map_overlap_speakers(lines[1], label_map)
        if new_line != lines[1]:
            lines[1] = new_line
            txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(f" text: {txt.name}: {new_line}")

    print(f"\nСпикеров: {len(ordered_roots)} -> {', '.join(root_to_label[r] for r in ordered_roots)}")
    print("Карта ярлыков:", label_map)
    return label_map


if __name__ == "__main__":
    align_speakers()
