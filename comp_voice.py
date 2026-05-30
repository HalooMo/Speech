"""Сравнение двух аудиофайлов по эмбеддингу голоса (SpeechBrain ECAPA)."""
from __future__ import annotations

import torch
import torch.nn.functional as F
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier

_classifier: EncoderClassifier | None = None
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _get_classifier() -> EncoderClassifier:
    global _classifier
    if _classifier is None:
        _classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": _DEVICE},
        )
    return _classifier


def _load_mono(path: str) -> tuple[torch.Tensor, int]:
    signal, fs = torchaudio.load(path)
    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)
    return signal, fs


def comp_voice(audio_path1: str, audio_path2: str, threshold: float = 0.5) -> bool:
    classifier = _get_classifier()
    s1, fs1 = _load_mono(audio_path1)
    s2, fs2 = _load_mono(audio_path2)
    if fs1 != fs2:
        s2 = torchaudio.functional.resample(s2, fs2, fs1)

    embeddings1 = classifier.encode_batch(s1)
    embeddings2 = classifier.encode_batch(s2)
    res = F.cosine_similarity(embeddings1, embeddings2, dim=2).item()
    return res > threshold
