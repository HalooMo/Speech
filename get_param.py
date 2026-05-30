"""Пол и эмоция по WAV реплики (wav2vec2). JSON-строки для voice_param.txt."""
from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel

EMOTION_MODEL = os.environ.get(
    "SPEECHLAB_EMOTION_MODEL", "Dpngtm/wav2vec2-emotion-recognition"
)
AGE_GENDER_MODEL = os.environ.get(
    "SPEECHLAB_AGE_GENDER_MODEL", "audeering/wav2vec2-large-robust-24-ft-age-gender"
)

_EMOTION: dict[str, Any] = {"fe": None, "model": None}
_AGE_GENDER: dict[str, Any] = {"processor": None, "model": None, "device": None}

_EMOTION_TO_DUB = {
    "neutral": "calm",
    "calm": "calm",
    "sad": "sad",
    "happy": "happy",
    "angry": "angry",
    "disgust": "disgust",
    "discust": "disgust",
    "fearful": "sad",
    "fear": "sad",
    "surprised": "calm",
    "surprise": "calm",
}


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_audio_16k(path: str | os.PathLike) -> np.ndarray:
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav.squeeze(0).numpy().astype(np.float32)


def _age_group(age: float) -> str:
    if age < 13:
        return "child"
    if age < 20:
        return "teenager"
    if age < 55:
        return "mature"
    return "elderly"


class _ModelHead(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return self.out_proj(x)


class _AgeGenderModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = _ModelHead(config, 1)
        self.gender = _ModelHead(config, 3)
        self.post_init()

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0].mean(dim=1)
        logits_age = self.age(hidden_states)
        logits_gender = torch.softmax(self.gender(hidden_states), dim=1)
        return hidden_states, logits_age, logits_gender


def _ensure_emotion():
    if _EMOTION["model"] is None:
        _EMOTION["fe"] = AutoFeatureExtractor.from_pretrained(EMOTION_MODEL)
        _EMOTION["model"] = AutoModelForAudioClassification.from_pretrained(
            EMOTION_MODEL
        ).eval().to(_device())


def _ensure_age_gender():
    dev = _device()
    if _AGE_GENDER["model"] is None or _AGE_GENDER["device"] != dev:
        _AGE_GENDER["processor"] = Wav2Vec2Processor.from_pretrained(AGE_GENDER_MODEL)
        _AGE_GENDER["model"] = _AgeGenderModel.from_pretrained(AGE_GENDER_MODEL).to(dev).eval()
        _AGE_GENDER["device"] = dev


def _predict_emotion_scores(audio_np: np.ndarray, top_k: int = 5) -> list[tuple[str, float]]:
    _ensure_emotion()
    fe, model = _EMOTION["fe"], _EMOTION["model"]
    inputs = fe(audio_np, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {k: v.to(_device()) for k, v in inputs.items()}
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=-1).squeeze(0)
    k = min(top_k, probs.shape[0])
    top_probs, top_ids = torch.topk(probs, k=k)
    return [
        (model.config.id2label[i.item()].lower(), round(p.item(), 4))
        for p, i in zip(top_probs, top_ids)
    ]


def _predict_age_gender(audio_np: np.ndarray, input_sr: int = 16000) -> dict:
    _ensure_age_gender()
    processor = _AGE_GENDER["processor"]
    model = _AGE_GENDER["model"]
    dev = _AGE_GENDER["device"]
    x = processor(audio_np, sampling_rate=input_sr, return_tensors="pt")["input_values"].to(dev)
    with torch.no_grad():
        _, logits_age, logits_gender = model(x)
        age = float(logits_age.squeeze().cpu().item())
        gf, gm, gc = logits_gender.squeeze().cpu().tolist()
    labels = ["female", "male", "child"]
    probs = {"female": gf, "male": gm, "child": gc}
    top = labels[int(np.argmax([gf, gm, gc]))]
    gender = "child" if top == "child" else top
    return {
        "gender": gender,
        "age_group": _age_group(age),
        "age": round(age, 1),
        "confidence": round(max(gf, gm, gc), 4),
        "probs": {k: round(v, 4) for k, v in probs.items()},
    }


def get_sex(audio_path: str) -> str:
    """JSON: gender, age_group, age, confidence, probs."""
    audio = load_audio_16k(audio_path)
    return json.dumps(_predict_age_gender(audio), ensure_ascii=False)


def get_emotion(audio_path: str) -> str:
    """JSON: primary_emotion, confidence, scores (для dubbing.py)."""
    audio = load_audio_16k(audio_path)
    scores = _predict_emotion_scores(audio)
    raw_label = scores[0][0] if scores else "neutral"
    primary = _EMOTION_TO_DUB.get(raw_label, raw_label)
    if primary not in ("calm", "sad", "disgust", "happy", "angry"):
        primary = "calm"
    return json.dumps(
        {
            "primary_emotion": primary,
            "raw_label": raw_label,
            "confidence": scores[0][1] if scores else 0.0,
            "scores": [{"label": l, "prob": p} for l, p in scores],
        },
        ensure_ascii=False,
    )


def unload_model() -> None:
    """Освободить VRAM перед TTS."""
    import gc

    global _EMOTION, _AGE_GENDER
    _EMOTION = {"fe": None, "model": None}
    _AGE_GENDER = {"processor": None, "model": None, "device": None}
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
