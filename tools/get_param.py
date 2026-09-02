"""Определение пола и возраста по WAV реплики → профиль для casting.json (PRD 3.4).

Используется в main.build_casting() перед TTS: по профилю выбирается
voice_key (male_mature и т.д.) в tools/dubbing.py.

Эмоции для Fish TTS задаёт LLM тегами [brackets] в target_text (prompt.py).
"""
import gc
import os

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel

AGE_MODEL = os.environ.get("SPEECHLAB_AGE_GENDER_MODEL", "audeering/wav2vec2-large-robust-24-ft-age-gender")

# Ленивая загрузка — модель в памяти до unload_model()
_age = {"proc": None, "model": None, "dev": None}


def _dev():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_audio_16k(path):
    """Привести любой WAV к mono float32 16 kHz (вход wav2vec)."""
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav.squeeze(0).numpy().astype(np.float32)


def _age_group(age):
    """Возраст в годах → группа для voice_key (как в dubbing.age_to_group)."""
    if age < 13:
        return "child"
    if age < 20:
        return "teenager"
    if age < 55:
        return "mature"
    return "elderly"


# =============================================================================
# Модель audeering: wav2vec2 + две головы (возраст регрессия, пол 3-class)
# =============================================================================
class _Head(nn.Module):
    def __init__(self, config, n):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, n)

    def forward(self, features, **kw):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return self.out_proj(x)


class _AgeGender(Wav2Vec2PreTrainedModel):
    """Кастомная архитектура как в README модели audeering на Hugging Face."""

    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = _Head(config, 1)
        self.gender = _Head(config, 3)
        self.init_weights()

    def forward(self, input_values):
        h = self.wav2vec2(input_values)[0]
        h = torch.mean(h, dim=1)
        return h, self.age(h), torch.softmax(self.gender(h), dim=1)


def _load_age():
    d = _dev()
    if _age["model"] is None or _age["dev"] != d:
        _age["proc"] = Wav2Vec2Processor.from_pretrained(AGE_MODEL)
        _age["model"] = _AgeGender.from_pretrained(AGE_MODEL).to(d).eval()
        _age["dev"] = d


def _predict_age_gender(audio):
    """Inference: gender (female/male/child), age в годах, age_group."""
    _load_age()
    x = _age["proc"](audio, sampling_rate=16000, return_tensors="pt")["input_values"].to(_age["dev"])
    with torch.no_grad():
        _, la, lg = _age["model"](x)
        age_norm = float(la.squeeze().cpu().item())
        age = round(age_norm * 100.0, 1)  # модель: 0..1 → 0..100 лет
        gf, gm, gc = lg.squeeze().cpu().tolist()
    labels = ["female", "male", "child"]
    probs = {"female": gf, "male": gm, "child": gc}
    top = labels[int(np.argmax([gf, gm, gc]))]
    gender = "child" if top == "child" else top
    return {
        "gender": gender, "age_group": _age_group(age), "age": round(age, 1),
        "confidence": round(max(gf, gm, gc), 4), "probs": {k: round(v, 4) for k, v in probs.items()},
    }


def profile_from_wav(audio_path):
    """Главная функция: путь к speech_*.wav → dict для casting.json['profile']."""
    ag = _predict_age_gender(load_audio_16k(audio_path))
    return {
        "gender": ag["gender"], "age_group": ag["age_group"], "age": ag["age"],
        "confidence_gender": ag["confidence"], "gender_probs": ag["probs"],
    }


def unload_model():
    """Выгрузить wav2vec из VRAM перед Fish TTS (main вызывает после casting)."""
    global _age
    _age = {"proc": None, "model": None, "dev": None}
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
