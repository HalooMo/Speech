"""Определение пола и возраста по WAV реплики → профиль для casting.json (PRD 3.4).

Используется в main.build_casting() перед TTS: по профилю выбирается
voice_key (male_mature и т.д.) в tools/dubbing.py.

Эмоция (Dpngtm/wav2vec2) опциональна — по умолчанию SPEECHLAB_SKIP_EMOTION=1,
т.к. эмоции для Fish TTS задаёт LLM тегами [brackets] в target_text.
"""
import gc
import os

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel

EMOTION_MODEL = os.environ.get("SPEECHLAB_EMOTION_MODEL", "Dpngtm/wav2vec2-emotion-recognition")
AGE_MODEL = os.environ.get("SPEECHLAB_AGE_GENDER_MODEL", "audeering/wav2vec2-large-robust-24-ft-age-gender")
SKIP_EMOTION = os.environ.get("SPEECHLAB_SKIP_EMOTION", "1").lower() not in ("0", "false", "no")

# Ленивая загрузка — модели держим в памяти до unload_model()
_emotion = {"fe": None, "model": None}
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


def _load_emotion():
    if _emotion["model"] is None:
        _emotion["fe"] = AutoFeatureExtractor.from_pretrained(EMOTION_MODEL)
        _emotion["model"] = AutoModelForAudioClassification.from_pretrained(EMOTION_MODEL).eval().to(_dev())


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


def _predict_emotion(audio):
    _load_emotion()
    inp = _emotion["fe"](audio, sampling_rate=16000, return_tensors="pt", padding=True)
    inp = {k: v.to(_dev()) for k, v in inp.items()}
    with torch.no_grad():
        probs = torch.softmax(_emotion["model"](**inp).logits, dim=-1).squeeze(0)
    i = int(probs.argmax().item())
    label = _emotion["model"].config.id2label[i].lower()
    return label, round(probs[i].item(), 4)


def profile_from_wav(audio_path, *, with_emotion=None):
    """Главная функция: путь к speech_*.wav → dict для casting.json['profile']."""
    use_emo = (not SKIP_EMOTION) if with_emotion is None else with_emotion
    audio = load_audio_16k(audio_path)
    ag = _predict_age_gender(audio)
    if use_emo:
        emo, conf = _predict_emotion(audio)
        emo_map = {
            "neutral": "calm", "calm": "calm", "sad": "sad", "happy": "happy",
            "angry": "angry", "disgust": "disgust",
        }
        primary = emo_map.get(emo, "calm")
    else:
        emo, conf, primary = "skipped", 0.0, "calm"
    return {
        "gender": ag["gender"], "age_group": ag["age_group"], "age": ag["age"],
        "confidence_gender": ag["confidence"], "gender_probs": ag["probs"],
        "primary_emotion": primary, "raw_emotion_label": emo, "confidence_emotion": conf,
    }


def unload_model():
    """Выгрузить wav2vec из VRAM перед Fish TTS (main вызывает после casting)."""
    global _emotion, _age
    _emotion = {"fe": None, "model": None}
    _age = {"proc": None, "model": None, "dev": None}
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
