"""Юнит-тесты mk_voice (без загрузки Qwen3-TTS)."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

import mk_voice


class TestAsDict(unittest.TestCase):
    def test_from_dict(self):
        d = {"gender": "female"}
        self.assertEqual(mk_voice._as_dict(d), d)

    def test_from_json_string(self):
        s = '{"primary_emotion": "happy"}'
        self.assertEqual(mk_voice._as_dict(s), {"primary_emotion": "happy"})

    def test_invalid_json(self):
        with self.assertRaises(ValueError):
            mk_voice._as_dict("not json")

    def test_json_array_rejected(self):
        with self.assertRaises(TypeError):
            mk_voice._as_dict("[1, 2]")

    def test_invalid_type(self):
        with self.assertRaises(TypeError):
            mk_voice._as_dict(42)


class TestToWavArray(unittest.TestCase):
    def test_numpy_1d(self):
        arr = np.array([0.1, -0.2], dtype=np.float32)
        out = mk_voice._to_wav_array(arr)
        self.assertEqual(out.shape, (2,))

    def test_squeeze_channel(self):
        arr = np.array([[0.1, -0.2]], dtype=np.float32)
        out = mk_voice._to_wav_array(arr)
        self.assertEqual(out.shape, (2,))

    def test_invalid_ndim(self):
        with self.assertRaises(ValueError):
            mk_voice._to_wav_array(np.zeros((2, 2, 2)))


class TestSynthVoice(unittest.TestCase):
    def test_empty_text_raises(self):
        with self.assertRaises(ValueError):
            mk_voice.synth_voice("", {}, {})

    def test_whitespace_text_raises(self):
        with self.assertRaises(ValueError):
            mk_voice.synth_voice("   ", {}, {})

    @patch.object(mk_voice, "_get_model")
    def test_writes_wav(self, mock_get_model: MagicMock):
        mock_model = MagicMock()
        mock_model.generate_voice_design.return_value = (
            [np.array([0.0, 0.25, -0.25], dtype=np.float32)],
            24000,
        )
        mock_get_model.return_value = mock_model

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "test.wav"
            path = mk_voice.synth_voice(
                text="Привет",
                voice_biometrics={"gender": "female"},
                voice_emotion={"primary_emotion": "neutral"},
                out_path=out,
            )
            self.assertTrue(Path(path).is_file())
            self.assertGreater(Path(path).stat().st_size, 0)

        mock_model.generate_voice_design.assert_called_once()
        call_kw = mock_model.generate_voice_design.call_args.kwargs
        instruct = json.loads(call_kw["instruct"])
        self.assertEqual(instruct["voice_biometrics"]["gender"], "female")
        self.assertEqual(instruct["voice_emotion"]["primary_emotion"], "neutral")

    @patch.object(mk_voice, "_get_model")
    def test_empty_wavs_raises(self, mock_get_model: MagicMock):
        mock_get_model.return_value.generate_voice_design.return_value = ([], 24000)
        with self.assertRaises(RuntimeError):
            mk_voice.synth_voice("текст", {}, {})


if __name__ == "__main__":
    unittest.main()
