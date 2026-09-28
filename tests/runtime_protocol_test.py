import base64
import io
import json
import struct
import sys
import unittest
from unittest import mock
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "stt"))
sys.path.insert(0, str(ROOT / "packages" / "tts"))
import stt_daemon as stt  # noqa: E402
import tts_pocket as tts  # noqa: E402


class ProtocolTests(unittest.TestCase):
    def test_stt_rejects_invalid_or_oversized_pcm_without_model_load(self):
        class NoNumpy:
            def __getattr__(self, name):
                raise AssertionError(f"NumPy must not be touched: {name}")

        invalid = (
            {"audio": base64.b64encode(b"\x00\x00").decode(), "sample_rate": 8000},
            {"audio": "not-base64", "sample_rate": 16000},
            {"audio": base64.b64encode(b"\x00").decode(), "sample_rate": 16000},
            {"audio": base64.b64encode(b"\x00\x00" * (stt.MAX_SAMPLES + 1)).decode(), "sample_rate": 16000},
        )
        for request in invalid:
            with self.subTest(request=request.get("sample_rate"), length=len(request["audio"])):
                with self.assertRaises(ValueError):
                    stt.decode_pcm(request, NoNumpy())

    def test_stt_language_and_request_identity_are_bounded(self):
        self.assertEqual(stt.normalize_language("vi-VN"), "vi")
        self.assertIsNone(stt.normalize_language("../../unsafe"))
        self.assertEqual(stt.request_id({"id": "turn-1"}), "turn-1")
        self.assertEqual(stt.request_id({"id": 9}), "unknown")

    def test_mlx_returns_unused_scratch_and_whisper_stays_independent(self):
        class Cache:
            calls = 0

            def clear_cache(self):
                self.calls += 1

        class Backend:
            engine = stt.MLX_ENGINE
            mx = Cache()

        mlx = Backend()
        stt.release_inference_scratch(mlx)
        self.assertEqual(mlx.mx.calls, 1)

        class Whisper:
            engine = stt.OPENAI_WHISPER_ENGINE

        stt.release_inference_scratch(Whisper())

    def test_prebuilt_mlx_runtime_never_repairs_packages_in_place(self):
        with mock.patch.dict('os.environ', {"CORTEX_STT_IMMUTABLE_RUNTIME": "1"}):
            with mock.patch.object(stt.subprocess, 'run', side_effect=AssertionError('pip must not run')):
                with self.assertRaisesRegex(RuntimeError, 'prebuilt speech runtime'):
                    stt.install_mlx_whisper_package()

    def test_tts_rejects_non_object_protocol_input(self):
        for line in ("null", "[]", '"hello"'):
            with self.subTest(line=line), self.assertRaises(ValueError):
                tts._read_request(line)
        self.assertEqual(tts._read_request(json.dumps({"text": "Hello"}))["text"], "Hello")

    def test_tts_pcm_stream_has_exact_frames_and_terminal_marker(self):
        class FakeSynth:
            def pcm_frames(self, text):
                self.text = text
                yield b"\x01\x00\x02\x00"
                yield b"\x03\x00"

        synth = FakeSynth()
        out = io.BytesIO()
        tts._write_frames(out, synth, "Hello")
        self.assertEqual(synth.text, "Hello")
        self.assertEqual(out.getvalue(),
                         struct.pack(">I", 4) + b"\x01\x00\x02\x00"
                         + struct.pack(">I", 2) + b"\x03\x00" + struct.pack(">I", 0))


if __name__ == "__main__":
    unittest.main()
