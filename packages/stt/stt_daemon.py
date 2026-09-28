#!/usr/bin/env python3
"""Persistent local speech-to-text service for Cortex realtime voice.

The process speaks newline-delimited JSON over stdin/stdout. It deliberately
does not open a network port: Electron owns the child lifecycle and is the only
client, so raw microphone audio never becomes a loopback service that another
local process can reach.

Requests:
  {"id":"...","op":"health"}
  {"id":"...","op":"transcribe_pcm","audio":"<base64 PCM16LE>",
   "sample_rate":16000}
  {"id":"...","op":"close"}

Responses always echo ``id`` and are one JSON object per line. Supported Apple
Silicon systems use MLX Whisper by default; other Macs and platforms use
OpenAI Whisper. The model is loaded once at startup and all audio is decoded
directly into a float32 numpy buffer, avoiding the file/ffmpeg hop used by the
legacy one-shot route.
"""

from __future__ import annotations

import base64
import binascii
import importlib
from importlib import metadata
import json
import math
import os
import platform
import shlex
import subprocess
import sys
import time
from typing import Any
from urllib.parse import urlparse


MLX_ENGINE = "mlx"
OPENAI_WHISPER_ENGINE = "whisper"
MLX_WHISPER_PACKAGE = "mlx-whisper==0.4.3"
MLX_WHISPER_VERSION = MLX_WHISPER_PACKAGE.split("==", 1)[1]
MLX_MODEL_NAME = "mlx-community/whisper-small-mlx"
OPENAI_WHISPER_MODEL_NAME = "small"
SAMPLE_RATE = 16_000
MAX_UTTERANCE_SECONDS = 60
MAX_SAMPLES = SAMPLE_RATE * MAX_UTTERANCE_SECONDS
WHISPER_WINDOW_SECONDS = 30
MLX_TRAILING_CLIP_MARGIN_SECONDS = 0.05
MLX_IMPOSSIBLE_TAIL_TOLERANCE_SECONDS = 1.0
MINIMUM_MLX_MACOS = (13, 5)


def macos_supports_mlx(version: str) -> bool:
    try:
        parts = tuple(int(part) for part in version.split(".")[:2])
    except ValueError:
        return False
    return parts >= MINIMUM_MLX_MACOS


def default_engine(
    system_name: str | None = None,
    machine: str | None = None,
    macos_version: str | None = None,
) -> str:
    system = (system_name or sys.platform).lower()
    architecture = (machine or platform.machine()).lower()
    if system != "darwin" or architecture != "arm64":
        return OPENAI_WHISPER_ENGINE
    version = platform.mac_ver()[0] if macos_version is None else macos_version
    return MLX_ENGINE if macos_supports_mlx(version) else OPENAI_WHISPER_ENGINE


def selected_engine() -> str:
    configured = os.environ.get("CORTEX_STT_ENGINE", "").strip().lower()
    return configured if configured in {MLX_ENGINE, OPENAI_WHISPER_ENGINE} else default_engine()


ENGINE_NAME = selected_engine()
MODEL_NAME = MLX_MODEL_NAME if ENGINE_NAME == MLX_ENGINE else OPENAI_WHISPER_MODEL_NAME


def emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), flush=True)


def normalize_language(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    clean = value.strip().lower()
    if not clean or clean == "auto":
        return None
    # Whisper expects its language codes, not a full BCP-47 locale.
    return clean.split("-", 1)[0] if clean.split("-", 1)[0].isalpha() else None


def request_id(request: dict[str, Any]) -> str:
    value = request.get("id")
    return value if isinstance(value, str) and value else "unknown"


def decode_pcm(request: dict[str, Any], numpy: Any) -> Any:
    sample_rate = request.get("sample_rate", SAMPLE_RATE)
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"Only {SAMPLE_RATE} Hz PCM is supported")
    encoded = request.get("audio")
    if not isinstance(encoded, str) or not encoded:
        raise ValueError("audio must be a non-empty base64 PCM string")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("audio is not valid base64 PCM") from error
    if len(raw) % 2:
        raise ValueError("PCM16 audio must contain whole samples")
    sample_count = len(raw) // 2
    if sample_count > MAX_SAMPLES:
        raise ValueError(f"audio exceeds the {MAX_UTTERANCE_SECONDS} second limit")
    if sample_count == 0:
        raise ValueError("audio must contain at least one PCM sample")
    return numpy.frombuffer(raw, dtype="<i2").astype(numpy.float32) / 32768.0


def warm_backend(backend: Any, numpy: Any) -> tuple[bool, str | None]:
    """Prime the first inference without making model readiness depend on it."""
    try:
        # A quarter-second of silence is enough to initialize the model's first
        # inference path while keeping launch work bounded.
        backend.transcribe_audio(
            numpy.zeros(SAMPLE_RATE // 4, dtype=numpy.float32),
            "en",
        )
        return True, None
    except Exception as error:  # A model can reject silent input but still work.
        return False, str(error)


def release_inference_scratch(backend: Any) -> None:
    """Return MLX's unused scratch while keeping the loaded model available."""
    if getattr(backend, "engine", None) != MLX_ENGINE:
        return
    try:
        backend.mx.clear_cache()
    except Exception:
        # Allocator cleanup must not turn a successful transcript into an error.
        pass


def installed_openai_model_path(whisper: Any) -> str | None:
    """Return a local model path without letting `load_model` download one."""
    if os.path.isfile(MODEL_NAME):
        return MODEL_NAME
    model_url = (getattr(whisper, "_MODELS", {}) or {}).get(MODEL_NAME)
    if not isinstance(model_url, str):
        return None
    cache_root = os.environ.get("WHISPER_DOWNLOAD_ROOT") or os.path.join(os.path.expanduser("~"), ".cache", "whisper")
    file_name = os.path.basename(urlparse(model_url).path)
    candidate = os.path.join(cache_root, file_name)
    return candidate if file_name and os.path.isfile(candidate) else None


def installed_mlx_model_path(snapshot_download: Any) -> str | None:
    """Resolve an MLX model only from disk; normal startup must not download."""
    if os.path.isdir(MODEL_NAME):
        return MODEL_NAME
    try:
        return str(snapshot_download(repo_id=MODEL_NAME, local_files_only=True))
    except Exception:
        return None


def installed_mlx_whisper_version() -> str | None:
    try:
        return metadata.version("mlx-whisper")
    except metadata.PackageNotFoundError:
        return None


def mlx_whisper_needs_install(installed_version: str | None) -> bool:
    return installed_version != MLX_WHISPER_VERSION


def install_mlx_whisper_package(force_reinstall: bool = False) -> None:
    if os.environ.get("CORTEX_STT_IMMUTABLE_RUNTIME") == "1":
        raise RuntimeError("The prebuilt speech runtime must be repaired from its verified artifact")
    command = [sys.executable, "-m", "pip", "install", "--upgrade"]
    if force_reinstall:
        command.append("--force-reinstall")
    command.append(MLX_WHISPER_PACKAGE)
    subprocess.run(command, check=True)


def load_mlx_install_dependencies() -> tuple[Any, Any]:
    """Pin a preexisting runtime before importing it during an update."""
    if mlx_whisper_needs_install(installed_mlx_whisper_version()):
        emit({"type": "install_progress", "stage": "package", "engine": MLX_ENGINE})
        install_mlx_whisper_package()
        if mlx_whisper_needs_install(installed_mlx_whisper_version()):
            raise RuntimeError(f"MLX Whisper {MLX_WHISPER_VERSION} was not installed")

    try:
        import mlx_whisper  # type: ignore
        from huggingface_hub import snapshot_download  # type: ignore
        return mlx_whisper, snapshot_download
    except Exception:
        # Package metadata can survive a broken or partial upgrade. Repair the
        # managed venv instead of repeatedly accepting that stale state.
        emit({"type": "install_progress", "stage": "package_repair", "engine": MLX_ENGINE})
        install_mlx_whisper_package(force_reinstall=True)
        import mlx_whisper  # type: ignore
        from huggingface_hub import snapshot_download  # type: ignore
        return mlx_whisper, snapshot_download


def install_model_command(model_name: str, engine: str = ENGINE_NAME) -> str:
    # Quote both the resolved Python executable and the full snippet. This
    # means a copied fallback command uses the same interpreter that Electron
    # selected in a packaged app rather than assuming `python3` is on PATH.
    if engine == MLX_ENGINE:
        program = (
            "from huggingface_hub import snapshot_download; "
            f"snapshot_download(repo_id={json.dumps(model_name)})"
        )
    else:
        program = f"import whisper; whisper.load_model({json.dumps(model_name)})"
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(program)}"


def transcription_confidence(result: Any) -> float | None:
    """Return a bounded, conservative confidence from Whisper segment data.

    Whisper does not expose a calibrated confidence score. This combines its
    no-speech probability with average log probability solely for the narrow
    spoken-approval safety gate; ordinary dictation never depends on it.
    """
    segments = (result or {}).get("segments")
    if not isinstance(segments, list) or not segments:
        return None
    weighted = 0.0
    weight = 0
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        text = str(segment.get("text") or "").strip()
        if not text:
            continue
        no_speech = segment.get("no_speech_prob")
        average = segment.get("avg_logprob")
        try:
            speech_score = max(0.0, min(1.0, 1.0 - float(no_speech)))
            # -1 is the conservative lower bound for a usable transcription;
            # 0 is ideal. Values below it should not grant spoken approval.
            text_score = max(0.0, min(1.0, 1.0 + float(average)))
        except (TypeError, ValueError):
            continue
        segment_weight = max(1, len(text))
        weighted += speech_score * text_score * segment_weight
        weight += segment_weight
    return round(weighted / weight, 3) if weight else None


def choose_auto_language(probabilities: Any) -> tuple[str, float | None]:
    """Choose the strongest Whisper language across its complete language set."""
    if not isinstance(probabilities, dict):
        return "en", None
    scores: list[tuple[float, str]] = []
    for language, probability in probabilities.items():
        normalized = normalize_language(language)
        if not normalized:
            continue
        try:
            score = float(probability)
        except (TypeError, ValueError):
            continue
        if math.isfinite(score):
            scores.append((score, normalized))
    if not scores:
        return "en", None
    score, language = max(scores, key=lambda item: item[0])
    return language, round(max(0.0, min(1.0, score)), 3)


def result_tuple(
    result: Any,
    selected_language: str,
    language_confidence: float | None,
) -> tuple[str, str | None, float | None, float | None]:
    text = str((result or {}).get("text") or "").strip()
    detected = (result or {}).get("language")
    return (
        text,
        detected if isinstance(detected, str) else selected_language,
        transcription_confidence(result),
        language_confidence,
    )


def trim_impossible_mlx_tail(result: Any, audio_duration: float) -> Any:
    """Drop MLX segments that can only come from padded audio after capture."""
    if not isinstance(result, dict):
        return result
    segments = result.get("segments")
    if not isinstance(segments, list) or not segments:
        return result
    retained: list[Any] = []
    for segment in segments:
        if not isinstance(segment, dict):
            retained.append(segment)
            continue
        try:
            start = float(segment.get("start"))
            end = float(segment.get("end"))
        except (TypeError, ValueError):
            retained.append(segment)
            continue
        starts_at_tail = start >= max(0.0, audio_duration - 0.25)
        extends_beyond_audio = end > audio_duration + MLX_IMPOSSIBLE_TAIL_TOLERANCE_SECONDS
        if not (starts_at_tail and extends_beyond_audio):
            retained.append(segment)
    if len(retained) == len(segments):
        return result
    trimmed = dict(result)
    trimmed["segments"] = retained
    trimmed["text"] = "".join(
        str(segment.get("text") or "")
        for segment in retained
        if isinstance(segment, dict)
    ).strip()
    return trimmed


class OpenAiWhisperBackend:
    engine = OPENAI_WHISPER_ENGINE
    display_name = "OpenAI Whisper"

    def __init__(self, model: Any, whisper: Any) -> None:
        self.model = model
        self.whisper = whisper

    def detect_auto_language(self, audio: Any) -> tuple[str, float | None]:
        if getattr(self.model, "is_multilingual", True) is False:
            return "en", 1.0
        padded = self.whisper.pad_or_trim(audio)
        mel = self.whisper.log_mel_spectrogram(
            padded,
            n_mels=self.model.dims.n_mels,
        ).to(self.model.device)
        _, probabilities = self.model.detect_language(mel)
        return choose_auto_language(probabilities)

    def transcribe_audio(
        self,
        audio: Any,
        requested_language: str | None,
    ) -> tuple[str, str | None, float | None]:
        if requested_language:
            selected_language, language_confidence = requested_language, None
        else:
            selected_language, language_confidence = self.detect_auto_language(audio)
        options: dict[str, Any] = {
            "language": selected_language,
            "fp16": False,
        }
        if len(audio) > SAMPLE_RATE * WHISPER_WINDOW_SECONDS:
            options["condition_on_previous_text"] = False
        result = self.model.transcribe(audio, **options)
        return result_tuple(result, selected_language, language_confidence)


class MlxWhisperBackend:
    engine = MLX_ENGINE
    display_name = "MLX Whisper"

    def __init__(self, mlx_whisper: Any, model_path: str) -> None:
        import mlx.core as mx  # type: ignore
        from mlx_whisper.transcribe import ModelHolder  # type: ignore

        self.mx = mx
        self.mlx_whisper = mlx_whisper
        self.model_path = model_path
        # MLX Whisper's holder is process-global. Loading it here means language
        # detection and every later transcribe call reuse the same fp16 model.
        self.model = ModelHolder.get_model(model_path, mx.float16)

    def detect_auto_language(self, audio: Any) -> tuple[str, float | None]:
        if getattr(self.model, "is_multilingual", True) is False:
            return "en", 1.0
        audio_module = self.mlx_whisper.audio
        mel = audio_module.log_mel_spectrogram(
            audio,
            n_mels=self.model.dims.n_mels,
            padding=audio_module.N_SAMPLES,
        )
        mel_segment = audio_module.pad_or_trim(
            mel,
            audio_module.N_FRAMES,
            axis=-2,
        ).astype(self.mx.float16)
        _, probabilities = self.model.detect_language(mel_segment)
        return choose_auto_language(probabilities)

    def transcribe_audio(
        self,
        audio: Any,
        requested_language: str | None,
    ) -> tuple[str, str | None, float | None]:
        if requested_language:
            selected_language, language_confidence = requested_language, None
        else:
            selected_language, language_confidence = self.detect_auto_language(audio)
        options: dict[str, Any] = {
            "path_or_hf_repo": self.model_path,
            "language": selected_language,
            "fp16": True,
            "verbose": None,
        }
        audio_duration = len(audio) / SAMPLE_RATE
        if audio_duration > MLX_TRAILING_CLIP_MARGIN_SECONDS:
            clip_end = audio_duration - MLX_TRAILING_CLIP_MARGIN_SECONDS
            options["clip_timestamps"] = f"0,{clip_end:.3f}"
        if len(audio) > SAMPLE_RATE * WHISPER_WINDOW_SECONDS:
            options["condition_on_previous_text"] = False
        result = self.mlx_whisper.transcribe(audio, **options)
        result = trim_impossible_mlx_tail(result, audio_duration)
        return result_tuple(result, selected_language, language_confidence)


def transcribe(
    backend: Any,
    request: dict[str, Any],
    numpy: Any,
) -> tuple[str, str | None, float | None, float | None]:
    audio = decode_pcm(request, numpy)
    return backend.transcribe_audio(audio, None)


def install_mlx() -> int:
    if default_engine() != MLX_ENGINE:
        emit({
            "type": "install_error",
            "engine": MLX_ENGINE,
            "error": "MLX Whisper requires Apple Silicon and macOS 13.5 or newer",
        })
        return 1
    try:
        mlx_whisper, snapshot_download = load_mlx_install_dependencies()

        emit({
            "type": "install_progress",
            "stage": "model",
            "engine": MLX_ENGINE,
            "model": MODEL_NAME,
        })
        # This is intentionally the one call site allowed to download. Normal
        # daemon startup resolves the Hugging Face cache with local-only mode.
        model_path = MODEL_NAME if os.path.isdir(MODEL_NAME) else snapshot_download(repo_id=MODEL_NAME)
        # Verify the downloaded files can instantiate before reporting success.
        MlxWhisperBackend(mlx_whisper, str(model_path))
        emit({
            "type": "install_complete",
            "engine": MLX_ENGINE,
            "model": MODEL_NAME,
        })
        return 0
    except Exception as error:
        emit({
            "type": "install_error",
            "engine": MLX_ENGINE,
            "error": f"Could not install MLX Whisper model '{MODEL_NAME}'",
            "detail": str(error)[:240],
        })
        return 1


def install_openai_whisper_package(force_reinstall: bool = False) -> None:
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
    ]
    if force_reinstall:
        command.append("--force-reinstall")
    command.append("openai-whisper")
    subprocess.run(command, check=True)


def load_openai_whisper_install_dependency() -> Any:
    try:
        return importlib.import_module("whisper")
    except Exception:
        emit({
            "type": "install_progress",
            "stage": "package",
            "engine": OPENAI_WHISPER_ENGINE,
        })
        install_openai_whisper_package()
    try:
        return importlib.import_module("whisper")
    except Exception:
        emit({
            "type": "install_progress",
            "stage": "package_repair",
            "engine": OPENAI_WHISPER_ENGINE,
        })
        install_openai_whisper_package(force_reinstall=True)
        return importlib.import_module("whisper")


def install_openai_whisper() -> int:
    try:
        whisper = load_openai_whisper_install_dependency()
        models = getattr(whisper, "_MODELS", {}) or {}
        if MODEL_NAME not in models:
            raise ValueError(f"Whisper model '{MODEL_NAME}' cannot be downloaded automatically")
        emit({
            "type": "install_progress",
            "stage": "model",
            "engine": OPENAI_WHISPER_ENGINE,
            "model": MODEL_NAME,
        })
        whisper.load_model(MODEL_NAME)
        emit({
            "type": "install_complete",
            "engine": OPENAI_WHISPER_ENGINE,
            "model": MODEL_NAME,
        })
        return 0
    except Exception as error:
        emit({
            "type": "install_error",
            "engine": OPENAI_WHISPER_ENGINE,
            "error": f"Could not install OpenAI Whisper model '{MODEL_NAME}'",
            "detail": str(error)[:240],
        })
        return 1


def install() -> int:
    """Perform the one explicit, Electron-requested STT setup action."""
    return install_mlx() if ENGINE_NAME == MLX_ENGINE else install_openai_whisper()


def load_backend(numpy: Any) -> Any:
    if ENGINE_NAME == MLX_ENGINE:
        if default_engine() != MLX_ENGINE:
            raise RuntimeError("MLX Whisper requires Apple Silicon and macOS 13.5 or newer")
        import mlx_whisper  # type: ignore
        from huggingface_hub import snapshot_download  # type: ignore

        model_path = installed_mlx_model_path(snapshot_download)
        if not model_path:
            return None
        return MlxWhisperBackend(mlx_whisper, model_path)

    import whisper  # type: ignore

    model_path = installed_openai_model_path(whisper)
    if not model_path:
        return None
    model = whisper.load_model(model_path)
    return OpenAiWhisperBackend(model, whisper)


def main() -> int:
    if ENGINE_NAME == MLX_ENGINE and default_engine() != MLX_ENGINE:
        emit({
            "type": "startup_error",
            "code": "mlx_unsupported",
            "engine": ENGINE_NAME,
            "model": MODEL_NAME,
            "error": "MLX Whisper requires Apple Silicon and macOS 13.5 or newer",
        })
        return 1
    if ENGINE_NAME == MLX_ENGINE:
        installed_version = installed_mlx_whisper_version()
        if installed_version is not None and mlx_whisper_needs_install(installed_version):
            emit({
                "type": "startup_error",
                "code": "mlx_version_mismatch",
                "engine": ENGINE_NAME,
                "model": MODEL_NAME,
                "error": f"MLX Whisper {MLX_WHISPER_VERSION} is required for realtime voice",
                "detail": f"Installed mlx-whisper version is {installed_version}",
            })
            return 1
    try:
        import numpy  # type: ignore
        if ENGINE_NAME == MLX_ENGINE:
            import mlx_whisper  # type: ignore  # noqa: F401
            from huggingface_hub import snapshot_download  # type: ignore  # noqa: F401
        else:
            import whisper  # type: ignore  # noqa: F401
    except Exception as error:
        package_label = "MLX Whisper" if ENGINE_NAME == MLX_ENGINE else "OpenAI Whisper"
        emit({
            "type": "startup_error",
            "code": "mlx_not_installed" if ENGINE_NAME == MLX_ENGINE else "whisper_not_installed",
            "engine": ENGINE_NAME,
            "model": MODEL_NAME,
            "error": f"{package_label} is not installed for realtime voice",
            "detail": str(error),
        })
        return 1

    started_at = time.monotonic()
    try:
        backend = load_backend(numpy)
    except Exception as error:
        emit({
            "type": "startup_error",
            "code": "model_load_failed",
            "engine": ENGINE_NAME,
            "model": MODEL_NAME,
            "error": f"Could not load {ENGINE_NAME} speech model '{MODEL_NAME}'",
            "detail": str(error),
        })
        return 1
    if backend is None:
        emit({
            "type": "startup_error",
            "code": "model_missing",
            "engine": ENGINE_NAME,
            "model": MODEL_NAME,
            "error": f"Speech model '{MODEL_NAME}' is not installed for realtime voice",
            "install_command": install_model_command(MODEL_NAME),
            "detail": "Install it explicitly with the provided command.",
        })
        return 1

    warm, warm_error = warm_backend(backend, numpy)
    release_inference_scratch(backend)
    emit({
        "type": "ready",
        "engine": ENGINE_NAME,
        "model": MODEL_NAME,
        "loaded": True,
        "warm": warm,
        "load_ms": round((time.monotonic() - started_at) * 1000),
        # Keep the reason operationally useful without exposing model internals.
        **({"warm_error": warm_error[:240]} if warm_error else {}),
    })

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("request must be a JSON object")
        except (json.JSONDecodeError, ValueError) as error:
            emit({"id": "unknown", "error": f"Invalid request: {error}"})
            continue

        identifier = request_id(request)
        operation = request.get("op")
        if operation == "health":
            emit({
                "id": identifier,
                "engine": ENGINE_NAME,
                "model": MODEL_NAME,
                "loaded": True,
                "warm": warm,
            })
            continue
        if operation == "close":
            emit({"id": identifier, "ok": True})
            return 0
        if operation != "transcribe_pcm":
            emit({"id": identifier, "error": "Unknown operation"})
            continue

        try:
            text, detected, confidence, language_confidence = transcribe(backend, request, numpy)
            response: dict[str, Any] = {"id": identifier, "text": text}
            if detected:
                response["detected_language"] = detected
            if language_confidence is not None:
                response["language_confidence"] = language_confidence
            if confidence is not None:
                response["confidence"] = confidence
            emit(response)
        except Exception as error:
            emit({
                "id": identifier,
                "error": f"Local {backend.display_name} transcription failed: {error}",
            })
        finally:
            release_inference_scratch(backend)

    return 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        assert default_engine("darwin", "arm64", "13.5") == MLX_ENGINE
        assert default_engine("darwin", "arm64", "13.4") == OPENAI_WHISPER_ENGINE
        assert default_engine("darwin", "x86_64", "14.0") == OPENAI_WHISPER_ENGINE
        assert default_engine("linux", "x86_64") == OPENAI_WHISPER_ENGINE
        assert mlx_whisper_needs_install("0.4.2")
        assert not mlx_whisper_needs_install(MLX_WHISPER_VERSION)
        assert choose_auto_language({"en": 0.02, "vi": 0.91, "ja": 0.07}) == ("vi", 0.91)
        assert choose_auto_language({"fr": 0.998, "<|invalid|>": 1.0}) == ("fr", 0.998)
        assert choose_auto_language(None) == ("en", None)
        class _CacheRecorder:
            calls = 0

            def clear_cache(self) -> None:
                self.calls += 1

        class _MlxFixture:
            engine = MLX_ENGINE
            mx = _CacheRecorder()

        mlx_fixture = _MlxFixture()
        release_inference_scratch(mlx_fixture)
        assert mlx_fixture.mx.calls == 1
        padded_tail = {
            "text": "Real speech Impossible tail",
            "segments": [
                {"start": 0.0, "end": 4.08, "text": "Real speech"},
                {"start": 4.08, "end": 34.06, "text": " Impossible tail"},
            ],
        }
        trimmed_tail = trim_impossible_mlx_tail(padded_tail, 4.148)
        assert trimmed_tail["text"] == "Real speech"
        assert len(trimmed_tail["segments"]) == 1
        emit({"type": "self_test", "ok": True})
        raise SystemExit(0)
    if sys.argv[1:] == ["--install"]:
        raise SystemExit(install())
    raise SystemExit(main())
