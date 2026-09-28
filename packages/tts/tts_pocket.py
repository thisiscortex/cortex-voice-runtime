#!/usr/bin/env python3
"""Persistent Pocket TTS helper for Cortex Live voice.

Pocket TTS yields audio incrementally on CPU.  Cortex keeps its existing
newline-JSON / length-framed-PCM protocol so the renderer and local TTS route
remain provider-agnostic:

* stdin: one JSON object per line, with ``text`` and optional ``lang``;
* stdout: repeated ``[uint32 BE length][PCM16LE]`` frames and a zero-length
  terminator per request;
* stderr: ``READY`` only after the model and exported Jarvis clone state are
  available.

The clone state is intentionally separate from the release-installed model.
It is an owner-only Pocket ``.safetensors`` state exported once from the
approved Jarvis 3 reference; normal startup is offline and never sends audio
or text to a remote service.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import hashlib
import importlib.metadata
import json
import os
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterator


POCKET_TTS_VERSION = "2.1.0"
SAMPLE_RATE = 24_000
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RUNTIME_ROOT = Path.home() / ".cortex-ai-sessions" / "voice-tts"
BUNDLED_VOICE_STATE_SHA256 = "3df5a2355cb6fa1fda5408d9099000ea966b741730ee320f6e6aadd4d7df75e5"
POCKET_MODEL_SHA256 = "473f47d99560bd50eb8b4509d3cacfe7f316ab20bdca86505403a2e6a936a6e9"
POCKET_MODEL_BYTES = 219_029_196
POCKET_TOKENIZER_SHA256 = "d461765ae179566678c93091c5fa6f2984c31bbe990bf1aa62d92c64d91bc3f6"
POCKET_TOKENIZER_BYTES = 59_339


def _path_env(name: str, default: Path) -> Path:
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser().resolve() if value else default.resolve()


def _profile_path() -> Path:
    return _path_env(
        "CORTEX_VOICE_TTS_POCKET_STATE",
        DEFAULT_RUNTIME_ROOT / "profiles" / "jarvis" / "pocket.safetensors",
    )


def _bundled_state_path() -> Path:
    return _path_env(
        "CORTEX_VOICE_TTS_POCKET_BUNDLED_STATE",
        SCRIPT_DIR / "voice-assets" / "jarvis-pocket.safetensors",
    )


def _bundled_assets_dir() -> Path:
    return _path_env(
        "CORTEX_VOICE_TTS_BUNDLED_ASSET_DIR",
        SCRIPT_DIR / "voice-assets",
    )


def _release_model_dir() -> Path:
    return _path_env(
        "CORTEX_VOICE_TTS_POCKET_MODEL_DIR",
        DEFAULT_RUNTIME_ROOT / "pocket-model",
    )


def _release_model_paths() -> tuple[Path, Path, Path]:
    model_dir = _release_model_dir()
    return (
        _bundled_assets_dir() / "pocket-model" / "english.yaml",
        model_dir / "model.safetensors",
        model_dir / "tokenizer.model",
    )


def _ensure_private_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass


def _fail(message: str) -> None:
    sys.stderr.write(f"tts_pocket: {message}\n")
    sys.stderr.flush()
    raise SystemExit(1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _provision_clone() -> None:
    source = _bundled_state_path()
    target = _profile_path()
    if not source.is_file():
        raise RuntimeError(f"bundled Pocket Jarvis clone is unavailable: {source}")
    if _sha256(source) != BUNDLED_VOICE_STATE_SHA256:
        raise RuntimeError("bundled Pocket Jarvis clone checksum does not match")
    _ensure_private_directory(target.parent)
    try:
        current = _sha256(target)
    except OSError:
        current = ""
    if current != BUNDLED_VOICE_STATE_SHA256:
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        try:
            with source.open("rb") as reader, temporary.open("wb") as writer:
                while chunk := reader.read(1024 * 1024):
                    writer.write(chunk)
            os.chmod(temporary, 0o600)
            os.replace(temporary, target)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
    else:
        try:
            os.chmod(target, 0o600)
        except OSError:
            pass


def _configure_offline_runtime() -> None:
    # The model and tokenizer are installed from Cortex's matching public
    # release asset. Forcing offline mode makes runtime network access
    # impossible even if Pocket changes its default resolver.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


@functools.lru_cache(maxsize=1)
def _verified_release_model_paths() -> tuple[Path, Path, Path]:
    template, model, tokenizer = _release_model_paths()
    if not template.is_file():
        raise RuntimeError(f"packaged Pocket model configuration is unavailable: {template}")
    for path, expected_bytes, expected_sha, name in (
        (model, POCKET_MODEL_BYTES, POCKET_MODEL_SHA256, "model"),
        (tokenizer, POCKET_TOKENIZER_BYTES, POCKET_TOKENIZER_SHA256, "tokenizer"),
    ):
        try:
            actual_bytes = path.stat().st_size
        except OSError as exc:
            raise RuntimeError(f"release Pocket {name} is unavailable: {path}") from exc
        if actual_bytes != expected_bytes or _sha256(path) != expected_sha:
            raise RuntimeError(f"release Pocket {name} checksum does not match")
    return template, model, tokenizer


def _load_release_model(tts_model: Any) -> Any:
    template, model, tokenizer = _verified_release_model_paths()
    contents = template.read_text(encoding="utf-8")
    replacements = {
        "__CORTEX_POCKET_MODEL__": json.dumps(str(model)),
        "__CORTEX_POCKET_TOKENIZER__": json.dumps(str(tokenizer)),
    }
    for placeholder, value in replacements.items():
        if placeholder not in contents:
            raise RuntimeError(f"packaged Pocket model configuration is missing {placeholder}")
        contents = contents.replace(placeholder, value)
    descriptor, temporary_path = tempfile.mkstemp(prefix="cortex-pocket-", suffix=".yaml")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as config:
            config.write(contents)
        return tts_model.load_model(config=temporary_path)
    finally:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass


class Synth:
    def __init__(self) -> None:
        _configure_offline_runtime()
        try:
            installed = importlib.metadata.version("pocket-tts")
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError("pocket-tts is not installed") from exc
        if installed != POCKET_TTS_VERSION:
            raise RuntimeError(
                f"pocket-tts {POCKET_TTS_VERSION} is required; found {installed}"
            )

        state_path = _profile_path()
        if not state_path.is_file():
            raise RuntimeError(
                "the reusable Pocket Jarvis clone is missing; re-run the latest Cortex update"
            )

        import numpy as np
        from pocket_tts import TTSModel

        # Pocket is deliberately CPU-first: its upstream implementation pins
        # one CPU thread and streams PCM while synthesis continues.  The
        # clone-capable model is loaded only from the checksummed release asset.
        self._np = np
        self._model = _load_release_model(TTSModel)
        self._state = self._model.get_state_for_audio_prompt(state_path)
        if self._model.sample_rate != SAMPLE_RATE:
            raise RuntimeError(
                f"unexpected Pocket sample rate: {self._model.sample_rate}"
            )

    def pcm_frames(self, text: str) -> Iterator[bytes]:
        cleaned = text.strip()
        if not cleaned:
            return
        for chunk in self._model.generate_audio_stream(self._state, cleaned):
            wav = self._np.asarray(chunk.detach().cpu().numpy(), dtype=self._np.float32)
            pcm = (self._np.clip(wav.reshape(-1), -1.0, 1.0) * 32767.0).astype("<i2")
            if pcm.size:
                yield pcm.tobytes()


def _read_request(line: str) -> dict[str, Any]:
    value = json.loads(line)
    if not isinstance(value, dict):
        raise ValueError("request must be an object")
    return value


def _write_frames(out: Any, synth: Synth, text: str) -> None:
    for pcm in synth.pcm_frames(text):
        out.write(struct.pack(">I", len(pcm)))
        out.write(pcm)
        out.flush()
    out.write(struct.pack(">I", 0))
    out.flush()


def run_once(out: Any) -> None:
    try:
        request = json.load(sys.stdin)
        if not isinstance(request, dict):
            raise ValueError("request must be an object")
    except (json.JSONDecodeError, ValueError) as exc:
        _fail(f"bad params: {exc}")
    text = str(request.get("text") or "").strip()
    if not text:
        return
    try:
        synth = Synth()
        for pcm in synth.pcm_frames(text):
            out.write(pcm)
        out.flush()
    except Exception as exc:  # noqa: BLE001
        _fail(f"synthesis failed: {exc}")


def run_daemon(out: Any) -> None:
    try:
        synth = Synth()
        # Exercise the stream before advertising readiness.  A non-empty PCM
        # chunk proves the cache, clone state, and model version agree.
        if not next(synth.pcm_frames("Ready."), b""):
            raise RuntimeError("warmup produced no audio")
    except Exception as exc:  # noqa: BLE001
        _fail(f"offline model/clone startup failed: {exc}")

    sys.stderr.write("READY\n")
    sys.stderr.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = _read_request(line)
        except (json.JSONDecodeError, ValueError) as exc:
            sys.stderr.write(f"tts_pocket: bad daemon request: {exc}\n")
            _write_frames(out, synth, "")
            continue
        if request.get("op") == "close":
            break
        try:
            _write_frames(out, synth, str(request.get("text") or ""))
        except Exception as exc:  # noqa: BLE001
            sys.stderr.write(f"tts_pocket: synth error: {exc}\n")
            sys.stderr.flush()
            _write_frames(out, synth, "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--daemon", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    # Pocket and a few optional audio helpers can print progress while loading.
    # The framed protocol reserves stdout exclusively for binary PCM.
    out = sys.stdout.buffer
    with contextlib.redirect_stdout(sys.stderr):
        if args.prepare:
            _provision_clone()
        if args.check or args.prepare:
            synth = Synth()
            if not next(synth.pcm_frames("Ready."), b""):
                _fail("check synthesis produced no audio")
            return
        if args.daemon:
            run_daemon(out)
        else:
            run_once(out)


if __name__ == "__main__":
    main()
