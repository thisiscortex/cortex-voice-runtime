# v0.1.1 — macOS 26 Apple Silicon voice runtimes

Separate STT and TTS Python 3.12 runtimes with hash-locked dependencies.
Both archives are relocatable and omit models, user audio, credentials and
the Jarvis clone. Cortex downloads them only after an explicit local voice
setup action, verifies their exact bytes and SHA-256, and retains its managed
installer if an artifact is missing or the platform is unsupported.

| Asset | Bytes | SHA-256 |
| --- | ---: | --- |
| `cortex-voice-stt-0.1.1-darwin-arm64-macos26.tar.gz` | 307024512 | `4dae7d13457e7fded62aecce2127b96ca9aa3577809085704c6dc3811b60d441` |
| `cortex-voice-tts-0.1.1-darwin-arm64-macos26.tar.gz` | 209106969 | `17e9b65a3ee967b1c52b56af1e9bd3c4403d12d9f2c297cd07393cfe4a24df0a` |

The STT release was exercised with the cached MLX Whisper Small model,
synthetic PCM and idle shutdown. The TTS release was installed with the
separate pinned Pocket English model and the already bundled Cortex Jarvis
profile, then produced nonempty PCM. The host integration, setup progress,
fallback, rollback and packaged Electron checks are recorded in Cortex's
voice extraction verification audit. These checks qualify the tested macOS
26 arm64 path; other operating systems and macOS versions retain the existing
installer until independently built and verified.
