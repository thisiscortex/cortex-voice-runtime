# Cortex voice runtime

Local voice components for Cortex. STT and TTS use separate Python environments,
models and processes. The host retains microphone access, authentication,
conversation state, playback and Stop semantics.

## Current state

The v0.1.1 prebuilt runtimes target Apple Silicon macOS 26. Cortex verifies
each archive's fixed size and SHA-256, contained entrypoints and import health.
The TTS installer also qualifies its separate model/profile with offline PCM
before activation; STT checks its separate model on use. On other systems, or when a release is
unavailable, Cortex retains its existing managed voice installer. The runtime
release contains neither speech model weights nor a cloned voice profile.

`packages/stt/stt_daemon.py` preserves the existing private newline JSON and
PCM16LE protocol. `packages/tts/tts_pocket.py` preserves the length framed PCM
protocol. The test suite exercises these contracts without downloading weights
or reading microphone data.

## Source and model boundaries

- Runtime source is MIT licensed. Third party Python packages keep their own
  licenses and are installed from pinned, reviewed sources.
- Pocket English weights/tokenizer remain a separate versioned download with
  their own attribution and license. The model revision and SHA-256 are listed
  in `packages/tts/pocket-model/README.md`.
- The existing Jarvis clone state is not included in this public source tree.
  Its public redistribution provenance needs a separate review. Cortex supplies
  its existing bundled state locally, stores an owner-only profile and proves
  nonempty offline PCM before activating a downloaded TTS artifact.
- No provider API key, private audio, user transcript, local profile, virtual
  environment or model cache belongs in this repository or its release source.

## Development verification

```sh
python3 -m unittest discover -s tests -p '*_test.py' -v
python3 packages/stt/stt_daemon.py --self-test
python3 packages/tts/tts_pocket.py --help
node scripts/verify-source.mjs
```

On a qualified Apple Silicon build host, prepare each component independently:

```sh
node scripts/build-runtime.mjs stt
node scripts/verify-release.mjs dist/cortex-voice-stt-0.1.1-darwin-arm64-macos26.tar.gz
node scripts/build-runtime.mjs tts
node scripts/verify-release.mjs dist/cortex-voice-tts-0.1.1-darwin-arm64-macos26.tar.gz
```

The build uses managed Python 3.12 and hash-locked wheels, then moves the
assembled runtime before import/self-test. The generated source archive has a
manifest and external `.release.json` with its exact byte count and SHA-256.
It carries no model weights or cloned voice. The current local trial was built
on macOS 26; MLX wheels selected there require that OS major, so its manifest
declares `minimumMacos: "26.0"`. Other platforms need their own build and real
qualification. Generated `dist/` files stay out of Git.

The host integration gate also exercises local voice start/stop, model cache
release, setup progress, first voice use, packaged launch and conversation
readability. A source test alone does not qualify a user install.
