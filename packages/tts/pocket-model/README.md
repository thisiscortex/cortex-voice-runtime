# Pocket English model identity

The runtime uses the Pocket English clone capable model separately from source.
The upstream [Pocket TTS source](https://github.com/kyutai-labs/pocket-tts) is
MIT licensed. The [model card](https://huggingface.co/kyutai/pocket-tts) lists
the model as CC BY 4.0. A model release must carry its attribution and terms.

| Asset | Upstream revision | SHA-256 |
| --- | --- | --- |
| `model.safetensors` | `kyutai/pocket-tts` `39592ff23c9ef80098bb74895d104c26275fe2c9` | `473f47d99560bd50eb8b4509d3cacfe7f316ab20bdca86505403a2e6a936a6e9` |
| `tokenizer.model` | `kyutai/pocket-tts-without-voice-cloning` `d29db7978e464fb90cb3359ee0c69a273b9142cc` | `d461765ae179566678c93091c5fa6f2984c31bbe990bf1aa62d92c64d91bc3f6` |

The model weights, tokenizer and voice clone states are excluded from Git. The
runtime validates exact bytes before load. The public source tree does not
include the Jarvis clone state; qualifying its redistribution is a release gate.
