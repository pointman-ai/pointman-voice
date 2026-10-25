# Voice

A Pointman extension for voice calls with your agents, run on your own machines: it listens, decides
when you've finished, and gives each agent its own voice. Audio never leaves the machine the call runs on.

This first version holds what a call needs on each machine:

- **The voice models,** fetched once into the standard Hugging Face cache (`~/.cache/huggingface/hub`):
  Kokoro for voices and Parakeet for listening on a Mac (MLX), Kokoro and Whisper turbo on Linux
  (PyTorch, CTranslate2). Health shows how far a download has got.
- **`/voice voices`:** Kokoro's English voices, 20 American and 8 British, best first by Kokoro's own grades.
- **`voice.sample`:** a few seconds of one voice at a given speed, as a WAV published to the workspace and
  never posted. The apps' voice picker plays it. The same voice, speed and text give the same file at once.
- **`/voice call @research @design`:** starts a call and opens it as a call pane on the device that ran it.
  Its post has a Join button (`voice.join`) that opens the same call on any other device. Until the agents
  speak in calls, a call plays each device's own mic back to it a second later, so the sound, the events and
  the controls (mute, raise a hand) can be tried end to end. The post says which agents could join.

| Command | Params | Result |
|---|---|---|
| `voice.voices` | none | `{ready, progress?, voices: [{id, name, language, gender, grade}], text}` |
| `voice.sample` | `voice`, `speed` (0.5–2, default 1), `name`, `text` (≤200) | `{asset_id, voice, speed, seconds, text}` |
| `voice.call` | `agents`: handles, as typed (optional) | `{text, blocks, pane, call, agents, can_join}` |
| `voice.join` | `call` | `{text, blocks, pane, call}`: the call's post again, and its pane |

A failure is one sentence to show as it comes: an unknown voice, a speed outside 0.5–2, text that's too
long, or "The voice model is still downloading (40%)."

**The call pane** (`[[panes]] call`, format `call`): Opus both ways, the voices at 24 kHz and the mic at
16 kHz. Each device gets the call as it is when its pane opens, and again whenever its connection comes
back: who's in it, then whose hands are up.

**A live call's mark:** while a call runs, `~/.local/state/pointman/voice/calls/<call-id>` is touched every
10 s and deleted when the call ends, so heavy jobs on the machine can wait for it.

## Platforms

`macos-arm64` and `linux-x86_64`. On Linux the extension starts itself once more with the CUDA libraries
pip installed on the loader's path; on NixOS also the GPU driver's folder and a libstdc++. Without a GPU,
Kokoro runs on the CPU.

## Tests

```sh
uv sync
POINTMAN_EXTENSIONS_SDK=<the node's SDK folder> PYTHONPATH=.:tests .venv/bin/python -m unittest discover -s tests
```

The tests that start the extension need the node's Python SDK and skip without it; those that speak need
Kokoro in the cache.
