"""Kokoro: the one voice model a call and a voice sample share, so a sample sounds as the call will.

The Mac runs it on MLX, Linux on PyTorch (on the GPU when there's one), each the model `models.py` fetches
into the standard Hugging Face cache. One generation runs at a time: calls and samples queue on the same
lock rather than contend for the GPU.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

from .. import models

RATE = 24_000  # Kokoro's own rate; nothing here resamples
SPEED_RANGE = (0.5, 2.0)
MAX_CHARS = 500  # a sample or one sentence of a call; longer text is the caller's to split


class VoiceError(ValueError):
    """Something the person can fix, as one sentence the apps show as it comes."""


_lock = threading.Lock()
_model = None  # the MLX model, or {lang_code: KPipeline} on Linux


def _folder() -> Path:
    model = next(m for m in models.models_here() if m.key == "voices")
    folder = models.local_snapshot(model)
    if folder is None:
        raise VoiceError("The voice model isn't on this machine yet.")
    return folder


def voices() -> list[str]:
    """Every voice id the model on this machine has, as Kokoro names them (af_heart, bm_george)."""
    suffix = ".safetensors" if models.ON_MAC else ".pt"
    return sorted(p.stem for p in (_folder() / "voices").glob(f"*{suffix}"))


def loaded() -> bool:
    return _model is not None


def load() -> None:
    """Load the model now (the first generation otherwise pays for it: about 5 s on an M4 Max)."""
    global _model
    with _lock:
        _load()


def _load() -> None:
    global _model
    if _model is not None:
        return
    if models.ON_MAC:
        from mlx_audio.tts.utils import load_model

        _folder()  # in the cache, or VoiceError
        # by repo id, not the snapshot's folder: mlx-audio tells the model's type from its name
        _model = load_model(next(m for m in models.models_here() if m.key == "voices").repo)
    else:
        _model = {}


def _generate(text: str, voice: str, speed: float):
    """Float32 samples at RATE for the whole text. Holds the lock while it runs."""
    import numpy as np

    lang_code = voice[0]  # Kokoro's prefix: a = American English, b = British English
    with _lock:
        _load()
        if models.ON_MAC:
            chunks = _model.generate(text=text, voice=str(_folder() / "voices" / f"{voice}.safetensors"),
                                     speed=speed, lang_code=lang_code)
            parts = [np.asarray(r.audio, dtype=np.float32) for r in chunks]
        else:
            from kokoro import KModel, KPipeline

            if "model" not in _model:
                import torch

                folder = _folder()
                device = "cuda" if torch.cuda.is_available() else "cpu"
                _model["model"] = KModel(repo_id="hexgrad/Kokoro-82M", config=str(folder / "config.json"),
                                         model=str(folder / "kokoro-v1_0.pth")).to(device).eval()
            pipeline = _model.get(lang_code)
            if pipeline is None:
                pipeline = _model[lang_code] = KPipeline(lang_code=lang_code, repo_id="hexgrad/Kokoro-82M",
                                                         model=_model["model"])
            voice_file = str(_folder() / "voices" / f"{voice}.pt")
            parts = [np.asarray(audio, dtype=np.float32) for _, _, audio in
                     pipeline(text, voice=voice_file, speed=speed)]
    parts = [p.reshape(-1) for p in parts if p is not None and p.size]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def to_pcm16(samples) -> bytes:
    """Float samples in [-1, 1] as 16-bit little-endian mono PCM."""
    import numpy as np

    return (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def check(text: str, voice: str, speed: float) -> str:
    """The text to say, or VoiceError for what a person can fix."""
    text = " ".join(str(text).split())
    if not text:
        raise VoiceError("There's no text to say.")
    if len(text) > MAX_CHARS:
        raise VoiceError(f"That's too long to say at once ({len(text)} characters, at most {MAX_CHARS}).")
    if not SPEED_RANGE[0] <= speed <= SPEED_RANGE[1]:
        raise VoiceError(f"The speed must be between {SPEED_RANGE[0]} and {SPEED_RANGE[1]}, not {speed}.")
    if voice not in voices():
        raise VoiceError(f"There's no voice called {voice!r} on this machine.")
    return text


def speak_now(text: str, voice: str, speed: float = 1.0) -> tuple[bytes, int]:
    """speak(), on the calling thread: for a command's own thread."""
    return to_pcm16(_generate(check(text, voice, speed), voice, speed)), RATE


async def speak(text: str, voice: str, speed: float = 1.0) -> tuple[bytes, int]:
    """One-shot speech: (16-bit little-endian mono PCM, sample rate). Raises VoiceError for what a person can fix."""
    text = check(text, voice, speed)
    samples = await asyncio.to_thread(_generate, text, voice, speed)
    return to_pcm16(samples), RATE
