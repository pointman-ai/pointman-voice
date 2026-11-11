"""Speech to text for one turn: Parakeet TDT v3 on MLX on a Mac, Whisper turbo (CTranslate2) on Linux.

One model per process, one transcription at a time (the lock): a call's turns are short and come one by one,
and the GPU is shared with the voices.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from ..models import ON_MAC

_model = None
_lock = threading.Lock()


def load() -> None:
    global _model
    with _lock:
        if _model is not None:
            return
        if ON_MAC:
            from parakeet_mlx import from_pretrained

            _model = from_pretrained("mlx-community/parakeet-tdt-0.6b-v3")
        else:
            from faster_whisper import WhisperModel

            try:
                _model = WhisperModel("mobiuslabsgmbh/faster-whisper-large-v3-turbo", device="cuda",
                                      compute_type="float16")
            except Exception:  # noqa: BLE001 - no GPU: the CPU, slower
                _model = WhisperModel("mobiuslabsgmbh/faster-whisper-large-v3-turbo", device="cpu",
                                      compute_type="int8")


def transcribe(pcm: np.ndarray) -> tuple[str, float]:
    """16 kHz int16 samples → (text, ms it took)."""
    load()
    audio = pcm.astype(np.float32) / 32768.0
    t = time.perf_counter()
    with _lock:
        if ON_MAC:
            import mlx.core as mx
            from parakeet_mlx.audio import get_logmel

            text = _model.generate(get_logmel(mx.array(audio), _model.preprocessor_config))[0].text
        else:
            segments, _ = _model.transcribe(audio, language="en", beam_size=1, vad_filter=False)
            text = " ".join(s.text for s in segments)
    return " ".join(text.split()), (time.perf_counter() - t) * 1000
