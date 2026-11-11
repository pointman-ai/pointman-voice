"""Silero VAD on ONNX: is someone speaking in each 32 ms of 16 kHz audio. Each listener has its own state."""

from __future__ import annotations

import threading

import numpy as np

REPO, FILE = "onnx-community/silero-vad", "onnx/model.onnx"
WINDOW = 512          # samples per decision at 16 kHz (32 ms), as Silero takes them
RATE = 16000

_session = None
_lock = threading.Lock()


def _model():
    global _session
    with _lock:
        if _session is None:
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            _session = ort.InferenceSession(hf_hub_download(REPO, FILE), opts, providers=["CPUExecutionProvider"])
    return _session


class Vad:
    """Feed 16 kHz int16 samples as they come; each whole window gives a speech probability."""

    def __init__(self):
        self._s = _model()
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._left = np.zeros(0, dtype=np.float32)
        self._sr = np.array(RATE, dtype=np.int64)

    def feed(self, pcm: np.ndarray) -> list[float]:
        x = np.concatenate([self._left, pcm.astype(np.float32) / 32768.0])
        out = []
        n = len(x) // WINDOW * WINDOW
        for i in range(0, n, WINDOW):
            prob, self._state = self._s.run(None, {"input": x[None, i:i + WINDOW], "state": self._state,
                                                   "sr": self._sr})
            out.append(float(prob.reshape(-1)[0]))
        self._left = x[n:]
        return out
