"""A call's sound as Opus packets, one each way (docs/dev/call.md): the mic comes up as 16 kHz packets, the
voices go down as 24 kHz ones, each 20 ms. PyAV carries libopus in its wheel, so nothing is installed system-wide.
"""

from __future__ import annotations

import numpy as np

UP_RATE = 16000       # the mic, as listening wants it
DOWN_RATE = 24000     # the voices, Kokoro's own rate
FRAME_MS = 20


class MicDecoder:
    """One device's mic: Opus packets in, 16 kHz int16 samples out (libopus decodes at 48 kHz; we resample)."""

    def __init__(self):
        import av

        self._av = av
        self._dec = av.CodecContext.create("libopus", "r")
        self._dec.sample_rate = 48000
        self._dec.layout = "mono"
        self._dec.open()
        self._resample = av.AudioResampler(format="s16", layout="mono", rate=UP_RATE)

    def decode(self, packet: bytes) -> np.ndarray:
        out = []
        try:
            frames = self._dec.decode(self._av.Packet(packet))
        except Exception:  # noqa: BLE001 - a packet that won't decode is a gap, never the end of the call
            return np.zeros(0, dtype=np.int16)
        for f in frames:
            for r in self._resample.resample(f):
                out.append(r.to_ndarray().reshape(-1))
        return np.concatenate(out).astype(np.int16) if out else np.zeros(0, dtype=np.int16)


class VoiceEncoder:
    """An agent's voice: 24 kHz int16 samples in, one Opus packet per 20 ms out, the last frame padded. Each
    sentence is encoded whole and flushed, so none of it waits in the encoder for the next."""

    SAMPLES = DOWN_RATE * FRAME_MS // 1000  # 480

    def __init__(self, bit_rate: int = 32000):
        import av

        self._av = av
        self.bit_rate = bit_rate

    def _encoder(self):
        enc = self._av.CodecContext.create("libopus", "w")
        enc.sample_rate = DOWN_RATE
        enc.layout = "mono"
        enc.format = "s16"
        enc.bit_rate = self.bit_rate
        enc.options = {"application": "voip", "frame_duration": str(FRAME_MS)}
        enc.open()
        return enc

    def frames(self, pcm: np.ndarray) -> list[bytes]:
        """Every 20 ms of `pcm` as a packet, the rest padded with silence."""
        n = self.SAMPLES
        pad = (-len(pcm)) % n
        if pad:
            pcm = np.concatenate([pcm, np.zeros(pad, dtype=np.int16)])
        enc = self._encoder()
        packets: list[bytes] = []
        for i in range(0, len(pcm), n):
            f = self._av.AudioFrame.from_ndarray(pcm[i:i + n].reshape(1, -1), format="s16", layout="mono")
            f.sample_rate = DOWN_RATE
            packets.extend(bytes(p) for p in enc.encode(f))
        packets.extend(bytes(p) for p in enc.encode(None))
        return packets
