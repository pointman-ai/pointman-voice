"""The models a call needs on this machine, fetched once into the standard Hugging Face cache.

Kokoro (the voices) comes first, since the voice picker waits on it; then speech to text. Nothing sets
HF_HOME: what's in ~/.cache/huggingface/hub already counts, so a machine that has them fetches nothing.
"""

from __future__ import annotations

import fnmatch
import platform
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

ON_MAC = sys.platform == "darwin" and platform.machine() == "arm64"
RETRY_EVERY = 300.0  # seconds before a failed fetch is tried again


@dataclass
class Model:
    key: str                 # "voices" or "listen"
    repo: str
    allow: list[str]         # the files it needs
    needs: list[str]         # each of these must match a file for it to count as there
    name: str                # what health and the picker call it


def models_here() -> list[Model]:
    if ON_MAC:
        return [
            Model("voices", "mlx-community/Kokoro-82M-bf16", ["config.json", "kokoro-v1_0.safetensors",
                  "voices/*.safetensors"], ["kokoro-v1_0.safetensors", "voices/af_heart.safetensors"], "Kokoro voices"),
            Model("listen", "mlx-community/parakeet-tdt-0.6b-v3", ["config.json", "model.safetensors",
                  "tokenizer.model", "tokenizer.vocab", "vocab.txt"], ["model.safetensors"], "Parakeet"),
        ]
    return [
        Model("voices", "hexgrad/Kokoro-82M", ["config.json", "kokoro-v1_0.pth", "voices/*.pt"],
              ["kokoro-v1_0.pth", "voices/af_heart.pt"], "Kokoro voices"),
        Model("listen", "mobiuslabsgmbh/faster-whisper-large-v3-turbo", ["*.json", "*.txt", "model.bin"],
              ["model.bin"], "Whisper turbo"),
    ]


def _wanted(path: str, allow: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in allow)


def local_snapshot(model: Model) -> Path | None:
    """Its snapshot folder when every file it needs is in the cache already; no network."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        folder = Path(snapshot_download(model.repo, allow_patterns=model.allow, local_files_only=True))
    except (LocalEntryNotFoundError, OSError):
        return None
    for pattern in model.needs:
        if not any(p.exists() for p in folder.glob(pattern)):  # a dangling link is a file not fetched yet
            return None
    return folder


@dataclass
class Fetch:
    """One model's state on this machine."""
    model: Model
    ready: bool = False
    done_bytes: int = 0
    total_bytes: int = 0
    error: str | None = None
    current: str | None = None       # the file being fetched now
    _blobs: Path | None = None

    def progress(self) -> float:
        if self.ready:
            return 1.0
        if not self.total_bytes:
            return 0.0
        partial = 0
        if self._blobs and self._blobs.is_dir():
            partial = sum(p.stat().st_size for p in self._blobs.glob("*.incomplete"))
        return min(0.99, (self.done_bytes + partial) / self.total_bytes)

    def as_dict(self) -> dict:
        out = {"name": self.model.name, "repo": self.model.repo, "ready": self.ready,
               "progress": round(self.progress(), 3)}
        if self.total_bytes:
            out["bytes"] = self.total_bytes
        if self.error:
            out["error"] = self.error
        return out


@dataclass
class Models:
    """Fetches what's missing on a thread of its own, voices first, and says how far it's got."""
    fetches: list[Fetch] = field(default_factory=lambda: [Fetch(m) for m in models_here()])
    _thread: threading.Thread | None = None
    _wake: threading.Event = field(default_factory=threading.Event)

    def get(self, key: str) -> Fetch:
        return next(f for f in self.fetches if f.model.key == key)

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="models", daemon=True)
            self._thread.start()

    def _run(self) -> None:
        while True:
            for f in self.fetches:
                if not f.ready:
                    self._fetch(f)
            if all(f.ready for f in self.fetches):
                return
            self._wake.wait(RETRY_EVERY)
            self._wake.clear()

    def _fetch(self, f: Fetch) -> None:
        if local_snapshot(f.model):
            f.ready, f.error = True, None
            return
        try:
            from huggingface_hub import HfApi, hf_hub_download, try_to_load_from_cache
            from huggingface_hub.constants import HF_HUB_CACHE

            info = HfApi().model_info(f.model.repo, files_metadata=True)
            files = [(s.rfilename, s.size or 0) for s in info.siblings if _wanted(s.rfilename, f.model.allow)]
            f.total_bytes = sum(size for _, size in files)
            f._blobs = Path(HF_HUB_CACHE) / f"models--{f.model.repo.replace('/', '--')}" / "blobs"
            f.done_bytes = 0
            for name, size in files:
                cached = try_to_load_from_cache(f.model.repo, name, revision=info.sha)
                if not isinstance(cached, str):
                    f.current = name
                    hf_hub_download(f.model.repo, name, revision=info.sha)
                f.done_bytes += size
            f.current, f.error = None, None
            f.ready = local_snapshot(f.model) is not None
        except Exception as e:  # noqa: BLE001 - offline, rate limited, disk full: say so, try again later
            f.current = None
            f.error = f"{type(e).__name__}: {e}"[:200]

    def summary(self) -> str:
        """One line for health: what's here, or how far the fetch has got."""
        if all(f.ready for f in self.fetches):
            return "The voice models are on this machine: " + ", ".join(f.model.name for f in self.fetches) + "."
        parts = []
        for f in self.fetches:
            if f.ready:
                parts.append(f"{f.model.name} ready")
            elif f.error:
                parts.append(f"{f.model.name} couldn't download yet ({f.error}); trying again in "
                             f"{int(RETRY_EVERY // 60)} min")
            elif f.total_bytes:
                parts.append(f"{f.model.name} downloading: {f.progress():.0%} of {f.total_bytes / 1e9:.1f} GB")
            else:
                parts.append(f"{f.model.name} waiting to download")
        return "; ".join(parts) + "."

    def wait(self, key: str, timeout: float) -> bool:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self.get(key).ready:
                return True
            time.sleep(0.2)
        return self.get(key).ready
