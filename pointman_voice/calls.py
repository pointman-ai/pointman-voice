"""A live call's mark for the machine's heavy-job lock: while a call runs, its file is touched every 10 s, and it's
deleted when the call ends. The lock's `take` waits while any file there is fresher than 60 s, so a crashed
call stops holding it after a minute."""

from __future__ import annotations

import os
import re
import secrets
import threading
from pathlib import Path

TOUCH_EVERY = 10.0
CALL_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


def calls_dir() -> Path:
    return Path(os.environ.get("POINTMAN_VOICE_CALLS") or "~/.local/state/pointman/voice/calls").expanduser()


def new_call_id() -> str:
    """A plain token, letters, digits and dashes: always a safe file name."""
    return f"call-{secrets.token_hex(6)}"


class Mark:
    """The file for one live call: made at start(), touched on its own thread, removed at stop()."""

    def __init__(self, call_id: str, folder: Path | None = None):
        if not CALL_ID.match(call_id):
            raise ValueError(f"not a call id: {call_id!r}")
        self.path = (folder or calls_dir()) / call_id
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> Mark:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch()
        self._thread = threading.Thread(target=self._touch, name=f"mark-{self.path.name}", daemon=True)
        self._thread.start()
        return self

    def _touch(self) -> None:
        while not self._stop.wait(TOUCH_EVERY):
            try:
                self.path.touch()
            except OSError:
                pass  # its folder went: the next touch makes nothing, and the lock sees no call

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
        self.path.unlink(missing_ok=True)
