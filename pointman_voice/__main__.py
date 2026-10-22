"""The voice extension's process: `python -m pointman_voice`, started and spoken to by the node.

- `/voice call @research @design` checks who can join a call with you and what this machine has ready. The
  call itself comes in a later version, with the node's two-way call audio and the call engine.
- `voice.voices` and `voice.sample` are the apps' voice picker's: Kokoro's English voices, best first, and a
  short sample in one of them, published as a file and never posted.
- Health has one part of its own: the voice models, ready or how far their download has got (never failing).
"""

from .libpaths import ensure

ensure()  # before anything loads a native library

import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import threading  # noqa: E402
import wave  # noqa: E402
from pathlib import Path  # noqa: E402

try:
    from pointman_extension import Extension
except ImportError:                 # a node from before the rename
    from mb_extension import Extension  # type: ignore[no-redef]

from . import models as model_files  # noqa: E402
from .engine import tts  # noqa: E402
from .voices import catalogue, known  # noqa: E402

ext = Extension()
models = model_files.Models()
lock = threading.Lock()

HANDLE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
LIVE_CLIENTS = {"claude"}          # who can hold a live session in v1 (a client's own `live` flag, later)
SAMPLE_TEXT_MAX = 200


def machine() -> str:
    return str(ext.node.get("machine") or "this machine")


def state_dir() -> Path:
    return Path(ext.node.get("state_dir") or Path.home() / ".local/state/pointman/voice")


@ext.on_start
def start() -> None:
    models.start()


def warm_voices() -> None:
    """Load Kokoro in the background once its files are here, so the picker's first sample doesn't wait."""
    if models.get("voices").ready and not tts.loaded():
        threading.Thread(target=_load_quietly, name="warm-voices", daemon=True).start()


def _load_quietly() -> None:
    try:
        tts.load()
    except Exception as e:  # noqa: BLE001 - the next sample says what's wrong
        ext.log(f"loading Kokoro: {e}")


@ext.command("voice.voices")
def voices(job) -> dict:
    f = models.get("voices")
    warm_voices()
    out: dict = {"ready": f.ready, "voices": catalogue()}
    if f.ready:
        out["text"] = f"{len(out['voices'])} English voices on {machine()}, best first."
    else:
        out["progress"] = round(f.progress(), 3)
        out["text"] = f"The voice model is still downloading on {machine()} ({f.progress():.0%})."
    return out


def _sample_index() -> tuple[Path, dict]:
    folder = state_dir() / "samples"
    folder.mkdir(parents=True, exist_ok=True)
    index = folder / "index.json"
    try:
        return folder, json.loads(index.read_text())
    except (OSError, ValueError):
        return folder, {}


@ext.command("voice.sample")
def sample(job) -> dict:
    p = job.params
    voice = str(p.get("voice") or "")
    if not known(voice):
        raise ValueError(f"There's no voice called {voice!r}: pick one from the voice list.")
    try:
        speed = round(float(p.get("speed") or 1.0), 2)
    except (TypeError, ValueError):
        raise ValueError(f"The speed must be a number between 0.5 and 2, not {p.get('speed')!r}.") from None
    name = " ".join(str(p.get("name") or "").split())[:40]
    text = " ".join(str(p.get("text") or "").split()) or (
        f"Hi, I'm {name}. This is how I'll sound on our calls." if name else "Hi. This is how I'll sound on our calls.")
    if len(text) > SAMPLE_TEXT_MAX:
        raise ValueError(f"The sample's text is too long ({len(text)} characters, at most {SAMPLE_TEXT_MAX}).")
    f = models.get("voices")
    if not f.ready:
        raise ValueError(f"The voice model is still downloading ({f.progress():.0%}).")

    key = hashlib.sha256(f"{voice}|{speed:.2f}|{text}".encode()).hexdigest()[:20]
    with lock:
        folder, index = _sample_index()
        if key in index:                           # the same voice, speed and text: the same file, at once
            return {**index[key], "voice": voice, "speed": speed, "text": text}

    pcm, rate = tts.speak_now(text, voice, speed)  # a VoiceError's sentence is the job's failure
    path = folder / f"{voice}-{key}.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    seconds = round(len(pcm) / 2 / rate, 2)
    answer = ext.request("files.publish", {"body": {
        "project": {"slug": "voice-samples", "name": "Voice samples"},
        "asset": {"path": f"samples/{path.name}", "title": f"{voice} at {speed:g}×", "kind": "audio"},
        "caption": text,
        "files": [{"role": "main", "file": str(path), "name": path.name, "mime": "audio/wav",
                   "size": path.stat().st_size, "duration": seconds}],
    }}, timeout=120)
    entry = {"asset_id": answer.get("asset_id"), "seconds": seconds}
    with lock:
        _, index = _sample_index()
        index[key] = entry
        (folder / "index.json").write_text(json.dumps(index))
    return {**entry, "voice": voice, "speed": speed, "text": text}


def _handles(raw: str) -> list[str]:
    seen: list[str] = []
    for word in re.split(r"[\s,]+", raw):
        h = word.strip().lstrip("@").lower()
        if h and h not in seen:
            seen.append(h)
    return seen


@ext.command("voice.call")
def call(job) -> dict:
    handles = _handles(str(job.params.get("agents") or ""))
    if not handles:
        raise ValueError("Who should join? Name them: /voice call @research @design")
    odd = [h for h in handles if not HANDLE.match(h)]
    if odd:
        raise ValueError(f"{', '.join(odd)} isn't an agent's handle.")
    try:
        here = {a.get("handle"): a for a in ext.request("agents.sessions", {}).get("agents", [])}
    except Exception as e:  # noqa: BLE001 - an older node: say what we can
        ext.log(f"agents.sessions: {e}")
        here = {}

    rows, ready = [], 0
    for h in handles:
        agent = here.get(h)
        if agent is None:
            rows.append(f"| @{h} | doesn't run on {machine()} |")
        elif (agent.get("client") or "claude") not in LIVE_CLIENTS:
            rows.append(f"| @{h} | a {agent.get('client')} agent: v1 calls are for Claude agents |")
        else:
            ready += 1
            rows.append(f"| @{h} | can join |")
    lines = [f"**Voice calls aren't open yet.** The call itself comes with the node's call audio and the "
             f"engine. Here's what {machine()} has ready for one with you and {len(handles)} "
             f"agent{'s' if len(handles) != 1 else ''}:", "", "| Agent | |", "|---|---|", *rows, "",
             models.summary()]
    return {"text": "\n".join(lines), "agents": handles, "can_join": ready}


@ext.method("health.check")
def health(_params) -> list[dict]:
    """Its own part under the extension's: the models, ok while they download (never failing)."""
    return [{"id": "extension/voice/models", "kind": "models", "name": "Voice models", "state": "ok",
             "why": models.summary()}]


def status() -> None:
    ext.event("voice.status", {"models": [f.as_dict() for f in models.fetches]})


ext.every(15, status, lane="status")

if __name__ == "__main__":
    ext.run()
