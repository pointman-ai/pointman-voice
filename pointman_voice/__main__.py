"""The voice extension's process: `python -m pointman_voice`, started and spoken to by the node.

- `/voice call @research @design` starts a call and opens it as a call pane on the device that ran it; its
  post's Join button (`voice.join`) opens it on any other. Until the engine speaks for the agents, the call
  plays each device's own mic back to it a second later, and says which agents could join.
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

import pointman_extension_types as types  # noqa: E402
from pointman_extension import Extension, Refused  # noqa: E402

from . import models as model_files  # noqa: E402
from .engine import tts  # noqa: E402
from .room import DOWN_RATE, UP_RATE, Calls, Ended  # noqa: E402
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
    answer = ext.request(types.FilesPublishRequest(body={
        "project": {"slug": "voice-samples", "name": "Voice samples"},
        "asset": {"path": f"samples/{path.name}", "title": f"{voice} at {speed:g}×", "kind": "audio"},
        "caption": text,
        "files": [{"role": "main", "file": str(path), "name": path.name, "mime": "audio/wav",
                   "size": path.stat().st_size, "duration": seconds}],
    }), timeout=120)
    entry = {"asset_id": _number(answer.file), "seconds": seconds}
    with lock:
        _, index = _sample_index()
        index[key] = entry
        (folder / "index.json").write_text(json.dumps(index))
    return {**entry, "voice": voice, "speed": speed, "text": text}


def _number(ref) -> int | None:
    """A typed id's number: file_4801 is 4801."""
    tail = (ref.value if ref is not None else "").rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else None


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
    odd = [h for h in handles if not HANDLE.match(h)]
    if odd:
        raise ValueError(f"{', '.join(odd)} isn't an agent's handle.")
    rows, ready = [], 0
    if handles:
        # the node lists the agents it has woken since it started: one not woken yet has no session here.
        # A failed request fails the job with the node's own reason (RuntimeError), never an empty list
        here = {a.handle.handle: a for a in ext.request(types.AgentsSessionsRequest()).agents if a.handle}
        for h in handles:
            agent = here.get(h)
            if agent is None:
                rows.append(f"| @{h} | no session on {machine()} yet |")
            elif (agent.client or "claude") not in LIVE_CLIENTS:
                rows.append(f"| @{h} | a {agent.client} agent: v1 calls are for Claude agents |")
            else:
                ready += 1
                rows.append(f"| @{h} | can join |")
    lines = ["**A voice call.** Agents don't speak in calls yet: until they do, the call plays your own mic "
             "back to you a second later, to try the sound. Join from another device with the button."]
    if rows:
        lines += ["", "| Agent | |", "|---|---|", *rows]
    lines += ["", models.summary()]
    text = "\n".join(lines)
    c = calls.start(agents=handles, text=text)
    return {**_answer(c), "agents": handles, "can_join": ready}


@ext.command("voice.join")
def join(job) -> dict:
    """The call post's Join button: the call's pane, on the device that tapped it. Its answer replaces the
    post's blocks, so it says the post again, Join button and all."""
    try:
        c = calls.get(str(job.params.get("call") or ""))
    except Ended as e:
        raise ValueError(str(e)) from None
    return _answer(c)


def _answer(c) -> dict:
    return {"text": c.text, "call": c.id,
            "blocks": [{"type": "actions", "buttons": [
                {"title": "Join", "command": "voice.join", "params": {"call": c.id}}]}],
            "pane": {"open": "call", "target": {"call": c.id}}}


# ---------------------------------------------------------------- the call pane

seats: dict[str, object] = {}                 # pane id → its Pane, for what the call sends it


def _send_event(pane: str, event: dict) -> None:
    p = seats.get(pane)
    if p is not None:
        p.send(types.PaneEventRequest(event=event))


def _send_audio(pane: str, captured_ms: int, opus: bytes) -> None:
    p = seats.get(pane)
    if p is not None:
        p.send(types.PaneAudioRequest(captured_ms=captured_ms, opus=opus))


calls = Calls(_send_event, _send_audio)


@ext.pane("call")
def call_pane(pane) -> types.PaneOpenResponse:
    person = pane.viewer.person if pane.viewer else None
    handle = (person.handle if person else None) or (person.id.value if person and person.id else "") or "you"
    seats[pane.id] = pane
    try:
        calls.sit(str(pane.target.get("call") or ""), pane.id, handle)
    except Ended as e:
        seats.pop(pane.id, None)
        raise ValueError(str(e)) from None

    @pane.on_input
    def control(event: types.PaneInput) -> None:
        try:
            calls.control(pane.id, event.kind, event.fields or {})
        except (Ended, ValueError) as e:
            raise Refused(str(e)) from None

    @pane.on_close
    def closed(_reason: str) -> None:
        calls.leave(pane.id)
        seats.pop(pane.id, None)

    opus = types.PaneOpenResponse.Call.Sound.Codec.CODEC_OPUS
    return types.PaneOpenResponse(title="Voice call", call=types.PaneOpenResponse.Call(
        down=types.PaneOpenResponse.Call.Sound(codec=opus, rate=DOWN_RATE, channels=1),
        up=types.PaneOpenResponse.Call.Sound(codec=opus, rate=UP_RATE, channels=1)))


@ext.method("pane.mic")
def mic(req: types.PaneMicRequest) -> None:
    calls.mic(req.pane.value if req.pane else "", req.captured_ms, req.opus)


@ext.method("pane.viewers")
def viewers(req: types.PaneViewersRequest) -> None:
    calls.viewers(req.pane.value if req.pane else "", req.attached)


@ext.method("health.check")
def health(_req: types.HealthCheckRequest) -> types.HealthCheckResponse:
    """Its own part under the extension's: the models, ok while they download (never failing)."""
    return types.HealthCheckResponse(parts=[types.Health.Part(
        id="extension/voice/models", kind="models", name="Voice models", state=types.Health.Part.State.STATE_OK,
        why=models.summary())])


def status() -> None:
    ext.event("voice.status", {"models": [f.as_dict() for f in models.fetches]})


ext.every(15, status, lane="status")

if __name__ == "__main__":
    ext.run()
