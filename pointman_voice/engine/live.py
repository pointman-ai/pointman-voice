"""The call engine: it listens to each device's mic, decides when a person has finished, gives the agents what
they said, speaks each agent's answer in its own voice, and stops an agent the moment someone talks over it.

It knows nothing of who hosts it. A host (the node, or tonight the voice extension) passes in each call's start,
seats, mic and live-session notifications, and gives the engine a Host to send sound and events and to reach the
agents. Spark's measured values set the timing: barge-in after 0.2 s of speech, a turn ends after 0.6 s of
silence, and a set line after 1.5 s of an agent working unheard.
"""

from __future__ import annotations

import itertools
import json
import queue
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

WINDOW_S = 0.032          # one VAD decision (512 samples at 16 kHz)
SPEECH_ON, SPEECH_OFF = 0.5, 0.35
START_WINDOWS = 3         # ~0.1 s of speech before it counts as speaking
MIN_TURN_S = 0.25         # shorter than this is a cough, not a turn
MAX_TURN_S = 30.0
PARTIAL_EVERY_S = 1.0
PREROLL_S = 0.2
VOICES = ["af_heart", "am_michael", "bf_emma", "am_fenrir", "af_bella", "bm_george", "af_nicole", "am_puck"]
SET_LINES = ["One moment.", "Let me check.", "Give me a second.", "Just a moment."]

INSTRUCTIONS = (
    "You're in a live voice call with {people}. Everything you write is spoken aloud by a text-to-speech voice, "
    "so talk as you would on a call: short spoken sentences, no markdown, lists, headings, tables or code, and "
    "say numbers, names and links plainly. Keep most answers to one to three sentences unless you're asked for "
    "more. You can still use your tools to look things up or act; say in a few words what you're doing first. "
    "If you post something to the thread, say so briefly.")
GREETING = "{person} has started a voice call with you. Say hello in one short sentence."


class Host(Protocol):
    def audio(self, pane: str, captured_ms: int, opus: bytes) -> None: ...
    def flush(self, pane: str) -> None: ...
    def event(self, pane: str, event: dict) -> None: ...
    def join(self, agent: str, pane: str, instructions: str, effort: str) -> dict: ...  # {live, voice, voice_speed}
    def say(self, live: str, text: str) -> None: ...
    def interrupt(self, live: str) -> None: ...
    def leave(self, live: str) -> None: ...
    def joined(self, call: str, handle: str) -> None: ...      # an agent is in: the host tells the devices
    def left(self, call: str, handle: str) -> None: ...
    def log(self, line: str) -> None: ...


@dataclass
class Settings:
    barge_in_s: float = 0.2
    commit_silence_s: float = 0.6
    max_wait_s: float = 2.0
    filler_after_s: float = 1.5

    @classmethod
    def of(cls, d: dict | None) -> Settings:
        d = d or {}
        return cls(**{k: float(d[k]) for k in ("barge_in_s", "commit_silence_s", "max_wait_s", "filler_after_s")
                      if isinstance(d.get(k), (int, float))})


# ---------------------------------------------------------------- what an agent says, made speakable

_FENCE = re.compile(r"```.*?```", re.S)
_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_URL = re.compile(r"https?://\S+")
_SENTENCE = re.compile(r"(.+?[.!?…]+[\"')\]]*)(?:\s+|$)", re.S)


def speakable(text: str) -> str:
    text = _FENCE.sub(" ", text)
    text = _LINK.sub(r"\1", text)
    text = _URL.sub("a link", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"[*_#>|]+", " ", text)
    text = re.sub(r"^\s*[-•]\s+", "", text, flags=re.M)
    return " ".join(text.split())


def sentences(buffer: str, *, done: bool) -> tuple[list[str], str]:
    """Whole sentences out of what's streamed so far, and what's left over. A long run without a full stop is cut
    at its last comma so the voice can start; at the turn's end everything goes."""
    out, rest = [], buffer
    while True:
        m = _SENTENCE.match(rest)
        if not m:
            break
        out.append(m.group(1))
        rest = rest[m.end():]
    if not done and len(rest) > 160 and "," in rest[40:]:
        cut = rest.rindex(",") + 1
        out.append(rest[:cut])
        rest = rest[cut:]
    if done and rest.strip():
        out.append(rest)
        rest = ""
    return [s for s in (speakable(x) for x in out) if s], rest


# ---------------------------------------------------------------- the pieces of a call


@dataclass
class Agent:
    handle: str
    live: str
    voice: str
    speed: float = 1.0
    next_seq: int = 0
    held: dict = field(default_factory=dict)          # out-of-order notifications, by seq
    pending: str = ""                                  # streamed words not yet a sentence
    turn: bool = False                                 # a turn is running (said to, not yet ended)
    dropping: bool = False                             # cut by a barge-in: drop the rest of this turn
    spoke: bool = False                                # this turn has made a sound
    filler: threading.Timer | None = None
    said_at: float = 0.0                               # when the person's words went to it, for timings
    first_text_at: float = 0.0
    cid: str | None = None                             # this turn's caption


@dataclass
class Listener:
    """One device's mic: voice detection, the turn being spoken, and its partial captions."""

    pane: str
    person: str
    decoder: object
    vad: object
    speaking: bool = False
    run: int = 0                     # speech windows in a row
    quiet: int = 0                   # silent windows in a row
    voiced: int = 0                  # speech windows in this turn
    audio: list = field(default_factory=list)
    preroll: list = field(default_factory=list)
    turn_no: int = 0
    last_partial: float = 0.0
    barged: bool = False


class Speaker:
    """The call's one voice out: sentences become sound on their own thread, and a pacer sends one 20 ms packet
    every 20 ms to every device, as a sound card takes it (call.md)."""

    def __init__(self, call: Call):
        self.call = call
        self._synth: queue.Queue = queue.Queue()
        self._play: queue.Queue = queue.Queue()
        self._gen = 0                     # bumped by a cut: everything older is dropped
        self._lock = threading.Lock()
        self.current: dict | None = None  # the turn being played: {agent, cid, heard: [..], playing}
        self.busy = False
        self._stop = False
        threading.Thread(target=self._synth_loop, name=f"synth-{call.id}", daemon=True).start()
        threading.Thread(target=self._play_loop, name=f"play-{call.id}", daemon=True).start()

    def say(self, agent: Agent, text: str, cid: str, *, last: bool = False) -> None:
        with self._lock:
            self.busy = True
            self._synth.put((self._gen, agent, text, cid, last))

    def end_turn(self, agent: Agent, cid: str) -> None:
        with self._lock:
            self._synth.put((self._gen, agent, None, cid, True))

    def cut(self) -> dict | None:
        """Stop now: what was playing is dropped on every device, and its caption ends with what was heard."""
        with self._lock:
            self._gen += 1
            cur, self.current = self.current, None
            self.busy = False
            for q in (self._synth, self._play):
                while not q.empty():
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        break
        for pane in self.call.panes():
            self.call.host.flush(pane)
        if cur:
            heard = " ".join(cur["heard"] + ([cur["partial"]] if cur.get("partial") else [])).strip()
            self.call.caption(cur["cid"], cur["agent"].handle, heard or "…", final=True)
            self.call.speaking(cur["agent"].handle, False)
        return cur

    def stop(self) -> None:
        self._stop = True
        self.cut()
        self._synth.put(None)
        self._play.put(None)

    def _synth_loop(self) -> None:
        while not self._stop:
            item = self._synth.get()
            if item is None:
                return
            gen, agent, text, cid, last = item
            if gen != self._gen:
                continue
            packets: list[bytes] = []
            took = 0.0
            if text:
                t = time.perf_counter()
                try:
                    pcm = self.call.engine.tts(text, agent.voice, agent.speed)
                    packets = self.call.engine.encoder.frames(pcm)
                except Exception as e:  # noqa: BLE001 - one sentence that won't speak is skipped, not the call
                    self.call.host.log(f"voice: {agent.handle}: {e}")
                took = (time.perf_counter() - t) * 1000
            self._play.put((gen, agent, text, cid, last, packets, took))

    def _play_loop(self) -> None:
        tick = 0.020
        while not self._stop:
            item = self._play.get()
            if item is None:
                return
            gen, agent, text, cid, last, packets, took = item
            with self._lock:
                if gen != self._gen:
                    continue
                if self.current is None or self.current["cid"] != cid:
                    self.current = {"agent": agent, "cid": cid, "heard": [], "partial": ""}
                    self.call.speaking(agent.handle, True)
                cur = self.current
            if text:
                shown = " ".join(cur["heard"] + [text])
                self.call.caption(cid, agent.handle, shown, final=False)
                if not agent.spoke:
                    agent.spoke = True
                    self.call.timing(agent, "first_sound", took)
                words = text.split()
                n = max(1, len(packets))
                nxt = time.monotonic()
                for i, p in enumerate(packets):
                    if gen != self._gen or self._stop:
                        break
                    for pane in self.call.panes():
                        self.call.host.audio(pane, int(time.time() * 1000), p)
                    cur["partial"] = " ".join(words[: max(1, round(len(words) * (i + 1) / n))])
                    nxt += tick
                    time.sleep(max(0.0, nxt - time.monotonic()))
                if gen != self._gen:
                    continue
                cur["heard"].append(text)
                cur["partial"] = ""
            if last:
                with self._lock:
                    if gen != self._gen:
                        continue
                    done, self.current = self.current, None
                    if self._synth.empty() and self._play.empty():
                        self.busy = False
                if done and done["heard"]:
                    self.call.caption(cid, agent.handle, " ".join(done["heard"]), final=True)
                if done:
                    self.call.speaking(agent.handle, False)


class Call:
    def __init__(self, engine: Engine, call_id: str, thread: str, agents: list[str]):
        self.engine, self.host = engine, engine.host
        self.id, self.thread = call_id, thread
        self.wanted = list(agents)
        self.agents: dict[str, Agent] = {}             # by live id
        self.listeners: dict[str, Listener] = {}       # by pane
        self.transcript: list[dict] = []
        self.speaker = Speaker(self)
        self.joining = False
        self.last_agent: Agent | None = None
        self._ids = itertools.count(1)
        self.lock = threading.RLock()

    # --------------------------------------------- to the devices

    def panes(self) -> list[str]:
        with self.lock:
            return list(self.listeners)

    def send(self, event: dict) -> None:
        for p in self.panes():
            self.host.event(p, event)

    def caption(self, cid: str, handle: str, text: str, *, final: bool) -> None:
        self.send({"kind": "caption", "id": cid, "handle": handle, "text": text, "final": final})
        if final and text.strip():
            self.transcript.append({"handle": handle, "text": text, "at": time.time()})

    def speaking(self, handle: str, on: bool) -> None:
        self.send({"kind": "speaking", "handle": handle, "on": on})

    def cid(self, handle: str) -> str:
        return f"{handle}-{next(self._ids)}"

    def timing(self, agent: Agent, mark: str, ms: float = 0.0) -> None:
        now = time.monotonic()
        line = {"call": self.id, "agent": agent.handle, "mark": mark,
                "since_said_ms": round((now - agent.said_at) * 1000) if agent.said_at else None}
        if ms:
            line["took_ms"] = round(ms)
        self.host.log("timing " + json.dumps(line))

    # --------------------------------------------- agents

    def join_all(self, pane: str) -> None:
        people = ", ".join(sorted({l.person for l in self.listeners.values()})) or "a person"
        for i, handle in enumerate(self.wanted):
            try:
                got = self.host.join(handle, pane, INSTRUCTIONS.format(people=people), "low")
            except Exception as e:  # noqa: BLE001 - one agent that can't join leaves the others in the call
                self.host.log(f"{handle} couldn't join call {self.id}: {e}")
                continue
            live = str(got.get("live") or "")
            if not live:
                continue
            a = Agent(handle=handle, live=live, voice=got.get("voice") or VOICES[i % len(VOICES)],
                      speed=float(got.get("voice_speed") or 1.0))
            with self.lock:
                self.agents[live] = a
            self.host.joined(self.id, handle)
            a.turn, a.said_at = True, time.monotonic()
            self.host.say(live, GREETING.format(person=people))
        self.joining = False

    def agent_for(self, text: str) -> Agent | None:
        with self.lock:
            agents = list(self.agents.values())
        if not agents:
            return None
        low = f" {text.lower()} "
        for a in agents:
            if re.search(rf"\b{re.escape(a.handle.lower())}\b", low):
                return a
        return self.last_agent if self.last_agent in agents else agents[0]

    def notified(self, live: str, seq: int, kind: str, payload: dict) -> None:
        """agents.text / tool / turn / left, put back in order by their seq before they're acted on."""
        with self.lock:
            a = self.agents.get(live)
            if a is None:
                return
            a.held[seq] = (kind, payload)
            ready = []
            while a.next_seq in a.held:
                ready.append(a.held.pop(a.next_seq))
                a.next_seq += 1
        for kind, payload in ready:
            getattr(self, f"_on_{kind}")(a, payload)

    def _on_text(self, a: Agent, p: dict) -> None:
        if a.dropping:
            return
        if not a.first_text_at:
            a.first_text_at = time.monotonic()
            self.timing(a, "first_text")
        self._cancel_filler(a)
        a.pending += p.get("text", "")
        out, a.pending = sentences(a.pending, done=False)
        if out and not a.cid:
            a.cid = self.cid(a.handle)
        for s in out:
            self.speaker.say(a, s, a.cid)
            self.last_agent = a

    def _on_tool(self, a: Agent, p: dict) -> None:
        if a.dropping or p.get("state") != "start" or a.spoke or a.filler is not None:
            return
        a.filler = threading.Timer(self.engine.settings.filler_after_s, self._fill, args=(a,))
        a.filler.daemon = True
        a.filler.start()

    def _fill(self, a: Agent) -> None:
        if a.spoke or a.dropping or not a.turn:
            return
        a.cid = a.cid or self.cid(a.handle)
        self.speaker.say(a, SET_LINES[next(self._ids) % len(SET_LINES)], a.cid)

    def _cancel_filler(self, a: Agent) -> None:
        if a.filler is not None:
            a.filler.cancel()

    def _on_turn(self, a: Agent, p: dict) -> None:
        self._cancel_filler(a)
        if not a.dropping:
            out, a.pending = sentences(a.pending, done=True)
            cid = a.cid or self.cid(a.handle)
            for s in out:
                self.speaker.say(a, s, cid)
            if a.cid or out:
                self.speaker.end_turn(a, cid)
        self.timing(a, "turn_end")
        a.pending, a.cid, a.turn, a.dropping, a.spoke, a.filler, a.first_text_at = "", None, False, False, False, None, 0.0

    def _on_left(self, a: Agent, p: dict) -> None:
        with self.lock:
            self.agents.pop(a.live, None)
        self._cancel_filler(a)
        self.host.left(self.id, a.handle)

    # --------------------------------------------- people

    def heard(self, pane: str, opus: bytes) -> None:
        with self.lock:
            lis = self.listeners.get(pane)
        if lis is None:
            return
        pcm = lis.decoder.decode(opus)
        if not len(pcm):
            return
        s = self.engine.settings
        for prob in lis.vad.feed(pcm):
            if prob >= SPEECH_ON:
                lis.run += 1
                lis.quiet = 0
            elif prob < SPEECH_OFF:
                lis.quiet += 1
                lis.run = 0
            if not lis.speaking and lis.run >= START_WINDOWS:
                lis.speaking, lis.voiced, lis.barged = True, 0, False
                lis.turn_no += 1
                lis.audio = list(lis.preroll)
                lis.last_partial = time.monotonic()
                self.speaking(lis.person, True)
            if lis.speaking:
                if prob >= SPEECH_ON:
                    lis.voiced += 1
                if not lis.barged and lis.voiced * WINDOW_S >= s.barge_in_s and self.speaker.busy:
                    lis.barged = True
                    self.barge_in()
                if lis.quiet * WINDOW_S >= s.commit_silence_s or lis.voiced * WINDOW_S >= MAX_TURN_S:
                    self.finish(lis)
        if lis.speaking:
            lis.audio.append(pcm)
            if time.monotonic() - lis.last_partial >= PARTIAL_EVERY_S:
                lis.last_partial = time.monotonic()
                self.engine.partial(self, lis, np.concatenate(lis.audio), f"{lis.person}-t{lis.turn_no}")
        else:
            lis.preroll = (lis.preroll + [pcm])[-int(PREROLL_S / 0.02) - 1:]

    def barge_in(self) -> None:
        cut = self.speaker.cut()
        if cut is None:
            return
        a: Agent = cut["agent"]
        self.host.log(f"barge-in on {a.handle} in call {self.id}")
        if a.turn:
            a.dropping = True
            try:
                self.host.interrupt(a.live)
            except Exception as e:  # noqa: BLE001
                self.host.log(f"interrupt {a.handle}: {e}")

    def finish(self, lis: Listener) -> None:
        audio = np.concatenate(lis.audio) if lis.audio else np.zeros(0, dtype=np.int16)
        voiced = lis.voiced * WINDOW_S
        lis.speaking, lis.audio, lis.run, lis.quiet, lis.voiced = False, [], 0, 0, 0
        self.speaking(lis.person, False)
        cid = f"{lis.person}-t{lis.turn_no}"
        if voiced < MIN_TURN_S:
            return
        self.engine.final(self, lis, audio, cid, time.monotonic())

    def said(self, person: str, text: str, ended_at: float, stt_ms: float) -> None:
        a = self.agent_for(text)
        if a is None:
            return
        words = text if len({l.person for l in self.listeners.values()}) < 2 else f"{person}: {text}"
        a.said_at, a.turn, a.first_text_at = ended_at, True, 0.0
        self.host.log("timing " + json.dumps({"call": self.id, "agent": a.handle, "mark": "said",
                                              "stt_ms": round(stt_ms),
                                              "since_end_ms": round((time.monotonic() - ended_at) * 1000)}))
        try:
            self.host.say(a.live, words)
        except Exception as e:  # noqa: BLE001
            self.host.log(f"say to {a.handle}: {e}")


class Engine:
    """Every call this process runs. tts, stt, vad and the codecs can be swapped (tests use fakes)."""

    def __init__(self, host: Host, settings: dict | None = None, *, tts: Callable | None = None,
                 stt: Callable | None = None, vad: Callable | None = None, decoder: Callable | None = None,
                 encoder=None):
        self.host = host
        self.settings = Settings.of(settings)
        if tts is None:
            from . import tts as kokoro

            def tts(text, voice, speed):
                pcm, _ = kokoro.speak_now(text, voice, speed)
                return np.frombuffer(pcm, dtype="<i2")
        if stt is None:
            from . import stt as listen

            stt = listen.transcribe
        if vad is None:
            from .vad import Vad as vad
        if decoder is None or encoder is None:
            from .codec import MicDecoder, VoiceEncoder

            decoder = decoder or MicDecoder
            encoder = encoder or VoiceEncoder()
        self.tts, self.stt, self.vad, self.decoder, self.encoder = tts, stt, vad, decoder, encoder
        self.calls: dict[str, Call] = {}
        self._work: queue.Queue = queue.Queue()        # speech to text, one at a time
        self._partial_busy = False
        threading.Thread(target=self._worker, name="listen", daemon=True).start()

    # --------------------------------------------- the host's calls

    def start(self, call: str, thread: str, agents: list[str]) -> None:
        self.calls.setdefault(call, Call(self, call, thread, agents))

    def seat(self, call: str, pane: str, person: str) -> None:
        c = self.calls[call]
        with c.lock:
            c.listeners[pane] = Listener(pane=pane, person=person, decoder=self.decoder(), vad=self.vad())
            first = c.wanted and not c.agents and not c.joining
            if first:
                c.joining = True
        if first:
            threading.Thread(target=c.join_all, args=(pane,), name=f"join-{call}", daemon=True).start()

    def unseat(self, call: str, pane: str) -> None:
        c = self.calls.get(call)
        if c:
            with c.lock:
                c.listeners.pop(pane, None)

    def end(self, call: str, reason: str = "ended") -> list[dict]:
        c = self.calls.pop(call, None)
        if c is None:
            return []
        c.speaker.stop()
        for a in list(c.agents.values()):
            try:
                self.host.leave(a.live)
            except Exception as e:  # noqa: BLE001
                self.host.log(f"leave {a.handle}: {e}")
        return c.transcript

    def mic(self, pane: str, opus: bytes) -> None:
        for c in list(self.calls.values()):
            if pane in c.listeners:
                c.heard(pane, opus)
                return

    def agents_event(self, live: str, seq: int, kind: str, payload: dict) -> None:
        for c in list(self.calls.values()):
            if live in c.agents:
                c.notified(live, seq, kind, payload)
                return

    def agents_in(self, call: str) -> int:
        c = self.calls.get(call)
        return len(c.agents) + (len(c.wanted) if c and c.joining else 0) if c else 0

    # --------------------------------------------- listening, off the mic's thread

    def partial(self, c: Call, lis: Listener, audio: np.ndarray, cid: str) -> None:
        if self._partial_busy:
            return
        self._partial_busy = True
        self._work.put(("partial", c, lis, audio, cid, 0.0))

    def final(self, c: Call, lis: Listener, audio: np.ndarray, cid: str, ended_at: float) -> None:
        self._work.put(("final", c, lis, audio, cid, ended_at))

    def _worker(self) -> None:
        while True:
            kind, c, lis, audio, cid, ended_at = self._work.get()
            try:
                text, ms = self.stt(audio)
            except Exception as e:  # noqa: BLE001
                self.host.log(f"listen: {e}")
                text, ms = "", 0.0
            if kind == "partial":
                self._partial_busy = False
                if text and lis.speaking:
                    c.send({"kind": "caption", "id": cid, "handle": lis.person, "text": text, "final": False})
                continue
            if text:
                c.caption(cid, lis.person, text, final=True)
                c.said(lis.person, text, ended_at, ms)
