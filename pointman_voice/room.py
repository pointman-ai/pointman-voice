"""The calls this machine holds: who's in each, on which panes, and what each pane is sent (docs/dev/call.md).

A call is made by `/voice call` and opened as a call pane on each device that joins it: the device that ran
the command, and any that tap the call post's Join (`voice.join`). Each device's pane is a seat. The call
ends when its last seat closes, or when nobody has opened it within EXPIRE seconds.

Until the engine speaks for the agents, a call plays each device's own mic back to it ECHO_DELAY later, so
the call view's sound, events and controls can be tried end to end.
"""

from __future__ import annotations

import heapq
import itertools
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from .calls import Mark, new_call_id

ECHO_DELAY = 1.0          # seconds before a device hears its own mic again
EXPIRE = 600.0            # seconds a call nobody has opened is kept for
DOWN_RATE = 24000         # the voices' sound, Kokoro's own rate
UP_RATE = 16000           # the mic, as listening wants it

Send = Callable[[str, dict], None]           # (pane, an event) → that pane's app
Audio = Callable[[str, int, bytes], None]    # (pane, captured_ms, one Opus packet) → that pane's app


class Ended(LookupError):
    """No such call here, or it's over: a sentence to show."""


class Echo:
    """One seat's mic, played back to it ECHO_DELAY later, paced as it was captured (one 20 ms packet every
    20 ms, call.md), never in a burst as the network may bring it."""

    def __init__(self, pane: str, audio: Audio, delay: float = ECHO_DELAY):
        self.pane, self._audio, self.delay = pane, audio, delay
        self._due: list[tuple[float, int, bytes]] = []
        self._order = itertools.count()
        self._base: float | None = None       # monotonic time of captured_ms 0, as first heard
        self._cv = threading.Condition()
        self._stopped = False
        self._thread = threading.Thread(target=self._loop, name=f"echo-{pane}", daemon=True)
        self._thread.start()

    def heard(self, captured_ms: int, opus: bytes) -> None:
        now = time.monotonic()
        with self._cv:
            if self._stopped:
                return
            at = (self._base or 0.0) + captured_ms / 1000
            if self._base is None or abs(now - at) > 0.5:     # the first, or the app's clock jumped: start again
                self._base = now - captured_ms / 1000
                at = now
            heapq.heappush(self._due, (at + self.delay, next(self._order), opus))
            self._cv.notify()

    def stop(self) -> None:
        with self._cv:
            self._stopped = True
            self._due.clear()
            self._cv.notify()
        self._thread.join(timeout=1)

    def _loop(self) -> None:
        while True:
            with self._cv:
                while not self._stopped and (not self._due or self._due[0][0] > time.monotonic()):
                    wait = self._due[0][0] - time.monotonic() if self._due else None
                    self._cv.wait(wait)
                if self._stopped:
                    return
                _, _, opus = heapq.heappop(self._due)
            self._audio(self.pane, int(time.time() * 1000), opus)


@dataclass
class Seat:
    """One device's pane in a call."""

    pane: str
    person: str
    echo: Echo | None = None
    attached: int = 0


@dataclass
class Call:
    id: str
    agents: list[str]                    # who was asked to join
    text: str = ""                       # its post's words, kept for the Join button's answer
    thread: str | None = None            # its thread, from the first pane: a pane in another is refused
    joined: list[str] = field(default_factory=list)    # agents in the call now
    voiced: bool = False                 # the engine runs it (agents were asked), so there's no echo
    made: float = field(default_factory=time.monotonic)
    seats: dict[str, Seat] = field(default_factory=dict)
    hands: list[str] = field(default_factory=list)
    mark: Mark | None = None

    def people(self) -> list[str]:
        seen: list[str] = []
        for s in self.seats.values():
            if s.person not in seen:
                seen.append(s.person)
        return seen


class Calls:
    """Every call on this machine. Thread-safe: panes, mics and commands come on threads of their own."""

    def __init__(self, send: Send, audio: Audio, *, echo: bool = True, marks: bool = True, engine=None,
                 ended: Callable[[Call, list], None] | None = None):
        """engine: what speaks for a call's agents (engine.live.Engine), made on first use by calling it.
        ended: told each call's transcript once it's over."""
        self._send, self._audio = send, audio
        self._echo, self._marks = echo, marks
        self._engine_of, self._engine = engine, None
        self._ended = ended
        self._calls: dict[str, Call] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------ calls

    def start(self, agents: list[str], text: str = "") -> Call:
        self.prune()
        with self._lock:
            c = Call(id=new_call_id(), agents=agents, text=text)
            self._calls[c.id] = c
            return c

    def get(self, call_id: str) -> Call:
        self.prune()
        with self._lock:
            c = self._calls.get(str(call_id or ""))
        if c is None:
            raise Ended("That call has ended. Start another with /voice call.")
        return c

    def prune(self) -> None:
        """Forget calls nobody ever opened, after EXPIRE."""
        now = time.monotonic()
        with self._lock:
            for cid in [cid for cid, c in self._calls.items() if not c.seats and now - c.made > EXPIRE]:
                self._end(cid)

    def __len__(self) -> int:
        with self._lock:
            return len(self._calls)

    # ------------------------------------------------------------ seats

    def engine(self):
        if self._engine is None and self._engine_of is not None:
            self._engine = self._engine_of()
        return self._engine

    def sit(self, call_id: str, pane: str, person: str, thread: str | None = None) -> Call:
        """A device opened the call's pane: it gets the call as it is now, and the others hear it joined. The
        call keeps the thread its first pane named, and refuses a pane that names another."""
        c = self.get(call_id)
        with self._lock:
            if c.thread is None:
                c.thread = thread or ""
            elif thread and c.thread and thread != c.thread:
                raise Ended("That call belongs to another thread.")
            new_person = person not in c.people()
            seat = Seat(pane=pane, person=person)
            engine = self.engine() if c.agents else None
            if engine is not None:
                if not c.voiced:
                    c.voiced = True
                    engine.start(c.id, c.thread, c.agents)
            elif self._echo:
                seat.echo = Echo(pane, self._audio)
            c.seats[pane] = seat
            if self._marks and c.mark is None:
                c.mark = Mark(c.id).start()
            if new_person:
                self._to_others(c, pane, {"kind": "joined", "handle": person, "role": "person"})
            self._replay(c, seat)
        if c.voiced:
            self.engine().seat(c.id, pane, person)
        return c

    def agent_joined(self, call_id: str, handle: str) -> None:
        with self._lock:
            c = self._calls.get(call_id)
            if c is None or handle in c.joined:
                return
            c.joined.append(handle)
            self._to_all(c, {"kind": "joined", "handle": handle, "role": "agent"})

    def agent_left(self, call_id: str, handle: str) -> None:
        with self._lock:
            c = self._calls.get(call_id)
            if c is None or handle not in c.joined:
                return
            c.joined.remove(handle)
            self._to_all(c, {"kind": "left", "handle": handle, "role": "agent"})

    def viewers(self, pane: str, attached: int) -> None:
        """A device's paths came and went: each 0 → 1 (a late device, or one whose path dropped) gets a replay,
        since the node keeps only the last 64 events while none is attached."""
        with self._lock:
            c, seat = self._seat(pane)
            if seat is None:
                return
            was, seat.attached = seat.attached, attached
            if was == 0 and attached > 0:
                self._replay(c, seat)

    def mic(self, pane: str, captured_ms: int, opus: bytes) -> None:
        with self._lock:
            c, seat = self._seat(pane)
        if seat is None or not opus:
            return
        if c.voiced:
            self.engine().mic(pane, opus)
        elif seat.echo is not None:
            seat.echo.heard(captured_ms, opus)

    def control(self, pane: str, kind: str, fields: dict) -> None:
        """The app's controls: mute (it sends no mic meanwhile, so there's nothing to do) and hand."""
        with self._lock:
            c, seat = self._seat(pane)
            if seat is None:
                raise Ended("This call has ended.")
            if kind == "mute":
                return
            if kind != "hand":
                raise ValueError(f"a call takes mute and hand, not {kind!r}")
            up = bool(fields.get("up", False))          # a hand without `up` is lowered, never raised
            if up and seat.person not in c.hands:
                c.hands.append(seat.person)
            elif not up and seat.person in c.hands:
                c.hands.remove(seat.person)
            else:
                return
            self._to_all(c, {"kind": "floor", "hands": list(c.hands)})

    def leave(self, pane: str) -> None:
        """A device's pane closed: the last one out ends the call."""
        ended = None
        with self._lock:
            c, seat = self._seat(pane)
            if seat is None:
                return
            del c.seats[pane]
            if c.voiced:
                self.engine().unseat(c.id, pane)
            if not c.seats:
                ended = self._take(c.id)
            elif seat.person not in c.people():
                if seat.person in c.hands:
                    c.hands.remove(seat.person)
                    self._to_all(c, {"kind": "floor", "hands": list(c.hands)})
                self._to_all(c, {"kind": "left", "handle": seat.person, "role": "person"})
        if seat.echo is not None:                     # outside the lock: its thread takes up to a second
            seat.echo.stop()
        if ended is not None:
            self._finish(ended, "ended")

    def end_all(self, reason: str = "shutdown") -> None:
        """The extension is stopping: every device hears why its call ended, then the calls go."""
        with self._lock:
            calls = [self._take(cid) for cid in list(self._calls)]
            for c in calls:
                for p in list(c.seats):
                    self._send(p, {"kind": "ended", "reason": reason})
        for c in calls:
            self._finish(c, reason)

    # ------------------------------------------------------------ inside

    def _seat(self, pane: str) -> tuple[Call | None, Seat | None]:
        for c in self._calls.values():
            if pane in c.seats:
                return c, c.seats[pane]
        return None, None

    def _replay(self, c: Call, seat: Seat) -> None:
        """The call as it is now, in call.md's order: members, the blocks showing, floor, who's talking."""
        members = [{"handle": h, "role": "person"} for h in c.people()]
        members += [{"handle": h, "role": "agent"} for h in c.joined]
        self._send(seat.pane, {"kind": "members", "members": members})
        self._send(seat.pane, {"kind": "floor", "hands": list(c.hands)})

    def _to_all(self, c: Call, event: dict) -> None:
        for p in list(c.seats):
            self._send(p, event)

    def _to_others(self, c: Call, pane: str, event: dict) -> None:
        for p in list(c.seats):
            if p != pane:
                self._send(p, event)

    def _end(self, call_id: str) -> None:
        c = self._take(call_id)
        if c is not None:
            self._finish(c, "ended")

    def _take(self, call_id: str) -> Call | None:
        return self._calls.pop(call_id, None)

    def _finish(self, c: Call, reason: str) -> None:
        for seat in list(c.seats.values()):
            if seat.echo is not None:
                seat.echo.stop()
        c.seats.clear()
        transcript = self.engine().end(c.id, reason) if c.voiced else []
        if c.mark is not None:
            c.mark.stop()
        if self._ended is not None:
            self._ended(c, transcript)
