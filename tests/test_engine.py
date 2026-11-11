"""The call engine with stand-ins for the models and codecs, so turns, voices and barge-in run in seconds.

A mic packet is 20 ms: b"S" is speech, b"." silence. Speech to text answers what the test says was spoken.
Each 480 samples of an agent's voice is one packet down.
"""

import threading
import time
import unittest

import numpy as np

from pointman_voice.engine.live import Engine, sentences, speakable


class FakeHost:
    def __init__(self):
        self.lock = threading.Lock()
        self.audio_n, self.flushes, self.events, self.said, self.interrupts, self.left_ = {}, [], [], [], [], []
        self.joined_ = []
        self.lines = []

    def audio(self, pane, ms, opus):
        with self.lock:
            self.audio_n[pane] = self.audio_n.get(pane, 0) + 1

    def flush(self, pane):
        self.flushes.append(pane)

    def event(self, pane, event):
        with self.lock:
            self.events.append((pane, event))

    def join(self, agent, pane, instructions, effort):
        assert "voice call" in instructions and effort == "low"
        return {"live": f"run_{agent}", "voice": None, "voice_speed": 1.0}

    def say(self, live, text):
        self.said.append((live, text))

    def interrupt(self, live):
        self.interrupts.append(live)

    def leave(self, live):
        self.left_.append(live)

    def joined(self, call, handle):
        self.joined_.append(handle)

    def left(self, call, handle):
        pass

    def log(self, line):
        self.lines.append(line)

    def kinds(self, kind):
        with self.lock:
            return [e for _, e in self.events if e["kind"] == kind]


class Decoder:
    def decode(self, packet):
        return np.full(320, 8000 if packet == b"S" else 0, dtype=np.int16)


class Vad:
    def __init__(self):
        self.left = np.zeros(0, dtype=np.int16)

    def feed(self, pcm):
        x = np.concatenate([self.left, pcm])
        n = len(x) // 512 * 512
        out = [1.0 if np.abs(x[i:i + 512]).mean() > 1000 else 0.0 for i in range(0, n, 512)]
        self.left = x[n:]
        return out


class Encoder:
    def frames(self, pcm):
        return [b"v"] * max(1, len(pcm) // 480)


def wait(test, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if test():
            return True
        time.sleep(0.01)
    return False


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.host = FakeHost()
        self.spoken = "research what's on the board today"
        self.engine = Engine(self.host, {"filler_after_s": 0.2},
                             tts=lambda text, voice, speed: np.zeros(480 * 5 * len(text.split()), dtype=np.int16),
                             stt=lambda audio: (self.spoken, 12.0), vad=Vad, decoder=Decoder, encoder=Encoder())
        self.engine.start("call-1", "thr_1", ["research"])
        self.engine.seat("call-1", "pane_1", "james")
        self.assertTrue(wait(lambda: self.host.said))

    def talk(self, seconds, packet=b"S"):
        for _ in range(int(seconds / 0.02)):
            self.engine.mic("pane_1", packet)

    def agent(self, *pieces, turn=True, start=0):
        seq = start
        for p in pieces:
            self.engine.agents_event("run_research", seq, "text", {"text": p})
            seq += 1
        if turn:
            self.engine.agents_event("run_research", seq, "turn", {"text": "", "interrupted": False})
            seq += 1
        return seq

    def test_an_agent_joins_on_the_first_seat_and_is_greeted(self):
        self.assertEqual(self.host.joined_, ["research"])
        live, text = self.host.said[0]
        self.assertEqual(live, "run_research")
        self.assertIn("james has started a voice call", text)

    def test_a_turn_ends_after_the_silence_and_goes_to_the_agent(self):
        seq = self.agent("Hi James.")             # the greeting's answer
        self.talk(0.8)
        self.talk(0.7, b".")
        self.assertTrue(wait(lambda: len(self.host.said) == 2))
        self.assertEqual(self.host.said[1], ("run_research", self.spoken))
        finals = [e for e in self.host.kinds("caption") if e["handle"] == "james" and e["final"]]
        self.assertEqual(finals[-1]["text"], self.spoken)
        self.assertIn({"kind": "speaking", "handle": "james", "on": True}, self.host.kinds("speaking"))
        self.assertTrue(any('"mark": "said"' in l for l in self.host.lines))
        del seq

    def test_a_cough_is_not_a_turn(self):
        self.talk(0.12)
        self.talk(0.8, b".")
        time.sleep(0.2)
        self.assertEqual(len(self.host.said), 1)

    def test_an_agents_words_are_spoken_in_order_with_captions(self):
        self.agent("Hello", " there James", ". The board has", " three tasks.")
        self.assertTrue(wait(lambda: any(e["final"] and e["handle"] == "research"
                                         for e in self.host.kinds("caption"))))
        final = [e for e in self.host.kinds("caption") if e["handle"] == "research" and e["final"]][-1]
        self.assertEqual(final["text"], "Hello there James. The board has three tasks.")
        self.assertGreater(self.host.audio_n.get("pane_1", 0), 20)
        on = [e["on"] for e in self.host.kinds("speaking") if e["handle"] == "research"]
        self.assertEqual(on[0], True)
        self.assertTrue(wait(lambda: [e["on"] for e in self.host.kinds("speaking") if e["handle"] == "research"][-1]
                             is False))

    def test_out_of_order_pieces_are_put_back_in_order(self):
        self.engine.agents_event("run_research", 1, "text", {"text": " world."})
        self.engine.agents_event("run_research", 0, "text", {"text": "Hello"})
        self.engine.agents_event("run_research", 2, "turn", {"text": "", "interrupted": False})
        self.assertTrue(wait(lambda: any(e["final"] and e["handle"] == "research"
                                         for e in self.host.kinds("caption"))))
        final = [e for e in self.host.kinds("caption") if e["handle"] == "research" and e["final"]][-1]
        self.assertEqual(final["text"], "Hello world.")

    def test_talking_over_an_agent_stops_it_within_the_barge_in_time(self):
        long = "This is a long answer that goes on. " * 6
        seq = self.agent(long, turn=False)
        self.assertTrue(wait(lambda: self.host.audio_n.get("pane_1", 0) > 5))
        t = time.monotonic()
        self.talk(0.3)
        self.assertTrue(wait(lambda: self.host.flushes, timeout=1))
        self.assertLess(time.monotonic() - t, 0.5)
        self.assertEqual(self.host.interrupts, ["run_research"])
        cut = [e for e in self.host.kinds("caption") if e["handle"] == "research" and e["final"]][-1]
        self.assertLess(len(cut["text"]), len(long.strip()))
        sent = self.host.audio_n["pane_1"]
        self.engine.agents_event("run_research", seq, "text", {"text": "More words that were already streaming."})
        self.engine.agents_event("run_research", seq + 1, "turn", {"text": "", "interrupted": True})
        time.sleep(0.3)
        self.assertEqual(self.host.audio_n["pane_1"], sent)   # the rest of the cut turn is dropped

    def test_a_tool_with_nothing_said_gets_a_set_line(self):
        self.engine.agents_event("run_research", 0, "tool", {"name": "list_tasks", "state": "start"})
        self.assertTrue(wait(lambda: any(e["handle"] == "research" for e in self.host.kinds("caption"))))
        first = [e for e in self.host.kinds("caption") if e["handle"] == "research"][0]
        self.assertIn(first["text"], ["One moment.", "Let me check.", "Give me a second.", "Just a moment."])

    def test_the_call_ends_with_the_agents_out_and_its_transcript(self):
        self.agent("Hi James.")
        self.assertTrue(wait(lambda: any(e["final"] for e in self.host.kinds("caption"))))
        transcript = self.engine.end("call-1")
        self.assertEqual(self.host.left_, ["run_research"])
        self.assertEqual([t["handle"] for t in transcript], ["research"])


class TextTests(unittest.TestCase):
    def test_sentences_come_out_whole_and_the_rest_waits(self):
        out, rest = sentences("Sure. I'll look that up", done=False)
        self.assertEqual((out, rest), (["Sure."], "I'll look that up"))
        self.assertEqual(sentences(rest, done=True), (["I'll look that up"], ""))

    def test_markdown_and_links_are_spoken_plainly(self):
        self.assertEqual(speakable("**Done**: see [the PR](https://x.y/1) or https://x.y/2 `ok`"),
                         "Done : see the PR or a link ok")


if __name__ == "__main__":
    unittest.main()
