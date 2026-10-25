"""A call through the SDK's stand-in node: /voice call opens its pane, Join opens it on another device, and the
pane gets the call's sound, events and controls as docs/dev/call.md says.

Needs the node's Python SDK (where $POINTMAN_EXTENSIONS_SDK names its folder) and the package's own
environment (uv sync); skipped elsewhere. No models are needed: until the engine speaks, a call echoes the mic.
"""

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SDK = os.environ.get("POINTMAN_EXTENSIONS_SDK")
if SDK:
    sys.path.insert(0, SDK)

try:
    import pointman_extension_types as types
    from pointman_extension_testing import ExtensionFailed, InputRefused, StandInNode
    import numpy  # noqa: F401  - the package's environment
except ImportError:
    StandInNode = None


def events(view, kind=None):
    out = [m.event for name, m in view.frames if name == "pane.event"]
    return [e for e in out if kind is None or e.get("kind") == kind]


def audio(view):
    return [m for name, m in view.frames if name == "pane.audio"]


@unittest.skipUnless(StandInNode, "needs the node's Python SDK and the package's environment")
class CallTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.calls = Path(self.tmp.name) / "calls"
        self.node = StandInNode(ROOT, methods={"agents.sessions": lambda req: types.AgentsSessionsResponse()},
                                env={"HF_HUB_OFFLINE": "1", "POINTMAN_VOICE_CALLS": str(self.calls)},
                                machine="mac-test")
        self.node.start()

    def tearDown(self):
        self.node.stop()
        self.tmp.cleanup()

    def start_call(self, agents=""):
        return self.node.run("voice.call", {"agents": agents} if agents else {}).result

    def open(self, call_id, person="james"):
        return self.node.open_pane("call", {"call": call_id},
                                   viewer=types.PaneViewer(person=types.MemberRef(handle=person)))

    def test_voice_call_answers_with_its_pane_and_a_join_button(self):
        got = self.start_call()
        cid = got["call"]
        self.assertRegex(cid, r"^call-[0-9a-f]{12}$")
        self.assertEqual(got["pane"], {"open": "call", "target": {"call": cid}})
        self.assertEqual(got["blocks"], [{"type": "actions", "buttons": [
            {"title": "Join", "command": "voice.join", "params": {"call": cid}}]}])
        self.assertIn("plays your own mic back", got["text"])

    def test_join_answers_with_the_same_post_and_the_pane_and_an_ended_call_says_so(self):
        first = self.start_call()
        again = self.node.run("voice.join", {"call": first["call"]}).result
        self.assertEqual(again["text"], first["text"])
        self.assertEqual(again["blocks"], first["blocks"])      # the post keeps its Join button
        self.assertEqual(again["pane"], first["pane"])
        with self.assertRaises(ExtensionFailed) as caught:
            self.node.run("voice.join", {"call": "call-000000000000"})
        self.assertIn("That call has ended", str(caught.exception))

    def test_the_pane_opens_with_the_calls_sound_and_the_call_as_it_is(self):
        cid = self.start_call()["call"]
        view = self.open(cid)
        sound = view.opened.call
        opus = types.PaneOpenResponse.Call.Sound.Codec.CODEC_OPUS
        self.assertEqual((sound.down.codec, sound.down.rate, sound.down.channels), (opus, 24000, 1))
        self.assertEqual((sound.up.codec, sound.up.rate, sound.up.channels), (opus, 16000, 1))
        view.wait(lambda v: len(events(v)) >= 2)
        self.assertEqual(events(view)[:2], [{"kind": "members", "members": [{"handle": "james", "role": "person"}]},
                                            {"kind": "floor", "hands": []}])
        self.assertEqual(view.problems, [])
        self.assertTrue((self.calls / cid).is_file())           # the heavy-job lock waits while it's on

    def test_the_mic_comes_back_a_second_later_paced_however_it_arrived(self):
        view = self.open(self.start_call()["call"])
        packets = [bytes([i]) * 40 for i in range(10)]
        t0 = time.monotonic()
        for i, p in enumerate(packets):                         # all at once, as a burst off the network
            self.node.notify(types.PaneMicRequest(pane=types.PaneId(value=view.id), seq=i, captured_ms=1000 + 20 * i,
                                                  opus=p))
        view.wait(lambda v: len(audio(v)) >= 10, timeout=5)
        took = time.monotonic() - t0
        self.assertEqual([a.opus for a in audio(view)], packets)
        self.assertGreater(took, 1.1)                           # a second's delay, then 9 more 20 ms packets
        sent = [a.captured_ms for a in audio(view)]
        self.assertGreaterEqual(sent[-1] - sent[0], 100)        # paced (180 ms), not sent in one go (~0)

    def test_a_second_device_joins_and_each_hears_only_its_own_mic(self):
        cid = self.start_call()["call"]
        mac = self.open(cid, "james")
        phone = self.open(cid, "loupe")
        mac.wait(lambda v: {"kind": "joined", "handle": "loupe", "role": "person"} in events(v))
        phone.wait(lambda v: events(v, "members"))
        self.assertEqual(events(phone, "members")[0]["members"],
                         [{"handle": "james", "role": "person"}, {"handle": "loupe", "role": "person"}])
        self.node.notify(types.PaneMicRequest(pane=types.PaneId(value=phone.id), seq=0, captured_ms=5, opus=b"p"))
        phone.wait(lambda v: audio(v), timeout=5)
        time.sleep(0.2)
        self.assertEqual(audio(mac), [])

    def test_hand_up_and_down_send_the_floor_and_other_inputs_are_refused(self):
        view = self.open(self.start_call()["call"])
        view.input("hand", up=True)
        view.input("mute", on=True)
        view.input("hand", up=False)
        floors = [e["hands"] for e in events(view, "floor")]
        self.assertEqual(floors[-2:], [["james"], []])
        with self.assertRaises(InputRefused):
            view.input("tap", block="x")

    def test_a_path_coming_back_gets_the_call_again(self):
        view = self.open(self.start_call()["call"])
        view.wait(lambda v: len(events(v, "members")) == 1)

        def viewers(n):
            self.node.notify(types.PaneViewersRequest(pane=types.PaneId(value=view.id), attached=n))

        viewers(1)
        view.wait(lambda v: len(events(v, "members")) == 2)     # the first path: a replay
        viewers(2)
        viewers(0)
        viewers(1)
        view.wait(lambda v: len(events(v, "members")) == 3)     # 0 → 1 again: another; 1 → 2 none
        time.sleep(0.2)
        self.assertEqual(len(events(view, "members")), 3)

    def test_the_last_one_out_ends_the_call(self):
        cid = self.start_call()["call"]
        mac, phone = self.open(cid, "james"), self.open(cid, "loupe")
        phone.close()
        mac.wait(lambda v: {"kind": "left", "handle": "loupe", "role": "person"} in events(v))
        mac.close()
        deadline = time.monotonic() + 5
        while (self.calls / cid).exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertFalse((self.calls / cid).exists())
        with self.assertRaises(ExtensionFailed) as caught:
            self.node.run("voice.join", {"call": cid})
        self.assertIn("That call has ended", str(caught.exception))

    def test_a_pane_for_a_call_that_isnt_here_doesnt_open(self):
        with self.assertRaises(ExtensionFailed) as caught:
            self.open("call-000000000000")
        self.assertIn("That call has ended", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
