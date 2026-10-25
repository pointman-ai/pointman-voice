"""The extension's process as a node runs it, through the SDK's stand-in node, on this machine's models.

Needs the node's Python SDK (where $POINTMAN_EXTENSIONS_SDK names its folder) and the package's own
environment (uv sync); skipped elsewhere. Offline: models count only if they're in the cache already.
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
    from pointman_extension_testing import ExtensionFailed, StandInNode
    import numpy  # noqa: F401  - the package's environment
except ImportError:
    StandInNode = None
else:
    AGENTS = [types.AgentSessionOnMachine(handle=types.MemberRef(handle=h), client=client, session_id=sid,
                                          thread=types.ThreadId(value="thr_1"), folder="/x")
              for h, client, sid in (("research", "claude", "s1"), ("oai", "codex", "s2"))]


@unittest.skipUnless(StandInNode, "needs the node's Python SDK and the package's environment")
class ExtensionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.published = []

        def publish(p):               # a FilesPublishRequest
            body = p.body
            self.published.append(body)
            self.assertTrue(Path(body["files"][0]["file"]).is_file())
            return types.FilesPublishResponse(file=types.FileId(value=f"file_{4800 + len(self.published)}"),
                                              version=types.FileVersionId(value="ver_1"))

        self.sessions_error = None

        def sessions(p):
            if self.sessions_error:
                raise RuntimeError(self.sessions_error)
            return types.AgentsSessionsResponse(agents=AGENTS)

        self.node = StandInNode(ROOT, methods={"files.publish": publish, "agents.sessions": sessions},
                                env={"HF_HUB_OFFLINE": "1",
                                     "POINTMAN_VOICE_CALLS": str(Path(self.tmp.name) / "calls")},
                                machine="mac-test")
        self.info = self.node.start()

    def tearDown(self):
        self.node.stop()
        self.tmp.cleanup()

    def ready(self) -> bool:
        for _ in range(100):
            if self.node.run("voice.voices").result["ready"]:
                return True
            time.sleep(0.1)
        return False

    def test_it_says_hello(self):
        self.assertEqual(self.info.id, "voice")
        self.assertEqual(self.info.version, "0.3.0")

    def test_the_voices_list_is_there_before_the_model_is(self):
        result = self.node.run("voice.voices").result
        self.assertEqual(len(result["voices"]), 28)
        self.assertEqual(result["voices"][0]["id"], "af_heart")
        self.assertIn("ready", result)
        self.assertIn("text", result)
        if not result["ready"]:
            self.assertTrue(0 <= result["progress"] <= 1)

    def test_a_sample_is_published_never_posted_and_kept(self):
        if not self.ready():
            self.skipTest("Kokoro isn't in this machine's cache")
        first = self.node.run("voice.sample", {"voice": "af_heart", "speed": 1.2, "name": "Loupe"},
                              timeout=120).result
        self.assertEqual(first["asset_id"], 4801)
        self.assertEqual(first["text"], "Hi, I'm Loupe. This is how I'll sound on our calls.")
        self.assertTrue(1.5 < first["seconds"] < 6, first)
        body = self.published[0]
        self.assertNotIn("post", body)
        self.assertEqual(body["files"][0]["mime"], "audio/wav")
        again = self.node.run("voice.sample", {"voice": "af_heart", "speed": 1.2, "name": "Loupe"}).result
        self.assertEqual(again["asset_id"], 4801)
        self.assertEqual(len(self.published), 1)               # the same voice, speed and text: kept

    def test_what_a_person_can_fix_is_one_sentence(self):
        for params, words in [({"voice": "zz_nobody"}, "no voice called"),
                              ({"voice": "af_heart", "text": "x" * 201}, "too long"),
                              ({"voice": "af_heart", "speed": "fast"}, "must be a number")]:
            with self.assertRaises(ExtensionFailed) as caught:
                self.node.run("voice.sample", params)
            self.assertIn(words, str(caught.exception))
        if self.ready():
            with self.assertRaises(ExtensionFailed) as caught:
                self.node.run("voice.sample", {"voice": "af_heart", "speed": 2.5}, timeout=60)
            self.assertIn("between 0.5 and 2", str(caught.exception))

    def test_voice_call_says_who_can_join(self):
        text = self.node.run("voice.call", {"agents": "@research @oai, @scene"}).result["text"]
        self.assertIn("| @research | can join |", text)
        self.assertIn("v1 calls are for Claude agents", text)
        self.assertIn("| @scene | no session on mac-test yet |", text)
        self.assertIn("plays your own mic back", text)
        alone = self.node.run("voice.call", {"agents": "  "}).result   # a call with nobody else: just the echo
        self.assertEqual(alone["agents"], [])
        self.assertNotIn("| Agent |", alone["text"])

    def test_a_failed_agents_list_fails_the_call_with_the_nodes_reason(self):
        self.sessions_error = "this node runs no agents"
        with self.assertRaises(ExtensionFailed) as caught:
            self.node.run("voice.call", {"agents": "@research"})
        self.assertIn("this node runs no agents", str(caught.exception))

    def test_health_has_the_models_and_never_fails_while_they_download(self):
        parts = self.node.call(types.HealthCheckRequest()).parts
        self.assertEqual(parts[0].id, "extension/voice/models")
        self.assertEqual(parts[0].state, types.Health.Part.State.STATE_OK)
        self.assertTrue(parts[0].why.endswith("."))

    def test_status_reports_the_models(self):
        for _ in range(50):
            if "voice.status" in self.node.events:
                break
            time.sleep(0.1)
        models = self.node.events["voice.status"]["models"]
        self.assertEqual([m["name"] for m in models][0], "Kokoro voices")


if __name__ == "__main__":
    unittest.main()
