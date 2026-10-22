"""The parts that need nothing installed: the voice list, a call's mark for the build lock, Linux's library paths."""

import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from pointman_voice import calls, libpaths, voices


class VoicesTests(unittest.TestCase):
    def test_twenty_american_and_eight_british_best_first(self):
        listed = voices.catalogue()
        self.assertEqual(len(listed), 28)
        self.assertEqual(sum(v["language"] == "en-US" for v in listed), 20)
        self.assertEqual(sum(v["language"] == "en-GB" for v in listed), 8)
        self.assertEqual(listed[0], {"id": "af_heart", "name": "Heart", "language": "en-US", "gender": "female",
                                     "grade": "A"})
        self.assertEqual(listed[-1]["id"], "am_adam")
        self.assertEqual(len({v["id"] for v in listed}), 28)

    def test_names_and_genders_come_from_the_id(self):
        self.assertEqual(voices.describe("bm_george"),
                         {"id": "bm_george", "name": "George", "language": "en-GB", "gender": "male"})
        self.assertTrue(voices.known("bf_emma"))
        self.assertFalse(voices.known("zf_xiaobei"))   # in Kokoro, but not one of the English 28


class MarkTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_call_holds_its_file_until_it_ends(self):
        with mock.patch.object(calls, "TOUCH_EVERY", 0.05):
            mark = calls.Mark("call-abc123", self.folder).start()
            path = self.folder / "call-abc123"
            self.assertTrue(path.exists())
            os.utime(path, (time.time() - 100, time.time() - 100))
            time.sleep(0.2)
            self.assertGreater(path.stat().st_mtime, time.time() - 5)   # touched again
            mark.stop()
        self.assertFalse(path.exists())

    def test_ids_are_safe_file_names(self):
        for _ in range(20):
            self.assertRegex(calls.new_call_id(), r"^call-[0-9a-f]{12}$")
        for bad in ("../x", "a/b", "", "Call", "x" * 65):
            with self.assertRaises(ValueError):
                calls.Mark(bad, self.folder)

    def test_the_default_folder_is_the_one_the_lock_watches(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("POINTMAN_VOICE_CALLS", None)
            self.assertEqual(calls.calls_dir(), Path.home() / ".local/state/pointman/voice/calls")


class LibPathTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Path(self.tmp.name)
        for name in ("aaa-gcc-13.3.0-lib", "bbb-gcc-14.3.0-lib", "ccc-gcc-14.2.0-lib", "ddd-gcc-15.1.0-lib"):
            (self.store / name / "lib").mkdir(parents=True)
        for name in ("aaa-gcc-13.3.0-lib", "bbb-gcc-14.3.0-lib", "ccc-gcc-14.2.0-lib"):
            (self.store / name / "lib" / "libstdc++.so.6").touch()   # 15.1 has none

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_newest_gcc_with_libstdcxx(self):
        with mock.patch.dict(os.environ, {"NIX_LD_LIBRARY_PATH": ""}):
            self.assertEqual(libpaths._libstdcxx(self.store), [str(self.store / "bbb-gcc-14.3.0-lib" / "lib")])

    def test_nix_ld_wins(self):
        with mock.patch.dict(os.environ, {"NIX_LD_LIBRARY_PATH": "/a:/b"}):
            self.assertEqual(libpaths._libstdcxx(self.store), ["/a", "/b"])

    def test_nothing_twice_and_what_was_there_stays(self):
        with mock.patch.object(libpaths, "_nvidia_libs", return_value=["/venv/nvidia/cublas/lib"]), \
                mock.patch.dict(os.environ, {"LD_LIBRARY_PATH": "/venv/nvidia/cublas/lib:/mine"}):
            self.assertEqual(libpaths.wanted(nixos=False), {})
        with mock.patch.object(libpaths, "_nvidia_libs", return_value=["/venv/nvidia/cudnn/lib"]), \
                mock.patch.dict(os.environ, {"LD_LIBRARY_PATH": "/mine"}):
            self.assertEqual(libpaths.wanted(nixos=False), {"LD_LIBRARY_PATH": "/venv/nvidia/cudnn/lib:/mine"})

    def test_off_linux_it_never_restarts(self):
        with mock.patch.object(libpaths.sys, "platform", "darwin"), mock.patch.object(libpaths.os, "execve") as ex:
            libpaths.ensure()
        ex.assert_not_called()


if __name__ == "__main__":
    unittest.main()
