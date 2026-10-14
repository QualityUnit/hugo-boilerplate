"""LINKBUILDING_THREADS: how many CPU threads the generator's embedding model may use."""

import os
import unittest
from unittest import mock

from support import generator


class ThreadCount(unittest.TestCase):
    def count(self, value):
        env = {k: v for k, v in os.environ.items() if k != "LINKBUILDING_THREADS"}
        if value is not None:
            env["LINKBUILDING_THREADS"] = value
        with mock.patch.dict(os.environ, env, clear=True):
            return generator._thread_count()

    def test_unset_keeps_one_thread(self):
        self.assertEqual(self.count(None), 1)

    def test_a_number(self):
        self.assertEqual(self.count("4"), 4)

    def test_auto_uses_every_core(self):
        with mock.patch.object(os, "cpu_count", return_value=8):
            self.assertEqual(self.count("auto"), 8)
            self.assertEqual(self.count(" AUTO "), 8)

    def test_auto_without_a_known_core_count_is_one(self):
        with mock.patch.object(os, "cpu_count", return_value=None):
            self.assertEqual(self.count("auto"), 1)

    def test_nonsense_and_zero_fall_back_to_one(self):
        for value in ("", "many", "0", "-3", "2.5"):
            with self.subTest(value=value):
                self.assertEqual(self.count(value), 1)


if __name__ == "__main__":
    unittest.main()
