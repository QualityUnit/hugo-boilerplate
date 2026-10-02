"""CjkWordCounter: word counts for Japanese and Chinese labels."""

import builtins
import unittest
from unittest import mock

import support  # noqa: F401  (import path)
from linkbuilding_cjk import CjkWordCounter


class Japanese(unittest.TestCase):
    counter = CjkWordCounter("ja")

    def test_counts(self):
        cases = {
            "ライブ": 1, "ライブチャット": 2, "エクスペリエンス": 1, "ヘルプデスクソフトウェア": 3,
            "生産性": 1,        # suffix 性 is not a word
            "統合する": 1,      # dependent verb する is not a word
            "中小企業": 3,      # 中 / 小 / 企業 — prefixes count: "small business"
            "LiveAgentの": 1,  # particle
            "チケッティングシステム": 1,  # dictionary compound: needs a site exception
        }
        for text, words in cases.items():
            with self.subTest(text=text):
                self.assertEqual(self.counter.count(text), words)


class Words(unittest.TestCase):
    def test_words_are_returned_in_order(self):
        self.assertEqual(CjkWordCounter("ja").words("AI機能"), ("AI", "機能"))
        self.assertEqual(CjkWordCounter("ja").words("統合する"), ("統合",))
        self.assertEqual(CjkWordCounter("zh").words("客户服务"), ("客户", "服务"))
        self.assertEqual(CjkWordCounter("zh").words("LiveAgent的"), ("LiveAgent",))


class Chinese(unittest.TestCase):
    counter = CjkWordCounter("zh")

    def test_counts(self):
        cases = {
            "客户服务": 2, "知识库": 2, "帮助台": 2, "转化率": 2,   # jieba keeps these whole
            "重要性": 1, "服务器": 1, "个性化": 1,                   # suffix
            "LiveAgent的": 1, "SLA的": 1,                           # Latin token + particle
        }
        for text, words in cases.items():
            with self.subTest(text=text):
                self.assertEqual(self.counter.count(text), words)


class Errors(unittest.TestCase):
    def test_unknown_script(self):
        with self.assertRaises(ValueError):
            CjkWordCounter("ko")

    def test_missing_library_raises_on_first_count_not_on_creation(self):
        real_import = builtins.__import__

        def no_segmenters(name, *args, **kwargs):
            if name in ("fugashi", "jieba"):
                raise ImportError(f"No module named {name!r}")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=no_segmenters):
            for script in ("ja", "zh"):
                counter = CjkWordCounter(script)  # creating it needs no library
                with self.subTest(script=script), self.assertRaises(RuntimeError):
                    counter.count("ヘルプデスク")


if __name__ == "__main__":
    unittest.main()
