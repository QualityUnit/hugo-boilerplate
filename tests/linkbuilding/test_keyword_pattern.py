"""keyword_pattern: the injector's match for an anchor text, shared with the generator."""

import random
import re
import threading
import unittest

from support import linkbuilding_html as lh

# The boundary before S7 (QualityUnit/web-issues#4254). On text without CJK the new
# matcher must find exactly what this found.
def old_pattern(keyword):
    return re.compile(rf"(?<![\w-]){re.escape(keyword.strip())}(?![\w-])", re.IGNORECASE)


# The same rule written as one regular expression (slow to compile, so the code does
# not use it): only non-CJK word characters and "-" continue a word.
def reference_pattern(keyword):
    cont = rf"[^\W{lh.CJK_CHARS}]|-"
    return re.compile(rf"(?<!{cont}){re.escape(keyword.strip())}(?!{cont})", re.IGNORECASE)


def span(match):
    return match.span() if match else None


class LatinBoundaries(unittest.TestCase):
    def test_whole_words_only(self):
        cases = [
            ("chat", "live chat here", True),
            ("chat", "a chat.", True),
            ("chat", "Chat now", True),
            ("chat", "chat-widget", False),
            ("chat", "multi-chat", False),
            ("chat", "livechat", False),
            ("chat", "chatbot", False),
            ("help desk", "the help desk tool", True),
            ("help desk", "helpdesk", False),
            ("live-agent", "the live-agent tool", True),
            ("Ticket", "Ticket-System", False),
            ("ticket", "Ticket.", True),
        ]
        for keyword, text, expected in cases:
            with self.subTest(keyword=keyword, text=text):
                self.assertEqual(lh.keyword_pattern(keyword).search(text) is not None, expected)

    def test_returns_the_text_as_written(self):
        match = lh.keyword_pattern("customer service").search("Great Customer Service matters")
        self.assertEqual(match.group(0), "Customer Service")

    def test_keyword_is_stripped(self):
        self.assertEqual(span(lh.keyword_pattern("  help desk ").search("a help desk")), (2, 11))


class CjkBoundaries(unittest.TestCase):
    def test_cjk_neighbours_are_boundaries(self):
        cases = [
            ("ヘルプデスク", "優れたヘルプデスクを", True),
            ("LiveAgent", "LiveAgentは", True),
            ("チャット", "ライブチャットで", True),
            ("客户服务", "提升您的客户服务。", True),
            ("ヘルプ", "様々ヘルプ", True),  # 々 is CJK, not a Latin word character
        ]
        for keyword, text, expected in cases:
            with self.subTest(keyword=keyword, text=text):
                self.assertEqual(lh.keyword_pattern(keyword).search(text) is not None, expected)

    def test_latin_and_digit_neighbours_still_block(self):
        self.assertIsNone(lh.keyword_pattern("時間").search("24時間"))
        self.assertIsNone(lh.keyword_pattern("チャット").search("AIチャット"))
        self.assertIsNone(lh.keyword_pattern("chat").search("は-chat"))

    def test_old_regex_missed_cjk_in_running_text(self):
        # The reason for the change: CJK characters are \w, so the old lookarounds
        # never let a CJK anchor match inside a Japanese sentence.
        self.assertIsNone(old_pattern("ヘルプデスク").search("優れたヘルプデスクを"))


class EmptyKeyword(unittest.TestCase):
    def test_empty_or_blank_keyword_matches_nothing(self):
        # Run in a thread with a deadline: without the guard in KeywordPattern.search an
        # empty keyword loops forever, and the test must fail rather than hang CI.
        results = []

        def search_all():
            for keyword in ("", "   "):
                for text in ("", "a b", " ", "ヘルプ"):
                    results.append((keyword, text, lh.keyword_pattern(keyword).search(text)))

        worker = threading.Thread(target=search_all, daemon=True)
        worker.start()
        worker.join(timeout=5)
        self.assertFalse(worker.is_alive(), "keyword_pattern('').search() does not return")
        self.assertEqual([r for r in results if r[2] is not None], [])


class EquivalenceFuzz(unittest.TestCase):
    """Random keywords against random text: same match as the reference rules."""

    ALPHABET = "ab AB-_.,1é ßẞſKİı" + "ヘルプデスクのはー・々" + "客户服务的。"

    def _cases(self, alphabet, n=4000, seed=4254):
        rng = random.Random(seed)
        for _ in range(n):
            text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 24)))
            if text and rng.random() < 0.7:
                start = rng.randrange(len(text))
                keyword = text[start:start + rng.randint(1, 5)]
            else:
                keyword = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 4)))
            if keyword.strip():
                yield keyword, text

    def test_matches_reference_regex(self):
        for keyword, text in self._cases(self.ALPHABET):
            self.assertEqual(span(lh.keyword_pattern(keyword).search(text)),
                             span(reference_pattern(keyword).search(text)), (keyword, text))

    def test_without_cjk_matches_old_regex(self):
        latin = "".join(ch for ch in self.ALPHABET if not lh.CJK_RE.match(ch))
        for keyword, text in self._cases(latin, seed=4253):
            self.assertEqual(span(lh.keyword_pattern(keyword).search(text)),
                             span(old_pattern(keyword).search(text)), (keyword, text))


if __name__ == "__main__":
    unittest.main()
