"""HTML path: paragraphs and clauses read from the built page (_html_blocks and friends)."""

import tempfile
import unittest
from pathlib import Path

from bs4 import BeautifulSoup

from support import generator as g, theme_rules

PAGE = """<html><body><div class="cookie-banner"><p>Cookie text outside main</p></div><main>
<p>Our <strong>help</strong> desk software, the <a href="/x/">ticketing system</a> and live chat.<br>Second line here</p>
<p class="no-linkbuilding">Hidden text</p><ul><li>One item, two</li></ul><pre>code here</pre></main></body></html>"""


class HtmlBlocks(unittest.TestCase):
    def setUp(self):
        self.blocks = g._html_blocks(BeautifulSoup(PAGE, "lxml"))
        self.texts = ["".join(piece for piece, _ in block).strip() for block in self.blocks]

    def test_reads_main_only_and_skips_opt_out_and_code(self):
        joined = " ".join(self.texts)
        self.assertNotIn("Cookie text", joined)  # a plain <div>: only the <main> rule drops it
        self.assertNotIn("Hidden text", joined)
        self.assertNotIn("code here", joined)

    def test_one_block_per_paragraph_with_inline_markup_kept(self):
        self.assertIn("Our help desk software, the ticketing system and live chat. Second line here", self.texts)
        self.assertIn("One item, two", self.texts)

    def test_link_text_is_wording_but_not_linkable(self):
        block = next(b for b in self.blocks if any(p == "ticketing system" for p, _ in b))
        self.assertIn(("ticketing system", False), block)
        self.assertIn(("Our ", True), block)

    def test_body_is_read_when_there_is_no_main(self):
        blocks = g._html_blocks(BeautifulSoup("<html><body><p>Only body text</p></body></html>", "lxml"))
        self.assertIn("Only body text", ["".join(p for p, _ in b).strip() for b in blocks])

    def test_br_separates_words(self):
        block = next(b for b in self.blocks if any(p == "Second line here" for p, _ in b))
        self.assertIn((" ", False), block)


class LinkableClauses(unittest.TestCase):
    def test_clauses_never_cross_text_nodes_or_punctuation(self):
        block = next(b for b in g._html_blocks(BeautifulSoup(PAGE, "lxml")) if any(p == "Our " for p, _ in b))
        self.assertEqual(g._linkable_clauses(block, theme_rules("en")),
                         ["Our ", "help", " desk software", " the ", " and live chat", "Second line here"])


class ClauseSplit(unittest.TestCase):
    def test_latin_languages_ignore_cjk_punctuation(self):
        text = "自動化は、便利です。Next one"
        self.assertEqual(g._split_clauses(text, theme_rules("en")), [text])

    def test_japanese_punctuation_and_brackets(self):
        self.assertEqual(
            g._split_clauses("カスタマーサービスは、エージェントなしで「サポート」を（向上）させます。LiveAgentは", theme_rules("jp")),
            ["カスタマーサービスは", "エージェントなしで", "サポート", "を", "向上", "させます", "LiveAgentは"],
        )

    def test_chinese_full_width_comma_and_title_marks(self):
        clauses = [c for c in g._split_clauses("提升您的客户服务，增强知识库；以及《帮助台》", theme_rules("zh-hans")) if c]
        self.assertEqual(clauses, ["提升您的客户服务", "增强知识库", "以及", "帮助台"])

    def test_every_cjk_mark_ends_a_clause(self):
        marks = "。！？、；：，「」『』【】《》〈〉（）"
        text = "".join(f"語{i}{mark}" for i, mark in enumerate(marks))
        clauses = [c for c in g._split_clauses(text, theme_rules("jp")) if c]
        self.assertEqual(clauses, [f"語{i}" for i in range(len(marks))])


class ParagraphLength(unittest.TestCase):
    def test_latin_needs_18_tokens(self):
        rules = theme_rules("en")
        self.assertTrue(g._long_enough("word " * 17 + "word", rules))
        self.assertFalse(g._long_enough("word " * 16 + "word", rules))

    def test_japanese_needs_54_characters(self):
        rules = theme_rules("jp")
        self.assertFalse(g._long_enough("あ" * 53, rules))
        self.assertTrue(g._long_enough("あ" * 54, rules))

    def test_chinese_needs_33_characters(self):
        rules = theme_rules("zh-hans")
        self.assertFalse(g._long_enough("客" * 32, rules))
        self.assertTrue(g._long_enough("客" * 33, rules))

    def test_cjk_counting_only_in_cjk_languages(self):
        # A Japanese paragraph on an English page is still measured in tokens.
        self.assertFalse(g._long_enough("あ" * 200, theme_rules("en")))

    def test_paragraph_eligible_keeps_cjk_paragraphs_on_the_html_path(self):
        text = "自動化されたカスタマーサービスはエージェントの参加なしにサポートを提供できるプロセスです" * 2
        self.assertTrue(g._paragraph_eligible(text, theme_rules("jp")))
        self.assertFalse(g._paragraph_eligible(text))  # markdown path: tokens, as before


class ParagraphsFromHtml(unittest.TestCase):
    def test_japanese_page(self):
        sentence = "自動化されたカスタマーサービスは、エージェントの参加なしにサポートを提供できるプロセスです。"
        html = f"<html><body><main><p>{sentence * 2}</p><p>短い文です。</p></main></body></html>"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "index.html"
            path.write_text(html, encoding="utf-8")
            paragraphs, clauses = g._paragraphs_from_html(path, theme_rules("jp"))
        self.assertEqual(len(paragraphs), 1)
        self.assertIn("自動化されたカスタマーサービスは", clauses[0])
        self.assertIn("エージェントの参加なしにサポートを提供できるプロセスです", clauses[0])


if __name__ == "__main__":
    unittest.main()
