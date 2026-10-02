"""estimated_words and the injector's link cap: CJK words estimated from characters."""

import random
import re
import unittest

from bs4 import BeautifulSoup

from support import injector, linkbuilding_html as lh


class EstimatedWords(unittest.TestCase):
    def test_without_cjk_it_is_the_plain_word_count(self):
        text = "The help-desk tool, version 2.0 — it's fast."
        self.assertEqual(lh.estimated_words([text]), len(re.findall(r"\w+", text)))

    def test_japanese_three_characters_per_word(self):
        # 30 CJK characters, kana-heavy -> Japanese -> 30 / 3.0 = 10
        text = "これはカスタマーサポートのためのヘルプデスクソフトウェアです"
        cjk = len(lh.CJK_RE.findall(text))
        self.assertEqual(lh.estimated_words([text]), int(cjk / lh.CJK_CHARS_PER_WORD_JA))

    def test_chinese_without_kana(self):
        text = "提升您的客户服务和客户满意度"
        cjk = len(lh.CJK_RE.findall(text))
        self.assertEqual(lh.estimated_words([text]), int(cjk / lh.CJK_CHARS_PER_WORD_ZH))

    def test_a_stray_japanese_name_does_not_make_chinese_text_japanese(self):
        text = "提升您的客户服务和客户满意度" * 5 + "カ"
        cjk = len(lh.CJK_RE.findall(text))
        self.assertEqual(lh.estimated_words([text]), int(cjk / lh.CJK_CHARS_PER_WORD_ZH))

    def test_kana_share_boundary(self):
        # Japanese from 10 % kana on: 1 kana in 10 CJK characters is Japanese, 1 in 11 is not.
        self.assertEqual(lh.estimated_words(["カ" + "漢" * 9]), int(10 / lh.CJK_CHARS_PER_WORD_JA))
        self.assertEqual(lh.estimated_words(["カ" + "漢" * 10]), int(11 / lh.CJK_CHARS_PER_WORD_ZH))

    def test_half_width_katakana_is_cjk(self):
        self.assertEqual(lh.estimated_words(["ﾍﾙﾌﾟﾃﾞｽｸ"]), int(len("ﾍﾙﾌﾟﾃﾞｽｸ") / lh.CJK_CHARS_PER_WORD_JA))
        self.assertIsNotNone(lh.keyword_pattern("ﾍﾙﾌﾟ").search("ｶｽﾀﾏｰﾍﾙﾌﾟ"))

    def test_latin_words_inside_cjk_text_count_as_words(self):
        self.assertEqual(lh.estimated_words(["LiveAgentは"]), 1)  # 1 run + int(1 / 3.0)

    def test_how_the_text_is_cut_does_not_matter(self):
        # The injector counts the page before and after inserting anchors, which splits
        # text nodes; the count must not move. keyword_pattern only cuts where no Latin
        # word continues across the cut, so only such cuts are tried.
        rng = random.Random(4254)
        latin = re.compile(rf"[^\W{lh.CJK_CHARS}]")

        def cuttable(text, i):
            return not (0 < i < len(text) and latin.match(text[i - 1]) and latin.match(text[i]))

        texts = ["自動化されたカスタマーサービスは、LiveAgentのヘルプデスクで改善します。",
                 "提升您的客户服务。LiveAgent帮助台软件和知识库。",
                 "The help desk tool for customer service teams."]
        for text in texts:
            cuts = [i for i in range(len(text) + 1) if cuttable(text, i)]
            for _ in range(200):
                a, b = sorted(rng.sample(cuts, 2))
                self.assertEqual(lh.estimated_words([text[:a], text[a:b], text[b:]]),
                                 lh.estimated_words([text]), (text, a, b))


class InjectorCap(unittest.TestCase):
    def _page(self, body):
        return f"<html><body><main><p>{body}</p></main></body></html>"

    def test_japanese_page_gets_a_cap_from_its_length(self):
        # 30 sentences of 44 CJK characters: 440 estimated words. The old \w+ count saw
        # one "word" per sentence (30), so the cap stayed at links_min.
        sentence = "自動化されたカスタマーサービスはエージェントの参加なしにサポートを提供できるプロセスです。"
        soup = BeautifulSoup(self._page(sentence * 30), "lxml")
        words = injector._page_words(injector._plain_texts(lh.linkable_text_nodes(soup)))
        self.assertEqual(words, 440)
        config = injector.LinkConfig(links_per_words=100, links_min=3, links_max=40)
        self.assertEqual(config.cap_for(words), 4)
        old_words = len(re.findall(r"\w+", sentence * 30))
        self.assertEqual(old_words, 30)
        self.assertEqual(config.cap_for(old_words), 3)

    def test_cap_is_stable_across_a_second_pass(self):
        sentence = "優れたヘルプデスクを選ぶと、カスタマーサポートとライブチャットが改善します。"
        html = self._page(sentence * 40)
        keywords = [injector.Keyword(keyword=k, url=f"/t{i}/") for i, k in
                    enumerate(["ヘルプデスク", "カスタマーサポート", "ライブチャット"])]
        config = injector.LinkConfig(links_per_words=50, links_min=1, links_max=40)

        first = BeautifulSoup(html, "lxml")
        builder = injector.LinkBuilder(keywords, config)
        added = builder._apply_links(first, builder.applicable_keywords(html))
        self.assertEqual(added, 3)

        second_html = str(first)
        second = BeautifulSoup(second_html, "lxml")
        again = injector.LinkBuilder(keywords, config)
        self.assertEqual(again._apply_links(second, again.applicable_keywords(second_html)), 0)
        self.assertEqual(builder.stats.total_words, again.stats.total_words)


if __name__ == "__main__":
    unittest.main()
