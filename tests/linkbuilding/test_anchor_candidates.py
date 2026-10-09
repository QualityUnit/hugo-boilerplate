"""HTML path anchors: word runs, fallback windows and exact target labels."""

import dataclasses
import unittest
from pathlib import Path

from support import generator as g, theme_rules


def page(url, title, keywords):
    return g.Page(path=Path("x.md"), rel_path="x.md", url=url, title=title, description="",
                  keywords=keywords, body="", paragraphs=[], clauses=[])


class ClauseWordRuns(unittest.TestCase):
    def test_runs_break_at_symbols_and_glued_words(self):
        runs = g._clause_word_runs("Zendesk’s own entry point at $19/mo for and/or teams")
        self.assertEqual([[word for _, _, word in run] for run in runs],
                         [["Zendesk’s", "own", "entry", "point", "at"], ["for"], ["teams"]])

    def test_double_space_breaks_a_run(self):
        runs = g._clause_word_runs("help desk  software")
        self.assertEqual([[word for _, _, word in run] for run in runs], [["help", "desk"], ["software"]])

    def test_spans_point_into_the_clause(self):
        clause = "help desk software"
        for start, end, word in g._clause_word_runs(clause)[0]:
            self.assertEqual(clause[start:end], word)


class ClauseCandidates(unittest.TestCase):
    def test_windows_longest_first_inside_one_clause(self):
        candidates = g._clause_candidates(["Our help desk software for small business teams"], theme_rules("en"))
        self.assertEqual(candidates[0], "help desk software for small")
        for phrase in ("help desk software", "help desk", "small business"):
            self.assertIn(phrase, candidates)
        self.assertNotIn("Our help", candidates)  # starts with a stopword

    def test_no_windows_in_cjk_languages(self):
        self.assertEqual(g._clause_candidates(["LiveAgent の AI ハンドオフ"], theme_rules("jp")), [])
        self.assertEqual(g._clause_candidates(["客户服务 软件"], theme_rules("zh-hans")), [])


class ExactLabelsLatin(unittest.TestCase):
    def setUp(self):
        target = page("/help-desk-software/", "Help Desk Software | LiveAgent",
                      ["help desk software", "help desk", "tool", "the ticketing system", "AI tool"])
        self.labels = g._exact_labels(target, theme_rules("en"))
        self.texts = [text for _, text in self.labels]

    def test_two_words_and_stopwords_trimmed(self):
        self.assertIn("help desk software", self.texts)
        self.assertIn("ticketing system", self.texts)
        self.assertNotIn("tool", self.texts)

    def test_generic_only_labels_are_dropped(self):
        self.assertNotIn("AI tool", self.texts)

    def test_most_words_first(self):
        self.assertEqual(self.texts[0], "help desk software")
        self.assertLess(self.texts.index("ticketing system"), self.texts.index("help desk"))

    def test_words_before_characters(self):
        target = page("/x/", "X", ["ticketing software", "live chat app"])
        self.assertEqual([text for _, text in g._exact_labels(target, theme_rules("en"))],
                         ["live chat app", "ticketing software"])  # 3 words beat 18 characters

    def test_latin_minimum_five_characters(self):
        target = page("/x/", "X", ["ui ux", "ui x"])
        self.assertEqual([text for _, text in g._exact_labels(target, theme_rules("en"))], ["ui ux"])


class ExactLabelsCjk(unittest.TestCase):
    def test_japanese_two_word_rule_and_length(self):
        target = page("/help-desk-software/", "AIエージェント搭載ヘルプデスクソフトウェア", [
            "ヘルプデスクソフトウェア",          # help desk software: kept
            "ヘルプデスク",                    # help desk: kept
            "ライブ",                          # "live": one word
            "エクスペリエンス",                 # "experience": one word
            "統合する",                        # "integrate": one word + dependent verb
            "ライブチャットソフトウェア",         # live chat software, 13 characters: kept
        ])
        texts = [text for _, text in g._exact_labels(target, theme_rules("jp"))]
        self.assertIn("ヘルプデスクソフトウェア", texts)
        self.assertIn("ヘルプデスク", texts)
        self.assertIn("ライブチャットソフトウェア", texts)
        for dropped in ("ライブ", "エクスペリエンス", "統合する",
                        "AIエージェント搭載ヘルプデスクソフトウェア"):
            self.assertNotIn(dropped, texts)

    def test_site_exceptions_for_dictionary_compounds(self):
        target = page("/x/", "チケット", ["チケッティングシステム"])
        rules = theme_rules("jp")
        self.assertEqual(g._exact_labels(target, rules), [])  # the segmenter reads one word
        rules = dataclasses.replace(rules, cjk_multiword_terms=frozenset({"チケッティングシステム"}))
        self.assertEqual(g._exact_labels(target, rules), [("keyword", "チケッティングシステム")])

    def test_cjk_generic_terms_whole_label(self):
        target = page("/email/", "メール", ["メールテンプレート", "電子メール管理"])
        rules = dataclasses.replace(theme_rules("jp"), nonspecific_terms=frozenset({"電子メール管理"}))
        self.assertEqual([text for _, text in g._exact_labels(target, rules)], ["メールテンプレート"])

    def test_cjk_generic_terms_per_word(self):
        # Like "AI feature" in English: every word generic or a brand -> dropped. One
        # generic word next to a specific one is fine ("customer support").
        target = page("/x/", "X", ["AI機能", "LiveAgentツール", "カスタマーサポート", "チャットボタン"])
        rules = dataclasses.replace(theme_rules("jp"), nonspecific_terms=frozenset({"ai", "liveagent", "機能", "ツール", "ボタン"}))
        self.assertEqual(sorted(text for _, text in g._exact_labels(target, rules)), ["カスタマーサポート", "チャットボタン"])

    def test_cjk_generic_terms_in_written_form(self):
        # The segmenter's word for 自動化 is 自動 (化 is a suffix) and it cuts AI工作流 into
        # 工作 + 流; the listed written forms must still make the label generic.
        jp = dataclasses.replace(theme_rules("jp"), nonspecific_terms=frozenset({"ai", "自動化"}))
        self.assertEqual(g._exact_labels(page("/x/", "X", ["AI自動化", "マーケティング自動化"]), jp),
                         [("keyword", "マーケティング自動化")])
        zh = dataclasses.replace(theme_rules("zh-hans"), nonspecific_terms=frozenset({"ai", "自动化", "工作流"}))
        self.assertEqual(sorted(t for _, t in g._exact_labels(page("/x/", "X", ["AI自动化", "AI工作流", "工作流管理"]), zh)),
                         ["工作流管理"])

    def test_cjk_label_length_limits(self):
        # 2+ characters and at most 6 estimated words: up to 20 characters of Japanese
        # with kana, 12 of Chinese, a Latin name counting as one word. Checked on site
        # exceptions, which skip the word rules.
        ja = ["客", "客服", "カ" * 20, "カ" * 21]
        rules = dataclasses.replace(theme_rules("jp"), cjk_multiword_terms=frozenset(ja))
        self.assertEqual(sorted(text for _, text in g._exact_labels(page("/x/", "X", ja), rules)),
                         sorted(["客服", "カ" * 20]))
        zh = ["客" * 12, "客" * 13, "WhatsApp集成"]
        rules = dataclasses.replace(theme_rules("zh-hans"), cjk_multiword_terms=frozenset(zh))
        self.assertEqual(sorted(text for _, text in g._exact_labels(page("/x/", "X", zh), rules)),
                         sorted(["客" * 12, "WhatsApp集成"]))

    def test_cjk_one_word_terms(self):
        # 电子邮件 ("e-mail") is two segmenter words: listed, it is dropped as a whole
        # label but stays an ordinary word inside a longer one.
        rules = dataclasses.replace(theme_rules("zh-hans"), cjk_one_word_terms=frozenset({"电子邮件"}))
        self.assertEqual(g._exact_labels(page("/x/", "X", ["电子邮件", "电子邮件自动化"]), rules),
                         [("keyword", "电子邮件自动化")])

    def test_title_is_split_on_cjk_punctuation(self):
        target = page("/x/", "ヘルプデスク、ライブチャット", [])
        texts = [text for _, text in g._exact_labels(target, theme_rules("jp"))]
        self.assertEqual(texts, ["ライブチャット", "ヘルプデスク"])

    def test_chinese(self):
        target = page("/bangzhutai/", "帮助台软件", ["客户服务", "知识库", "重要性", "服务器", "帮助台"])
        texts = [text for _, text in g._exact_labels(target, theme_rules("zh-hans"))]
        self.assertEqual(texts, ["帮助台软件", "客户服务", "知识库", "帮助台"])

    def test_cjk_label_in_a_latin_language_needs_two_words_as_before(self):
        target = page("/x/", "X", ["ヘルプデスク"])
        self.assertEqual(g._exact_labels(target, theme_rules("en")), [])


class ExactAnchor(unittest.TestCase):
    def test_finds_the_label_inside_a_japanese_clause(self):
        labels = [("keyword", "ヘルプデスク")]
        self.assertEqual(g._exact_anchor(["優れたヘルプデスクを選ぶ"], labels, set()), ("keyword", "ヘルプデスク"))
        self.assertIsNone(g._exact_anchor(["優れたヘルプデスクを選ぶ"], labels, {"ヘルプデスク"}))


if __name__ == "__main__":
    unittest.main()
