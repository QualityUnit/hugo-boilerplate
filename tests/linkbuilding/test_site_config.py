"""Site config: cjk_languages and cjk_multiword_terms in generator.yaml."""

import tempfile
import unittest
from pathlib import Path

from support import generator as g, quiet


def load(yaml_text):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "generator.yaml"
        path.write_text(yaml_text, encoding="utf-8")
        return quiet(g._load_site_config, path)


class CjkLanguages(unittest.TestCase):
    def test_theme_default(self):
        config = load("brand_terms: [acme]\n")
        self.assertEqual(config.cjk_languages, {"jp": "ja", "zh-hans": "zh"})
        self.assertEqual(config.cjk_multiword_terms, frozenset())

    def test_site_override_and_normalisation(self):
        config = load("cjk_languages: {JA: ja, zh: ZH}\ncjk_multiword_terms: [セルフサービス]\ncjk_one_word_terms: [电子邮件]\n")
        self.assertEqual(config.cjk_languages, {"ja": "ja", "zh": "zh"})
        self.assertEqual(config.cjk_multiword_terms, frozenset({"セルフサービス"}))
        self.assertEqual(config.cjk_one_word_terms, frozenset({"电子邮件"}))

    def test_switched_off(self):
        self.assertEqual(load("cjk_languages: {}\n").cjk_languages, {})

    def test_invalid_values_are_rejected(self):
        for bad in ("cjk_languages: [jp, zh-hans]\n", "cjk_languages: {jp: ko}\n"):
            with self.subTest(yaml=bad), self.assertRaises(ValueError):
                load(bad)

    def test_rules_per_language(self):
        config = load("brand_terms: [acme]\n")
        jp = quiet(g._site_rules, config, "jp")
        de = quiet(g._site_rules, config, "de")
        self.assertTrue(jp.cjk)
        self.assertEqual(jp.cjk_words.script, "ja")
        self.assertFalse(de.cjk)
        self.assertIsNone(de.cjk_words)

    def test_one_word_terms_reach_the_rules(self):
        config = load("cjk_one_word_terms: [电子邮件]\n")
        self.assertEqual(quiet(g._site_rules, config, "zh-hans").cjk_one_word_terms, frozenset({"电子邮件"}))

    def test_multiword_terms_reach_the_rules(self):
        config = load("cjk_multiword_terms: [セルフサービス]\n")
        self.assertEqual(quiet(g._site_rules, config, "jp").cjk_multiword_terms, frozenset({"セルフサービス"}))


if __name__ == "__main__":
    unittest.main()
