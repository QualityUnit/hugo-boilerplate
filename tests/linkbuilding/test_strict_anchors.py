"""Strict anchors (QualityUnit/web-issues#4258): an anchor is a keyword of its target.

strict_anchor_languages in generator.yaml switches it on per language, HTML path only.
The anchor must be one of the target's canonical phrases or keywords; no word window is
scored, so the target gets no link from a paragraph without one. The two cases from the
issue ("Agile CRM" -> /integrations/close/, "plan anytime" -> /free-plan/) came from
windows that only shared one word with the target; measurement run #29 showed that
windows around a whole keyword mostly add filler ("Help desk typically").
"""

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from support import FIXTURES, FakeEmbedder, generator as g, quiet, theme_rules
from test_markdown_fixture import written_files


def load(yaml_text):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "generator.yaml"
        path.write_text(yaml_text, encoding="utf-8")
        return quiet(g._load_site_config, path)


def page(url, title, keywords):
    return g.Page(path=Path("x.md"), rel_path="x.md", url=url, title=title, description="",
                  keywords=keywords, body="", paragraphs=[], clauses=[])


CLOSE = page("/integrations/close/", "Close", ["close", "integrate close crm", "close crm", "integrate close"])
# The keywords of the live pages (content/en, after LiveAgent-hugo#1339).
FREE_PLAN = page("/free-plan/", "Free plan", ["free plan", "free forever", "free support", "free customer portal",
                                              "free customization options", "free multi-channel support"])
FREE_HELP_DESK = page("/blog/free-help-desk-software/", "Free Help Desk Software", ["free help desk", "free service desk"])
HELP_DESK = page("/help-desk-software/", "Help Desk Software", ["help desk software", "ticketing tool"])
ROUTING = page("/features/ticket-routing/", "Ticket Routing Rules", ["routing automation"])


class Config(unittest.TestCase):
    def test_off_by_default(self):
        config = load("brand_terms: [acme]\n")
        self.assertEqual(config.strict_anchor_languages, frozenset())
        self.assertFalse(quiet(g._site_rules, config, "en").strict_anchors)

    def test_per_language(self):
        config = load("strict_anchor_languages: [EN]\n")
        self.assertTrue(quiet(g._site_rules, config, "en").strict_anchors)
        self.assertFalse(quiet(g._site_rules, config, "de").strict_anchors)

    def test_invalid_values_are_rejected(self):
        for bad in ("strict_anchor_languages: en\n", "strict_anchor_languages: [[en]]\n"):
            with self.subTest(yaml=bad), self.assertRaises(ValueError):
                load(bad)


class Anchor(unittest.TestCase):
    def choose(self, target, clauses, *, strict=True, used=()):
        rules = theme_rules("en")
        rules.strict_anchors = strict
        info = g._target_info(target, rules)
        candidates = g._anchor_candidate_infos(g._clause_candidates(clauses, rules), rules)
        return g._html_anchor_for_target(clauses, candidates, target, info,
                                         fit=0.9, lift=0.2, rules=rules, used=set(used))

    def test_issue_cases_get_no_link(self):
        # The texts of the two links left from run #26 (blog/best-crm-software.md,
        # ai-powered-software.md); the loose rule links both from one shared word.
        clause = "Agile CRM integrates with LiveAgent to enhance customer service efficiency"
        self.assertEqual(self.choose(CLOSE, [clause], strict=False)[0], "Agile CRM")
        self.assertIsNone(self.choose(CLOSE, [clause]))
        clause = "or cancel your plan anytime with no contracts"
        self.assertIn("plan", self.choose(FREE_PLAN, [clause], strict=False)[0])
        self.assertIsNone(self.choose(FREE_PLAN, [clause]))

    def test_keyword_is_kept_whole(self):
        # "free" is an English stopword: trimmed, "free help desk" would hand out "help desk".
        clause = "A good help desk keeps every ticket in one place"
        self.assertEqual(self.choose(FREE_HELP_DESK, [clause], strict=False)[0], "help desk")
        self.assertIsNone(self.choose(FREE_HELP_DESK, [clause]))
        self.assertEqual(self.choose(FREE_HELP_DESK, ["Try a free help desk for your team"])[0], "free help desk")
        self.assertEqual(self.choose(FREE_PLAN, ["Get free multi-channel support with every account"])[0],
                         "free multi-channel support")

    def test_keyword_of_one_word_without_function_words_is_no_anchor(self):
        # Kept whole only when the trimmed keyword still passes: "free customer" and "free
        # plan" are one word without "free" and give no link (the loose rule drops them too).
        chat_client = page("/customer-support-glossary/free-chat-client/", "Free Chat Client",
                           ["free to chat", "free client", "free chat client", "free customer"])
        self.assertIsNone(self.choose(chat_client, ["LiveAgent gives small teams free customer support over chat"]))
        self.assertIsNone(self.choose(FREE_PLAN, ["Start on the free plan today"]))
        labels = [label for _, label in g._target_info(chat_client, self._strict_rules()).exact_labels]
        self.assertEqual(labels, ["free chat client"])

    def _strict_rules(self):
        rules = theme_rules("en")
        rules.strict_anchors = True
        return rules

    def test_keyword_is_still_an_anchor(self):
        choice = self.choose(HELP_DESK, ["Teams pick reliable help desk software"])
        self.assertEqual(choice[0], "help desk software")
        self.assertEqual(choice[3], "exact:keyword")

    def test_no_window_around_a_used_keyword(self):
        # The keyword text is already a link on the page: the loose rule takes a longer
        # window around it, the strict one gives no link.
        clauses = ["Teams pick reliable help desk software"]
        choice = self.choose(HELP_DESK, clauses, strict=False, used=["help desk software"])
        self.assertEqual(choice[3], "window")
        self.assertIsNone(self.choose(HELP_DESK, clauses, used=["help desk software"]))

    def test_window_cut_out_of_a_keyword_is_rejected(self):
        clauses = ["Agencies compare desk software"]
        self.assertEqual(self.choose(HELP_DESK, clauses, strict=False)[0], "desk software")
        self.assertIsNone(self.choose(HELP_DESK, clauses))

    def test_title_and_slug_are_not_keywords(self):
        clauses = ["Set up ticket routing rules for every team"]
        self.assertEqual(self.choose(ROUTING, clauses, strict=False)[3], "exact:title")
        self.assertIsNone(self.choose(ROUTING, clauses))

    def test_one_word_keyword_is_no_anchor(self):
        # "close" is a keyword of /integrations/close/, and a common word.
        self.assertIsNone(self.choose(CLOSE, ["Close the ticket when the customer is happy"]))

    def test_canonical_phrase_counts_as_keyword(self):
        rules = theme_rules("en")
        rules.strict_anchors = True
        rules.canonical_owner["ai help desk software"] = "/help-desk-software/"
        rules.canonical_by_owner["/help-desk-software/"] = ["AI help desk software"]
        target = page("/help-desk-software/", "Help Desk Software", [])
        clauses = ["Compare AI help desk software for small teams"]
        info = g._target_info(target, rules)
        candidates = g._anchor_candidate_infos(g._clause_candidates(clauses, rules), rules)
        choice = g._html_anchor_for_target(clauses, candidates, target, info,
                                           fit=0.9, lift=0.2, rules=rules, used=set())
        self.assertEqual(choice[:1] + choice[3:], ("AI help desk software", "exact:canonical"))

    def test_keyword_blocked_by_a_foreign_phrase_gives_no_window(self):
        # Another page owns "AI help desk software", which also occurs on the source page,
        # so the exact "help desk software" is blocked there; no window takes its place.
        clauses = ["Teams pick reliable help desk software"]
        source = page("/blog/source/", "Source", [])
        source.linkable_clauses = ["Try AI help desk software today", *clauses]
        for strict in (False, True):
            rules = theme_rules("en")
            rules.strict_anchors = strict
            rules.canonical_owner["ai help desk software"] = "/ai-help-desk/"
            info = g._target_info(HELP_DESK, rules)
            candidates = g._anchor_candidate_infos(g._clause_candidates(clauses, rules), rules)
            choice = g._html_anchor_for_target(clauses, candidates, HELP_DESK, info, fit=0.9, lift=0.2, rules=rules,
                                               used=set(), guard=g._canonical_guard(source, rules))
            with self.subTest(strict=strict):
                if strict:
                    self.assertIsNone(choice)
                else:
                    self.assertEqual(choice[3], "window")


class MarkdownPath(unittest.TestCase):
    def test_strict_is_reported_and_has_no_effect_without_public_dir(self):
        # The markdown path never reads strict_anchors: same output as test_markdown_fixture.py,
        # plus a warning so a run without --public-dir does not pass for a strict one.
        fixture = FIXTURES / "markdown-en"
        cwd, original_model, original_argv = os.getcwd(), g.LazySentenceTransformer, sys.argv
        stderr = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "site"
            shutil.copytree(fixture / "site", work)
            config = work / "data" / "linkbuilding" / "generator.yaml"
            config.write_text(config.read_text(encoding="utf-8") + "\nstrict_anchor_languages: [en]\n", encoding="utf-8")
            os.chdir(work)
            g.LazySentenceTransformer = FakeEmbedder
            sys.argv = ["generate_paragraph_linkbuilding.py", "--lang", "en", "--no-cache", "--write",
                        "--remove-old-linkbuilding", "--similarity-floor", "-1", "--lift-floor", "-1",
                        "--anchor-floor", "0", "--top-k-per-page", "30", "--output", str(Path(tmp) / "report.json")]
            try:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
                    rc = g.main()
            finally:
                os.chdir(cwd)
                g.LazySentenceTransformer, sys.argv = original_model, original_argv
            got = written_files(work / "content")
        self.assertEqual(rc, 0)
        self.assertIn("strict_anchor_languages applies with --public-dir only", stderr.getvalue())
        expected = json.loads((fixture / "expected.json").read_text(encoding="utf-8"))
        self.assertEqual(got, expected)


if __name__ == "__main__":
    unittest.main()
