"""Canonical phrases (QualityUnit/web-issues#4255): one owner page per phrase.

canonical-phrases.toml in the linkbuilding directory gives a phrase to one page. The
generator lets only that page take the phrase as an anchor, the injector links the
phrase to it on every page (below [[lnks]], above <lang>.json). Without the file
nothing changes — test_markdown_fixture.py covers that.
"""

import contextlib
import io
import json
import os
import re
import shutil
import tempfile
import tomllib
import unittest
from pathlib import Path

from support import FIXTURES, generator as g, injector, linkbuilding_html, quiet, run_generator, theme_rules

TABLE = """\
[[en]]
phrase = "help desk software"
url = "/help-desk-software/"
type = "secondary"

[[en]]
phrase = "AI help desk software"
url = "/help-desk-software"
type = "primary"
gsc_impr = 1313

[[en]]
phrase = "social media customer service software"
url = "/social-media-customer-service/"

[[en]]
phrase = "customer service software"
url = "/customer-service-software/"

[[en]]
phrase = "Help Desk Software"
url = "/ticketing-software/"

[[en]]
phrase = "ticketing system"

[[de]]
phrase = "Helpdesk-Software"
url = "/helpdesk-software/"
"""

def write_table(directory: Path, text: str = TABLE) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / linkbuilding_html.CANONICAL_PHRASES_FILE).write_text(text, encoding="utf-8")
    return directory


def page(url, title, keywords, *, clauses=True):
    return g.Page(path=Path("x.md"), rel_path="x.md", url=url, title=title, description="",
                  keywords=keywords, body="", paragraphs=[], clauses=[] if clauses else None)


def rules_with_table(directory: Path, lang: str = "en"):
    rules = theme_rules(lang)
    with contextlib.redirect_stderr(io.StringIO()):  # TABLE's conflicting rows are reported there
        quiet(g._apply_canonical_phrases, rules, directory, lang)
    return rules


class Loader(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = write_table(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_primary_first_paths_normalised(self):
        rows, _ = linkbuilding_html.load_canonical_phrases(self.dir, "en")
        self.assertEqual(rows[0], ("AI help desk software", "/help-desk-software/"))
        self.assertIn(("help desk software", "/help-desk-software/"), rows)

    def test_first_owner_wins_and_problems_are_named(self):
        rows, problems = linkbuilding_html.load_canonical_phrases(self.dir, "en")
        owners = [url for phrase, url in rows if phrase.casefold() == "help desk software"]
        self.assertEqual(owners, ["/help-desk-software/"])
        self.assertTrue(any("/ticketing-software/" in p for p in problems))
        self.assertTrue(any("needs both" in p for p in problems))  # "ticketing system" has no url

    def test_languages_are_separate(self):
        rows, _ = linkbuilding_html.load_canonical_phrases(self.dir, "de")
        self.assertEqual(rows, [("Helpdesk-Software", "/helpdesk-software/")])
        self.assertEqual(linkbuilding_html.load_canonical_phrases(self.dir, "sk"), ([], []))

    def test_broken_file_is_a_problem_not_a_crash(self):
        (self.dir / linkbuilding_html.CANONICAL_PHRASES_FILE).write_text("[[en]\nphrase = ", encoding="utf-8")
        rows, problems = linkbuilding_html.load_canonical_phrases(self.dir, "en")
        self.assertEqual(rows, [])
        self.assertTrue(problems and "does not parse" in problems[0])

    def test_no_file_no_rows(self):
        self.assertEqual(linkbuilding_html.load_canonical_phrases(Path(self.tmp.name) / "missing", "en"), ([], []))


class Injector(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = write_table(Path(self.tmp.name))
        (self.dir / "en.json").write_text(json.dumps({"keywords": [
            {"Keyword": "help desk software", "URL": "/blog/help-desk/", "Priority": 100},
            {"Keyword": "live chat", "URL": "/live-chat-software/", "Priority": 100},
        ]}), encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def keywords(self):
        with contextlib.redirect_stderr(io.StringIO()):  # the conflicting rows are reported there
            return injector._load_global_keywords(self.dir, "en")

    def test_canonical_rows_rank_between_lnks_and_json(self):
        by_text = {(kw.keyword.casefold(), kw.url): kw.priority for kw in self.keywords()}
        canonical = by_text[("help desk software", "/help-desk-software/")]
        self.assertEqual(canonical, injector.CANONICAL_PRIORITY)
        self.assertGreater(canonical, by_text[("help desk software", "/blog/help-desk/")])
        self.assertLess(canonical, 1000 - 100)  # every [[lnks]] entry of a page

    def test_owner_wins_the_phrase_over_json_and_longest_phrase_first(self):
        html = ("<html><body><main><p>Compare social media customer service software, "
                "customer service software and help desk software with live chat.</p></main></body></html>")
        run = injector.dry_run_page(html, {}, self.keywords(), page_url="/blog/x/")
        self.assertIn(("/help-desk-software/", "help desk software"), run.applied)
        self.assertIn(("/social-media-customer-service/", "social media customer service software"), run.applied)
        self.assertIn(("/customer-service-software/", "customer service software"), run.applied)
        self.assertNotIn(("/blog/help-desk/", "help desk software"), run.applied)

    def test_lnks_still_go_first(self):
        html = "<html><body><main><p>Our help desk software is fast.</p></main></body></html>"
        meta = {"lnks": [{"text": "help desk software", "path": "/blog/other/"}]}
        run = injector.dry_run_page(html, meta, self.keywords(), page_url="/blog/x/")
        self.assertEqual(run.applied, {("/blog/other/", "help desk software")})

    def test_no_self_link_on_the_owner(self):
        html = "<html><body><main><p>Our help desk software is fast.</p></main></body></html>"
        run = injector.dry_run_page(html, {}, self.keywords(), page_url="/help-desk-software/")
        self.assertNotIn(("/help-desk-software/", "help desk software"), run.applied)


class Generator(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.rules = rules_with_table(write_table(Path(self.tmp.name)))
        self.owner = page("/help-desk-software/", "Help Desk | LiveAgent", ["help desk tool"])
        self.other = page("/blog/free-help-desk-software/", "Free Help Desk Software",
                          ["help desk software", "free help desk software"])

    def tearDown(self):
        self.tmp.cleanup()

    def test_owner_tries_its_phrases_first_longest_first(self):
        info = g._target_info(self.owner, self.rules)
        self.assertEqual(info.exact_labels[:2], [("canonical", "AI help desk software"), ("canonical", "help desk software")])

    def test_other_page_loses_the_phrase_as_keyword_and_label(self):
        info = g._target_info(self.other, self.rules)
        self.assertEqual(info.keywords, ["free help desk software"])
        self.assertNotIn("help desk software", [label.casefold() for _, label in info.exact_labels])
        self.assertEqual(g._exact_keyword_bonus("help desk software", self.other, info, rules=self.rules), 0.08)

    def test_windows_never_give_another_page_the_phrase(self):
        clauses = ["Pick the right help desk software today"]
        candidates = g._anchor_candidate_infos(g._clause_candidates(clauses, self.rules), self.rules)
        other = g._target_info(self.other, self.rules)
        choice = g._html_anchor_for_target(clauses, candidates, self.other, other,
                                           fit=0.9, lift=0.2, rules=self.rules, used=set())
        self.assertNotEqual((choice or ("",))[0].casefold(), "help desk software")
        owner = g._target_info(self.owner, self.rules)
        choice = g._html_anchor_for_target(clauses, candidates, self.owner, owner,
                                           fit=0.9, lift=0.2, rules=self.rules, used=set())
        self.assertEqual(choice[0], "help desk software")
        self.assertEqual(choice[3], "exact:canonical")

    def choose(self, target, clauses, *, page_text=None, guard=True):
        """The HTML-path anchor for ``target`` from ``clauses`` of a source page whose
        linkable text is ``page_text`` (default: just those clauses)."""
        source = page("/blog/source/", "Source", [])
        source.linkable_clauses = list(page_text if page_text is not None else clauses)
        info = g._target_info(target, self.rules)
        candidates = g._anchor_candidate_infos(g._clause_candidates(clauses, self.rules), self.rules)
        return g._html_anchor_for_target(
            clauses, candidates, target, info, fit=0.9, lift=0.2, rules=self.rules, used=set(),
            guard=g._canonical_guard(source, self.rules) if guard else None,
        )

    def test_window_cut_out_of_a_foreign_phrase_is_rejected(self):
        # Review of hugo-boilerplate#420: "desk software" -> another page inside "help desk software".
        target = page("/business/agency/", "Desk Software for Agencies", ["desk software", "agency software"])
        clauses = ["Agencies compare help desk software"]
        self.assertEqual(self.choose(target, clauses, guard=False)[0], "desk software")  # the bug without the guard
        choice = self.choose(target, clauses)
        self.assertNotIn("desk software", (choice or ("",))[0].casefold())

    def test_window_containing_a_foreign_phrase_is_rejected(self):
        # "Reliable call center software" -> /callback-software/ swallowed the owner's phrase.
        self.rules.canonical_owner["call center software"] = "/call-center-software/"
        target = page("/callback-software/", "Callback Software", ["reliable callback", "call center"])
        clauses = ["Reliable call center software with callback"]
        choice = self.choose(target, clauses)
        self.assertFalse(choice and "call center software" in choice[0].casefold(), choice)

    def test_foreign_phrase_elsewhere_on_the_page_blocks_the_anchor(self):
        # The injector links an anchor's first occurrence on the page: "live chat" from this
        # paragraph would land inside "live chat software" in an earlier list item.
        self.rules.canonical_owner["live chat software"] = "/live-chat-software/"
        target = page("/free-live-chat-software/", "Free Live Chat", ["live chat", "free chat"])
        clauses = ["Add live chat to every page"]
        self.assertEqual(self.choose(target, clauses)[0], "live chat")
        choice = self.choose(target, clauses, page_text=["Best live chat software", *clauses])
        self.assertNotEqual((choice or ("",))[0].casefold(), "live chat")

    def test_exact_label_inside_a_foreign_phrase_is_rejected(self):
        target = page("/customer-service-software/", "Customer Service Software", [])
        clauses = ["the best social media customer service software for teams"]
        choice = self.choose(target, clauses)
        self.assertFalse(choice and choice[3] == "exact:canonical", choice)

    def test_owner_keeps_its_phrase_and_unrelated_windows_pass(self):
        clauses = ["Agencies compare help desk software", "ticket routing rules"]
        self.assertEqual(self.choose(self.owner, clauses)[0], "help desk software")
        source = page("/blog/source/", "Source", [])
        source.linkable_clauses = clauses
        guard = g._canonical_guard(source, self.rules)
        self.assertFalse(guard.blocks("ticket routing", "/blog/x/"))
        self.assertFalse(guard.blocks("help desk software", "/help-desk-software/"))
        self.assertTrue(guard.blocks("help desk software", "/blog/x/"))

    def test_plural_is_not_a_canonical_phrase(self):
        # Same word boundaries as the injector, which would never link "ticketing system" in "systems".
        self.rules.canonical_owner["ticketing system"] = "/ticketing-software/"
        source = page("/blog/source/", "Source", [])
        source.linkable_clauses = ["two ticketing systems compared"]
        self.assertFalse(g._canonical_guard(source, self.rules).blocks("ticketing systems", "/blog/x/"))

    def test_no_guard_without_a_table(self):
        self.assertIsNone(g._canonical_guard(page("/x/", "X", []), theme_rules("en")))

    def test_short_blocks_are_part_of_the_page_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            html = Path(tmp) / "index.html"
            html.write_text("<html><body><main><ul><li>Best live chat software</li></ul>"
                            "<p>" + "Our team adds live chat to every page so visitors get answers fast and the agents can help them right away. " * 2
                            + "</p></main></body></html>", encoding="utf-8")
            everything: list[str] = []
            paragraphs, _ = g._paragraphs_from_html(html, self.rules, everything)
        self.assertEqual(len(paragraphs), 1)  # the list item is too short to be a paragraph
        self.assertIn("Best live chat software", everything)

    def test_no_table_no_change(self):
        rules = theme_rules("en")
        info = g._target_info(self.other, rules)
        self.assertIn("help desk software", info.keywords)
        self.assertEqual(rules.canonical_owner, {})

    def test_multi_target_report(self):
        recs = [g.LinkRec(Path("a"), "a", "/a/", 0, "", url, "", text, "", 0, 0, 0, 0)
                for text, url in [("live chat", "/x/"), ("Live chat", "/y/"), ("live chat", "/x/"), ("help desk", "/x/")]]
        report = g._multi_target_report(recs)
        self.assertEqual((report["phrases"], report["links"], report["share"]), (1, 3, 0.75))
        self.assertEqual(report["top"][0], {"text": "live chat", "targets": 2, "links": 3})


class MarkdownFixtureWithTable(unittest.TestCase):
    """End to end on the 10-page fixture: the phrases that pointed at 3-6 targets get one."""

    TABLE = """\
[[en]]
phrase = "customer support"
url = "/ticketing-software/"

[[en]]
phrase = "help desk"
url = "/help-desk-software/"

[[en]]
phrase = "customer service"
url = "/live-chat-software/"
"""
    def test_each_phrase_points_at_its_owner_only(self):
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "site"
            shutil.copytree(FIXTURES / "markdown-en" / "site", work)
            write_table(work / "data" / "linkbuilding", self.TABLE)
            os.chdir(work)
            try:
                rc = run_generator([
                    "--lang", "en", "--no-cache", "--write", "--remove-old-linkbuilding",
                    "--similarity-floor", "-1", "--lift-floor", "-1", "--anchor-floor", "0",
                    "--top-k-per-page", "30", "--output", str(Path(tmp) / "report.json"),
                ])
            finally:
                os.chdir(cwd)
            self.assertEqual(rc, 0)
            owners = {"customer support": "/ticketing-software/", "help desk": "/help-desk-software/",
                      "customer service": "/live-chat-software/"}
            seen = 0
            for md in (work / "content").rglob("*.md"):
                text = md.read_text(encoding="utf-8")
                meta = tomllib.loads(re.match(r"^\+\+\+\n(.*?)\n\+\+\+", text, re.S).group(1))
                for item in meta.get("lnks") or []:
                    owner = owners.get(item["text"].casefold())
                    if owner:
                        seen += 1
                        self.assertEqual(item["path"], owner, f"{md.name}: {item['text']!r}")
                    # No other page's anchor swallows a table phrase ("help desk software" -> a blog post).
                    for phrase, phrase_owner in owners.items():
                        if phrase_owner != item["path"] and linkbuilding_html.keyword_pattern(phrase).search(item["text"]):
                            self.fail(f"{md.name}: {item['text']!r} -> {item['path']} contains {phrase!r}")
            self.assertGreater(seen, 0)


if __name__ == "__main__":
    unittest.main()
