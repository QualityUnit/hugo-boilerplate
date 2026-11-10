"""Protected phrases (QualityUnit/PostAffiliatePro-hugo#983): a name a link takes whole or leaves alone.

protected-phrases.toml in the linkbuilding directory lists multi-word names with the one
page a link holding the whole name may point to. The injector never starts or ends a
link inside such a name ("Post <a>Affiliate</a> Pro") and never sends the whole name to
another page ("<a>program with Post Affiliate Pro</a>" → a feature page). The generator
checks its suggestions with the injector, so it drops those anchors as
``cuts_protected_phrase``. Without the file nothing changes.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

from support import generator as g, injector, linkbuilding_html, quiet

BRAND = [("Post Affiliate Pro", "/")]


def page(text: str) -> str:
    return f"<html><body><main><p>{text}</p></main></body></html>"


def kw(keyword: str, url: str, priority: int = 0):
    return injector.Keyword(keyword=keyword, url=url, priority=priority)


def links(text: str, keywords, protected=BRAND, page_url: str = "/blog/x/"):
    return injector.dry_run_page(page(text), {}, keywords, page_url=page_url, protected_phrases=protected)


class LoaderTest(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    def write(self, text: str) -> None:
        (self.dir / linkbuilding_html.PROTECTED_PHRASES_FILE).write_text(text, encoding="utf-8")

    def test_no_file_no_rows(self):
        self.assertEqual(linkbuilding_html.load_protected_phrases(self.dir), ([], []))

    def test_rows_in_file_order_with_canonical_paths(self):
        self.write('[[protected]]\nphrase = "Post  Affiliate Pro"\nurl = "/"\n\n'
                   '[[protected]]\nphrase = "Live Agent"\nurl = "/live-agent"\n')
        rows, problems = linkbuilding_html.load_protected_phrases(self.dir)
        self.assertEqual(rows, [("Post Affiliate Pro", "/"), ("Live Agent", "/live-agent/")])
        self.assertEqual(problems, [])

    def test_incomplete_and_conflicting_rows_are_problems(self):
        self.write('[[protected]]\nphrase = "Post Affiliate Pro"\nurl = "/"\n\n'
                   '[[protected]]\nphrase = "post affiliate pro"\nurl = "/pricing/"\n\n'
                   '[[protected]]\nphrase = "No URL"\n')
        rows, problems = linkbuilding_html.load_protected_phrases(self.dir)
        self.assertEqual(rows, [("Post Affiliate Pro", "/")])
        self.assertEqual(len(problems), 2)

    def test_broken_file_is_a_problem_not_a_crash(self):
        self.write("[[protected]\nphrase = ")
        rows, problems = linkbuilding_html.load_protected_phrases(self.dir)
        self.assertEqual(rows, [])
        self.assertEqual(len(problems), 1)


class CutsProtectedPhraseTest(unittest.TestCase):
    TEXT = "Integrate Post Affiliate Pro today"

    def span(self, part: str):
        start = self.TEXT.index(part)
        return start, start + len(part)

    def test_partial_overlaps_are_cuts(self):
        for part in ("Affiliate", "Post Affiliate", "Affiliate Pro", "Integrate Post", "Pro today"):
            with self.subTest(part=part):
                self.assertTrue(linkbuilding_html.cuts_protected_phrase(self.TEXT, *self.span(part), "/", BRAND))

    def test_whole_phrase_only_to_its_own_page(self):
        for part in ("Post Affiliate Pro", "Integrate Post Affiliate Pro"):
            with self.subTest(part=part):
                self.assertFalse(linkbuilding_html.cuts_protected_phrase(self.TEXT, *self.span(part), "/", BRAND))
                self.assertTrue(linkbuilding_html.cuts_protected_phrase(self.TEXT, *self.span(part), "/features/", BRAND))

    def test_text_outside_the_phrase_is_free(self):
        self.assertFalse(linkbuilding_html.cuts_protected_phrase(self.TEXT, *self.span("Integrate"), "/x/", BRAND))

    def test_an_inflection_glued_to_the_end_keeps_the_name(self):
        for text in ("Post Affiliate Pros Vorteil", "a Post Affiliate Proban"):
            with self.subTest(text=text):
                start = text.index("Post")
                # "Post Affiliate" alone, and the name without its suffix, both cut it
                self.assertTrue(linkbuilding_html.cuts_protected_phrase(text, start, start + 14, "/", BRAND))
                self.assertTrue(linkbuilding_html.cuts_protected_phrase(text, start, start + 18, "/", BRAND))
                word_end = text.index(" ", start + 15) if " " in text[start + 15:] else len(text)
                self.assertFalse(linkbuilding_html.cuts_protected_phrase(text, start, word_end, "/", BRAND))

    def test_a_glued_start_is_not_the_name(self):
        text = "XPost Affiliate Pro"
        self.assertFalse(linkbuilding_html.cuts_protected_phrase(text, 6, 15, "/x/", BRAND))


class InjectorTest(unittest.TestCase):
    def test_issue_examples_get_no_link_inside_the_brand(self):
        cases = [
            ("For integration with Post Affiliate Pro follow these steps", kw("affiliate", "/glossary/affiliate/")),
            ("Manage your program with Post Affiliate Pro.", kw("post affiliate", "/")),
            ("Post Affiliate Pro hilft Ihnen", kw("Affiliate Pro", "/integrationsmethoden/shopsite/")),
            ("Post Affiliate Pro-val egyszerű", kw("post affiliate", "/")),  # HU: the hyphen keeps the name whole
            ("Post Affiliate Pros Vorteil", kw("Post Affiliate", "/funktionen/x/")),  # DE genitive
            ("a Post Affiliate Proban", kw("post affiliate", "/")),  # HU suffix without a hyphen
            ("Post\u00a0Affiliate\u00a0Pro tracks clicks", kw("affiliate", "/glossary/affiliate/")),
        ]
        for text, keyword in cases:
            with self.subTest(text=text):
                run = links(text, [keyword])
                self.assertEqual(run.applied, set())
                self.assertIn(keyword.keyword.casefold(), run.protected)

    def test_whole_brand_links_to_its_own_page_only(self):
        text = "Manage your program with Post Affiliate Pro for growth."
        self.assertEqual(links(text, [kw("post affiliate pro", "/")]).applied, {("/", "post affiliate pro")})
        self.assertEqual(links(text, [kw("Post Affiliate Pro", "/features/x/")]).applied, set())
        self.assertEqual(links(text, [kw("program with Post Affiliate Pro", "/features/x/")]).applied, set())
        self.assertEqual(links(text, [kw("Post Affiliate Pro for growth", "/")]).applied,
                         {("/", "post affiliate pro for growth")})

    def test_a_later_match_outside_the_brand_is_taken(self):
        text = "Post Affiliate Pro helps every affiliate earn more."
        run = links(text, [kw("affiliate", "/glossary/affiliate/")])
        self.assertEqual(run.applied, {("/glossary/affiliate/", "affiliate")})
        self.assertEqual(run.protected, set())  # linked after all: not reported as cut

    def test_keyword_falls_through_to_the_next_text_node(self):
        html = "<html><body><main><p>Post Affiliate Pro</p><p>Become an affiliate.</p></main></body></html>"
        run = injector.dry_run_page(html, {}, [kw("affiliate", "/glossary/affiliate/")], page_url="/blog/x/",
                                    protected_phrases=BRAND)
        self.assertEqual(run.applied, {("/glossary/affiliate/", "affiliate")})

    def test_lnks_from_front_matter_are_held_to_the_same_rule(self):
        meta = {"lnks": [{"text": "Affiliate Pro", "path": "/integration-methods/x-cart/"}]}
        html = page("Connect Post Affiliate Pro with your shop.")
        run = injector.dry_run_page(html, meta, [], page_url="/blog/x/", protected_phrases=BRAND)
        self.assertEqual(run.applied, set())

    def test_without_protected_phrases_nothing_changes(self):
        run = links("For integration with Post Affiliate Pro follow these steps",
                    [kw("affiliate", "/glossary/affiliate/")], protected=None)
        self.assertEqual(run.applied, {("/glossary/affiliate/", "affiliate")})
        self.assertEqual(run.protected, set())

    def test_deploy_counts_skipped_matches(self):
        config = injector.LinkConfig(protected_phrases=BRAND)
        builder = injector.LinkBuilder([kw("affiliate", "/glossary/affiliate/")], config, page_url="/blog/x/")
        soup = injector.BeautifulSoup(page("Post Affiliate Pro and Post Affiliate Pro"), "lxml")
        self.assertEqual(builder._apply_links(soup, builder.keywords), 0)
        self.assertEqual(builder.stats.protected_skipped, 2)
        self.assertEqual(soup.find_all("a"), [])


class GeneratorValidationTest(unittest.TestCase):
    """_validate_with_injector reads protected-phrases.toml and names the rejection."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.html = self.dir / "index.html"
        self.html.write_text(page("Connect Post Affiliate Pro with affiliate programs you run."), encoding="utf-8")
        self.page = g.Page(path=self.dir / "x.md", rel_path="x.md", url="/blog/x/", title="", description="",
                           keywords=[], body="", paragraphs=[], html_path=self.html)

    def rec(self, text: str, target: str):
        return g.LinkRec(source_path=self.page.path, source_rel_path="x.md", source_url="/blog/x/",
                         paragraph_index=0, paragraph="", target_url=target, target_title="", text=text,
                         title="", fit=0.5, lift=0.1, anchor_score=0.5, anchor_confidence=0.5)

    def validate(self):
        recs = [self.rec("Affiliate Pro", "/integration-methods/x-cart/"), self.rec("affiliate programs", "/programs/")]
        kept, rejected = quiet(g._validate_with_injector, [self.page], recs, self.dir, "en")
        return [r.text for r in kept], rejected

    def test_suggestion_inside_the_brand_is_rejected_with_its_reason(self):
        (self.dir / linkbuilding_html.PROTECTED_PHRASES_FILE).write_text(
            '[[protected]]\nphrase = "Post Affiliate Pro"\nurl = "/"\n', encoding="utf-8")
        kept, rejected = self.validate()
        self.assertEqual(kept, ["affiliate programs"])
        self.assertEqual(rejected["cuts_protected_phrase"], 1)

    def test_without_the_file_both_suggestions_stay(self):
        kept, rejected = self.validate()
        self.assertEqual(sorted(kept), ["Affiliate Pro", "affiliate programs"])
        self.assertEqual(rejected["cuts_protected_phrase"], 0)


if __name__ == "__main__":
    unittest.main()
