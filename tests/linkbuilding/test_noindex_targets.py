"""Pages kept out of search results take no part in linkbuilding (QualityUnit/LiveAgent-hugo#1436).

Front matter (noindex, robots, private, draft) and the site's `indexing` params
(noindexSections, noindexPaths) both count: a page they hide is neither a link
source nor a link target.
"""

import tempfile
import unittest
from pathlib import Path

from support import generator as g, quiet, theme_rules

BODY = "Our help desk software answers every ticket quickly and keeps customers happy all year.\n"


def page(title, extra=""):
    # Every page has its url: without one the generator derives it from the content
    # path through sync_translation_urls, which reads the theme's own site, not tmp.
    url = "/" + title.lower() + "/"
    return f'+++\ntitle = "{title}"\ndescription = "About {title}."\nurl = "{url}"\n{extra}+++\n\n{BODY}'


class NoindexTargets(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(g._site_indexing.cache_clear)

    def load(self, files, config=None, config_name="params.toml"):
        root = Path(self.tmp.name)
        (root / "config" / "_default").mkdir(parents=True, exist_ok=True)
        if config is not None:
            (root / "config" / "_default" / config_name).write_text(config, encoding="utf-8")
        content = root / "content" / "en"
        for rel, text in files.items():
            (content / rel).parent.mkdir(parents=True, exist_ok=True)
            (content / rel).write_text(text, encoding="utf-8")
        g._site_indexing.cache_clear()
        return {p.rel_path for p in quiet(g._load_pages, content, theme_rules("en"))}

    def test_site_indexing_params_hide_sections_and_paths(self):
        loaded = self.load({
            "academy/kept.md": page("Kept"),
            "landing/ad.md": page("Ad"),
            "landing/deep/nested.md": page("Nested"),
            "search/_index.md": page("Search"),
            "author/_index.md": page("Authors"),
            "author/jane.md": page("Jane"),
        }, '[indexing]\n  noindexSections = ["landing", "search"]\n  noindexPaths = ["/author"]\n')
        # noindexPaths is the page's own path, as in Hugo's .Path: the pages below stay
        self.assertEqual(loaded, {"academy/kept.md", "author/jane.md"})

    def test_a_leaf_bundle_is_at_its_folder(self):
        loaded = self.load({"kept.md": page("Kept"), "about/index.md": page("About")},
                           '[indexing]\n  noindexPaths = ["/about"]\n')
        self.assertEqual(loaded, {"kept.md"})

    def test_case_is_ignored_like_hugo(self):
        loaded = self.load({"kept.md": page("Kept"), "Landing/ad.md": page("Ad")},
                           '[Indexing]\n  noindexsections = ["LANDING"]\n')
        self.assertEqual(loaded, {"kept.md"})

    def test_front_matter_noindex_hides_the_page(self):
        loaded = self.load({"kept.md": page("Kept"),
                            "thank-you.md": page("Thanks", 'robots = "noindex, nofollow"\n')})
        self.assertEqual(loaded, {"kept.md"})

    def test_without_indexing_params_every_section_takes_part(self):
        loaded = self.load({"kept.md": page("Kept"), "landing/ad.md": page("Ad")},
                           '[other]\n  noindexSections = ["landing"]\n')
        self.assertEqual(loaded, {"kept.md", "landing/ad.md"})

    def test_params_in_hugo_toml_count_too(self):
        loaded = self.load({"kept.md": page("Kept"), "landing/ad.md": page("Ad")},
                           '[params.indexing]\n  noindexSections = ["landing"]\n', config_name="hugo.toml")
        self.assertEqual(loaded, {"kept.md"})


if __name__ == "__main__":
    unittest.main()
