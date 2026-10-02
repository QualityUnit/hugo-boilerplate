"""HTML path end to end for Japanese and Chinese: generator, then the real injector.

fixtures/html-cjk/site is a small synthetic site (5 jp + 5 zh-hans pages) built as the
LiveAgent workflow builds it: one language per build, pages at the root of
public-<lang>/ (--content-at-root). Two targets, /live/ (ライブ "live") and
/zhongyaoxing/ (重要性 "importance"), have nothing but a one-word keyword, and the
word appears in other pages' prose: without the two-word rule for CJK labels they
would receive links. The fake embedder makes fit and lift arbitrary but repeatable,
so the expected links are fixed.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from bs4 import BeautifulSoup

from support import FIXTURES, run_generator, run_injector

FIXTURE = FIXTURES / "html-cjk"

EXPECTED = {
    "jp": {
        ("/help-desk-software/", "/call-center/", "コールセンター"),
        ("/help-desk-software/", "/knowledge-base/", "ナレッジベース"),
        ("/knowledge-base/", "/help-desk-software/", "ヘルプデスクソフトウェア"),
        ("/live-chat-software/", "/help-desk-software/", "ヘルプデスク"),
    },
    "zh-hans": {
        ("/bangzhutai/", "/hujiao-zhongxin/", "呼叫中心"),
        ("/bangzhutai/", "/zaixian-liaotian/", "在线聊天"),
        ("/hujiao-zhongxin/", "/zaixian-liaotian/", "在线聊天"),
        ("/zaixian-liaotian/", "/bangzhutai/", "帮助台"),
        ("/zaixian-liaotian/", "/hujiao-zhongxin/", "呼叫中心"),
        ("/zhishiku/", "/bangzhutai/", "帮助台软件"),
        ("/zhishiku/", "/zaixian-liaotian/", "在线聊天"),
    },
}


class HtmlCjkFixture(unittest.TestCase):
    def _run(self, lang):
        """Generate with --write, inject into a copy of the build; returns {page url: [(href, text)]}."""
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "site"
            shutil.copytree(FIXTURE / "site", work)
            os.chdir(work)
            try:
                rc = run_generator([
                    "--lang", lang, "--public-dir", f"public-{lang}", "--content-at-root", "--no-cache",
                    "--write", "--remove-old-linkbuilding",
                    "--similarity-floor", "-1", "--lift-floor", "-1", "--top-k-per-page", "10",
                ])
                self.assertEqual(rc, 0)
                injected = Path(tmp) / "injected"
                shutil.copytree(work / f"public-{lang}", injected)
                rc = run_injector([
                    "--content-root", "content", "--public-dir", str(injected), "--lang", lang,
                    "--content-at-root", "--linkbuilding-dir", "data/linkbuilding",
                    "--links-per-words", "200", "--links-min", "3", "--links-max", "40", "--file-workers", "1",
                ])
                self.assertEqual(rc, 0)
            finally:
                os.chdir(cwd)
            links = {}
            for page in injected.rglob("index.html"):
                soup = BeautifulSoup(page.read_text(encoding="utf-8"), "lxml")
                url = "/" + page.parent.relative_to(injected).as_posix() + "/"
                links[url] = [(a["href"], a.get_text()) for a in soup.find_all("a", class_="prose-links")]
                original = BeautifulSoup((work / f"public-{lang}" / page.relative_to(injected)).read_text(encoding="utf-8"), "lxml")
                visible = original.find("main").get_text()
                for _, text in links[url]:
                    self.assertIn(text, visible)  # contiguous visible text of the page
            return links

    def test_generated_links_are_injected(self):
        for lang, expected in EXPECTED.items():
            with self.subTest(lang=lang):
                links = self._run(lang)
                got = {(url, href, text) for url, items in links.items() for href, text in items}
                self.assertEqual(got, expected)
                for _, href, text in got:
                    self.assertNotIn(href, {"/live/", "/zhongyaoxing/"})  # one-word keywords only
                    self.assertNotIn('="', text)


if __name__ == "__main__":
    unittest.main()
