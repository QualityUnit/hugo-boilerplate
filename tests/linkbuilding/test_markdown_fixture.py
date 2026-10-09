"""Markdown path regression: 10 English pages, output must stay byte-identical.

The markdown path (no --public-dir) is what every site used before the HTML path
existed; every later change keeps its output identical. fixtures/markdown-en/site is a
copy of 10 LiveAgent pages (content, a trimmed config, data) and expected.json holds,
per written file, the SHA-256 of its text with \\n line endings (so the test also
passes on a Windows checkout with core.autocrlf) and its [[lnks]] as (text, path)
pairs for a readable diff when the hash changes.

Refreshing expected.json after an intended change of the markdown path — say why in
the commit message:

    LINKBUILDING_UPDATE_FIXTURES=1 python -B -m unittest discover -s tests/linkbuilding -t tests/linkbuilding -p test_markdown_fixture.py
"""

import hashlib
import json
import os
import re
import shutil
import tempfile
import tomllib
import unittest
from pathlib import Path

from support import FIXTURES, run_generator

FIXTURE = FIXTURES / "markdown-en"
UPDATE = os.environ.get("LINKBUILDING_UPDATE_FIXTURES") == "1"


def written_files(content: Path) -> dict:
    """{relative path: {"sha256", "lnks"}} for every page the generator wrote."""
    out = {}
    for page in sorted(content.rglob("*.md"), key=lambda p: p.relative_to(content).as_posix()):
        text = page.read_bytes().decode("utf-8").replace("\r\n", "\n")
        meta = tomllib.loads(re.match(r"^\+\+\+\n(.*?)\n\+\+\+", text, re.S).group(1))
        out[page.relative_to(content).as_posix()] = {
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "lnks": [[item.get("text"), item.get("path")] for item in meta.get("lnks") or []],
        }
    return out


class MarkdownFixture(unittest.TestCase):
    def test_output_is_unchanged(self):
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "site"
            shutil.copytree(FIXTURE / "site", work)
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
            got = written_files(work / "content")

        if UPDATE:
            (FIXTURE / "expected.json").write_text(json.dumps(got, ensure_ascii=False, indent=1) + "\n",
                                                   encoding="utf-8", newline="\n")
            self.skipTest("expected.json rewritten")
        expected = json.loads((FIXTURE / "expected.json").read_text(encoding="utf-8"))
        self.assertEqual(sorted(got), sorted(expected))
        for rel, want in expected.items():
            with self.subTest(file=rel):
                self.assertEqual(got[rel]["lnks"], want["lnks"])
                self.assertEqual(got[rel]["sha256"], want["sha256"])


if __name__ == "__main__":
    unittest.main()
