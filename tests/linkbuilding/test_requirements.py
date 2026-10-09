"""tests/linkbuilding/requirements.txt follows scripts/requirements.txt.

The test list is a subset (no torch / transformers), so it cannot simply include the
scripts list. Every package it names must be in scripts/requirements.txt with the same
version bounds, otherwise the tests run against versions the sites never install.
"""

import re
import unittest

from support import SCRIPTS, THEME

_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def requirements(path):
    """Package name (PEP 503 normalised) -> the rest of its line; option lines are skipped."""
    found = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        match = _NAME_RE.match(line)
        if match is None or "://" in line:  # empty, -r / --index-url, or a URL
            continue
        name = re.sub(r"[-_.]+", "-", match.group(0)).lower()
        found[name] = line[match.end():].replace(" ", "")
    return found


class Requirements(unittest.TestCase):
    def test_test_requirements_match_scripts_requirements(self):
        scripts = requirements(SCRIPTS / "requirements.txt")
        tests = requirements(THEME / "tests" / "linkbuilding" / "requirements.txt")
        self.assertTrue(tests)
        for name, bounds in tests.items():
            with self.subTest(name):
                self.assertIn(name, scripts, "missing from scripts/requirements.txt")
                self.assertEqual(bounds, scripts[name])


if __name__ == "__main__":
    unittest.main()
