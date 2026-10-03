"""Shared setup for the linkbuilding tests: import path, quiet config loading, fake embedder.

The scripts live in scripts/ and import each other by module name, so that directory
goes on sys.path. Bytecode writing is switched off before anything is imported, so a
test run leaves no .pyc files next to the scripts (scripts/__pycache__/ also holds two
tracked ones).
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import sys
from pathlib import Path

sys.dont_write_bytecode = True

THEME = Path(__file__).resolve().parents[2]
SCRIPTS = THEME / "scripts"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import numpy as np  # noqa: E402

import generate_paragraph_linkbuilding as generator  # noqa: E402,F401
import linkbuilding_frontmatter as injector  # noqa: E402,F401
import linkbuilding_html  # noqa: E402,F401


def quiet(fn, *args, **kwargs):
    """Call ``fn`` with its stdout swallowed (the config loaders log every call)."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def theme_rules(lang: str):
    """SiteRules for ``lang`` from the theme defaults alone (no site generator.yaml)."""
    # The file does not exist on purpose: a missing site config means theme defaults.
    config = quiet(generator._load_site_config, FIXTURES / "no-site-config.yaml")
    return quiet(generator._site_rules, config, lang)


class FakeEmbedder:
    """Deterministic stand-in for the sentence-transformers model: unit vectors from a hash.

    Fit and lift become arbitrary but repeatable, which is enough for tests that check
    which paragraphs, clauses and anchors the generator produces, not which target wins.
    The vector is the text's SHAKE-256 digest, not a seeded random generator: numpy does
    not promise the same random stream across versions, a hash does not change.
    """

    def __init__(self, *args, **kwargs):
        self.resolved_device = "cpu"

    def encode(self, texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True, **kwargs):
        vectors = []
        for text in texts:
            digest = hashlib.shake_256(text.encode("utf-8")).digest(64)
            vector = np.frombuffer(digest, dtype=np.uint8).astype(np.float32) - 127.5
            vectors.append(vector / np.linalg.norm(vector))
        return np.asarray(vectors, dtype=np.float32)


def run_injector(argv: list[str]) -> int:
    """Run the injector's main() with ``argv``, output swallowed."""
    original_argv = sys.argv
    sys.argv = ["linkbuilding_frontmatter.py", *argv]
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return injector.main()
    finally:
        sys.argv = original_argv


def run_generator(argv: list[str]) -> int:
    """Run the generator's main() with ``argv`` and the fake embedder, output swallowed."""
    original_model, original_argv = generator.LazySentenceTransformer, sys.argv
    generator.LazySentenceTransformer = FakeEmbedder
    sys.argv = ["generate_paragraph_linkbuilding.py", *argv]
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return generator.main()
    finally:
        generator.LazySentenceTransformer, sys.argv = original_model, original_argv
