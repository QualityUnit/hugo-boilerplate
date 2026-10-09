# Linkbuilding tests

Tests for the linkbuilding scripts in `scripts/`: the generator
(`generate_paragraph_linkbuilding.py`), the deploy-time injector
(`linkbuilding_frontmatter.py`) and their shared helpers (`linkbuilding_html.py`,
`linkbuilding_cjk.py`). They run in CI on every pull request
(`.github/workflows/linkbuilding-tests.yml`).

## Run locally

```bash
python -m pip install -r tests/linkbuilding/requirements.txt
python -B -m unittest discover -s tests/linkbuilding -t tests/linkbuilding -v
```

From the theme root. `-B` writes no bytecode, so a test run leaves no `.pyc` files
next to the scripts (`scripts/__pycache__/` also holds two tracked ones). No embedding
model is needed: `support.FakeEmbedder` derives each vector from a hash of the text,
so the tests check which paragraphs, clauses and anchors are produced, not which
target the model would prefer.

## What is covered

| File | What |
|---|---|
| `test_keyword_pattern.py` | The injector's anchor match: word boundaries for Latin and CJK text, equivalence with the previous regex on text without CJK |
| `test_word_counts.py` | `estimated_words` and the injector's link cap (CJK words from characters), same count before and after anchors are inserted |
| `test_html_paragraphs.py` | HTML path: `_html_blocks`, linkable clauses, clause punctuation, paragraph length |
| `test_anchor_candidates.py` | `_clause_word_runs`, `_clause_candidates`, `_exact_labels` (Latin and CJK), `_exact_anchor` |
| `test_cjk_words.py` | Japanese / Chinese word counts (fugashi, jieba) |
| `test_site_config.py` | `cjk_languages` / `cjk_multiword_terms` in `generator.yaml` |
| `test_requirements.py` | `tests/linkbuilding/requirements.txt` names each package with the same version bounds as `scripts/requirements.txt` |
| `test_markdown_fixture.py` | Markdown path on 10 English pages: output byte-identical (`fixtures/markdown-en`) |
| `test_html_cjk_fixture.py` | HTML path end to end on a small Japanese and Chinese site, generator then injector; one-word keywords never get a link (`fixtures/html-cjk`) |

`fixtures/markdown-en/expected.json` holds, for every file the generator writes, the
SHA-256 of its text (with `\n` line endings) and its `[[lnks]]`. It was generated
with the theme before S7 (99f80c8), so the test proves the markdown path did not change.
If a change of the markdown path is intended, rewrite it and say why in the commit
message:

```bash
LINKBUILDING_UPDATE_FIXTURES=1 python -B -m unittest discover -s tests/linkbuilding -t tests/linkbuilding -p test_markdown_fixture.py
```

`test_html_cjk_fixture.py` keeps its expected links in the test file itself.

Dependencies are lower bounds, like `scripts/requirements.txt`, not exact pins: if a
new release of markdown, bs4 or lxml changes what the generator writes, the fixture
test fails — the same change would reach the sites.
