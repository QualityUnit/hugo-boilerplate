#!/usr/bin/env python3
"""What the linkbuilding injector may link in built HTML, and where that HTML lives.

Shared by ``linkbuilding_frontmatter.py`` (the injector, which inserts the links
into ``public/``) and ``generate_paragraph_linkbuilding.py`` (the generator, which
chooses them). Both must agree on which text is linkable: an anchor the generator
picks from text the injector skips can never become a link.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from bs4 import BeautifulSoup, NavigableString


SKIP_TEXT_PARENTS = {
    # Inline/form elements that must not contain <a>
    "a", "button", "label", "summary", "legend",
    # Machine/metadata — never user-visible body content
    "script", "style", "title", "meta", "link",
    # Form inputs
    "textarea", "select", "option", "input",
    # Code / technical content
    "code", "pre", "kbd", "samp",
    # Embedded / special content
    "svg", "math", "noscript",
    # Headings — keep them link-free
    "h1", "h2", "h3", "h4", "h5", "h6",
    # Caption / table header / inline semantic — not prose
    "figcaption", "caption", "th", "cite", "time",
}
# Structural ancestors: skip the entire subtree of these elements.
# Links are only inserted in visible body prose — not in page chrome,
# document head, quotes, sidebars, forms, figures, or contact blocks.
SKIP_TEXT_ANCESTORS = {
    # Already-linked or heading context
    "a", "h1", "h2", "h3", "h4", "h5", "h6",
    # Document head — title, meta, OG tags, etc.
    "head",
    # Page chrome — navigation, site header, footer
    "header", "footer", "nav",
    # Structural containers that should stay link-free
    "aside", "form", "figure", "blockquote", "address",
}

# Opt-out marker: any element carrying this class excludes its ENTIRE subtree
# from linkbuilding. Checked via find_parent (ancestor walk), so the opt-out is
# inherited by every descendant text node at any depth — put it once on a
# banner/section wrapper and nothing inside gets auto-linked.
NO_LINKBUILDING_CLASS = "no-linkbuilding"


def is_linkbuilding_excluded(tag: Any) -> bool:
    """True if this element opts out of linkbuilding via the marker class.

    Used as a find_parent predicate so the opt-out is inherited by the whole
    subtree: a text node is excluded if it OR any ancestor carries the class.
    """
    get = getattr(tag, "get", None)
    if get is None:  # NavigableString / non-Tag — no attributes
        return False
    return NO_LINKBUILDING_CLASS in (get("class") or [])


def is_linkable_text_node(node: Any) -> bool:
    """May the injector insert a link into this text node?"""
    return (
        isinstance(node, NavigableString)
        and node.parent is not None
        and node.parent.name not in SKIP_TEXT_PARENTS
        and not node.find_parent(SKIP_TEXT_ANCESTORS)
        and not node.find_parent(is_linkbuilding_excluded)
    )


def linkable_text_nodes(soup: BeautifulSoup) -> list[NavigableString]:
    """Every text node of the document the injector may link, in document order."""
    return [node for node in soup.find_all(string=True) if is_linkable_text_node(node)]


# Japanese and Chinese script: the iteration and closing marks 々 〆 〇, kana, CJK
# ideographs (+ extension A, compatibility), half-width katakana. These languages write no spaces, so a CJK character next to
# an anchor is not "inside a word" — every CJK neighbour is a valid boundary.
CJK_CHARS = "\u3005-\u3007\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff66-\uff9f"
CJK_RE = re.compile(f"[{CJK_CHARS}]")
# A character that continues a word: \w that is not CJK, or a hyphen.
_WORD_CONTINUATION_RE = re.compile(rf"[^\W{CJK_CHARS}]|-")
_WORD_RUN_RE = re.compile(rf"[^\W{CJK_CHARS}]+")
_KANA_RE = re.compile("[\u3041-\u3096\u30a1-\u30fa\uff66-\uff9d]")
# Japanese and Chinese write no spaces, so their words are estimated from characters.
# Measured on 60 LiveAgent academy pages against the same German pages (2026-10):
# Japanese ~3.1 CJK characters per German word, Chinese ~1.8. Japanese prose is
# largely kana, Chinese has none — text whose CJK characters are at least 10 % kana
# is Japanese (a stray Japanese name in Chinese text does not flip it).
CJK_CHARS_PER_WORD_JA = 3.0
CJK_CHARS_PER_WORD_ZH = 1.8


def estimated_words(texts: list[str]) -> int:
    """Words in ``texts``: runs of non-CJK letters/digits, plus CJK characters / chars-per-word.

    The CJK characters are summed over all texts before dividing, so how a text is
    cut into pieces does not change the result. Without CJK this is the plain
    ``\\w+`` count.
    """
    runs = sum(len(_WORD_RUN_RE.findall(text)) for text in texts)
    cjk = sum(len(CJK_RE.findall(text)) for text in texts)
    if not cjk:
        return runs
    kana = sum(len(_KANA_RE.findall(text)) for text in texts)
    per_word = CJK_CHARS_PER_WORD_JA if kana * 10 >= cjk else CJK_CHARS_PER_WORD_ZH
    return runs + int(cjk / per_word)


class KeywordPattern:
    """``search`` like a compiled pattern, for an anchor text with word boundaries.

    Not a single regular expression with lookarounds: a character class holding the
    CJK ranges costs ~7 ms to compile (~0.05 ms for ``[\\w-]``), once per keyword, and
    the generator and the injector build tens of thousands of these per run. Here the
    keyword itself is a plain case-insensitive literal and the two neighbours of a
    match are checked with one shared, precompiled class.
    """

    __slots__ = ("_literal",)

    def __init__(self, keyword: str) -> None:
        self._literal = re.compile(re.escape(keyword.strip()), re.IGNORECASE)

    def search(self, text: str) -> re.Match[str] | None:
        if not self._literal.pattern:  # an empty keyword links nothing (and would never advance)
            return None
        pos = 0
        while True:
            match = self._literal.search(text, pos)
            if match is None:
                return None
            start, end = match.span()
            if not (start > 0 and _WORD_CONTINUATION_RE.match(text, start - 1)) and not _WORD_CONTINUATION_RE.match(text, end):
                return match
            pos = start + 1


def keyword_pattern(keyword: str) -> KeywordPattern:
    """The injector's match for an anchor text: case-insensitive, not inside a word or a hyphenated compound.

    Only Latin-script letters, digits and ``-`` count as "inside a word". A CJK
    neighbour does not: "ヘルプデスク" matches in "優れたヘルプデスクを", and
    "LiveAgent" in "LiveAgentは". On text without CJK this is the
    ``(?<![\\w-])…(?![\\w-])`` boundary it always was.
    """
    return KeywordPattern(keyword)


def canonical_path(url: str) -> str:
    path = urlparse(str(url or "")).path or str(url or "")
    if not path.startswith("/"):
        path = "/" + path
    if path != "/" and not path.endswith("/"):
        path += "/"
    return path


def html_path_for_url(public_dir: Path, url: str) -> Path:
    path = canonical_path(url).strip("/")
    if not path:
        return public_dir / "index.html"
    return public_dir / path / "index.html"


def lang_url_carries_prefix(hugo_config: Any, lang: str) -> bool:
    """Does the URL Hugo produces for this language already contain its language segment?

    Hugo decides this by whether the language has its own baseURL:

      own baseURL   -> the language is its own site; URLs carry no language segment
                       LiveAgent: liveagent.cz + "/chaport-migrace/"
      no baseURL    -> the language is a subfolder of one site; Hugo prefixes it
                       FlowHunt: flowhunt.io + "/fr/ai-flow-templates/"

    Read it from config rather than probing the filesystem: a built site also contains
    alias stubs (public/es/es/... on FlowHunt), and guessing from what exists on disk
    picks those up and injects into a redirect page instead of the real one.
    """
    try:
        languages = (hugo_config or {}).get("languages") or {}
        entry = languages.get(lang) or {}
        return not str(entry.get("baseURL") or "").strip()
    except Exception:
        return False


def language_html_root(
    public_dir: Path,
    lang: str,
    *,
    content_at_root: bool,
    url_carries_prefix: Any,
) -> tuple[Path, Path, str]:
    """Where one language's built HTML lives in ``public_dir``.

    Returns ``(lang_public_dir, html_root, layout)``: ``lang_public_dir`` holds the
    language's files, ``html_root`` is what its page URLs resolve against and
    ``layout`` names the case for the log. ``url_carries_prefix`` is a zero-argument
    callable (see ``lang_url_carries_prefix``), only called when the answer matters.

    --content-at-root: each language is built as the default at public/ root
    (per-language / per-domain deploys, e.g. PostAffiliatePro), so the HTML for the
    current language lives at public/ root, not public/<lang>/. Otherwise English sits
    at the root and every other language under public/<lang>/.

    The URL root is not always lang_public_dir. Hugo prefixes URLs with the language
    only when that language has no baseURL of its own, so the two site layouts need
    different roots:

      own baseURL (LiveAgent)  url "/chaport-migrace/"  -> public/cs/ + url
      no baseURL  (FlowHunt)   url "/fr/ai-flow..."     -> public/   + url
    """
    if content_at_root:
        lang_public_dir = public_dir
    else:
        lang_public_dir = public_dir if lang == "en" else public_dir / lang
    if content_at_root or lang_public_dir == public_dir:
        return lang_public_dir, public_dir, "content at root"
    if url_carries_prefix():
        return lang_public_dir, public_dir, "shared domain, language in URL"
    return lang_public_dir, lang_public_dir, "per-language domain"
