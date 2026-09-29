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


def keyword_pattern(keyword: str) -> re.Pattern[str]:
    """The injector's match for an anchor text: case-insensitive, not inside a word or a hyphenated compound."""
    escaped = re.escape(keyword.strip())
    return re.compile(rf"(?<![\w-]){escaped}(?![\w-])", re.IGNORECASE)


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
