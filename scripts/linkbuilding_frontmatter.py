#!/usr/bin/env python3
"""Apply Hugo linkbuilding from three sources:

1. Page-local ``[[lnks_man]]`` frontmatter — hand-authored links that automated
   generation must never touch. Highest priority.
2. Page-local ``[[lnks]]`` frontmatter — links written by
   ``generate_paragraph_linkbuilding.py``; regenerated on every run.
3. Global ``data/linkbuilding/<lang>.json`` — manually maintained keyword→URL list.

Both are applied in a single pass per HTML file. Global keywords are pre-filtered
against raw HTML before BeautifulSoup is invoked, so only keywords that actually
appear in the page text reach the DOM search — keeping the apply step fast.

How many links a page gets (``LinkConfig.cap_for``):

- default: a flat ``--max-links-per-page`` (8) — the behaviour every site gets when
  it passes nothing;
- ``--links-per-words W``: one link per W words of linkable prose, clamped to
  ``--links-min`` / ``--links-max`` (3 / 40);
- frontmatter ``linkbuilding_max = N`` on a page overrides both; ``0`` disables
  injection on that page.

The run is idempotent: ``<a class="prose-links">`` anchors already in the HTML count
against the cap and block their URL / anchor text, so a second pass over the same
``public/`` adds nothing. A keyword whose URL is the page's own URL is never applied.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from bs4 import BeautifulSoup, NavigableString
import toml_frontmatter as frontmatter
from sync_translation_urls import ensure_url_slashes, get_directory_url_path, get_hugo_config


LANG_CODES = {
    "ar", "cs", "da", "de", "en", "es", "fi", "fr", "it", "ja", "ko",
    "nl", "no", "pl", "pt", "ro", "sk", "sv", "tr", "vi", "zh",
}

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


def _is_linkbuilding_excluded(tag: Any) -> bool:
    """True if this element opts out of linkbuilding via the marker class.

    Used as a find_parent predicate so the opt-out is inherited by the whole
    subtree: a text node is excluded if it OR any ancestor carries the class.
    """
    get = getattr(tag, "get", None)
    if get is None:  # NavigableString / non-Tag — no attributes
        return False
    return NO_LINKBUILDING_CLASS in (get("class") or [])

_hugo_config_cache: dict[str, Any] = {}


def _get_hugo_config_cached(hugo_root: Path) -> Any:
    key = str(hugo_root.resolve())
    if key not in _hugo_config_cache:
        _hugo_config_cache[key] = get_hugo_config(hugo_root)
    return _hugo_config_cache[key]


@dataclass
class Keyword:
    keyword: str
    url: str
    title: str = ""
    priority: int = 0


@dataclass
class LinkConfig:
    max_links_per_page: int = 8
    max_same_url_per_page: int = 1
    # Word-count policy — off (0) unless --links-per-words is passed, so a site that
    # passes nothing keeps the flat max_links_per_page cap.
    links_per_words: int = 0
    links_min: int = 3
    links_max: int = 40

    def cap_for(self, words: int, page_max: int | None = None) -> int:
        """Total number of prose-links anchors this page may carry.

        Frontmatter ``linkbuilding_max`` wins outright (0 = no injection). Otherwise
        one link per ``links_per_words`` words, clamped to [links_min, links_max];
        with the policy off, the flat ``max_links_per_page``.
        """
        if page_max is not None:
            return max(0, int(page_max))
        if self.links_per_words > 0:
            by_words = words // self.links_per_words
            return max(self.links_min, min(self.links_max, by_words))
        return self.max_links_per_page


@dataclass
class LinkStats:
    total_files_processed: int = 0
    total_files_modified: int = 0
    total_links_added: int = 0
    total_words: int = 0           # linkable prose words on pages that reached the DOM stage
    existing_links: int = 0        # prose-links anchors already present before this run
    pages_at_cap: int = 0          # pages whose cap was reached (pre-existing + added)
    pages_disabled: int = 0        # pages with linkbuilding_max = 0
    self_links_skipped: int = 0    # keyword candidates pointing at the page's own URL


def _add_stats(total: LinkStats, part: dict[str, int]) -> None:
    for key, value in part.items():
        setattr(total, key, getattr(total, key) + value)


def _links_per_1000_words(stats: LinkStats) -> float:
    """Density of injected links (pre-existing + added) over the prose that was measured."""
    if stats.total_words <= 0:
        return 0.0
    return round(1000 * (stats.existing_links + stats.total_links_added) / stats.total_words, 2)


_WORD_RE = re.compile(r"\w+")


def _count_words(text: str) -> int:
    """Words = runs of letters/digits (``\\w+``).

    Deliberately not ``str.split()``: an inserted anchor splits a text node at a word
    boundary (``_keyword_pattern`` guarantees the characters around the match are not
    ``\\w``), so every ``\\w+`` run survives the split intact and pass 1 and pass 2 count
    the same words. With ``split()`` a ``.`` left behind after an anchor becomes an extra
    token and the cap could drift by one on a re-run.
    """
    return len(_WORD_RE.findall(text))


def _word_count(nodes: list[NavigableString]) -> int:
    """Word count across plain text nodes (comments, CDATA etc. excluded)."""
    return sum(_count_words(str(node)) for node in nodes if type(node) is NavigableString)


def _keyword_pattern(keyword: str) -> re.Pattern[str]:
    escaped = re.escape(keyword.strip())
    return re.compile(rf"(?<![\w-]){escaped}(?![\w-])", re.IGNORECASE)


class LinkBuilder:
    def __init__(
        self,
        keywords: list[Keyword],
        config: LinkConfig,
        page_url: str = "",
        page_max: int | None = None,
    ) -> None:
        self.keywords = sorted(keywords, key=lambda kw: (-kw.priority, -len(kw.keyword)))
        self.config = config
        self.stats = LinkStats()
        # The page's own URL path ("" = unknown) — keywords pointing at it are skipped.
        self.page_url = _canonical_path(page_url) if page_url else ""
        # Frontmatter linkbuilding_max for this page, None when not set.
        self.page_max = page_max

    def process_file(self, html_path: Path) -> bool:
        self.stats.total_files_processed += 1
        if self.page_max is not None and self.page_max <= 0:
            # linkbuilding_max = 0: this page never receives injected links.
            self.stats.pages_disabled += 1
            return False
        try:
            html = html_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            html = html_path.read_text(encoding="utf-8", errors="ignore")

        # Pre-filter: only keep keywords that appear anywhere in the raw HTML.
        # This eliminates most global keywords per page before the expensive DOM parse.
        html_lower = html.lower()
        applicable = [kw for kw in self.keywords if kw.keyword.lower() in html_lower]
        if not applicable:
            return False

        soup = BeautifulSoup(html, "lxml")
        links_added = self._apply_links(soup, applicable)
        if links_added <= 0:
            return False

        html_path.write_text(str(soup), encoding="utf-8")
        self.stats.total_files_modified += 1
        self.stats.total_links_added += links_added
        return True

    def _apply_links(self, soup: BeautifulSoup, keywords: list[Keyword]) -> int:
        added = 0
        used_urls: dict[str, int] = {}
        used_texts: set[str] = set()

        # Anchors injected by an earlier run count against the cap and block their
        # URL / anchor text, so re-running over an already-processed public/ is a no-op.
        existing = soup.find_all("a", class_="prose-links")
        pre_existing = len(existing)
        for anchor in existing:
            href = str(anchor.get("href") or "")
            if href:
                used_urls[href] = used_urls.get(href, 0) + 1
            used_texts.add(anchor.get_text().strip().lower())

        # Collect valid text nodes ONCE — reused across all keywords.
        # After inserting a link, the replaced node is detached (parent → None)
        # and the surrounding text fragments are appended so later keywords can match them.
        valid_nodes: list[NavigableString] = [
            node for node in soup.find_all(string=True)
            if isinstance(node, NavigableString)
            and node.parent
            and node.parent.name not in SKIP_TEXT_PARENTS
            and not node.find_parent(SKIP_TEXT_ANCESTORS)
            and not node.find_parent(_is_linkbuilding_excluded)
        ]

        # Linkable prose words. Existing prose-links anchors sit inside <a>, which the
        # list above excludes, so their text is added back — pass 1 and pass 2 must
        # measure the same page the same way.
        words = _word_count(valid_nodes) + sum(_count_words(a.get_text()) for a in existing)
        cap = self.config.cap_for(words, self.page_max)
        self.stats.total_words += words
        self.stats.existing_links += pre_existing

        for keyword in keywords:
            if pre_existing + added >= cap:
                break
            if used_urls.get(keyword.url, 0) >= self.config.max_same_url_per_page:
                continue
            if keyword.keyword.lower() in used_texts:
                continue
            if self.page_url and _canonical_path(keyword.url) == self.page_url:
                self.stats.self_links_skipped += 1
                continue

            pattern = _keyword_pattern(keyword.keyword)
            new_fragments: list[NavigableString] = []

            for text_node in valid_nodes:
                if text_node.parent is None:  # already replaced by a prior insertion
                    continue
                text = str(text_node)
                match = pattern.search(text)
                if not match:
                    continue

                before = text[:match.start()]
                label = text[match.start():match.end()]
                after = text[match.end():]

                anchor = soup.new_tag("a", href=keyword.url)
                if keyword.title:
                    anchor["title"] = keyword.title
                anchor["class"] = anchor.get("class", []) + ["prose-links"]
                anchor.string = label

                replacements: list[Any] = []
                if before:
                    b_node = NavigableString(before)
                    replacements.append(b_node)
                    new_fragments.append(b_node)
                replacements.append(anchor)
                if after:
                    a_node = NavigableString(after)
                    replacements.append(a_node)
                    new_fragments.append(a_node)

                text_node.replace_with(*replacements)
                added += 1
                used_urls[keyword.url] = used_urls.get(keyword.url, 0) + 1
                used_texts.add(keyword.keyword.lower())
                break

            valid_nodes.extend(new_fragments)

        if pre_existing + added >= cap:
            self.stats.pages_at_cap += 1
        return added


# Per-worker globals set via initializer — avoids pickling global keywords per file
_worker_global_keywords: list[Keyword] = []
_worker_config: LinkConfig = LinkConfig()


def _worker_init(global_kw_data: list[dict], config_data: dict) -> None:
    global _worker_global_keywords, _worker_config
    _worker_global_keywords = [Keyword(**d) for d in global_kw_data]
    _worker_config = LinkConfig(**config_data)


def _process_file_worker(
    html_path_str: str,
    page_kw_data: list[dict],
    page_meta: dict | None = None,
) -> dict[str, int]:
    """Worker function — global keywords are already in process memory via initializer.

    ``page_meta`` carries the page's own URL (self-link check) and its frontmatter
    ``linkbuilding_max``; the per-file ``LinkStats`` come back as a dict.
    """
    page_kws = [Keyword(**d) for d in page_kw_data]
    keywords = _dedupe_keywords(_worker_global_keywords + page_kws)
    meta = page_meta or {}
    builder = LinkBuilder(
        keywords,
        _worker_config,
        page_url=str(meta.get("url") or ""),
        page_max=meta.get("max_links"),
    )
    builder.process_file(Path(html_path_str))
    return asdict(builder.stats)


def _canonical_path(url: str) -> str:
    path = urlparse(str(url or "")).path or str(url or "")
    if not path.startswith("/"):
        path = "/" + path
    if path != "/" and not path.endswith("/"):
        path += "/"
    return path


def _url_for_file(file_path: Path, content_dir: Path, metadata: dict[str, Any]) -> str:
    url = str(metadata.get("url") or "").strip()
    if url:
        return _canonical_path(url)

    abs_file_path = file_path.resolve()
    abs_content_dir = content_dir.resolve()
    hugo_config = _get_hugo_config_cached(abs_content_dir.parent.parent)
    base_url = get_directory_url_path(abs_file_path.parent, abs_content_dir.name, hugo_config)
    if base_url:
        if file_path.name == "_index.md":
            return _canonical_path(ensure_url_slashes(base_url))
        return _canonical_path(ensure_url_slashes(f"{base_url.rstrip('/')}/{file_path.stem}/"))

    rel = file_path.relative_to(content_dir)
    path = str(rel).replace("\\", "/")
    path = path.removesuffix(".md")
    if path.endswith("/_index"):
        path = path[:-len("/_index")] + "/"
    elif path == "_index":
        path = "/"
    elif path.endswith("/index"):
        path = path[:-len("/index")] + "/"
    return _canonical_path(path)


def _html_path_for_url(public_dir: Path, url: str) -> Path:
    path = _canonical_path(url).strip("/")
    if not path:
        return public_dir / "index.html"
    return public_dir / path / "index.html"


def _url_for_html_path(html_root: Path, html_path: Path) -> str:
    """Inverse of _html_path_for_url: the URL path a built HTML file is served at.

    ``help-desk-software/index.html`` under the root → ``/help-desk-software/``;
    a bare ``404.html`` → ``/404.html``; a file outside the root → "".
    """
    try:
        rel = html_path.resolve().relative_to(html_root.resolve())
    except ValueError:
        return ""
    parts = list(rel.parts)
    if parts and parts[-1] == "index.html":
        return _canonical_path("/" + "/".join(parts[:-1]))
    return "/" + "/".join(parts)


PAGE_MAX_KEY = "linkbuilding_max"


@dataclass
class PageMeta:
    """Per-page settings read from frontmatter next to [[lnks]]."""
    max_links: int | None = None


def _page_max_from_metadata(metadata: dict, source: str = "") -> int | None:
    """``linkbuilding_max`` as a non-negative int, or None when absent / unusable."""
    if PAGE_MAX_KEY not in (metadata or {}):
        return None
    raw = metadata[PAGE_MAX_KEY]
    value: int | None = None
    if isinstance(raw, int) and not isinstance(raw, bool):
        value = raw
    elif isinstance(raw, str):
        try:
            value = int(raw.strip())
        except ValueError:
            value = None
    if value is None:
        print(f"Warning: {source}: {PAGE_MAX_KEY} must be an integer, got {raw!r} — ignored", file=sys.stderr)
        return None
    if value < 0:
        print(f"Warning: {source}: {PAGE_MAX_KEY} is negative ({value}) — treated as 0", file=sys.stderr)
        return 0
    return value


# Page-local link sources, highest priority first. ``lnks_man`` is hand-authored
# and is never rewritten by generate_paragraph_linkbuilding.py, so it wins over a
# generated ``lnks`` entry that claims the same anchor text.
PAGE_LINK_KEYS = (("lnks_man", 2000), ("lnks", 1000))


def _keywords_from_metadata(metadata: dict) -> list[Keyword]:
    """Build page-local keywords from both frontmatter link tables."""
    keywords: list[Keyword] = []
    seen: set[str] = set()
    for key, base_priority in PAGE_LINK_KEYS:
        items = metadata.get(key) or []
        if not isinstance(items, list):
            continue
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or item.get("keyword") or "").strip()
            url = str(item.get("path") or item.get("url") or "").strip()
            title = str(item.get("title") or "").strip()
            if not text or not url:
                continue
            dedup = text.casefold()
            if dedup in seen:
                continue
            seen.add(dedup)
            keywords.append(Keyword(keyword=text, url=url, title=title,
                                    priority=base_priority - idx))
    return keywords


def _lang_url_carries_prefix(hugo_root: Path, lang: str) -> bool:
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
        languages = _get_hugo_config_cached(hugo_root).get("languages") or {}
        entry = languages.get(lang) or {}
        return not str(entry.get("baseURL") or "").strip()
    except Exception:
        return False


def _load_page_keywords(
    content_dir: Path, html_root: Path
) -> tuple[dict[Path, list[Keyword]], dict[Path, PageMeta]]:
    """Read [[lnks_man]], [[lnks]] and linkbuilding_max from every .md file, keyed by HTML path in public/."""
    page_keywords: dict[Path, list[Keyword]] = {}
    page_meta: dict[Path, PageMeta] = {}
    for file_path in sorted(content_dir.rglob("*.md")):
        if any(part.startswith(".") for part in file_path.parts):
            continue
        try:
            raw = file_path.read_text(encoding="utf-8")
            post = frontmatter.loads(raw, handler=frontmatter.TOMLHandler())
        except Exception as exc:
            print(f"Warning: failed to parse {file_path}: {exc}", file=sys.stderr)
            continue

        keywords = _keywords_from_metadata(post.metadata)
        max_links = _page_max_from_metadata(post.metadata, str(file_path))
        if not keywords and max_links is None:
            continue

        page_url = _url_for_file(file_path, content_dir, post.metadata or {})
        html_path = _html_path_for_url(html_root, page_url)
        if keywords:
            page_keywords.setdefault(html_path, []).extend(keywords)
        if max_links is not None:
            page_meta[html_path] = PageMeta(max_links=max_links)
    return page_keywords, page_meta


def _parse_keyword_items(items: Any) -> list[Keyword]:
    keywords: list[Keyword] = []
    if not isinstance(items, list):
        return keywords
    for item in items:
        if not isinstance(item, dict):
            continue
        text = str(item.get("Keyword") or item.get("keyword") or "").strip()
        url = str(item.get("URL") or item.get("url") or "").strip()
        title = str(item.get("Title") or item.get("title") or "").strip()
        if not text or not url:
            continue
        priority = item.get("Priority", item.get("priority", 0))
        try:
            priority = int(priority)
        except (TypeError, ValueError):
            priority = 0
        # A legacy "Exact" field may still be present in the JSON; it was never read.
        keywords.append(Keyword(keyword=text, url=url, title=title, priority=priority))
    return keywords


def _load_global_keywords(linkbuilding_dir: Path, lang: str) -> list[Keyword]:
    """Load the manually maintained global keywords for this language.

    Only ``data/linkbuilding/<lang>.json`` is read. A language-independent
    ``all.json`` used to be merged in as well; it pushed English anchors and
    404 targets onto every non-English domain (LiveAgent-hugo#624), so a file by
    that name is now ignored with a warning instead of silently re-enabling that.
    """
    keywords: list[Keyword] = []
    stray = linkbuilding_dir / "all.json"
    if stray.exists():
        print(
            f"::warning::{stray} is ignored — the injector reads only <lang>.json; "
            f"move its rows into the language files or delete it.",
            file=sys.stderr,
        )
    for filename in (f"{lang}.json",):
        path = linkbuilding_dir / filename
        if not path.exists():
            continue
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            items = data.get("keywords", data) if isinstance(data, dict) else data
            keywords.extend(_parse_keyword_items(items))
        except Exception as exc:
            print(f"Warning: failed to load {path}: {exc}", file=sys.stderr)
    return _dedupe_keywords(keywords)


def _dedupe_keywords(keywords: list[Keyword]) -> list[Keyword]:
    seen: set[tuple[str, str]] = set()
    out: list[Keyword] = []
    for kw in sorted(keywords, key=lambda k: (-k.priority, -len(k.keyword))):
        key = (kw.keyword.lower(), kw.url)
        if key in seen:
            continue
        seen.add(key)
        out.append(kw)
    return out


def _load_page_keywords_fast(
    html_files: list[Path], content_dir: Path, public_dir: Path
) -> tuple[dict[Path, list[Keyword]], dict[Path, PageMeta]]:
    """Fast path for --since-seconds mode: derive the .md file from the HTML path directly.

    Avoids scanning all content files. Works for standard Hugo URL layouts where the
    public path mirrors the content path. Falls back gracefully when no .md is found.
    """
    page_keywords: dict[Path, list[Keyword]] = {}
    page_meta: dict[Path, PageMeta] = {}
    public_dir_abs = public_dir.resolve()
    for html_path in html_files:
        # Derive the relative URL from the HTML file path, e.g.
        # public/blog/ai-support-paradox/index.html → /blog/ai-support-paradox/
        # Resolve both paths to handle mix of absolute/relative inputs.
        try:
            rel = html_path.resolve().relative_to(public_dir_abs)
        except ValueError:
            continue
        parts = rel.parts
        if not parts or parts[-1] != "index.html":
            continue
        slug_parts = parts[:-1]  # drop index.html
        # Try candidate .md file paths
        candidates: list[Path] = []
        if slug_parts:
            # content/en/blog/ai-support-paradox.md
            candidates.append(content_dir / Path(*slug_parts[:-1]) / (slug_parts[-1] + ".md") if len(slug_parts) > 1 else content_dir / (slug_parts[0] + ".md"))
            # content/en/blog/ai-support-paradox/index.md
            candidates.append(content_dir / Path(*slug_parts) / "index.md")
        else:
            candidates.append(content_dir / "_index.md")

        for md_path in candidates:
            if not md_path.exists():
                continue
            try:
                raw = md_path.read_text(encoding="utf-8")
                post = frontmatter.loads(raw, handler=frontmatter.TOMLHandler())
            except Exception:
                break
            keywords = _keywords_from_metadata(post.metadata)
            if keywords:
                page_keywords.setdefault(html_path, []).extend(keywords)
            max_links = _page_max_from_metadata(post.metadata, str(md_path))
            if max_links is not None:
                page_meta[html_path] = PageMeta(max_links=max_links)
            break
    return page_keywords, page_meta


def _content_dirs(content_root: Path, lang: str | None) -> list[Path]:
    if lang:
        d = content_root / lang
        return [d] if d.exists() else []
    return [
        p for p in sorted(content_root.iterdir())
        if p.is_dir() and p.name in LANG_CODES
    ]


def run(args: argparse.Namespace) -> int:
    content_root = Path(args.content_root)
    public_dir = Path(args.public_dir)
    linkbuilding_dir = Path(args.linkbuilding_dir)
    config = LinkConfig(
        max_links_per_page=max(0, int(args.max_links_per_page)),
        links_per_words=max(0, int(args.links_per_words)),
        links_min=max(0, int(args.links_min)),
        links_max=max(0, int(args.links_max)),
    )
    config_data = asdict(config)
    if config.links_per_words > 0:
        print(f"Link cap: 1 per {config.links_per_words} words, clamped to "
              f"[{config.links_min}, {config.links_max}]; frontmatter {PAGE_MAX_KEY} overrides")
    else:
        print(f"Link cap: flat {config.max_links_per_page} per page; frontmatter {PAGE_MAX_KEY} overrides")

    total_pages = 0
    totals = LinkStats()

    for content_dir in _content_dirs(content_root, args.lang):
        lang = content_dir.name
        # --content-at-root: each language is built as the default at public/ root
        # (per-language / per-domain deploys, e.g. PostAffiliatePro). The HTML for the
        # current language lives at public/ root, not public/<lang>/.
        if args.content_at_root:
            lang_public_dir = public_dir
        else:
            lang_public_dir = public_dir if lang == "en" else public_dir / lang

        # Where the built HTML for THIS language's pages lives, which is not always
        # lang_public_dir. Hugo prefixes URLs with the language only when that language
        # has no baseURL of its own, so the two site layouts need different roots:
        #
        #   own baseURL (LiveAgent)  url "/chaport-migrace/"  -> public/cs/ + url
        #   no baseURL  (FlowHunt)   url "/fr/ai-flow..."     -> public/   + url
        #
        # --content-at-root stays an explicit override for per-language root builds.
        if args.content_at_root or lang_public_dir == public_dir:
            html_root, layout = public_dir, "content at root"
        elif _lang_url_carries_prefix(content_root.parent, lang):
            html_root, layout = public_dir, "shared domain, language in URL"
        else:
            html_root, layout = lang_public_dir, "per-language domain"
        print(f"[{lang}] layout: {layout} -> HTML under {html_root}")

        # --file: per-file dev mode — no rglob, no content scan, no global keywords.
        dev_mode = bool(args.files)

        if dev_mode:
            # Only process the explicitly listed HTML files (resolve to absolute paths).
            html_files = [Path(f).resolve() for f in args.files if Path(f).exists()]
        elif lang_public_dir.exists():
            # Build the full HTML file work list for normal (production) mode.
            # For a root build (English at public/ root, or --content-at-root per-language
            # deploys) exclude any stray subdirs named after other languages.
            if lang == "en" or args.content_at_root:
                html_files = sorted(
                    p for p in lang_public_dir.rglob("*.html")
                    if p.relative_to(lang_public_dir).parts
                    and p.relative_to(lang_public_dir).parts[0] not in LANG_CODES
                )
            else:
                html_files = sorted(lang_public_dir.rglob("*.html"))

            # --since-seconds: further limit to recently-modified files.
            if args.since_seconds > 0:
                cutoff = time.time() - args.since_seconds
                html_files = [p for p in html_files if p.stat().st_mtime >= cutoff]
        else:
            html_files = []

        # Source 1: page-specific links from [[lnks_man]] and [[lnks]] frontmatter.
        # Dev mode uses a fast direct-path lookup to avoid scanning all content files.
        if dev_mode:
            page_keywords, page_meta = _load_page_keywords_fast(html_files, content_dir, html_root)
            global_keywords: list[Keyword] = []  # skip global keywords in dev mode
        else:
            page_keywords, page_meta = _load_page_keywords(content_dir, html_root)
            # Source 2: global manual keywords from data/linkbuilding/<lang>.json
            # Applied to ALL HTML files; pre-filtered per page against raw HTML before DOM parse.
            global_keywords = _load_global_keywords(linkbuilding_dir, lang)
        global_kw_data = [asdict(kw) for kw in global_keywords]

        lang_pages = len(page_keywords)
        total_pages += lang_pages

        # Work list: per-file only passes page-specific keywords plus the page's own
        # URL and linkbuilding_max. Global keywords are loaded once per worker via initializer.
        items: list[tuple[str, list[dict], dict]] = []
        for html_path in html_files:
            page_kws = page_keywords.get(html_path, [])
            # Include file if it has page-specific links OR global keywords exist
            if page_kws or global_keywords:
                meta = page_meta.get(html_path)
                items.append((
                    str(html_path),
                    [asdict(kw) for kw in page_kws],
                    {
                        "url": _url_for_html_path(html_root, html_path),
                        "max_links": meta.max_links if meta else None,
                    },
                ))

        lang_stats = LinkStats()

        if items and args.file_workers > 1:
            with ProcessPoolExecutor(
                max_workers=args.file_workers,
                initializer=_worker_init,
                initargs=(global_kw_data, config_data),
            ) as executor:
                futures = [executor.submit(_process_file_worker, *item) for item in items]
                for future in as_completed(futures):
                    try:
                        _add_stats(lang_stats, future.result())
                    except Exception as exc:
                        print(f"Warning: worker error: {exc}", file=sys.stderr)
        else:
            _worker_init(global_kw_data, config_data)
            for item in items:
                _add_stats(lang_stats, _process_file_worker(*item))

        _add_stats(totals, asdict(lang_stats))
        lang_processed = lang_stats.total_files_processed
        print(
            f"[{lang}] processed {lang_processed} files, modified {lang_stats.total_files_modified}, "
            f"added {lang_stats.total_links_added} links "
            f"({lang_stats.existing_links} pre-existing, {lang_stats.pages_at_cap} pages at cap, "
            f"{lang_stats.pages_disabled} disabled, {lang_stats.self_links_skipped} self-URL keywords skipped, "
            f"{_links_per_1000_words(lang_stats)} links/1000 words)"
        )

        # Frontmatter said there is work to do, yet no HTML file was touched. That is
        # never a content problem — it means the [[lnks]]/[[lnks_man]] entries were
        # mapped to HTML paths that do not exist, so the whole page-local source was
        # silently discarded. Shout, because the run still "succeeds" otherwise.
        if lang_pages > 0 and lang_processed == 0:
            print(
                f"::warning::[{lang}] {lang_pages} page(s) carry [[lnks]] but no HTML file was "
                f"processed — page-local links were dropped. Expected HTML under "
                f"{html_root}; check that this is where the build actually wrote them.",
                file=sys.stderr,
            )

    summary = {
        "pages_with_lnks": total_pages,
        "files_processed": totals.total_files_processed,
        "files_modified": totals.total_files_modified,
        "links_added": totals.total_links_added,
        "existing_links": totals.existing_links,
        "pages_at_cap": totals.pages_at_cap,
        "pages_disabled": totals.pages_disabled,
        "self_links_skipped": totals.self_links_skipped,
        "words": totals.total_words,
        "links_per_1000_words": _links_per_1000_words(totals),
    }
    print("Frontmatter linkbuilding completed:", json.dumps(summary, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply linkbuilding from Hugo [[lnks_man]] and [[lnks]] frontmatter")
    parser.add_argument("--content-root", default="content")
    parser.add_argument("--public-dir", default="public")
    parser.add_argument("--linkbuilding-dir", default="data/linkbuilding")
    parser.add_argument("--lang", default="", help="Optional language code, e.g. en")
    parser.add_argument("--content-at-root", action="store_true",
                        help="Language is built as the default at public/ root "
                             "(per-language / per-domain deploys, e.g. PostAffiliatePro). "
                             "Forces lang_public_dir = public_dir for every language.")
    parser.add_argument("--file-workers", type=int, default=os.cpu_count() or 4,
                        help="Number of parallel worker processes")
    parser.add_argument("--max-links-per-page", type=int, default=8,
                        help="Flat cap on injected links per page, used when --links-per-words is not set (default 8).")
    parser.add_argument("--links-per-words", type=int, default=0,
                        help="Word-count policy: one link per N words of linkable prose, clamped to "
                             "--links-min/--links-max. 0 (default) keeps the flat --max-links-per-page cap. "
                             f"Frontmatter {PAGE_MAX_KEY} = N overrides either policy per page (0 disables).")
    parser.add_argument("--links-min", type=int, default=3,
                        help="Lower clamp for the word-count policy (default 3).")
    parser.add_argument("--links-max", type=int, default=40,
                        help="Upper clamp for the word-count policy (default 40).")
    parser.add_argument("--since-seconds", type=float, default=0,
                        help="Only process HTML files modified in the last N seconds.")
    parser.add_argument("--file", action="append", dest="files", default=[],
                        help="Dev mode: process only this specific HTML file (repeatable). "
                             "Skips rglob, content scan, and global keywords for instant per-page rebuilds.")
    # Deprecated no-op, hidden from --help. Still sent by FlowHunt-hugo's gulpfile
    # (gulpfile.js:604); dropping it would make their gulp linkbuilding fail with
    # "unrecognized arguments" on the next theme bump. Remove only after that
    # caller has been updated.
    parser.add_argument("--include-manual", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.include_manual:
        print("::warning::--include-manual is deprecated and ignored (manual links come from "
              "[[lnks_man]] frontmatter); remove the flag from the calling script.", file=sys.stderr)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
