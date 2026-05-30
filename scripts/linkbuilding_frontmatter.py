#!/usr/bin/env python3
"""Apply Hugo linkbuilding from two sources:

1. Page-local ``[[lnks]]`` frontmatter — manually curated per-page links.
2. Global ``data/linkbuilding/<lang>.json`` — manually maintained keyword→URL list.

Both are applied in a single pass per HTML file. Global keywords are pre-filtered
against raw HTML before BeautifulSoup is invoked, so only keywords that actually
appear in the page text reach the DOM search — keeping the apply step fast.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
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
    exact_match: bool = False


@dataclass
class LinkConfig:
    max_links_per_page: int = 8
    max_same_url_per_page: int = 1


@dataclass
class LinkStats:
    total_files_processed: int = 0
    total_files_modified: int = 0
    total_links_added: int = 0


def _keyword_pattern(keyword: str) -> re.Pattern[str]:
    escaped = re.escape(keyword.strip())
    return re.compile(rf"(?<![\w-]){escaped}(?![\w-])", re.IGNORECASE)


class LinkBuilder:
    def __init__(self, keywords: list[Keyword], config: LinkConfig) -> None:
        self.keywords = sorted(keywords, key=lambda kw: (-kw.priority, -len(kw.keyword)))
        self.config = config
        self.stats = LinkStats()

    def process_file(self, html_path: Path) -> bool:
        self.stats.total_files_processed += 1
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

        # Collect valid text nodes ONCE — reused across all keywords.
        # After inserting a link, the replaced node is detached (parent → None)
        # and the surrounding text fragments are appended so later keywords can match them.
        valid_nodes: list[NavigableString] = [
            node for node in soup.find_all(string=True)
            if isinstance(node, NavigableString)
            and node.parent
            and node.parent.name not in SKIP_TEXT_PARENTS
            and not node.find_parent(SKIP_TEXT_ANCESTORS)
        ]

        for keyword in keywords:
            if added >= self.config.max_links_per_page:
                break
            if used_urls.get(keyword.url, 0) >= self.config.max_same_url_per_page:
                continue
            if keyword.keyword.lower() in used_texts:
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
) -> tuple[int, int, int]:
    """Worker function — global keywords are already in process memory via initializer."""
    page_kws = [Keyword(**d) for d in page_kw_data]
    keywords = _dedupe_keywords(_worker_global_keywords + page_kws)
    builder = LinkBuilder(keywords, _worker_config)
    builder.process_file(Path(html_path_str))
    s = builder.stats
    return s.total_files_processed, s.total_files_modified, s.total_links_added


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


def _load_page_keywords(content_dir: Path, public_dir: Path) -> dict[Path, list[Keyword]]:
    """Read [[lnks]] from every .md file and map to HTML paths in public/."""
    page_keywords: dict[Path, list[Keyword]] = {}
    for file_path in sorted(content_dir.rglob("*.md")):
        if any(part.startswith(".") for part in file_path.parts):
            continue
        try:
            raw = file_path.read_text(encoding="utf-8")
            post = frontmatter.loads(raw, handler=frontmatter.TOMLHandler())
        except Exception as exc:
            print(f"Warning: failed to parse {file_path}: {exc}", file=sys.stderr)
            continue

        lnks = post.metadata.get("lnks") or []
        if not isinstance(lnks, list) or not lnks:
            continue

        keywords: list[Keyword] = []
        for idx, item in enumerate(lnks):
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or item.get("keyword") or "").strip()
            url = str(item.get("path") or item.get("url") or "").strip()
            title = str(item.get("title") or "").strip()
            if not text or not url:
                continue
            keywords.append(Keyword(keyword=text, url=url, title=title,
                                    priority=1000 - idx, exact_match=True))

        if not keywords:
            continue

        page_url = _url_for_file(file_path, content_dir, post.metadata or {})
        html_path = _html_path_for_url(public_dir, page_url)
        page_keywords.setdefault(html_path, []).extend(keywords)
    return page_keywords


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
        exact = bool(item.get("Exact", item.get("exact", False)))
        keywords.append(Keyword(keyword=text, url=url, title=title,
                                priority=priority, exact_match=exact))
    return keywords


def _load_global_keywords(linkbuilding_dir: Path, lang: str) -> list[Keyword]:
    """Load manually maintained global keywords for this language.

    Merges two sources (highest priority wins on dedup):
      - data/linkbuilding/all.json   — applied to every language
      - data/linkbuilding/<lang>.json — language-specific additions
    """
    keywords: list[Keyword] = []
    for filename in ("all.json", f"{lang}.json"):
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
    config = LinkConfig()
    config_data = asdict(config)

    total_pages = 0
    total_processed = 0
    total_modified = 0
    total_links = 0

    for content_dir in _content_dirs(content_root, args.lang):
        lang = content_dir.name
        lang_public_dir = public_dir if lang == "en" else public_dir / lang

        # Source 1: page-specific links from [[lnks]] frontmatter
        page_keywords: dict[Path, list[Keyword]] = _load_page_keywords(content_dir, public_dir)

        # Source 2: global manual keywords from data/linkbuilding/<lang>.json
        # Applied to ALL HTML files; pre-filtered per page against raw HTML before DOM parse.
        global_keywords = _load_global_keywords(linkbuilding_dir, lang)
        global_kw_data = [asdict(kw) for kw in global_keywords]

        # Build the work list: HTML files in the language dir.
        # For English (served at public/ root) exclude subdirs for other languages.
        if lang_public_dir.exists():
            if lang == "en":
                html_files = sorted(
                    p for p in lang_public_dir.rglob("*.html")
                    if p.relative_to(lang_public_dir).parts
                    and p.relative_to(lang_public_dir).parts[0] not in LANG_CODES
                )
            else:
                html_files = sorted(lang_public_dir.rglob("*.html"))
        else:
            html_files = []
        total_pages += len(page_keywords)

        # Work list: per-file only passes page-specific keywords.
        # Global keywords are loaded once per worker via initializer.
        items: list[tuple[str, list[dict]]] = []
        for html_path in html_files:
            page_kws = page_keywords.get(html_path, [])
            # Include file if it has page-specific links OR global keywords exist
            if page_kws or global_keywords:
                items.append((str(html_path), [asdict(kw) for kw in page_kws]))

        lang_processed = lang_modified = lang_links = 0

        if items and args.file_workers > 1:
            with ProcessPoolExecutor(
                max_workers=args.file_workers,
                initializer=_worker_init,
                initargs=(global_kw_data, config_data),
            ) as executor:
                futures = [executor.submit(_process_file_worker, *item) for item in items]
                for future in as_completed(futures):
                    try:
                        p, m, lk = future.result()
                        lang_processed += p
                        lang_modified += m
                        lang_links += lk
                    except Exception as exc:
                        print(f"Warning: worker error: {exc}", file=sys.stderr)
        else:
            _worker_init(global_kw_data, config_data)
            for item in items:
                p, m, lk = _process_file_worker(*item)
                lang_processed += p
                lang_modified += m
                lang_links += lk

        total_processed += lang_processed
        total_modified += lang_modified
        total_links += lang_links
        print(f"[{lang}] processed {lang_processed} files, modified {lang_modified}, added {lang_links} links")

    summary = {
        "pages_with_lnks": total_pages,
        "files_processed": total_processed,
        "files_modified": total_modified,
        "links_added": total_links,
    }
    print("Frontmatter linkbuilding completed:", json.dumps(summary, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply linkbuilding from Hugo [[lnks]] frontmatter")
    parser.add_argument("--content-root", default="content")
    parser.add_argument("--public-dir", default="public")
    parser.add_argument("--linkbuilding-dir", default="data/linkbuilding")
    parser.add_argument("--lang", default="", help="Optional language code, e.g. en")
    parser.add_argument("--file-workers", type=int, default=os.cpu_count() or 4,
                        help="Number of parallel worker processes")
    parser.add_argument("--include-manual", action="store_true",
                        help="Kept for backwards compatibility, no longer used.")
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
