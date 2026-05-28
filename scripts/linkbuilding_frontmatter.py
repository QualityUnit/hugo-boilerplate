#!/usr/bin/env python3
"""Apply Hugo linkbuilding from page frontmatter.

Reads page-local ``[[lnks]]`` entries from content files and applies those
exact text/url/title pairs to the matching generated HTML file. Manual
``data/linkbuilding/<lang>.json`` keyword files are still supported as global
fallbacks for the few hand-maintained links.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
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
    "a", "button", "script", "style", "textarea", "select", "option",
    "code", "pre", "kbd", "samp", "svg", "math", "noscript",
    "h1", "h2", "h3", "h4", "h5", "h6",
}
SKIP_TEXT_ANCESTORS = {"a", "h1", "h2", "h3", "h4", "h5", "h6"}


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


def _keyword_from_mapping(item: dict[str, Any], default_priority: int = 0) -> Keyword | None:
    text = str(item.get("Keyword") or item.get("keyword") or item.get("text") or "").strip()
    url = str(item.get("URL") or item.get("url") or item.get("path") or "").strip()
    title = str(item.get("Title") or item.get("title") or "").strip()
    if not text or not url:
        return None
    priority = item.get("Priority", item.get("priority", default_priority))
    try:
        priority = int(priority)
    except (TypeError, ValueError):
        priority = default_priority
    exact = bool(item.get("Exact", item.get("exact", item.get("exact_match", False))))
    return Keyword(keyword=text, url=url, title=title, priority=priority, exact_match=exact)


def load_keywords_from_json(path: str) -> list[Keyword]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    items = data.get("keywords", data) if isinstance(data, dict) else data
    if not isinstance(items, list):
        return []
    keywords: list[Keyword] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        keyword = _keyword_from_mapping(item)
        if keyword:
            keywords.append(keyword)
    return keywords


class LinkBuilder:
    def __init__(self, keywords: list[Keyword], config: LinkConfig, language: str = "") -> None:
        self.keywords = sorted(keywords, key=lambda kw: (-kw.priority, -len(kw.keyword)))
        self.config = config
        self.language = language
        self.stats = LinkStats()

    def process_directory(
        self,
        directory: str,
        exclude_dirs: list[str] | None = None,
        is_english: bool = False,
        max_workers: int = 1,
    ) -> LinkStats:
        base_dir = Path(directory)
        exclude = set(exclude_dirs or [])
        for html_path in sorted(base_dir.rglob("*.html")):
            if is_english and html_path.relative_to(base_dir).parts[:1] and html_path.relative_to(base_dir).parts[0] in exclude:
                continue
            self.process_file(html_path)
        return self.stats

    def process_file(self, html_path: Path) -> bool:
        self.stats.total_files_processed += 1
        try:
            html = html_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            html = html_path.read_text(encoding="utf-8", errors="ignore")

        soup = BeautifulSoup(html, "html.parser")
        links_added = self._apply_links(soup)
        if links_added <= 0:
            return False

        html_path.write_text(str(soup), encoding="utf-8")
        self.stats.total_files_modified += 1
        self.stats.total_links_added += links_added
        return True

    def _apply_links(self, soup: BeautifulSoup) -> int:
        added = 0
        used_urls: dict[str, int] = {}
        used_texts: set[str] = set()

        for keyword in self.keywords:
            if added >= self.config.max_links_per_page:
                break
            if used_urls.get(keyword.url, 0) >= self.config.max_same_url_per_page:
                continue
            if keyword.keyword.lower() in used_texts:
                continue
            if self._insert_first_match(soup, keyword):
                added += 1
                used_urls[keyword.url] = used_urls.get(keyword.url, 0) + 1
                used_texts.add(keyword.keyword.lower())
        return added

    def _insert_first_match(self, soup: BeautifulSoup, keyword: Keyword) -> bool:
        pattern = _keyword_pattern(keyword.keyword)
        for text_node in list(soup.find_all(string=True)):
            if not isinstance(text_node, NavigableString):
                continue
            parent = text_node.parent
            if not parent or parent.name in SKIP_TEXT_PARENTS:
                continue
            if parent.find_parent(SKIP_TEXT_ANCESTORS):
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
                replacements.append(NavigableString(before))
            replacements.append(anchor)
            if after:
                replacements.append(NavigableString(after))
            text_node.replace_with(*replacements)
            return True
        return False


def _keyword_pattern(keyword: str) -> re.Pattern[str]:
    escaped = re.escape(keyword.strip())
    return re.compile(rf"(?<![\w-]){escaped}(?![\w-])", re.IGNORECASE)


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
    hugo_config = get_hugo_config(abs_content_dir.parent.parent)
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


def _keyword_from_lnk(item: dict[str, Any], priority: int) -> Keyword | None:
    text = str(item.get("text") or item.get("keyword") or "").strip()
    url = str(item.get("path") or item.get("url") or "").strip()
    title = str(item.get("title") or "").strip()
    if not text or not url:
        return None
    return Keyword(keyword=text, url=url, title=title, priority=priority, exact_match=True)


def _load_page_keywords(content_dir: Path, public_dir: Path) -> dict[Path, list[Keyword]]:
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
        if not isinstance(lnks, list):
            continue

        keywords: list[Keyword] = []
        for idx, item in enumerate(lnks):
            if not isinstance(item, dict):
                continue
            keyword = _keyword_from_lnk(item, priority=1000 - idx)
            if keyword:
                keywords.append(keyword)

        if not keywords:
            continue

        url = _url_for_file(file_path, content_dir, post.metadata or {})
        html_path = _html_path_for_url(public_dir, url)
        page_keywords.setdefault(html_path, []).extend(keywords)
    return page_keywords


def _load_manual_keywords(linkbuilding_dir: Path, lang: str) -> list[Keyword]:
    manual_file = linkbuilding_dir / f"{lang}.json"
    if not manual_file.exists():
        return []
    try:
        keywords = load_keywords_from_json(str(manual_file))
    except Exception as exc:
        print(f"Warning: failed to load manual linkbuilding file {manual_file}: {exc}", file=sys.stderr)
        return []
    for keyword in keywords:
        keyword.priority += 2000
    return keywords


def _content_dirs(content_root: Path, lang: str | None) -> list[Path]:
    if lang:
        content_dir = content_root / lang
        return [content_dir] if content_dir.exists() else []
    return [
        path for path in sorted(content_root.iterdir())
        if path.is_dir() and path.name in LANG_CODES
    ]


def _dedupe_keywords(keywords: list[Keyword]) -> list[Keyword]:
    seen: set[tuple[str, str]] = set()
    out: list[Keyword] = []
    for keyword in sorted(keywords, key=lambda kw: (-kw.priority, -len(kw.keyword))):
        key = (keyword.keyword.lower(), keyword.url)
        if key in seen:
            continue
        seen.add(key)
        out.append(keyword)
    return out


def run(args: argparse.Namespace) -> int:
    content_root = Path(args.content_root)
    public_dir = Path(args.public_dir)
    linkbuilding_dir = Path(args.linkbuilding_dir)
    config = LinkConfig()

    total_pages_with_lnks = 0
    total_files_processed = 0
    total_files_modified = 0
    total_links_added = 0

    for content_dir in _content_dirs(content_root, args.lang):
        lang = content_dir.name
        page_keywords = _load_page_keywords(content_dir, public_dir)
        manual_keywords = _load_manual_keywords(linkbuilding_dir, lang) if args.include_manual else []
        total_pages_with_lnks += len(page_keywords)

        lang_links = 0
        lang_modified = 0
        lang_processed = 0
        html_files = set(page_keywords)

        if manual_keywords:
            manual_dir = public_dir if lang == "en" else public_dir / lang
            if manual_dir.exists():
                builder = LinkBuilder(manual_keywords, config, language=lang.upper())
                stats = builder.process_directory(
                    str(manual_dir),
                    exclude_dirs=sorted(LANG_CODES - {"en"}) if lang == "en" else [],
                    is_english=(lang == "en"),
                    max_workers=args.file_workers,
                )
                lang_processed += stats.total_files_processed
                lang_modified += stats.total_files_modified
                lang_links += stats.total_links_added

        for html_path in sorted(html_files):
            if not html_path.exists():
                continue
            keywords = _dedupe_keywords(page_keywords.get(html_path, []))
            if not keywords:
                continue
            builder = LinkBuilder(keywords, config, language=lang.upper())
            if builder.process_file(html_path):
                lang_modified += builder.stats.total_files_modified
            lang_processed += builder.stats.total_files_processed
            lang_links += builder.stats.total_links_added

        total_files_processed += lang_processed
        total_files_modified += lang_modified
        total_links_added += lang_links
        print(f"[{lang}] processed {lang_processed} files, modified {lang_modified}, added {lang_links} links")

    summary = {
        "pages_with_lnks": total_pages_with_lnks,
        "files_processed": total_files_processed,
        "files_modified": total_files_modified,
        "links_added": total_links_added,
    }
    print("Frontmatter linkbuilding completed:", json.dumps(summary, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply linkbuilding from Hugo [[lnks]] frontmatter")
    parser.add_argument("--content-root", default="content")
    parser.add_argument("--public-dir", default="public")
    parser.add_argument("--linkbuilding-dir", default="data/linkbuilding")
    parser.add_argument("--lang", default="", help="Optional language code, e.g. en")
    parser.add_argument("--file-workers", type=int, default=4)
    parser.add_argument("--include-manual", action="store_true", help="Also apply global data/linkbuilding/<lang>.json keywords")
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
