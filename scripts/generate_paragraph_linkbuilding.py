#!/usr/bin/env python3
"""Generate paragraph-aware Hugo linkbuilding frontmatter.

This replaces the old page-level ``linkbuilding = [...]`` keyword list with
explicit ``[[lnks]]`` entries:

    [[lnks]]
    text = "AI chatbot"
    path = "/ai-chatbot/"
    title = "AI Chatbot Solutions"

The algorithm mirrors the site-audit paragraph-link recommender:

1. embed every source paragraph and every target page;
2. find target pages whose page vector fits the paragraph vector;
3. require lift over the source page baseline so we link from paragraphs that
   are unusually relevant to the target;
4. select an exact anchor span already present in the paragraph by combining
   semantic anchor-target similarity, lexical target overlap, and specificity.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import markdown
import numpy as np
import yaml
from bs4 import BeautifulSoup
from tqdm import tqdm
import faiss

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embedding_cache import EmbeddingCache, default_embedding_device, shared_sqlite_cache_path

import toml_frontmatter as frontmatter
from sync_translation_urls import ensure_url_slashes, get_directory_url_path, get_hugo_config


MODEL_NAME = "Alibaba-NLP/gte-multilingual-base"

_TOKEN_RE = re.compile(r"[^\W\d_][\w'’.-]*", re.UNICODE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_MARKDOWN_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
_SHORTCODE_RE = re.compile(r"{{[<%].*?[>%]}}", re.DOTALL)
_SHORTCODE_OPEN_RE = re.compile(r"^\s*{{[<%]\s*([A-Za-z0-9_-]+)\b")
_NAV_PATH_RE = re.compile(
    r"^/(about-us|cart|checkout|login|sign[-_]?in|sign[-_]?up|account|search|contact|terms|privacy|legal|admin|wp-admin|cdn-cgi)/?",
    re.I,
)
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "at", "for",
    "with", "by", "from", "as", "is", "was", "are", "were", "be", "been",
    "this", "that", "these", "those", "it", "its", "our", "your", "their",
    "you", "we", "they", "i", "us", "them", "if", "so", "than", "then",
    "do", "does", "did", "have", "has", "had", "will", "would", "could",
    "should", "can", "may", "might", "into", "out", "up", "down", "very",
    "more", "most", "some", "any", "no", "not", "only", "also", "other",
    "such", "about", "over", "under", "all", "learn", "more", "click", "here",
    "text", "title", "image", "images", "alt", "url", "link", "links", "try",
    "now", "free", "schedule", "demo",
}
_BRAND_ONLY = {"flowhunt", "ai", "mcp", "llm"}
_GENERIC_TARGET_TERMS = {
    "flowhunt", "copilot", "ai", "mcp", "llm", "tool", "tools", "agent", "agents",
    "automation", "workflow", "workflows", "platform", "chatbot", "chatbots",
    "ready", "use", "using", "help", "helps", "key", "point", "points", "inside",
    "paste", "better", "best", "create", "build", "make", "content", "page",
    "pages", "guide", "learn", "need", "needs", "feature", "features",
    "text", "title", "image", "images", "link", "links", "button",
}
_BAD_ANCHORS = {
    "link text", "learn more", "read more", "try now", "try it now", "try it free",
    "schedule a demo", "book a demo", "get started", "click here",
}


@dataclass
class Page:
    path: Path
    rel_path: str
    url: str
    title: str
    description: str
    keywords: list[str]
    body: str
    paragraphs: list[str]
    existing_targets: set[str] = field(default_factory=set)


@dataclass
class LinkRec:
    source_path: Path
    source_rel_path: str
    source_url: str
    paragraph_index: int
    paragraph: str
    target_url: str
    target_title: str
    text: str
    title: str
    fit: float
    lift: float
    anchor_score: float
    anchor_confidence: float


@dataclass
class PreferredTargetSettings:
    min_links_per_page: int = 1
    max_targets_per_language: int = 20
    score_boost: float = 0.08
    similarity_floor: float = 0.46
    lift_floor: float = -0.05
    anchor_floor: float = 0.38


@dataclass
class TargetInfo:
    terms: set[str]
    keywords: list[str]
    title_lower: str
    slug_lower: str


@dataclass
class AnchorCandidate:
    phrase: str
    lower: str
    content_tokens: list[str]
    index: int
    length_bonus: float


class LazySentenceTransformer:
    def __init__(self, model_name: str, device: str):
        self.model_name = model_name
        self.device = device
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            try:
                import torch
                torch.set_num_threads(1)
                torch.set_num_interop_threads(1)
            except Exception:
                pass
            print(f"Loading embedding model once: {self.model_name} on {self.device}")
            self._model = SentenceTransformer(self.model_name, trust_remote_code=True, device=self.device)
        return self._model

    def encode(self, *args, **kwargs):
        return self._ensure_model().encode(*args, **kwargs)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paragraph-aware [[lnks]] frontmatter")
    parser.add_argument("--lang", default="en", help="Language code under content/, comma-separated list, or 'all' (default: en)")
    parser.add_argument("--all-languages", action="store_true", help="Process every language directory under content/")
    parser.add_argument("--content-root", default="content", help="Hugo content root (default: content)")
    parser.add_argument("--model", default=MODEL_NAME, help=f"Embedding model (default: {MODEL_NAME})")
    parser.add_argument("--device", default=default_embedding_device(), help="Embedding device (default: cpu; set FLOWHUNT_EMBEDDING_DEVICE or pass mps/cuda explicitly)")
    parser.add_argument("--top-k-per-page", type=int, default=12, help="Maximum lnks entries per source page")
    parser.add_argument("--top-k-per-paragraph", type=int, default=1, help="Maximum links per paragraph")
    parser.add_argument("--similarity-floor", type=float, default=0.52, help="Minimum paragraph-target similarity")
    parser.add_argument("--lift-floor", type=float, default=0.015, help="Minimum lift over source page baseline")
    parser.add_argument("--anchor-floor", type=float, default=0.34, help="Minimum final anchor score")
    parser.add_argument("--allow-root-target", action="store_true", help="Allow links to the language homepage")
    parser.add_argument("--max-pages", type=int, default=0, help="Limit pages for testing")
    parser.add_argument("--page-batch-size", type=int, default=48, help="Embedding batch size for pages")
    parser.add_argument("--paragraph-batch-size", type=int, default=24, help="Embedding batch size for paragraphs")
    parser.add_argument("--anchor-batch-size", type=int, default=64, help="Embedding batch size for anchor candidates")
    parser.add_argument("--max-anchor-candidates", type=int, default=8, help="Maximum lexical anchor finalists embedded per target")
    parser.add_argument("--semantic-anchor-fallback", action="store_true", help="Embed anchor finalists only when lexical scoring does not find an accepted anchor")
    parser.add_argument("--initial-targets-per-paragraph", type=int, default=5, help="Target candidates tried before expanding to the full candidate set")
    parser.add_argument("--top-targets-per-paragraph", type=int, default=12, help="Candidate target pages checked per paragraph")
    parser.add_argument("--score-batch-size", type=int, default=512, help="Paragraph rows scored per matrix batch")
    parser.add_argument("--max-page-chars", type=int, default=900, help="Maximum page label chars sent to embedder")
    parser.add_argument("--max-paragraph-chars", type=int, default=900, help="Maximum paragraph chars sent to embedder")
    parser.add_argument("--cache-path", default="", help="SQLite embedding cache path (default: .audit_cache/embedding-cache.sqlite3)")
    parser.add_argument("--no-cache", action="store_true", help="Disable persistent embedding cache")
    parser.add_argument("--preferred-targets", default="data/linkbuilding/preferred_targets.yaml", help="YAML file with weighted preferred target pages")
    parser.add_argument("--output", default="", help="Optional JSON report path")
    parser.add_argument("--write", action="store_true", help="Write [[lnks]] and remove old linkbuilding frontmatter")
    parser.add_argument("--remove-old-linkbuilding", action="store_true", help="Remove linkbuilding even for pages without generated lnks")
    return parser.parse_args()


def _tokens(text: str) -> list[str]:
    return [m.group(0).strip(".,;:!?()[]{}\"“”") for m in _TOKEN_RE.finditer(text or "") if m.group(0).strip(".,;:!?()[]{}\"“”")]


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _clip_text(text: str, max_chars: int) -> str:
    text = _normalize_space(text)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    clipped = text[:max_chars]
    # Avoid cutting in the middle of a token when possible.
    return clipped.rsplit(" ", 1)[0] or clipped


def _slug_to_text(url: str) -> str:
    path = urlparse(url).path or url
    return path.replace("/", " ").replace("-", " ").replace("_", " ")


def _canonical_path(url: str) -> str:
    path = urlparse(str(url or "")).path or str(url or "")
    if not path.startswith("/"):
        path = "/" + path
    if path != "/" and not path.endswith("/"):
        path += "/"
    return path


def _is_nav_url(url: str) -> bool:
    return bool(_NAV_PATH_RE.match(_canonical_path(url)))


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
    path = re.sub(r"\.md$", "", path)
    path = re.sub(r"(^|/)_index$", r"\1", path)
    path = re.sub(r"(^|/)index$", r"\1", path)
    return _canonical_path(path)


def _markdown_text(body: str) -> str:
    html = markdown.markdown(_strip_hugo_markup(body), extensions=["tables", "fenced_code"])
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "code", "pre"]):
        tag.decompose()
    return _normalize_space(soup.get_text(" "))


def _strip_hugo_markup(body: str) -> str:
    """Keep prose markdown, but remove Hugo shortcode/template data blocks."""
    lines = (body or "").splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        match = _SHORTCODE_OPEN_RE.match(line)
        if match:
            name = match.group(1)
            close_re = re.compile(r"{{[<%]\s*/\s*" + re.escape(name) + r"\s*[>%]}}")
            j = i + 1
            while j < len(lines) and not close_re.search(lines[j]):
                j += 1
            if j < len(lines):
                i = j + 1
                continue
            i += 1
            continue
        if stripped.startswith("{{") and stripped.endswith("}}"):
            i += 1
            continue
        out.append(_SHORTCODE_RE.sub(" ", line))
        i += 1
    return "\n".join(out)


def _looks_like_structured_data(text: str) -> bool:
    if not text:
        return True
    structural = sum(text.count(ch) for ch in "{}[]=:\"")
    if structural / max(1, len(text)) > 0.08:
        return True
    lower = text.lower()
    ui_terms = sum(1 for term in ("imaged", "imagealt", "linktext", "categorycolor", "primarycta") if term in lower)
    return ui_terms >= 2


def _paragraphs_from_markdown(body: str) -> list[str]:
    cleaned = _strip_hugo_markup(body)
    html = markdown.markdown(cleaned, extensions=["tables", "fenced_code"])
    soup = BeautifulSoup(html, "html.parser")
    paragraphs: list[str] = []
    for node in soup.find_all(["p", "li"]):
        text = _normalize_space(node.get_text(" "))
        words = _tokens(text)
        if len(words) >= 18 and not text.startswith("{{") and not _looks_like_structured_data(text):
            paragraphs.append(text)
    if paragraphs:
        return paragraphs
    text = _markdown_text(cleaned)
    return [p for p in _SENTENCE_SPLIT_RE.split(text) if len(_tokens(p)) >= 18 and not _looks_like_structured_data(p)]


def _existing_link_targets(body: str) -> set[str]:
    targets = set()
    for href in _MARKDOWN_LINK_RE.findall(body or ""):
        href = href.strip()
        if href.startswith(("http://", "https://", "#", "mailto:", "tel:")):
            continue
        targets.add(_canonical_path(href))
    return targets


def _load_pages(content_dir: Path, max_pages: int = 0) -> list[Page]:
    pages: list[Page] = []
    for file_path in _content_files(content_dir, max_pages=max_pages):
        raw = file_path.read_text(encoding="utf-8")
        post = frontmatter.loads(raw, handler=frontmatter.TOMLHandler())
        meta = post.metadata or {}
        title = str(meta.get("title") or "").strip()
        description = str(meta.get("description") or meta.get("shortDescription") or "").strip()
        if not title and not description:
            continue
        url = _url_for_file(file_path, content_dir, meta)
        if _is_nav_url(url):
            continue
        paragraphs = _paragraphs_from_markdown(post.content)
        if not paragraphs:
            continue
        keywords = [str(k).strip() for k in (meta.get("keywords") or []) if str(k).strip()]
        pages.append(Page(
            path=file_path,
            rel_path=str(file_path.relative_to(content_dir)).replace("\\", "/"),
            url=url,
            title=title,
            description=description,
            keywords=keywords,
            body=post.content,
            paragraphs=paragraphs,
            existing_targets=_existing_link_targets(post.content),
        ))
    return pages


def _content_files(content_dir: Path, max_pages: int = 0) -> list[Path]:
    files: list[Path] = []
    for file_path in sorted(content_dir.rglob("*.md")):
        if any(part.startswith(".") for part in file_path.parts):
            continue
        files.append(file_path)
        if max_pages and len(files) >= max_pages:
            break
    return files


def _candidate_ngrams(text: str, min_n: int = 2, max_n: int = 5) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for segment in _SENTENCE_SPLIT_RE.split(text or ""):
        tokens = _tokens(segment)
        if len(tokens) < min_n:
            continue
        for n in range(max_n, min_n - 1, -1):
            for i in range(0, len(tokens) - n + 1):
                window = tokens[i:i + n]
                lower_tokens = [t.lower().strip(".'’") for t in window]
                if all(t in _STOPWORDS for t in lower_tokens):
                    continue
                if lower_tokens[0] in _STOPWORDS or lower_tokens[-1] in _STOPWORDS:
                    continue
                if all(t in _BRAND_ONLY for t in lower_tokens):
                    continue
                phrase = _normalize_space(" ".join(window)).strip(".,;:!?")
                if phrase.lower() in _BAD_ANCHORS:
                    continue
                if re.search(r"\d", phrase):
                    continue
                if phrase.count(".") or phrase.count(":") or phrase.count(";"):
                    continue
                lower = phrase.lower()
                if 5 <= len(phrase) <= 72 and lower not in seen:
                    seen.add(lower)
                    out.append(phrase)
                if len(out) >= 160:
                    return out
    return out


def _target_terms(page: Page) -> set[str]:
    # Anchor eligibility should be based on stable target labels, not every
    # descriptive word. Descriptions contain generic terms like "daily" or
    # "seconds" that make poor exact anchors.
    text = " ".join([page.title, _slug_to_text(page.url), " ".join(page.keywords)]).lower()
    return {t.lower() for t in _tokens(text) if len(t) > 2 and t.lower() not in _STOPWORDS}


def _target_info(page: Page) -> TargetInfo:
    return TargetInfo(
        terms=_target_terms(page),
        keywords=[k.lower().strip() for k in page.keywords if k.lower().strip()],
        title_lower=page.title.lower(),
        slug_lower=_slug_to_text(page.url).lower().strip(),
    )


def _anchor_candidate_infos(candidates: list[str]) -> list[AnchorCandidate]:
    infos: list[AnchorCandidate] = []
    for idx, phrase in enumerate(candidates):
        phrase_tokens = [t.lower().strip(".'’") for t in _tokens(phrase)]
        content_tokens = [t for t in phrase_tokens if t not in _STOPWORDS]
        if not content_tokens:
            continue
        infos.append(AnchorCandidate(
            phrase=phrase,
            lower=phrase.lower(),
            content_tokens=content_tokens,
            index=idx,
            length_bonus=min(0.10, max(0, len(content_tokens) - 2) * 0.025),
        ))
    return infos


def _exact_keyword_bonus(anchor: str, target: Page, target_info: TargetInfo | None = None) -> float:
    a = anchor.lower()
    keywords = target_info.keywords if target_info is not None else [k.lower().strip() for k in target.keywords]
    for k in keywords:
        if not k:
            continue
        keyword_tokens = [t.lower() for t in _tokens(k)]
        if not any(t not in _GENERIC_TARGET_TERMS and t not in _STOPWORDS for t in keyword_tokens):
            continue
        if a == k:
            return 0.16
        if a in k or k in a:
            return 0.08
    return 0.0


def _exact_keyword_bonus_for_candidate(candidate: AnchorCandidate, target_info: TargetInfo) -> float:
    for keyword in target_info.keywords:
        if not keyword:
            continue
        keyword_tokens = [t.lower() for t in _tokens(keyword)]
        if not any(t not in _GENERIC_TARGET_TERMS and t not in _STOPWORDS for t in keyword_tokens):
            continue
        if candidate.lower == keyword:
            return 0.16
        if candidate.lower in keyword or keyword in candidate.lower:
            return 0.08
    return 0.0


def _anchor_is_target_specific(anchor: str, target_terms: set[str], overlap: float, exact_bonus: float) -> bool:
    if exact_bonus > 0:
        return True
    tokens = [t.lower().strip(".'’") for t in _tokens(anchor)]
    content_tokens = [t for t in tokens if t not in _STOPWORDS]
    if len(content_tokens) < 2:
        return False
    matched = [t for t in content_tokens if t in target_terms]
    specific_matches = [t for t in matched if t not in _GENERIC_TARGET_TERMS]
    # Require a non-generic target token so "FlowHunt" or "Copilot" alone does
    # not make a vague phrase eligible.
    return bool(specific_matches)


def _anchor_candidate_is_target_specific(candidate: AnchorCandidate, target_terms: set[str], exact_bonus: float) -> bool:
    if exact_bonus > 0:
        return True
    if len(candidate.content_tokens) < 2:
        return False
    matched = [t for t in candidate.content_tokens if t in target_terms]
    specific_matches = [t for t in matched if t not in _GENERIC_TARGET_TERMS]
    return bool(specific_matches)


def _lexical_anchor_score(anchor: str, target: Page, target_info: TargetInfo | None = None) -> tuple[float, float, float]:
    terms = target_info.terms if target_info is not None else _target_terms(target)
    phrase_tokens = [t.lower().strip(".'’") for t in _tokens(anchor)]
    content_tokens = [t for t in phrase_tokens if t not in _STOPWORDS]
    if not content_tokens:
        return 0.0, 0.0, 0.0
    matched = [t for t in content_tokens if t in terms]
    specific_matches = [t for t in matched if t not in _GENERIC_TARGET_TERMS]
    overlap = len(matched) / max(1, len(content_tokens))
    specific_overlap = len(specific_matches) / max(1, len(content_tokens))
    exact_bonus = _exact_keyword_bonus(anchor, target, target_info)
    length_bonus = min(0.10, max(0, len(content_tokens) - 2) * 0.025)
    title = target_info.title_lower if target_info is not None else target.title.lower()
    slug_text = target_info.slug_lower if target_info is not None else _slug_to_text(target.url).lower().strip()
    lower = anchor.lower()
    label_bonus = 0.0
    if lower == title or lower == slug_text.strip():
        label_bonus = 0.25
    elif lower in title or lower in slug_text:
        label_bonus = 0.12
    elif title in lower:
        label_bonus = 0.18
    score = (
        exact_bonus
        + label_bonus
        + overlap * 0.45
        + specific_overlap * 0.35
        + length_bonus
    )
    return score, overlap, exact_bonus


def _lexical_anchor_score_candidate(candidate: AnchorCandidate, target_info: TargetInfo) -> tuple[float, float, float]:
    matched = [t for t in candidate.content_tokens if t in target_info.terms]
    specific_matches = [t for t in matched if t not in _GENERIC_TARGET_TERMS]
    overlap = len(matched) / max(1, len(candidate.content_tokens))
    specific_overlap = len(specific_matches) / max(1, len(candidate.content_tokens))
    exact_bonus = _exact_keyword_bonus_for_candidate(candidate, target_info)
    label_bonus = 0.0
    if candidate.lower == target_info.title_lower or candidate.lower == target_info.slug_lower:
        label_bonus = 0.25
    elif candidate.lower in target_info.title_lower or candidate.lower in target_info.slug_lower:
        label_bonus = 0.12
    elif target_info.title_lower in candidate.lower:
        label_bonus = 0.18
    score = (
        exact_bonus
        + label_bonus
        + overlap * 0.45
        + specific_overlap * 0.35
        + candidate.length_bonus
    )
    return score, overlap, exact_bonus


def _anchor_finalists(candidates: list[str], target: Page, target_info: TargetInfo, max_candidates: int) -> list[str]:
    if not candidates:
        return []
    terms = target_info.terms
    scored: list[tuple[float, int, str]] = []
    for idx, phrase in enumerate(candidates):
        lexical_score, overlap, exact_bonus = _lexical_anchor_score(phrase, target, target_info)
        if not _anchor_is_target_specific(phrase, terms, overlap, exact_bonus):
            continue
        if lexical_score <= 0:
            continue
        scored.append((lexical_score, -idx, phrase))
    scored.sort(reverse=True)
    if max_candidates <= 0:
        return [phrase for _, _, phrase in scored]
    return [phrase for _, _, phrase in scored[:max_candidates]]


def _anchor_finalist_infos(candidates: list[AnchorCandidate], target_info: TargetInfo, max_candidates: int) -> list[AnchorCandidate]:
    if not candidates:
        return []
    scored: list[tuple[float, int, AnchorCandidate]] = []
    for candidate in candidates:
        lexical_score, _, exact_bonus = _lexical_anchor_score_candidate(candidate, target_info)
        if not _anchor_candidate_is_target_specific(candidate, target_info.terms, exact_bonus):
            continue
        if lexical_score <= 0:
            continue
        scored.append((lexical_score, -candidate.index, candidate))
    scored.sort(reverse=True, key=lambda item: (item[0], item[1]))
    if max_candidates <= 0:
        return [candidate for _, _, candidate in scored]
    return [candidate for _, _, candidate in scored[:max_candidates]]


def _best_lexical_anchor(
    candidates: list[str],
    target: Page,
    target_info: TargetInfo,
    *,
    fit: float,
    lift: float,
) -> tuple[str, float, float]:
    best = ("", -1.0, 0.0)
    for phrase in candidates:
        lexical_score, overlap, exact_bonus = _lexical_anchor_score(phrase, target, target_info)
        if not _anchor_is_target_specific(phrase, target_info.terms, overlap, exact_bonus):
            continue
        confidence = min(1.0, 0.62 + lexical_score * 0.35)
        score = (
            lexical_score * 0.72
            + min(0.14, max(0.0, fit - 0.55) * 0.45)
            + min(0.10, max(0.0, lift) * 1.8)
        )
        if score > best[1]:
            best = (phrase, score, confidence)
    return best


def _best_lexical_anchor_from_infos(
    candidates: list[AnchorCandidate],
    target_info: TargetInfo,
    *,
    fit: float,
    lift: float,
) -> tuple[str, float, float]:
    best = ("", -1.0, 0.0)
    for candidate in candidates:
        lexical_score, overlap, exact_bonus = _lexical_anchor_score_candidate(candidate, target_info)
        if not _anchor_candidate_is_target_specific(candidate, target_info.terms, exact_bonus):
            continue
        confidence = min(1.0, 0.62 + lexical_score * 0.35)
        score = (
            lexical_score * 0.72
            + min(0.14, max(0.0, fit - 0.55) * 0.45)
            + min(0.10, max(0.0, lift) * 1.8)
        )
        if score > best[1]:
            best = (candidate.phrase, score, confidence)
    return best


def _paragraph_eligible(paragraph: str) -> bool:
    tokens = _tokens(paragraph)
    if len(tokens) < 18:
        return False
    lower = paragraph.lower()
    cta_terms = ("try it free", "get started", "book a demo", "schedule a demo", "sign up")
    if any(term in lower for term in cta_terms) and len(tokens) < 35:
        return False
    return True


def _candidate_embeddings_for_paragraph(
    candidates: list[str],
    vector_cache: dict[str, np.ndarray],
    embedder,
    batch_size: int,
) -> np.ndarray:
    missing = [phrase for phrase in candidates if phrase not in vector_cache]
    if missing:
        vectors = embedder.encode(
            missing,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        vectors = np.asarray(vectors, dtype=np.float32)
        for phrase, vec in zip(missing, vectors):
            vector_cache[phrase] = vec
    return np.asarray([vector_cache[phrase] for phrase in candidates], dtype=np.float32)


def _best_anchor_for_target(
    candidates: list[str],
    candidate_embs: np.ndarray,
    target: Page,
    target_info: TargetInfo,
    target_vec: np.ndarray,
    *,
    fit: float,
    lift: float,
) -> tuple[str, float, float]:
    if not candidates or candidate_embs.size == 0:
        return "", 0.0, 0.0

    semantic_scores = np.clip(candidate_embs @ target_vec, -1.0, 1.0)
    terms = target_info.terms

    best = ("", -1.0, 0.0)
    for idx, phrase in enumerate(candidates):
        phrase_tokens = [t.lower().strip(".'’") for t in _tokens(phrase)]
        content_tokens = [t for t in phrase_tokens if t not in _STOPWORDS]
        if not content_tokens:
            continue
        overlap = sum(1 for t in content_tokens if t in terms) / max(1, len(content_tokens))
        length_bonus = min(0.10, max(0, len(content_tokens) - 2) * 0.025)
        exact_bonus = _exact_keyword_bonus(phrase, target, target_info)
        semantic = float(semantic_scores[idx])
        confidence = max(0.0, min(1.0, (semantic + 1.0) / 2.0))
        if not _anchor_is_target_specific(phrase, terms, overlap, exact_bonus):
            continue
        score = (
            confidence * 0.48
            + overlap * 0.26
            + min(0.14, max(0.0, fit - 0.55) * 0.45)
            + min(0.10, max(0.0, lift) * 1.8)
            + length_bonus
            + exact_bonus
        )
        if score > best[1]:
            best = (phrase, score, confidence)
    return best


def _best_anchor_from_vector_map(
    candidates: list[AnchorCandidate],
    vector_map: dict[str, np.ndarray],
    target: Page,
    target_info: TargetInfo,
    target_vec: np.ndarray,
    *,
    fit: float,
    lift: float,
) -> tuple[str, float, float]:
    if not candidates:
        return "", 0.0, 0.0
    phrases = [candidate.phrase for candidate in candidates if candidate.phrase in vector_map]
    candidate_embs = np.asarray([vector_map[p] for p in phrases], dtype=np.float32)
    if not phrases or candidate_embs.size == 0:
        return "", 0.0, 0.0
    return _best_anchor_for_target(phrases, candidate_embs, target, target_info, target_vec, fit=fit, lift=lift)


def _page_text(page: Page) -> str:
    return _normalize_space(" ".join([
        page.url,
        _slug_to_text(page.url),
        page.title,
        page.description,
        " ".join(page.keywords[:10]),
    ]))


def _load_preferred_target_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except Exception as exc:
        print(f"Warning: failed to load preferred target config {config_path}: {exc}", file=sys.stderr)
        return {}
    return data if isinstance(data, dict) else {}


def _preferred_settings(config: dict[str, Any]) -> PreferredTargetSettings:
    raw = config.get("settings") or {}
    if not isinstance(raw, dict):
        raw = {}

    def _int(name: str, default: int) -> int:
        try:
            return int(raw.get(name, default))
        except (TypeError, ValueError):
            return default

    def _float(name: str, default: float) -> float:
        try:
            return float(raw.get(name, default))
        except (TypeError, ValueError):
            return default

    return PreferredTargetSettings(
        min_links_per_page=max(0, _int("min_links_per_page", 1)),
        max_targets_per_language=max(0, _int("max_targets_per_language", 20)),
        score_boost=_float("score_boost", 0.08),
        similarity_floor=_float("similarity_floor", 0.46),
        lift_floor=_float("lift_floor", -0.05),
        anchor_floor=_float("anchor_floor", 0.38),
    )


def _iter_preferred_items(config: dict[str, Any], lang: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for item in config.get("targets") or []:
        if isinstance(item, dict):
            items.append(item)
    language_items = (config.get("languages") or {}).get(lang) or []
    for item in language_items:
        if isinstance(item, dict):
            items.append(item)
    return items


def _preferred_targets_for_language(
    pages: list[Page],
    config: dict[str, Any],
    lang: str,
    settings: PreferredTargetSettings,
) -> dict[str, float]:
    if not config or settings.max_targets_per_language <= 0:
        return {}
    by_rel = {page.rel_path: page for page in pages}
    by_url = {_canonical_path(page.url): page for page in pages}
    preferred: dict[str, float] = {}
    for item in _iter_preferred_items(config, lang):
        weight = item.get("weight", 1.0)
        try:
            weight = float(weight)
        except (TypeError, ValueError):
            weight = 1.0
        url = str(item.get("url") or item.get("path") or "").strip()
        file_path = str(item.get("file") or item.get("source") or "").strip()
        page: Page | None = None
        if file_path:
            page = by_rel.get(file_path)
        if not page and url:
            page = by_url.get(_canonical_path(url))
        if not page:
            continue
        preferred[_canonical_path(page.url)] = max(weight, preferred.get(_canonical_path(page.url), 0.0))
        if len(preferred) >= settings.max_targets_per_language:
            break
    return preferred


def _candidate_target_order(
    scores: dict[int, float],
    preferred_indices: dict[int, float],
    para_vec: np.ndarray,
    page_embs: np.ndarray,
    *,
    source_i: int,
    top_targets: int,
    initial_targets: int,
    preferred_needed: bool,
    score_boost: float,
) -> tuple[list[int], dict[int, float]]:
    scored: dict[int, float] = dict(scores)
    if preferred_needed:
        for idx, weight in preferred_indices.items():
            fit = float(para_vec @ page_embs[idx])
            scored[idx] = max(scored.get(idx, -1.0), fit + weight * score_boost)
    scored.pop(source_i, None)
    ordered = [idx for idx, _ in sorted(scored.items(), key=lambda item: item[1], reverse=True)[:top_targets]]
    initial = max(1, min(initial_targets, len(ordered)))
    staged = ordered[:initial] + ordered[initial:]
    return staged, scored


def _page_index(page_embs: np.ndarray):
    index = faiss.IndexFlatIP(page_embs.shape[1])
    index.add(page_embs.astype(np.float32))
    return index


def _recommend_links(
    pages: list[Page],
    embedder,
    args: argparse.Namespace,
    page_cache: EmbeddingCache,
    paragraph_cache: EmbeddingCache,
    preferred_urls: dict[str, float] | None = None,
    preferred_settings: PreferredTargetSettings | None = None,
) -> list[LinkRec]:
    preferred_urls = preferred_urls or {}
    preferred_settings = preferred_settings or PreferredTargetSettings(min_links_per_page=0)
    target_infos = [_target_info(page) for page in pages]
    page_texts = [_clip_text(_page_text(page), args.max_page_chars) for page in pages]
    page_embs = page_cache.encode(
        embedder,
        page_texts,
        batch_size=args.page_batch_size,
        show_progress_bar=True,
        desc="Page embeddings",
    )
    page_embs = np.asarray(page_embs, dtype=np.float32)

    paragraph_rows: list[tuple[int, int, str]] = []
    for page_i, page in enumerate(pages):
        for para_i, paragraph in enumerate(page.paragraphs):
            if not _paragraph_eligible(paragraph):
                continue
            paragraph_rows.append((page_i, para_i, paragraph))

    if not paragraph_rows:
        return []

    para_texts = [_clip_text(row[2], args.max_paragraph_chars) for row in paragraph_rows]
    para_embs = paragraph_cache.encode(
        embedder,
        para_texts,
        batch_size=args.paragraph_batch_size,
        show_progress_bar=True,
        desc="Paragraph embeddings",
    )
    para_embs = np.asarray(para_embs, dtype=np.float32)

    by_page: dict[int, list[int]] = {}
    for row_i, (page_i, _, _) in enumerate(paragraph_rows):
        by_page.setdefault(page_i, []).append(row_i)

    page_para_centroids = np.zeros_like(page_embs, dtype=np.float32)
    for page_i, row_idxs in by_page.items():
        page_para_centroids[page_i] = para_embs[row_idxs].mean(axis=0)

    page_index = _page_index(page_embs)

    recs: list[LinkRec] = []
    per_source_count: dict[int, int] = {}
    per_source_preferred_count: dict[int, int] = {}
    per_paragraph_count: dict[tuple[int, int], int] = {}
    per_source_target_seen: set[tuple[int, int]] = set()
    preferred_indices = {
        idx: weight for idx, page in enumerate(pages)
        if (weight := preferred_urls.get(_canonical_path(page.url))) is not None
    }

    score_batch_size = max(1, int(args.score_batch_size))
    top_targets = max(1, min(int(args.top_targets_per_paragraph), len(pages)))
    initial_targets = max(1, min(int(args.initial_targets_per_paragraph), top_targets))
    total_rows = len(paragraph_rows)
    progress = tqdm(total=total_rows, desc="Scoring paragraph targets")
    for start in range(0, total_rows, score_batch_size):
        end = min(start + score_batch_size, total_rows)
        sims_chunk, idxs_chunk = page_index.search(para_embs[start:end].astype(np.float32), top_targets)
        for local_i, (source_i, para_i, paragraph) in enumerate(paragraph_rows[start:end]):
            progress.update(1)
            if per_source_count.get(source_i, 0) >= args.top_k_per_page:
                if per_source_preferred_count.get(source_i, 0) >= preferred_settings.min_links_per_page:
                    continue
            source = pages[source_i]
            para_vec = para_embs[start + local_i]
            search_scores = {
                int(idx): float(score)
                for idx, score in zip(idxs_chunk[local_i], sims_chunk[local_i])
                if int(idx) >= 0
            }
            preferred_needed = per_source_preferred_count.get(source_i, 0) < preferred_settings.min_links_per_page
            order, target_scores = _candidate_target_order(
                search_scores,
                preferred_indices,
                para_vec,
                page_embs,
                source_i=source_i,
                top_targets=args.top_targets_per_paragraph,
                initial_targets=initial_targets,
                preferred_needed=preferred_needed,
                score_boost=preferred_settings.score_boost,
            )
            paragraph_candidates: list[str] | None = None
            paragraph_candidate_infos: list[AnchorCandidate] | None = None
            paragraph_vector_cache: dict[str, np.ndarray] = {}
            accepted_for_paragraph = False
            for stage_order in (order[:initial_targets], order[initial_targets:]):
                if accepted_for_paragraph:
                    break
                if not stage_order:
                    continue
                stage_candidates: list[tuple[int, Page, TargetInfo, float, float, bool, list[AnchorCandidate]]] = []
                for target_i_raw in stage_order:
                    target_i = int(target_i_raw)
                    is_preferred = target_i in preferred_indices
                    if per_source_count.get(source_i, 0) >= args.top_k_per_page and not (is_preferred and preferred_needed):
                        continue
                    if target_i == source_i:
                        continue
                    target = pages[target_i]
                    if target.url == "/" and not args.allow_root_target:
                        continue
                    if target.url in source.existing_targets or _canonical_path(target.url) == _canonical_path(source.url):
                        continue
                    if (source_i, target_i) in per_source_target_seen:
                        continue
                    if per_paragraph_count.get((source_i, para_i), 0) >= args.top_k_per_paragraph:
                        break
                    fit = float(target_scores.get(target_i, para_vec @ page_embs[target_i]))
                    similarity_floor = preferred_settings.similarity_floor if is_preferred and preferred_needed else args.similarity_floor
                    if fit < similarity_floor:
                        continue
                    lift = fit - float(page_para_centroids[source_i] @ page_embs[target_i])
                    lift_floor = preferred_settings.lift_floor if is_preferred and preferred_needed else args.lift_floor
                    if lift < lift_floor:
                        continue

                    if paragraph_candidates is None:
                        paragraph_candidates = _candidate_ngrams(paragraph)
                        if not paragraph_candidates:
                            break
                        paragraph_candidate_infos = _anchor_candidate_infos(paragraph_candidates)
                        if not paragraph_candidate_infos:
                            break

                    target_info = target_infos[target_i]
                    candidates = _anchor_finalist_infos(paragraph_candidate_infos or [], target_info, args.max_anchor_candidates)
                    if not candidates:
                        continue
                    stage_candidates.append((target_i, target, target_info, fit, lift, is_preferred, candidates))

                if not stage_candidates:
                    continue

                for target_i, target, target_info, fit, lift, is_preferred, candidates in stage_candidates:
                    lexical_anchor, lexical_score, lexical_conf = _best_lexical_anchor_from_infos(
                        candidates, target_info, fit=fit, lift=lift
                    )
                    anchor_floor = preferred_settings.anchor_floor if is_preferred and preferred_needed else args.anchor_floor
                    if not lexical_anchor or lexical_score < anchor_floor:
                        continue

                    recs.append(LinkRec(
                        source_path=source.path,
                        source_rel_path=source.rel_path,
                        source_url=source.url,
                        paragraph_index=para_i,
                        paragraph=paragraph[:360],
                        target_url=target.url,
                        target_title=target.title,
                        text=lexical_anchor,
                        title=target.description or target.title,
                        fit=round(fit, 4),
                        lift=round(lift, 4),
                        anchor_score=round(lexical_score, 4),
                        anchor_confidence=round(lexical_conf, 4),
                    ))
                    per_source_count[source_i] = per_source_count.get(source_i, 0) + 1
                    if is_preferred:
                        per_source_preferred_count[source_i] = per_source_preferred_count.get(source_i, 0) + 1
                    per_paragraph_count[(source_i, para_i)] = per_paragraph_count.get((source_i, para_i), 0) + 1
                    per_source_target_seen.add((source_i, target_i))
                    accepted_for_paragraph = True
                    if per_source_count[source_i] >= args.top_k_per_page or accepted_for_paragraph:
                        break

                if accepted_for_paragraph:
                    break
                if not args.semantic_anchor_fallback:
                    continue

                union_candidates = sorted({candidate.phrase for *_, phrases in stage_candidates for candidate in phrases})
                if union_candidates:
                    _candidate_embeddings_for_paragraph(
                        union_candidates,
                        paragraph_vector_cache,
                        embedder,
                        args.anchor_batch_size,
                    )

                for target_i, target, target_info, fit, lift, is_preferred, candidates in stage_candidates:
                    anchor, anchor_score, anchor_conf = _best_anchor_from_vector_map(
                        candidates,
                        paragraph_vector_cache,
                        target,
                        target_info,
                        page_embs[target_i],
                        fit=fit,
                        lift=lift,
                    )
                    anchor_floor = preferred_settings.anchor_floor if is_preferred and preferred_needed else args.anchor_floor
                    if not anchor or anchor_score < anchor_floor:
                        continue

                    recs.append(LinkRec(
                        source_path=source.path,
                        source_rel_path=source.rel_path,
                        source_url=source.url,
                        paragraph_index=para_i,
                        paragraph=paragraph[:360],
                        target_url=target.url,
                        target_title=target.title,
                        text=anchor,
                        title=target.description or target.title,
                        fit=round(fit, 4),
                        lift=round(lift, 4),
                        anchor_score=round(anchor_score, 4),
                        anchor_confidence=round(anchor_conf, 4),
                    ))
                    per_source_count[source_i] = per_source_count.get(source_i, 0) + 1
                    if is_preferred:
                        per_source_preferred_count[source_i] = per_source_preferred_count.get(source_i, 0) + 1
                    per_paragraph_count[(source_i, para_i)] = per_paragraph_count.get((source_i, para_i), 0) + 1
                    per_source_target_seen.add((source_i, target_i))
                    accepted_for_paragraph = True
                    if per_source_count[source_i] >= args.top_k_per_page or accepted_for_paragraph:
                        break
            if paragraph_candidates is not None:
                del paragraph_candidates
            if paragraph_vector_cache:
                del paragraph_vector_cache
        del sims_chunk, idxs_chunk
    progress.close()

    recs.sort(key=lambda r: (r.source_rel_path, -r.lift, -r.anchor_score))
    return recs


def _format_toml_string(value: str) -> str:
    return json.dumps(str(value or ""), ensure_ascii=False)


def _lnks_toml(recs: list[LinkRec]) -> str:
    blocks: list[str] = []
    seen: set[tuple[str, str]] = set()
    for rec in sorted(recs, key=lambda r: (-r.lift, -r.anchor_score, r.text.lower())):
        key = (rec.text.lower(), rec.target_url)
        if key in seen:
            continue
        seen.add(key)
        blocks.append("\n".join([
            "[[lnks]]",
            f"text = {_format_toml_string(rec.text)}",
            f"path = {_format_toml_string(rec.target_url)}",
            f"title = {_format_toml_string(rec.title or rec.target_title)}",
        ]))
    return "\n\n".join(blocks)


def _remove_key_line(lines: list[str], key: str) -> list[str]:
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(f"{key} ") and "=" in stripped and stripped.split("=", 1)[0].strip() == key:
            continue
        out.append(line)
    return out


def _remove_array_table(lines: list[str], table: str) -> list[str]:
    out: list[str] = []
    skipping = False
    header = f"[[{table}]]"
    for line in lines:
        stripped = line.strip()
        if stripped == header:
            skipping = True
            continue
        if skipping and stripped.startswith("[") and stripped.endswith("]"):
            skipping = False
        if not skipping:
            out.append(line)
    return out


def _update_frontmatter(path: Path, recs: list[LinkRec], *, remove_old_linkbuilding: bool) -> bool:
    raw = path.read_text(encoding="utf-8")
    match = re.match(r"^(\+\+\+\s*\n)(.*?)(\n\+\+\+\s*\n?)(.*)", raw, re.DOTALL)
    if not match:
        return False
    opening, fm, closing, body = match.groups()
    lines = fm.splitlines()
    if remove_old_linkbuilding:
        lines = _remove_key_line(lines, "linkbuilding")
    lines = _remove_array_table(lines, "lnks")
    while lines and not lines[-1].strip():
        lines.pop()
    if recs:
        if lines:
            lines.append("")
        lines.append(_lnks_toml(recs))
    updated = f"{opening}{chr(10).join(lines)}{closing}{body}"
    if updated != raw:
        path.write_text(updated, encoding="utf-8")
        return True
    return False


def _to_json_rows(recs: list[LinkRec]) -> list[dict[str, Any]]:
    return [
        {
            "source_file": rec.source_rel_path,
            "source_url": rec.source_url,
            "paragraph_index": rec.paragraph_index,
            "paragraph_excerpt": rec.paragraph,
            "text": rec.text,
            "path": rec.target_url,
            "title": rec.title,
            "target_title": rec.target_title,
            "fit": rec.fit,
            "lift": rec.lift,
            "anchor_score": rec.anchor_score,
            "anchor_confidence": rec.anchor_confidence,
        }
        for rec in recs
    ]


def _language_dirs(content_root: Path, args: argparse.Namespace) -> list[Path]:
    if args.all_languages or str(args.lang).strip().lower() == "all":
        return [
            path for path in sorted(content_root.iterdir())
            if path.is_dir() and not path.name.startswith(".")
        ]
    langs = [part.strip() for part in str(args.lang).split(",") if part.strip()]
    return [content_root / lang for lang in langs]


def _process_language(
    content_dir: Path,
    embedder,
    page_cache: EmbeddingCache,
    paragraph_cache: EmbeddingCache,
    args: argparse.Namespace,
    preferred_config: dict[str, Any],
    preferred_settings: PreferredTargetSettings,
    *,
    multi_language: bool,
) -> tuple[bool, int]:
    lang = content_dir.name
    if not content_dir.exists():
        print(f"Content directory not found: {content_dir}", file=sys.stderr)
        return False, 0

    pages = _load_pages(content_dir, max_pages=args.max_pages)
    if not pages:
        print("No pages found with title/description and paragraphs.", file=sys.stderr)
        return False, 0

    print(f"[{lang}] Loaded {len(pages)} pages from {content_dir}")
    preferred_urls = _preferred_targets_for_language(pages, preferred_config, lang, preferred_settings)
    if preferred_urls:
        print(f"[{lang}] Preferred targets: {len(preferred_urls)} URLs, min {preferred_settings.min_links_per_page} per page")
    recs = _recommend_links(pages, embedder, args, page_cache, paragraph_cache, preferred_urls, preferred_settings)
    rows = _to_json_rows(recs)
    print(f"[{lang}] Generated {len(recs)} paragraph link recommendations")

    if args.output:
        out_path = Path(args.output)
        if multi_language:
            out_path = out_path.with_name(f"{out_path.stem}-{lang}{out_path.suffix}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"recommendations": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[{lang}] Wrote {out_path}")

    changed = 0
    if args.write:
        grouped: dict[Path, list[LinkRec]] = {}
        for rec in recs:
            grouped.setdefault(rec.source_path, []).append(rec)
        target_paths = set(_content_files(content_dir, max_pages=args.max_pages)) if args.remove_old_linkbuilding else set(grouped)
        for path in sorted(target_paths):
            if _update_frontmatter(path, grouped.get(path, []), remove_old_linkbuilding=True):
                changed += 1
        print(f"[{lang}] Updated {changed} content files")
    else:
        print(f"[{lang}] Dry run only. Use --write to update frontmatter.")

    if rows:
        print(f"\n[{lang}] Top recommendations:")
        for row in sorted(rows, key=lambda r: (r["lift"], r["anchor_score"]), reverse=True)[:10]:
            print(f"[{lang}] - {row['source_file']} p{row['paragraph_index']}: \"{row['text']}\" -> {row['path']} "
                  f"(fit={row['fit']}, lift={row['lift']}, anchor={row['anchor_score']})")
    del pages, recs, rows
    gc.collect()
    return True, changed


def main() -> int:
    args = _parse_args()
    content_root = Path(args.content_root)
    content_dirs = _language_dirs(content_root, args)
    if not content_dirs:
        print(f"No language directories found under {content_root}", file=sys.stderr)
        return 2

    embedder = LazySentenceTransformer(args.model, args.device)
    cache_path = args.cache_path if args.cache_path else str(shared_sqlite_cache_path(Path(".")))
    page_cache = EmbeddingCache(cache_path, args.model, enabled=not args.no_cache, device=args.device, cache_type="page")
    paragraph_cache = EmbeddingCache(cache_path, args.model, enabled=not args.no_cache, device=args.device, cache_type="paragraph")
    preferred_config = _load_preferred_target_config(args.preferred_targets)
    preferred_settings = _preferred_settings(preferred_config)
    failed_langs: list[str] = []
    processed = 0
    changed_total = 0
    try:
        for content_dir in content_dirs:
            ok, changed = _process_language(
                content_dir,
                embedder,
                page_cache,
                paragraph_cache,
                args,
                preferred_config,
                preferred_settings,
                multi_language=len(content_dirs) > 1,
            )
            if ok:
                processed += 1
                changed_total += changed
            else:
                failed_langs.append(content_dir.name)
    finally:
        page_cache.close()
        paragraph_cache.close()

    if failed_langs:
        print(f"Failed languages: {', '.join(failed_langs)}", file=sys.stderr)
        return 1

    print(f"Processed {processed} languages; updated {changed_total} content files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
