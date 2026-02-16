#!/usr/bin/env python3
"""
===============================================================================
Universal Broken Link Checker
===============================================================================

A reusable tool that checks any website for broken links by crawling its
sitemap, extracting all links from every page, and verifying each one.

Works with any domain — no hardcoded values. All configuration is done via
command-line arguments.

-------------------------------------------------------------------------------
REQUIREMENTS
-------------------------------------------------------------------------------

    pip install requests beautifulsoup4 lxml

-------------------------------------------------------------------------------
QUICK START
-------------------------------------------------------------------------------

    # Check a single domain (auto-discovers sitemap from robots.txt):
    python3 check-broken-links.py --domain www.liveagent.com

    # Check a domain with a custom sitemap URL (skips robots.txt discovery):
    python3 check-broken-links.py --domain www.example.com \
        --sitemap https://www.example.com/custom-sitemap.xml

-------------------------------------------------------------------------------
ALL OPTIONS
-------------------------------------------------------------------------------

    --domain DOMAIN       (required) The domain to check, e.g. www.liveagent.com
    --sitemap URL         Custom sitemap URL. Default: auto-discovered from robots.txt
    --concurrency N       Number of parallel requests (default: 20)
    --timeout SECONDS     Per-request timeout in seconds (default: 15)
    --output FILE         Output report path (default: broken-links-<domain>.txt)
    --cache-dir DIR       Directory to store the URL cache (default: script dir)
    --no-cache            Ignore existing cache and re-check all URLs
    --user-agent STRING   Custom User-Agent header for requests
    --internal-only       Only check links pointing to the same domain
    --external-only       Only check links pointing to external domains
    --format FORMAT       Report format: "text" (default) or "json"
    --skip-patterns PAT   Additional URL prefixes to skip (can be repeated)

-------------------------------------------------------------------------------
EXAMPLES
-------------------------------------------------------------------------------

    # 1. Basic check for liveagent.com
    python3 check-broken-links.py --domain www.liveagent.com

    # 2. Check postaffiliatepro.com with higher concurrency
    python3 check-broken-links.py --domain www.postaffiliatepro.com --concurrency 40

    # 3. Only check internal links on your staging site
    python3 check-broken-links.py --domain staging.example.com --internal-only

    # 4. Only check external/outbound links
    python3 check-broken-links.py --domain www.liveagent.com --external-only

    # 5. Custom sitemap, custom output, JSON format
    python3 check-broken-links.py \
        --domain www.example.com \
        --sitemap https://www.example.com/post-sitemap.xml \
        --output /tmp/report.json \
        --format json

    # 6. Skip additional URL patterns (e.g. social media, CDN)
    python3 check-broken-links.py --domain www.example.com \
        --skip-patterns "https://facebook.com" \
        --skip-patterns "https://cdn.example.com"

    # 7. Fresh scan ignoring the cache
    python3 check-broken-links.py --domain www.liveagent.com --no-cache

-------------------------------------------------------------------------------
HOW IT WORKS
-------------------------------------------------------------------------------

    Phase 0 — Sitemap Discovery (automatic)
        Fetches robots.txt from the domain to find Sitemap: directives.
        If no --sitemap is given and robots.txt has no sitemaps, falls back
        to https://<domain>/sitemap.xml.

    Phase 1 — Sitemap Parsing
        Fetches each discovered sitemap (supports sitemap index files with
        child sitemaps). Collects all page URLs listed in the sitemaps.

    Phase 2 — Page Crawling & Link Extraction
        Downloads each page from the sitemap concurrently.
        Extracts every URL from HTML: <a href>, <img src>, <script src>,
        <link href>, srcset attributes, inline CSS url(), meta og:image, etc.

    Phase 3 — Link Verification
        Checks each unique URL with HTTP HEAD (falls back to GET on 405/403).
        Uses a persistent per-domain JSON cache so subsequent runs are fast.
        Only URLs not already in the cache are re-checked.

    Phase 4 — Report Generation
        Produces a report (text or JSON) listing every broken URL, its HTTP
        status, and all pages where it was found.

-------------------------------------------------------------------------------
CACHE
-------------------------------------------------------------------------------

    The cache file is stored as `.link-check-cache-<domain>.json` in the
    --cache-dir directory (defaults to the same directory as this script).
    Each URL maps to its HTTP status code (or a negative code for errors).

    Cache entries persist across runs so you don't re-check thousands of
    URLs every time. Use --no-cache to force a full re-check.

===============================================================================
"""

import argparse
import json
import os
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Skip patterns for non-HTTP URLs (always skipped regardless of config)
# ---------------------------------------------------------------------------
_BUILTIN_SKIP_PATTERNS = [
    re.compile(r"^mailto:"),
    re.compile(r"^tel:"),
    re.compile(r"^javascript:"),
    re.compile(r"^data:"),
    re.compile(r"^#"),
    re.compile(r"^ftp:"),
]


def _build_skip_checker(extra_prefixes: list[str] | None = None):
    """Return a function that checks whether a URL should be skipped."""
    extra = [p.rstrip("/") for p in (extra_prefixes or [])]

    def should_skip(url: str) -> bool:
        for pat in _BUILTIN_SKIP_PATTERNS:
            if pat.match(url):
                return True
        for prefix in extra:
            if url.startswith(prefix):
                return True
        return False

    return should_skip


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
class URLCache:
    """Thread-safe persistent JSON cache mapping URL -> HTTP status code."""

    def __init__(self, path: str):
        self.path = path
        self.data: dict[str, int] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r") as f:
                    self.data = json.load(f)
                print(f"  Loaded cache with {len(self.data)} entries from {self.path}")
            except (json.JSONDecodeError, IOError):
                self.data = {}

    def save(self):
        with self._lock:
            snapshot = dict(self.data)
        with open(self.path, "w") as f:
            json.dump(snapshot, f)

    def get(self, url: str):
        return self.data.get(url)

    def set(self, url: str, status: int):
        with self._lock:
            self.data[url] = status

    def __contains__(self, url: str):
        return url in self.data

    def __len__(self):
        return len(self.data)


# ---------------------------------------------------------------------------
# Sitemap discovery from robots.txt
# ---------------------------------------------------------------------------
def discover_sitemaps_from_robots_txt(
    session: requests.Session, domain: str, headers: dict, timeout: int
) -> list[str]:
    """Parse robots.txt and return all Sitemap: URLs found in it.

    Returns an empty list if robots.txt is missing or contains no Sitemap
    directives. Never raises — errors are printed and silently handled.
    """
    robots_url = f"https://{domain}/robots.txt"
    print(f"Discovering sitemaps from: {robots_url}")
    try:
        resp = session.get(robots_url, headers=headers, timeout=timeout)
        if resp.status_code != 200:
            print(f"  robots.txt returned HTTP {resp.status_code}, skipping discovery")
            return []
    except Exception as e:
        print(f"  Could not fetch robots.txt: {e}")
        return []

    sitemaps: list[str] = []
    for line in resp.text.splitlines():
        line = line.strip()
        # The directive is case-insensitive per the spec
        if line.lower().startswith("sitemap:"):
            url = line.split(":", 1)[1].strip()
            if url:
                sitemaps.append(url)

    if sitemaps:
        print(f"  Found {len(sitemaps)} sitemap(s) in robots.txt:")
        for s in sitemaps:
            print(f"    - {s}")
    else:
        print("  No Sitemap directives found in robots.txt")

    return sitemaps


# ---------------------------------------------------------------------------
# Sitemap parsing
# ---------------------------------------------------------------------------
def fetch_sitemap_urls(
    session: requests.Session, sitemap_url: str, headers: dict, timeout: int
) -> list[str]:
    """Fetch all page URLs from a sitemap (supports sitemap index files).

    Returns an empty list (instead of raising) if the sitemap cannot be fetched.
    """
    print(f"Fetching sitemap: {sitemap_url}")
    try:
        resp = session.get(sitemap_url, headers=headers, timeout=timeout)
        if resp.status_code != 200:
            print(f"  WARNING: Sitemap returned HTTP {resp.status_code} — skipping")
            return []
    except Exception as e:
        print(f"  WARNING: Could not fetch sitemap: {e} — skipping")
        return []

    soup = BeautifulSoup(resp.content, "lxml-xml")
    page_urls: list[str] = []

    # Check if this is a sitemap index (contains <sitemap> entries)
    sitemap_entries = soup.find_all("sitemap")
    if sitemap_entries:
        child_locs = [
            s.find("loc").text.strip()
            for s in sitemap_entries
            if s.find("loc")
        ]
        print(f"  Found {len(child_locs)} child sitemaps")
        for child_url in child_locs:
            print(f"  Fetching child sitemap: {child_url}")
            try:
                child_resp = session.get(child_url, headers=headers, timeout=timeout)
                child_resp.raise_for_status()
                child_soup = BeautifulSoup(child_resp.content, "lxml-xml")
                urls_in_child = [
                    u.find("loc").text.strip()
                    for u in child_soup.find_all("url")
                    if u.find("loc")
                ]
                page_urls.extend(urls_in_child)
                print(f"    -> {len(urls_in_child)} URLs")
            except Exception as e:
                print(f"    ERROR fetching child sitemap {child_url}: {e}")
    else:
        # Flat sitemap
        urls = [
            u.find("loc").text.strip()
            for u in soup.find_all("url")
            if u.find("loc")
        ]
        page_urls.extend(urls)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for u in page_urls:
        if u not in seen:
            seen.add(u)
            unique.append(u)

    print(f"Total unique sitemap URLs: {len(unique)}")
    return unique


# ---------------------------------------------------------------------------
# Link extraction from HTML
# ---------------------------------------------------------------------------
URL_ATTRS = {
    "a": ["href"],
    "img": ["src", "srcset"],
    "script": ["src"],
    "link": ["href"],
    "source": ["src", "srcset"],
    "video": ["src", "poster"],
    "audio": ["src"],
    "iframe": ["src"],
    "embed": ["src"],
    "object": ["data"],
    "form": ["action"],
    "area": ["href"],
    "base": ["href"],
    "meta": ["content"],
}


def _parse_srcset(srcset: str) -> list[str]:
    """Extract URLs from an HTML srcset attribute value."""
    urls = []
    for entry in srcset.split(","):
        entry = entry.strip()
        if entry:
            parts = entry.split()
            if parts:
                urls.append(parts[0])
    return urls


def extract_urls_from_html(html: str, base_url: str, should_skip) -> set[str]:
    """Extract every URL referenced in *html*, resolved against *base_url*."""
    soup = BeautifulSoup(html, "lxml")
    found: set[str] = set()

    for tag_name, attrs in URL_ATTRS.items():
        for tag in soup.find_all(tag_name):
            for attr in attrs:
                value = tag.get(attr)
                if not value:
                    continue

                # meta tags: only extract URL-like content values
                if tag_name == "meta" and attr == "content":
                    prop = tag.get("property", "") or tag.get("name", "")
                    if not any(
                        kw in prop.lower()
                        for kw in ["image", "url", "og:", "twitter:"]
                    ):
                        continue
                    if not value.startswith(("http://", "https://", "/")):
                        continue

                if attr == "srcset":
                    for u in _parse_srcset(value):
                        if not should_skip(u):
                            found.add(urljoin(base_url, u))
                else:
                    if should_skip(value):
                        continue
                    found.add(urljoin(base_url, value))

    # Inline CSS url() references
    for tag in soup.find_all(style=True):
        style = tag["style"]
        for match in re.finditer(r'url\(["\']?(.*?)["\']?\)', style):
            u = match.group(1).strip()
            if not should_skip(u):
                found.add(urljoin(base_url, u))

    # Keep only valid http(s) URLs, strip fragments
    filtered: set[str] = set()
    for u in found:
        parsed = urlparse(u)
        if parsed.scheme in ("http", "https") and parsed.netloc:
            clean = u.split("#")[0]
            if clean:
                filtered.add(clean)

    return filtered


# ---------------------------------------------------------------------------
# URL checking
# ---------------------------------------------------------------------------
STATUS_LABELS = {
    -1: "Connection Error",
    -2: "SSL Error",
    -3: "Timeout",
    310: "Too Many Redirects",
}


def check_url(session: requests.Session, url: str, headers: dict, timeout: int) -> int:
    """Return HTTP status code for *url*. Negative codes indicate errors."""
    try:
        resp = session.head(url, headers=headers, timeout=timeout, allow_redirects=True)
        if resp.status_code in (405, 403, 501):
            resp = session.get(
                url, headers=headers, timeout=timeout, allow_redirects=True, stream=True
            )
            resp.close()
        return resp.status_code
    except requests.exceptions.TooManyRedirects:
        return 310
    except requests.exceptions.SSLError:
        return -2
    except requests.exceptions.ConnectionError:
        return -1
    except requests.exceptions.Timeout:
        return -3
    except Exception:
        return -1


# ---------------------------------------------------------------------------
# Main crawl logic
# ---------------------------------------------------------------------------
def crawl_and_check(
    sitemap_urls: list[str],
    domain: str,
    headers: dict,
    should_skip,
    concurrency: int = 20,
    timeout: int = 15,
    cache: URLCache | None = None,
    internal_only: bool = False,
    external_only: bool = False,
):
    """
    1. Fetch each sitemap page and extract all links.
    2. Optionally filter to internal-only or external-only.
    3. Check every extracted link.
    4. Return (broken_links dict, failed_pages list).
    """
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=concurrency, pool_maxsize=concurrency, max_retries=1
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    url_sources: dict[str, set[str]] = defaultdict(set)

    # --- Phase 1: Fetch pages and extract links ---
    print("\n" + "=" * 70)
    print("PHASE 1: Crawling sitemap pages and extracting links")
    print("=" * 70)

    total_pages = len(sitemap_urls)
    fetched = 0
    failed_pages = []

    def fetch_page(page_url: str):
        try:
            resp = session.get(page_url, headers=headers, timeout=timeout)
            if resp.status_code != 200:
                return page_url, None, resp.status_code
            return page_url, resp.text, resp.status_code
        except Exception as e:
            return page_url, None, str(e)

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(fetch_page, url): url for url in sitemap_urls}
        for future in as_completed(futures):
            fetched += 1
            page_url, html, status = future.result()

            if html is None:
                failed_pages.append((page_url, status))
            else:
                links = extract_urls_from_html(html, page_url, should_skip)
                for link in links:
                    # Apply internal/external filter
                    link_domain = urlparse(link).netloc
                    if internal_only and link_domain != domain:
                        continue
                    if external_only and link_domain == domain:
                        continue
                    url_sources[link].add(page_url)

            if fetched % 50 == 0 or fetched == total_pages:
                print(
                    f"  [{fetched}/{total_pages}] pages fetched, "
                    f"{len(url_sources)} unique links found so far..."
                )

    if failed_pages:
        print(f"\n  WARNING: {len(failed_pages)} sitemap pages could not be fetched")

    total_links = len(url_sources)
    print(f"\nTotal unique links to check: {total_links}")

    # --- Phase 2: Check all extracted links ---
    print("\n" + "=" * 70)
    print("PHASE 2: Checking all extracted links")
    print("=" * 70)

    to_check = []
    already_ok = 0
    already_broken = 0

    for url in url_sources:
        if cache is not None and url in cache:
            status = cache.get(url)
            if status and 200 <= status < 400:
                already_ok += 1
            else:
                already_broken += 1
        else:
            to_check.append(url)

    print(f"  Already cached OK: {already_ok}")
    print(f"  Already cached broken: {already_broken}")
    print(f"  Need to check: {len(to_check)}")

    checked = 0
    broken_count = 0

    def check_single(url: str):
        status = check_url(session, url, headers, timeout)
        if cache is not None:
            cache.set(url, status)
        return url, status

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(check_single, url): url for url in to_check}
        for future in as_completed(futures):
            checked += 1
            url, status = future.result()
            if not (200 <= status < 400):
                broken_count += 1
            if checked % 100 == 0 or checked == len(to_check):
                print(
                    f"  [{checked}/{len(to_check)}] checked, "
                    f"{broken_count} broken so far..."
                )
            # Save cache periodically
            if cache is not None and checked % 500 == 0:
                cache.save()

    if cache is not None:
        cache.save()

    # --- Phase 3: Compile broken links ---
    print("\n" + "=" * 70)
    print("PHASE 3: Compiling report")
    print("=" * 70)

    broken_links: dict[str, dict] = {}
    for url, sources in url_sources.items():
        status = cache.get(url) if cache is not None else None
        if status is None:
            continue
        if not (200 <= status < 400):
            broken_links[url] = {
                "status": status,
                "pages": sorted(sources),
            }

    return broken_links, failed_pages


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def generate_report(
    broken_links: dict,
    failed_pages: list,
    output_path: str,
    duration: float,
    domain: str,
    fmt: str = "text",
):
    """Write the broken-link report to *output_path* in the chosen format."""
    if fmt == "json":
        report_data = {
            "domain": domain,
            "generated": datetime.now().isoformat(),
            "scan_duration_seconds": round(duration, 1),
            "total_broken_urls": len(broken_links),
            "failed_sitemap_pages": [
                {"url": url, "status": str(status)} for url, status in failed_pages
            ],
            "broken_links": [
                {
                    "url": url,
                    "status": info["status"],
                    "status_label": STATUS_LABELS.get(info["status"], str(info["status"])),
                    "found_on_pages": info["pages"],
                }
                for url, info in sorted(
                    broken_links.items(), key=lambda x: (x[1]["status"], x[0])
                )
            ],
        }
        with open(output_path, "w") as f:
            json.dump(report_data, f, indent=2)
    else:
        lines = []
        lines.append("=" * 80)
        lines.append(f"BROKEN LINK REPORT — {domain}")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Scan duration: {duration:.0f} seconds")
        lines.append(f"Total broken URLs found: {len(broken_links)}")
        lines.append("=" * 80)

        if failed_pages:
            lines.append("")
            lines.append("-" * 80)
            lines.append(
                f"SITEMAP PAGES THAT COULD NOT BE FETCHED ({len(failed_pages)})"
            )
            lines.append("-" * 80)
            for page_url, status in sorted(failed_pages):
                lines.append(f"  [{status}] {page_url}")

        if not broken_links:
            lines.append("")
            lines.append("No broken links found!")
        else:
            sorted_broken = sorted(
                broken_links.items(), key=lambda x: (x[1]["status"], x[0])
            )
            lines.append("")
            lines.append("-" * 80)
            lines.append("BROKEN LINKS")
            lines.append("-" * 80)
            for url, info in sorted_broken:
                status = info["status"]
                label = STATUS_LABELS.get(status, str(status))
                lines.append("")
                lines.append(f"[{label}] {url}")
                lines.append(f"  Found on {len(info['pages'])} page(s):")
                for page in info["pages"]:
                    lines.append(f"    - {page}")

        with open(output_path, "w") as f:
            f.write("\n".join(lines) + "\n")

    print(f"\nReport written to: {output_path}")
    print(f"Total broken URLs: {len(broken_links)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Universal broken link checker — works with any domain.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  %(prog)s --domain www.liveagent.com
  %(prog)s --domain www.example.com --concurrency 40 --internal-only
  %(prog)s --domain staging.example.com --sitemap https://staging.example.com/sitemap_index.xml
  %(prog)s --domain www.example.com --format json --output report.json
  %(prog)s --domain www.example.com --external-only --no-cache
  %(prog)s --domain www.example.com --skip-patterns "https://facebook.com" --skip-patterns "https://cdn.example.com"
        """,
    )
    parser.add_argument(
        "--domain",
        required=True,
        help="Domain to check, e.g. www.liveagent.com (required)",
    )
    parser.add_argument(
        "--sitemap",
        default=None,
        help="Full sitemap URL (default: auto-discovered from robots.txt)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=20,
        help="Number of concurrent requests (default: 20)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=15,
        help="Per-request timeout in seconds (default: 15)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output report path (default: broken-links-<domain>.txt or .json)",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Directory to store cache file (default: same dir as this script)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore existing cache and re-check all URLs",
    )
    parser.add_argument(
        "--user-agent",
        default=None,
        help="Custom User-Agent string (default: auto-generated with domain name)",
    )
    parser.add_argument(
        "--internal-only",
        action="store_true",
        help="Only check links pointing to the same domain",
    )
    parser.add_argument(
        "--external-only",
        action="store_true",
        help="Only check links pointing to external domains",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help='Report format: "text" (default) or "json"',
    )
    parser.add_argument(
        "--skip-patterns",
        action="append",
        default=[],
        help="Additional URL prefixes to skip (can be repeated)",
    )
    args = parser.parse_args()

    if args.internal_only and args.external_only:
        parser.error("--internal-only and --external-only are mutually exclusive")

    domain = args.domain
    user_agent = args.user_agent or (
        f"Mozilla/5.0 (compatible; BrokenLinkChecker/1.0; +https://{domain})"
    )
    headers = {"User-Agent": user_agent}

    cache_dir = args.cache_dir or os.path.dirname(os.path.abspath(__file__))
    cache_file = os.path.join(cache_dir, f".link-check-cache-{domain}.json")

    ext = "json" if args.format == "json" else "txt"
    output_path = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"broken-links-{domain}.{ext}",
    )

    should_skip = _build_skip_checker(args.skip_patterns)

    # Print configuration summary
    print(f"Broken Link Checker — {domain}")
    print("=" * 40)
    print(f"Sitemap:      {args.sitemap or '(auto-discover from robots.txt)'}")
    print(f"Concurrency:  {args.concurrency}")
    print(f"Timeout:      {args.timeout}s")
    print(f"Output:       {output_path}")
    print(f"Format:       {args.format}")
    print(f"Cache:        {cache_file}")
    if args.internal_only:
        print("Filter:       internal links only")
    elif args.external_only:
        print("Filter:       external links only")
    if args.skip_patterns:
        print(f"Skip:         {', '.join(args.skip_patterns)}")
    print()

    # Load cache
    cache = URLCache(cache_file)
    if args.no_cache:
        cache.data = {}
        print("Cache cleared (--no-cache)")

    session = requests.Session()

    # Step 1: Discover sitemap URL(s)
    if args.sitemap:
        # User provided an explicit sitemap — use only that one
        sitemap_sources = [args.sitemap]
    else:
        # Auto-discover from robots.txt
        sitemap_sources = discover_sitemaps_from_robots_txt(
            session, domain, headers, args.timeout
        )
        if not sitemap_sources:
            # Fallback: try the conventional /sitemap.xml location
            fallback = f"https://{domain}/sitemap.xml"
            print(f"  Falling back to: {fallback}")
            sitemap_sources = [fallback]

    # Step 2: Fetch page URLs from all discovered sitemaps
    sitemap_urls: list[str] = []
    for sitemap_url in sitemap_sources:
        urls = fetch_sitemap_urls(session, sitemap_url, headers, args.timeout)
        sitemap_urls.extend(urls)

    # Deduplicate (multiple sitemaps may overlap)
    seen: set[str] = set()
    unique_urls: list[str] = []
    for u in sitemap_urls:
        if u not in seen:
            seen.add(u)
            unique_urls.append(u)
    sitemap_urls = unique_urls

    if not sitemap_urls:
        print("ERROR: No URLs found in any sitemap!")
        sys.exit(1)

    print(f"\nTotal pages to crawl: {len(sitemap_urls)}")

    start = time.time()

    # Step 3: Crawl and check
    broken_links, failed_pages = crawl_and_check(
        sitemap_urls,
        domain=domain,
        headers=headers,
        should_skip=should_skip,
        concurrency=args.concurrency,
        timeout=args.timeout,
        cache=cache,
        internal_only=args.internal_only,
        external_only=args.external_only,
    )

    duration = time.time() - start

    # Step 4: Generate report
    generate_report(
        broken_links, failed_pages, output_path, duration, domain, fmt=args.format
    )

    # Save final cache
    cache.save()
    print(f"Cache saved with {len(cache)} entries to {cache_file}")


if __name__ == "__main__":
    main()
