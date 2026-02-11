#!/usr/bin/env python3
"""Split a monolithic sitemap.xml into per-sport sitemaps with a sitemapindex.

Handles Hugo multi-site builds where <loc> values are either:
  - Absolute URLs: https://betmana.co.uk/football/...
  - Domain-prefixed paths: /betmana.co.uk/football/...

The script groups URLs by sport section (the path segment after the domain),
chunks large groups into multiple sitemaps (max 10,000 URLs each), and
replaces sitemap.xml with a sitemapindex.

Usage:
    python split_sitemap.py --public-dir public/
"""

import argparse
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

MAX_URLS_PER_SITEMAP = 10_000

SITEMAP_NS = "http://www.sitemaps.org/schemas/sitemap/0.9"
NS = {"sm": SITEMAP_NS}

# Known sport sections on the site
SPORT_SECTIONS = {
    "football", "basketball", "nfl", "hockey", "baseball",
    "handball", "rugby", "volleyball", "mma", "boxing",
    "tennis", "cricket", "motorsport", "afl", "formula-1",
}


def parse_sitemap(sitemap_path: Path) -> list[ET.Element]:
    """Parse sitemap.xml and return all <url> elements."""
    tree = ET.parse(sitemap_path)
    root = tree.getroot()
    return list(root.findall("sm:url", NS))


def detect_base_url(url_elements: list[ET.Element]) -> str | None:
    """Detect the base URL if <loc> values contain scheme+host.

    Returns the base URL string (e.g. "https://betmana.co.uk") or None if
    the locs are relative paths (e.g. "/betmana.co.uk/football/...").
    """
    for url_el in url_elements:
        loc = url_el.find("sm:loc", NS)
        if loc is not None and loc.text:
            text = loc.text.strip()
            parsed = urlparse(text)
            if parsed.scheme and parsed.netloc:
                return f"{parsed.scheme}://{parsed.netloc}"
            return None
    return None


def extract_sport(loc_text: str) -> str:
    """Extract the sport section from a <loc> value.

    Handles two formats:
      - Absolute URL: https://betmana.co.uk/football/clubs/...
        -> sport is the first path segment
      - Domain-prefixed path: /betmana.co.uk/football/clubs/...
        -> first segment is a domain (contains '.'), sport is the second segment
    """
    parsed = urlparse(loc_text.strip())
    path = parsed.path.strip("/")
    if not path:
        return "general"

    segments = path.split("/")

    # If it's an absolute URL, the domain is in netloc and first segment is sport
    if parsed.scheme and parsed.netloc:
        candidate = segments[0]
    else:
        # Relative path — first segment might be a domain like "betmana.co.uk"
        if "." in segments[0] and len(segments) > 1:
            candidate = segments[1]
        else:
            candidate = segments[0]

    return candidate if candidate in SPORT_SECTIONS else "general"


def group_urls_by_sport(url_elements: list[ET.Element]) -> dict[str, list[ET.Element]]:
    """Group URL elements by sport section."""
    groups: dict[str, list[ET.Element]] = defaultdict(list)

    for url_el in url_elements:
        loc = url_el.find("sm:loc", NS)
        if loc is None or not loc.text:
            groups["general"].append(url_el)
            continue
        sport = extract_sport(loc.text)
        groups[sport].append(url_el)

    return dict(groups)


def write_sitemap(urls: list[ET.Element], output_path: Path) -> None:
    """Write a list of <url> elements as a sitemap XML file."""
    root = ET.Element("urlset")
    root.set("xmlns", SITEMAP_NS)

    for url_el in urls:
        root.append(url_el)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")

    with open(output_path, "wb") as f:
        f.write(b'<?xml version="1.0" encoding="UTF-8"?>\n')
        tree.write(f, encoding=None, xml_declaration=False)


def get_latest_lastmod(urls: list[ET.Element]) -> str | None:
    """Get the latest <lastmod> value from a list of URL elements."""
    dates = []
    for url_el in urls:
        lastmod = url_el.find("sm:lastmod", NS)
        if lastmod is not None and lastmod.text:
            dates.append(lastmod.text.strip())
    return max(dates) if dates else None


def write_sitemapindex(sitemaps: list[dict], output_path: Path) -> None:
    """Write a sitemapindex XML file."""
    root = ET.Element("sitemapindex")
    root.set("xmlns", SITEMAP_NS)

    for sm in sitemaps:
        sitemap_el = ET.SubElement(root, "sitemap")
        loc_el = ET.SubElement(sitemap_el, "loc")
        loc_el.text = sm["loc"]
        if sm.get("lastmod"):
            lastmod_el = ET.SubElement(sitemap_el, "lastmod")
            lastmod_el.text = sm["lastmod"]

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")

    with open(output_path, "wb") as f:
        f.write(b'<?xml version="1.0" encoding="UTF-8"?>\n')
        tree.write(f, encoding=None, xml_declaration=False)


def split_sitemap(public_dir: Path) -> None:
    """Main entry point: read sitemap.xml, split by sport, write index."""
    sitemap_path = public_dir / "sitemap.xml"
    if not sitemap_path.exists():
        raise FileNotFoundError(f"sitemap.xml not found at {sitemap_path}")

    print(f"Reading {sitemap_path} ...")
    url_elements = parse_sitemap(sitemap_path)
    total_urls = len(url_elements)
    print(f"  Found {total_urls:,} URLs")

    base_url = detect_base_url(url_elements)
    if base_url:
        print(f"  Base URL: {base_url}")
        loc_prefix = base_url.rstrip("/")
    else:
        print("  URLs use relative paths (no scheme+host)")
        loc_prefix = ""

    groups = group_urls_by_sport(url_elements)

    sitemap_entries: list[dict] = []
    total_sitemaps = 0

    # Sort groups for deterministic output (general first, then alphabetical)
    sorted_groups = sorted(groups.keys(), key=lambda k: ("" if k == "general" else k))

    for group_name in sorted_groups:
        urls = groups[group_name]
        print(f"  {group_name}: {len(urls):,} URLs", end="")

        if len(urls) <= MAX_URLS_PER_SITEMAP:
            filename = f"sitemap-{group_name}.xml"
            write_sitemap(urls, public_dir / filename)
            lastmod = get_latest_lastmod(urls)
            sitemap_entries.append({
                "loc": f"{loc_prefix}/{filename}",
                "lastmod": lastmod,
            })
            total_sitemaps += 1
            print(f" -> {filename}")
        else:
            chunks = [urls[i:i + MAX_URLS_PER_SITEMAP] for i in range(0, len(urls), MAX_URLS_PER_SITEMAP)]
            filenames = []
            for idx, chunk in enumerate(chunks, start=1):
                filename = f"sitemap-{group_name}-{idx}.xml"
                write_sitemap(chunk, public_dir / filename)
                lastmod = get_latest_lastmod(chunk)
                sitemap_entries.append({
                    "loc": f"{loc_prefix}/{filename}",
                    "lastmod": lastmod,
                })
                filenames.append(filename)
                total_sitemaps += 1
            print(f" -> {', '.join(filenames)}")

    # Write sitemapindex as sitemap.xml (replaces the original)
    write_sitemapindex(sitemap_entries, sitemap_path)
    print(f"\nWrote sitemapindex to {sitemap_path} with {total_sitemaps} sitemap(s)")
    print(f"Total URLs across all sitemaps: {total_urls:,}")


def main():
    parser = argparse.ArgumentParser(description="Split sitemap.xml into per-sport sitemaps")
    parser.add_argument(
        "--public-dir",
        type=Path,
        required=True,
        help="Path to Hugo's public output directory",
    )
    args = parser.parse_args()

    if not args.public_dir.is_dir():
        parser.error(f"Directory not found: {args.public_dir}")

    split_sitemap(args.public_dir)


if __name__ == "__main__":
    main()
