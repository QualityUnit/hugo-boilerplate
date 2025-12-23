#!/usr/bin/env python3
"""
YAKE Keyword Extractor - Uses YAKE for fast, unsupervised keyword extraction.

Extracts SEO keywords from Hugo markdown files using YAKE algorithm.
Works on any platform, no GPU required, language-independent.

Usage:
    python extract_keywords.py content/ --recursive
    python extract_keywords.py content/en/blog/ --recursive --force
    python extract_keywords.py content/ -r -f --workers 4
"""

import re
import sys
import argparse
import time
import tomllib
from pathlib import Path
from typing import Optional
from multiprocessing import Pool, cpu_count

# Try to import tomli_w for TOML writing
try:
    import tomli_w
except ImportError:
    tomli_w = None

# Try to import yake
try:
    import yake
    YAKE_AVAILABLE = True
except ImportError:
    YAKE_AVAILABLE = False
    print("Installing yake...")
    import subprocess
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "yake", "-q"])
        import yake
        YAKE_AVAILABLE = True
        print("yake installed successfully")
    except Exception as e:
        print(f"Failed to install yake: {e}")

# ============================================================================
# CONFIGURATION
# ============================================================================

MAX_KEYWORDS = 8

# Language code mapping for YAKE
LANG_MAP = {
    'en': 'en', 'de': 'de', 'es': 'es', 'fr': 'fr', 'it': 'it',
    'pt': 'pt', 'nl': 'nl', 'pl': 'pl', 'ru': 'ru', 'ar': 'ar',
    'cs': 'cs', 'da': 'da', 'fi': 'fi', 'hu': 'hu', 'id': 'id',
    'ja': 'ja', 'ko': 'ko', 'no': 'no', 'ro': 'ro', 'sk': 'sk',
    'sv': 'sv', 'tr': 'tr', 'uk': 'uk', 'vi': 'vi', 'zh': 'zh',
    'he': 'he', 'hi': 'hi', 'th': 'th', 'el': 'el', 'bg': 'bg',
    'hr': 'hr', 'lt': 'lt', 'lv': 'lv', 'sl': 'sl', 'et': 'et',
}


# ============================================================================
# PARSING & EXTRACTION
# ============================================================================

def parse_frontmatter(content: str) -> tuple[dict, str, str]:
    """Parse TOML frontmatter using tomllib.

    Returns:
        tuple: (frontmatter_dict, body, raw_frontmatter_str)
    """
    frontmatter = {}
    body = content
    raw_fm = ""
    toml_match = re.match(r'^\+\+\+\s*\n(.*?)\n\+\+\+\s*\n?(.*)', content, re.DOTALL)
    if toml_match:
        raw_fm = toml_match.group(1)
        body = toml_match.group(2)
        try:
            frontmatter = tomllib.loads(raw_fm)
        except tomllib.TOMLDecodeError:
            # Fallback to basic regex parsing if TOML is malformed
            for line in raw_fm.split('\n'):
                if line.strip().startswith('[['):
                    continue
                kv = re.match(r'^(\w+)\s*=\s*(.+)$', line.strip())
                if kv:
                    key, val = kv.groups()
                    if val.startswith('['):
                        frontmatter[key] = re.findall(r'"([^"]*)"', val)
                    elif val.startswith('"'):
                        frontmatter[key] = val.strip('"')
                    else:
                        frontmatter[key] = val
    return frontmatter, body, raw_fm


def detect_language(file_path: Path) -> str:
    """Detect language from file path (e.g., content/de/... -> de)."""
    parts = file_path.parts
    for i, part in enumerate(parts):
        if part == 'content' and i + 1 < len(parts):
            lang = parts[i + 1]
            return LANG_MAP.get(lang, 'en')
    return 'en'


def clean_body(body: str) -> str:
    """Clean markdown body for text extraction."""
    # Remove paired Hugo shortcodes with their content
    text = re.sub(r'\{\{<\s*(\w+)[^>]*>\}\}.*?\{\{<\s*/\1\s*>\}\}', ' ', body, flags=re.DOTALL)
    # Remove remaining standalone shortcodes
    text = re.sub(r'\{\{<[^>]*>\}\}', ' ', text)
    # Remove JSON arrays/objects
    text = re.sub(r'\[\s*\{[^]]*\}\s*\]', ' ', text, flags=re.DOTALL)
    text = re.sub(r'\{[^{}]*"[^"]*"[^{}]*\}', ' ', text)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Convert markdown links to just the text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # Remove code blocks
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    # Remove inline code
    text = re.sub(r'`[^`]+`', '', text)
    # Remove markdown formatting
    text = re.sub(r'[#*_~]', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_headers(body: str) -> list[str]:
    """Extract headers from markdown body."""
    headers = re.findall(r'^#{1,3}\s+(.+)$', body, re.MULTILINE)
    return [h.strip() for h in headers]


def extract_keywords_yake(title: str, description: str, body: str, lang: str = 'en') -> list[str]:
    """Extract keywords using YAKE algorithm.

    YAKE is unsupervised, language-independent, and produces meaningful phrases.
    """
    if not YAKE_AVAILABLE:
        return []

    # Build text from title, description, and headers (not full body to focus on key content)
    headers = extract_headers(body)

    # Weight title and description more by repeating them
    text_parts = [
        title, title,  # Title twice for emphasis
        description,
        ' '.join(headers)
    ]
    text = '. '.join(filter(None, text_parts))

    if len(text) < 10:
        return []

    # Configure YAKE extractor
    kw_extractor = yake.KeywordExtractor(
        lan=lang,
        n=3,  # max 3-word phrases
        dedupLim=0.7,  # deduplication threshold
        dedupFunc='seqm',  # sequence matcher for dedup
        windowsSize=1,
        top=MAX_KEYWORDS * 2,  # get more, filter later
        features=None
    )

    try:
        keywords = kw_extractor.extract_keywords(text)
    except Exception:
        # Fallback to English if language not supported
        kw_extractor = yake.KeywordExtractor(lan='en', n=3, dedupLim=0.7, top=MAX_KEYWORDS * 2)
        keywords = kw_extractor.extract_keywords(text)

    # Filter and clean keywords
    result = []
    seen = set()

    for kw, score in keywords:
        # Clean the keyword
        kw_clean = kw.lower().strip()

        # Remove quotes (single and double) - they break TOML syntax
        kw_clean = kw_clean.replace('"', '').replace("'", '').replace('"', '').replace('"', '').replace('«', '').replace('»', '')
        kw_clean = re.sub(r'\s+', ' ', kw_clean).strip()

        # Skip too short
        if len(kw_clean) < 3:
            continue

        # Skip if just numbers
        if re.match(r'^[\d\s]+$', kw_clean):
            continue

        # Skip duplicates
        if kw_clean in seen:
            continue

        # Skip single common words
        if len(kw_clean.split()) == 1 and kw_clean in {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'give', 'get', 'use', 'make'}:
            continue

        seen.add(kw_clean)
        result.append(kw_clean)

        if len(result) >= MAX_KEYWORDS:
            break

    return result


def update_frontmatter(content: str, keywords: list[str]) -> str:
    """Update frontmatter with new keywords using line-based approach.

    NOTE: We intentionally do NOT use tomli_w for full rewrite because it
    corrupts complex TOML structures (nested tables, inline tables, booleans).
    The line-based approach safely adds/updates only the keywords line.
    """
    if not content.startswith('+++'):
        return content

    toml_match = re.match(r'^(\+\+\+\s*\n)(.*?)(\n\+\+\+\s*\n?)(.*)', content, re.DOTALL)
    if not toml_match:
        return content

    opening, raw_fm, closing, body = toml_match.groups()

    # Line-based approach: filter out existing keywords lines, then add new one
    lines = raw_fm.split('\n')
    filtered_lines = []
    title_idx = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('keywords') and '=' in stripped:
            continue
        filtered_lines.append(line)
        if stripped.startswith('title') and '=' in stripped:
            title_idx = len(filtered_lines)

    # Build new keywords line
    kw_str = ', '.join(f'"{k}"' for k in keywords)
    kw_line = f'keywords = [ {kw_str} ]'

    # Insert after title
    insert_idx = title_idx if title_idx > 0 else 0
    filtered_lines.insert(insert_idx, kw_line)

    new_fm = '\n'.join(filtered_lines)
    return f"{opening}{new_fm}{closing}{body}"


def drop_keywords_from_file(file_path: Path) -> bool:
    """Remove keywords from a single file's frontmatter."""
    try:
        content = file_path.read_text(encoding='utf-8')
        if not content.startswith('+++'):
            return False

        toml_match = re.match(r'^(\+\+\+\s*\n)(.*?)(\n\+\+\+\s*\n?)(.*)', content, re.DOTALL)
        if not toml_match:
            return False

        opening, raw_fm, closing, body = toml_match.groups()

        if 'keywords' not in raw_fm:
            return False

        lines = raw_fm.split('\n')
        filtered_lines = [line for line in lines
                         if not (line.strip().startswith('keywords') and '=' in line.strip())]

        if len(filtered_lines) == len(lines):
            return False

        new_fm = '\n'.join(filtered_lines)
        new_content = f"{opening}{new_fm}{closing}{body}"
        file_path.write_text(new_content, encoding='utf-8')
        return True
    except Exception as e:
        print(f"Error dropping keywords from {file_path}: {e}")
        return False


def drop_all_keywords(files: list[Path]) -> int:
    """Drop keywords from all files."""
    print(f"Dropping existing keywords from {len(files)} files...")
    dropped = 0
    for i, file_path in enumerate(files):
        if drop_keywords_from_file(file_path):
            dropped += 1
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(files)} files, dropped {dropped} keywords so far")
    print(f"Dropped keywords from {dropped} files")
    return dropped


def process_single_file(file_path: Path, force: bool = False) -> Optional[dict]:
    """Process a single file."""
    try:
        content = file_path.read_text(encoding='utf-8')
        fm, body, _ = parse_frontmatter(content)

        if fm.get('keywords') and not force:
            return None

        title = str(fm.get('title', ''))
        if len(title) < 5:
            return None

        # Get description, ignoring linkbuilding attribute
        desc = str(fm.get('description', ''))

        # Detect language from path
        lang = detect_language(file_path)

        # Extract keywords using YAKE
        keywords = extract_keywords_yake(title, desc, body, lang)

        if keywords:
            updated = update_frontmatter(content, keywords)
            file_path.write_text(updated, encoding='utf-8')

        return {'file': str(file_path), 'keywords': keywords}
    except Exception as e:
        return {'file': str(file_path), 'error': str(e)}


def process_file_wrapper(args):
    """Wrapper for multiprocessing."""
    file_path, force = args
    return process_single_file(Path(file_path), force)


def main():
    parser = argparse.ArgumentParser(description='YAKE keyword extraction for Hugo markdown files')
    parser.add_argument('path', type=str, help='Directory to process')
    parser.add_argument('--force', '-f', action='store_true', help='Overwrite existing keywords')
    parser.add_argument('--workers', '-w', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--recursive', '-r', action='store_true', help='Process recursively')

    args = parser.parse_args()

    if not YAKE_AVAILABLE:
        print("Error: YAKE not available. Install with: pip install yake")
        sys.exit(1)

    path = Path(args.path)

    if not path.exists():
        print(f"Error: {path} does not exist")
        sys.exit(1)

    # Collect all files
    if args.recursive:
        files = list(path.rglob('*.md'))
    else:
        files = list(path.glob('*.md'))

    print(f"Found {len(files)} markdown files")
    print(f"Using {args.workers} workers")

    start = time.time()

    # Process with multiprocessing pool
    with Pool(processes=args.workers) as pool:
        file_args = [(str(f), args.force) for f in files]

        processed = 0
        last_report_time = time.time()
        last_report_count = 0

        for i, result in enumerate(pool.imap_unordered(process_file_wrapper, file_args, chunksize=50)):
            if result and 'keywords' in result:
                processed += 1

            if (i + 1) % 100 == 0:
                now = time.time()
                step_elapsed = now - last_report_time
                step_count = (i + 1) - last_report_count
                rate = step_count / step_elapsed if step_elapsed > 0 else 0

                pct = (i + 1) / len(files) * 100
                remaining = len(files) - (i + 1)
                eta_sec = remaining / rate if rate > 0 else 0
                eta_m = int(eta_sec // 60)
                eta_s = int(eta_sec % 60)

                print(f"[{i+1}/{len(files)}] {pct:.1f}% | {rate:.1f} files/sec | ETA: {eta_m}m {eta_s}s")

                last_report_time = now
                last_report_count = i + 1

    elapsed = time.time() - start
    print(f"\nCompleted: {processed} files in {elapsed:.1f}s ({len(files)/elapsed:.1f} files/sec avg)")


if __name__ == '__main__':
    main()
