#!/usr/bin/env python3
"""
Precompute Linkbuilding Files
Creates file-centric JSON files mapping HTML files to their matching keywords.
This enables O(1) keyword lookup during deployment instead of O(n) search.

Output structure:
    data/linkbuilding/precomputed/
    ├── en/
    │   ├── academy.json
    │   ├── features.json
    │   └── _root.json
    ├── de/
    │   └── ...

JSON format:
    {
      "/academy/index.html": ["affiliate marketing", "tracking software"],
      "/academy/other/index.html": []
    }
"""

import os
import sys
import json
import re
import argparse
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
from datetime import datetime
import concurrent.futures
import time

try:
    from bs4 import BeautifulSoup
except ImportError:
    print("Error: beautifulsoup4 is required. Run: pip install beautifulsoup4 lxml")
    sys.exit(1)

try:
    import ahocorasick
    AHOCORASICK_AVAILABLE = True
except ImportError:
    AHOCORASICK_AVAILABLE = False
    print("Warning: pyahocorasick not installed, using fallback regex matching (slower)")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def get_default_lang_in_subdir(hugo_root: Path) -> bool:
    """Read defaultContentLanguageInSubdir from Hugo config.

    Returns True if default language (English) is in subdirectory (/en/),
    False if English is at root level.
    """
    # Check config/_default/hugo.toml first
    config_file = hugo_root / 'config' / '_default' / 'hugo.toml'
    if config_file.exists():
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if 'defaultContentLanguageInSubdir' in line:
                        # Parse TOML line: defaultContentLanguageInSubdir = true
                        if '=' in line:
                            value = line.split('=', 1)[1].strip().lower()
                            return value == 'true'
        except Exception as e:
            logger.warning(f"Error reading Hugo config: {e}")

    # Fallback: check config.toml at root
    for config_name in ['config.toml', 'hugo.toml']:
        config_file = hugo_root / config_name
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if 'defaultContentLanguageInSubdir' in line:
                            if '=' in line:
                                value = line.split('=', 1)[1].strip().lower()
                                return value == 'true'
            except Exception as e:
                logger.warning(f"Error reading Hugo config: {e}")

    # Default: assume English in subdirectory (safer)
    logger.warning("Could not find defaultContentLanguageInSubdir in Hugo config, assuming true")
    return True


class KeywordMatcher:
    """Matches keywords in HTML content using Aho-Corasick algorithm."""

    SKIP_TAGS = {
        'script', 'style', 'code', 'pre', 'meta', 'link', 'svg',
        'iframe', 'video', 'audio', 'canvas', 'map', 'area'
    }

    SKIP_PATHS = {
        '/tags/', '/categories/', '/page/', '/author/',
        '/404.html', '/search/', '/index.xml', '/sitemap.xml',
        '/feed.xml', '/rss.xml', '/atom.xml', '/flags/'
    }

    def __init__(self, keywords: Dict[str, Dict]):
        """Initialize with keyword dictionary.

        Args:
            keywords: Dict mapping keyword text to metadata (url, title, priority, exact)
        """
        self.keywords = keywords
        self.automaton = None
        self.keyword_patterns = []
        self._build_matcher()

    def _build_matcher(self):
        """Build Aho-Corasick automaton or regex patterns for matching."""
        if AHOCORASICK_AVAILABLE:
            self.automaton = ahocorasick.Automaton()
            for keyword in self.keywords:
                keyword_lower = keyword.lower()
                self.automaton.add_word(keyword_lower, keyword)
            self.automaton.make_automaton()
            logger.debug(f"Built Aho-Corasick automaton with {len(self.keywords)} keywords")
        else:
            # Fallback to compiled regex patterns
            for keyword in self.keywords:
                if any(ord(char) > 127 for char in keyword):
                    pattern = re.compile(re.escape(keyword.lower()))
                else:
                    pattern = re.compile(r'\b' + re.escape(keyword.lower()) + r'\b')
                self.keyword_patterns.append((keyword, pattern))
            logger.debug(f"Built {len(self.keyword_patterns)} regex patterns")

    def extract_text_from_html(self, html_content: str) -> str:
        """Extract text content from HTML, removing scripts/styles."""
        try:
            soup = BeautifulSoup(html_content, 'lxml')
        except:
            soup = BeautifulSoup(html_content, 'html.parser')

        for tag in self.SKIP_TAGS:
            for element in soup.find_all(tag):
                element.decompose()

        return soup.get_text(separator=' ', strip=True)

    def find_keywords_in_text(self, text: str, page_url: str = None) -> List[str]:
        """Find all matching keywords in text.

        Args:
            text: Text content to search
            page_url: URL of the page (to skip self-references)

        Returns:
            List of matching keyword names
        """
        text_lower = text.lower()
        found_keywords = []

        if AHOCORASICK_AVAILABLE and self.automaton:
            for end_idx, keyword in self.automaton.iter(text_lower):
                start_idx = end_idx - len(keyword) + 1
                keyword_lower = keyword.lower()

                # Verify word boundaries for Latin scripts
                if not any(ord(char) > 127 for char in keyword):
                    if start_idx > 0 and text_lower[start_idx - 1].isalnum():
                        continue
                    if end_idx + 1 < len(text_lower) and text_lower[end_idx + 1].isalnum():
                        continue

                # Skip self-references
                if page_url and keyword in self.keywords:
                    keyword_url = self.keywords[keyword].get('url', '')
                    if self._is_self_reference(page_url, keyword_url):
                        continue

                if keyword not in found_keywords:
                    found_keywords.append(keyword)
        else:
            # Fallback regex matching
            for keyword, pattern in self.keyword_patterns:
                if pattern.search(text_lower):
                    # Skip self-references
                    if page_url and keyword in self.keywords:
                        keyword_url = self.keywords[keyword].get('url', '')
                        if self._is_self_reference(page_url, keyword_url):
                            continue
                    if keyword not in found_keywords:
                        found_keywords.append(keyword)

        return found_keywords

    def _is_self_reference(self, page_url: str, keyword_url: str) -> bool:
        """Check if keyword URL points to the same page."""
        if not page_url or not keyword_url:
            return False

        # Normalize URLs (skip external URLs)
        page_normalized = page_url.strip()
        if not page_normalized.startswith(('http://', 'https://', '//')):
            if not page_normalized.startswith('/'):
                page_normalized = '/' + page_normalized
            if not page_normalized.endswith('/'):
                page_normalized += '/'

        keyword_normalized = keyword_url.strip()
        if not keyword_normalized.startswith('/'):
            if keyword_normalized.startswith(('http://', 'https://', '//')):
                return False
            keyword_normalized = '/' + keyword_normalized
        if not keyword_normalized.endswith('/'):
            keyword_normalized += '/'

        return page_normalized == keyword_normalized

    def should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped based on path."""
        path_str = str(file_path).replace('\\', '/')
        for skip_pattern in self.SKIP_PATHS:
            if skip_pattern in path_str:
                return True
        return False

    def should_skip_for_english_at_root(self, file_path: Path, public_dir: Path) -> bool:
        """When English is at root, skip other language subdirectories."""
        try:
            rel_path = file_path.relative_to(public_dir)
            first_part = rel_path.parts[0] if rel_path.parts else ''
            # Skip if first directory looks like a language code (2-5 chars, lowercase)
            if first_part and len(first_part) <= 5 and first_part.islower() and first_part.isalpha():
                # Check if it's a known language directory (has index.html)
                lang_index = public_dir / first_part / 'index.html'
                if lang_index.exists():
                    return True
        except ValueError:
            pass
        return False


def load_keywords(linkbuilding_dir: Path, lang: str) -> Dict[str, Dict]:
    """Load keywords from automatic and manual files.

    Returns:
        Dict mapping keyword text to metadata {url, title, priority, exact}
    """
    keywords = {}

    # Load automatic keywords
    auto_file = linkbuilding_dir / 'automatic' / f"{lang}_automatic.json"
    if auto_file.exists():
        with open(auto_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        items = data.get('keywords', data) if isinstance(data, dict) else data
        for item in items:
            if isinstance(item, dict):
                keyword = item.get('Keyword', item.get('keyword', ''))
                if keyword:
                    keywords[keyword] = {
                        'url': item.get('URL', item.get('url', '')),
                        'title': item.get('Title', item.get('title', '')),
                        'priority': item.get('Priority', item.get('priority', 0)),
                        'exact': item.get('Exact', item.get('exact', False))
                    }
        logger.info(f"  Loaded {len(keywords)} automatic keywords from {auto_file.name}")

    # Load manual keywords (higher priority)
    manual_file = linkbuilding_dir / f"{lang}.json"
    if manual_file.exists():
        with open(manual_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        items = data.get('keywords', data) if isinstance(data, dict) else data
        manual_count = 0
        for item in items:
            if isinstance(item, dict):
                keyword = item.get('Keyword', item.get('keyword', ''))
                if keyword:
                    keywords[keyword] = {
                        'url': item.get('URL', item.get('url', '')),
                        'title': item.get('Title', item.get('title', '')),
                        'priority': item.get('Priority', item.get('priority', 0)) + 10,  # Boost manual
                        'exact': item.get('Exact', item.get('exact', False))
                    }
                    manual_count += 1
        logger.info(f"  Loaded {manual_count} manual keywords from {manual_file.name}")

    return keywords


def get_keywords_hash(linkbuilding_dir: Path, lang: str) -> str:
    """Compute a hash of keyword file contents to detect actual changes.

    Returns a hash string that changes only when keyword content changes,
    not when files are regenerated with the same content.
    """
    hasher = hashlib.md5()

    # Hash automatic keywords file content
    auto_file = linkbuilding_dir / 'automatic' / f"{lang}_automatic.json"
    if auto_file.exists():
        try:
            with open(auto_file, 'rb') as f:
                hasher.update(f.read())
        except:
            pass

    # Hash manual keywords file content
    manual_file = linkbuilding_dir / f"{lang}.json"
    if manual_file.exists():
        try:
            with open(manual_file, 'rb') as f:
                hasher.update(f.read())
        except:
            pass

    return hasher.hexdigest()


def load_cache(cache_file: Path) -> Dict:
    """Load incremental build cache."""
    if cache_file.exists():
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return {'file_mtimes': {}, 'keywords_hash': ''}


def save_cache(cache_file: Path, cache: Dict):
    """Save incremental build cache."""
    cache['generated_at'] = datetime.now().isoformat()
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2)


def get_folder_for_file(file_path: Path, html_dir: Path, lang: str) -> str:
    """Determine which folder category a file belongs to."""
    try:
        rel_path = file_path.relative_to(html_dir)
        parts = rel_path.parts

        if len(parts) <= 1:
            return '_root'

        # For English, files are at root level
        # For other languages, first part is the language code
        if lang == 'en':
            return parts[0]
        else:
            # Skip language code prefix if present
            if len(parts) > 1:
                return parts[0]
            return '_root'
    except ValueError:
        return '_root'


def get_page_url(file_path: Path, html_dir: Path, lang: str) -> str:
    """Extract page URL from file path."""
    try:
        rel_path = file_path.relative_to(html_dir)
        url_path = '/' + str(rel_path).replace('\\', '/')
        url_path = url_path.replace('/index.html', '/')
        if not url_path.endswith('/'):
            url_path = url_path.rsplit('.html', 1)[0] + '/'
        return url_path
    except ValueError:
        return None


def process_file(file_path: Path, matcher: KeywordMatcher, html_dir: Path,
                 lang: str) -> Tuple[str, str, List[str], float]:
    """Process a single HTML file.

    Returns:
        Tuple of (relative_path, folder, keywords_list, mtime)
    """
    try:
        mtime = file_path.stat().st_mtime

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        text = matcher.extract_text_from_html(content)
        page_url = get_page_url(file_path, html_dir, lang)
        keywords = matcher.find_keywords_in_text(text, page_url)

        # Get relative path for JSON key
        try:
            rel_path = file_path.relative_to(html_dir)
            rel_path_str = '/' + str(rel_path).replace('\\', '/')
        except ValueError:
            rel_path_str = str(file_path)

        folder = get_folder_for_file(file_path, html_dir, lang)

        return (rel_path_str, folder, keywords, mtime)

    except Exception as e:
        logger.warning(f"Error processing {file_path}: {e}")
        return (str(file_path), '_root', [], 0.0)


def process_language(lang: str,
                    linkbuilding_dir: Path,
                    public_dir: Path,
                    output_dir: Path,
                    default_lang_in_subdir: bool = True,
                    max_workers: int = 8,
                    force: bool = False) -> Dict:
    """Process a single language and generate JSON files."""
    logger.info(f"\nProcessing language: {lang}")

    # Determine HTML directory based on Hugo config
    # defaultContentLanguageInSubdir=true  -> all languages in subdirs (public/en/, public/de/)
    # defaultContentLanguageInSubdir=false -> English at root, others in subdirs
    if lang == 'en' and not default_lang_in_subdir:
        # English is at root level
        html_dir = public_dir
        english_at_root = True
        logger.info(f"  English at root level (defaultContentLanguageInSubdir=false)")
    else:
        # Language in subdirectory
        html_dir = public_dir / lang
        english_at_root = False

    if not html_dir.exists():
        logger.warning(f"HTML directory not found: {html_dir}")
        return {}

    # Create output directory
    lang_output_dir = output_dir / lang
    lang_output_dir.mkdir(parents=True, exist_ok=True)

    # Load cache for incremental builds
    cache_file = lang_output_dir / '.cache.json'
    cache = load_cache(cache_file)

    # Check if keywords source changed using content hash (not mtime)
    keywords_hash = get_keywords_hash(linkbuilding_dir, lang)
    cached_hash = cache.get('keywords_hash', '')
    keywords_changed = keywords_hash != cached_hash

    if keywords_changed:
        logger.info("  Keywords content changed - full recomputation required")
        cache = {'file_mtimes': {}, 'keywords_hash': keywords_hash}
    else:
        logger.info("  Keywords unchanged (hash match)")

    # Load keywords
    keywords = load_keywords(linkbuilding_dir, lang)
    if not keywords:
        logger.warning(f"No keywords found for {lang}")
        return {}

    logger.info(f"  Total keywords: {len(keywords)}")

    # Find HTML files
    all_html_files = list(html_dir.rglob('*.html'))

    # Filter files (use class method for filtering before creating matcher)
    html_files = []
    for f in all_html_files:
        # Check skip paths
        path_str = str(f).replace('\\', '/')
        skip = False
        for skip_pattern in KeywordMatcher.SKIP_PATHS:
            if skip_pattern in path_str:
                skip = True
                break
        if skip:
            continue
        # When English is at root, skip other language subdirectories
        if english_at_root:
            try:
                rel_path = f.relative_to(public_dir)
                first_part = rel_path.parts[0] if rel_path.parts else ''
                if first_part and len(first_part) <= 5 and first_part.islower() and first_part.isalpha():
                    lang_index = public_dir / first_part / 'index.html'
                    if lang_index.exists():
                        continue
            except ValueError:
                pass
        html_files.append(f)

    logger.info(f"  Found {len(html_files)} HTML files to process")

    # Determine which files need processing
    files_to_process = []
    cached_results = {}  # folder -> {path: keywords}

    for file_path in html_files:
        rel_path_str = '/' + str(file_path.relative_to(html_dir)).replace('\\', '/')
        file_mtime = file_path.stat().st_mtime
        cached_mtime = cache['file_mtimes'].get(rel_path_str, 0.0)

        if force or file_mtime > cached_mtime:
            files_to_process.append(file_path)
        else:
            # File hasn't changed - load from existing JSON
            folder = get_folder_for_file(file_path, html_dir, lang)
            if folder not in cached_results:
                cached_results[folder] = {}
            # We'll load existing results later

    logger.info(f"  Files to process: {len(files_to_process)} (skipping {len(html_files) - len(files_to_process)} cached)")

    # Fast path: if nothing to process and keywords unchanged, skip entirely
    if not files_to_process and not keywords_changed:
        logger.info("  Nothing changed - skipping (cached)")
        return {
            'language': lang,
            'html_files': len(html_files),
            'json_files': len(list(lang_output_dir.glob('*.json'))) - 1,  # Exclude .cache.json
            'total_keywords': 0,  # Not recounting, but exists
            'files_processed': 0,
            'files_cached': len(html_files),
            'skipped': True
        }

    # Build keyword matcher only when needed (expensive operation)
    matcher = KeywordMatcher(keywords)

    # Load existing JSON files for cached files
    existing_jsons = {}
    for json_file in lang_output_dir.glob('*.json'):
        folder = json_file.stem
        if folder == '.cache':
            continue
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                existing_jsons[folder] = json.load(f) or {}
        except:
            existing_jsons[folder] = {}

    # Process files in parallel
    results_by_folder = defaultdict(dict)  # folder -> {path: keywords}
    new_mtimes = {}

    if files_to_process:
        start_time = time.time()
        processed = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_file, f, matcher, html_dir, lang): f
                for f in files_to_process
            }

            for future in concurrent.futures.as_completed(futures):
                try:
                    rel_path, folder, keywords_list, mtime = future.result()
                    results_by_folder[folder][rel_path] = keywords_list
                    new_mtimes[rel_path] = mtime
                    processed += 1

                    if processed % 500 == 0:
                        logger.info(f"  Processed {processed}/{len(files_to_process)} files...")
                except Exception as e:
                    logger.error(f"Error processing file: {e}")

        elapsed = time.time() - start_time
        logger.info(f"  Processed {processed} files in {elapsed:.1f}s")

    # Merge with existing results for unchanged files
    for file_path in html_files:
        rel_path_str = '/' + str(file_path.relative_to(html_dir)).replace('\\', '/')
        folder = get_folder_for_file(file_path, html_dir, lang)

        if rel_path_str not in results_by_folder.get(folder, {}):
            # Load from existing JSON
            if folder in existing_jsons and rel_path_str in existing_jsons[folder]:
                if folder not in results_by_folder:
                    results_by_folder[folder] = {}
                results_by_folder[folder][rel_path_str] = existing_jsons[folder][rel_path_str]
                new_mtimes[rel_path_str] = cache['file_mtimes'].get(rel_path_str, 0.0)

    # Write JSON files per folder
    total_files = 0
    total_keywords = 0

    for folder, file_keywords in results_by_folder.items():
        if not file_keywords:
            continue

        json_file = lang_output_dir / f"{folder}.json"

        # Sort by file path for consistent output
        sorted_data = dict(sorted(file_keywords.items()))

        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(sorted_data, f, ensure_ascii=False, indent=2)

        files_in_folder = len(file_keywords)
        keywords_in_folder = sum(len(kw) for kw in file_keywords.values())
        total_files += files_in_folder
        total_keywords += keywords_in_folder

        logger.debug(f"  Wrote {json_file.name}: {files_in_folder} files, {keywords_in_folder} keyword matches")

    # Update and save cache
    cache['file_mtimes'].update(new_mtimes)
    cache['keywords_hash'] = keywords_hash
    save_cache(cache_file, cache)

    logger.info(f"  Output: {len(results_by_folder)} JSON files, {total_files} HTML files, {total_keywords} keyword matches")

    return {
        'language': lang,
        'html_files': total_files,
        'json_files': len(results_by_folder),
        'total_keywords': total_keywords,
        'files_processed': len(files_to_process),
        'files_cached': len(html_files) - len(files_to_process)
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Precompute file-centric linkbuilding JSON files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Creates JSON files mapping HTML files to their matching keywords.
This enables O(1) keyword lookup during deployment.

Output structure:
  data/linkbuilding/precomputed/[lang]/[folder].json

Examples:
  python precompute_linkbuilding_files.py \\
    --linkbuilding-dir data/linkbuilding \\
    --public-dir public \\
    --output-dir data/linkbuilding/precomputed

  # Force full recomputation
  python precompute_linkbuilding_files.py \\
    --linkbuilding-dir data/linkbuilding \\
    --public-dir public \\
    --output-dir data/linkbuilding/precomputed \\
    --force
        """
    )

    parser.add_argument('--linkbuilding-dir', required=True,
                       help='Directory containing keyword JSON files')
    parser.add_argument('--public-dir', required=True,
                       help='Public directory with built HTML files')
    parser.add_argument('--output-dir', required=True,
                       help='Output directory for precomputed JSON files')
    parser.add_argument('--languages', nargs='+',
                       help='Specific languages to process (default: all)')
    parser.add_argument('--max-workers', type=int, default=8,
                       help='Maximum parallel workers for file processing (default: 8)')
    parser.add_argument('--parallel-languages', type=int, default=3,
                       help='Number of languages to process in parallel (default: 3)')
    parser.add_argument('--force', action='store_true',
                       help='Force full recomputation, ignoring cache')
    parser.add_argument('--hugo-root',
                       help='Hugo root directory (to read config). Defaults to parent of public-dir')

    args = parser.parse_args()

    linkbuilding_dir = Path(args.linkbuilding_dir)
    public_dir = Path(args.public_dir)
    output_dir = Path(args.output_dir)

    # Determine Hugo root and read config
    if args.hugo_root:
        hugo_root = Path(args.hugo_root)
    else:
        # Default: assume public_dir is inside Hugo root
        hugo_root = public_dir.parent

    default_lang_in_subdir = get_default_lang_in_subdir(hugo_root)
    logger.info(f"Hugo config: defaultContentLanguageInSubdir = {default_lang_in_subdir}")

    # Validate directories
    if not linkbuilding_dir.exists():
        logger.error(f"Linkbuilding directory not found: {linkbuilding_dir}")
        sys.exit(1)
    if not public_dir.exists():
        logger.error(f"Public directory not found: {public_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find languages
    if args.languages:
        languages = args.languages
    else:
        auto_files = list((linkbuilding_dir / 'automatic').glob('*_automatic.json'))
        languages = [f.stem.replace('_automatic', '') for f in auto_files]

    if not languages:
        logger.error("No languages found to process")
        sys.exit(1)

    logger.info(f"Processing {len(languages)} languages: {', '.join(languages)}")
    logger.info("=" * 60)

    start_time = time.time()
    results = []

    # Process languages (can be parallelized, but sequential is simpler for now)
    for lang in languages:
        try:
            stats = process_language(
                lang, linkbuilding_dir, public_dir, output_dir,
                default_lang_in_subdir=default_lang_in_subdir,
                max_workers=args.max_workers,
                force=args.force
            )
            if stats:
                results.append(stats)
        except Exception as e:
            logger.error(f"Error processing {lang}: {e}")

    elapsed = time.time() - start_time

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("PRECOMPUTATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total time: {elapsed:.1f} seconds")
    logger.info(f"Languages processed: {len(results)}")

    total_html = sum(r.get('html_files', 0) for r in results)
    total_json = sum(r.get('json_files', 0) for r in results)
    total_kw = sum(r.get('total_keywords', 0) for r in results)
    total_processed = sum(r.get('files_processed', 0) for r in results)
    total_cached = sum(r.get('files_cached', 0) for r in results)

    logger.info(f"Total HTML files: {total_html:,}")
    logger.info(f"Total JSON files created: {total_json}")
    logger.info(f"Total keyword matches: {total_kw:,}")
    logger.info(f"Files processed: {total_processed:,}")
    logger.info(f"Files from cache: {total_cached:,}")

    logger.info("\nPer-language results:")
    logger.info("-" * 40)
    for r in sorted(results, key=lambda x: x.get('total_keywords', 0), reverse=True):
        status = "(cached)" if r.get('skipped') else ""
        logger.info(f"  {r['language'].upper():3} | {r['html_files']:5,} files | "
                   f"{r['json_files']:3} JSONs | {r.get('files_processed', 0):5,} processed {status}")

    # Save summary
    summary_file = output_dir / 'precomputation_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            'languages': results,
            'summary': {
                'total_languages': len(results),
                'total_html_files': total_html,
                'total_json_files': total_json,
                'total_keyword_matches': total_kw,
                'elapsed_seconds': elapsed
            },
            'generated_at': datetime.now().isoformat()
        }, f, indent=2)

    logger.info(f"\nSummary saved to: {summary_file}")
    logger.info("Precomputed linkbuilding files ready!")


if __name__ == '__main__':
    main()
