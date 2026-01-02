#!/usr/bin/env python3
"""
Linkbuilding Module for Hugo Static Sites
Processes HTML files and adds internal links based on keyword definitions.
Supports multilingual sites with different link definitions per language.

Optimized version with:
- lxml parser (2-3x faster than html.parser)
- Aho-Corasick algorithm for multi-pattern matching
- Parallel file processing within each language
- Early file skip optimization
- Reduced console output
"""

import os
import sys
import json
import csv
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup, NavigableString, Tag
import html
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Try to import ahocorasick for fast multi-pattern matching
try:
    import ahocorasick
    AHOCORASICK_AVAILABLE = True
except ImportError:
    AHOCORASICK_AVAILABLE = False
    print("Warning: pyahocorasick not installed, using fallback regex matching (slower)")


@dataclass
class Keyword:
    """Represents a keyword to be linked"""
    keyword: str
    url: str
    title: str = ""
    priority: int = 0
    exact_match: bool = False
    
    def __post_init__(self):
        # Normalize keyword for case-insensitive matching
        self.keyword_lower = self.keyword.lower()
        self.keyword_pattern = self._create_pattern()
    
    def _create_pattern(self) -> re.Pattern:
        """Create regex pattern for keyword matching"""
        escaped = re.escape(self.keyword)
        if self.exact_match:
            # Exact word boundary matching
            pattern = r'\b' + escaped + r'\b'
        else:
            # Word boundary matching
            pattern = r'\b' + escaped + r'\b'
        return re.compile(pattern, re.IGNORECASE)


@dataclass
class LinkConfig:
    """Configuration for linkbuilding"""
    max_replacements_per_keyword: int = 2  # Reduced from 2
    max_replacements_per_url: int = 2  # Reduced from 5
    max_replacements_per_keyword_url: int = 1
    max_links_on_page: int = 50  # Dramatically reduced from 500!
    max_replacements_per_page: int = 30  # Reduced from 50
    max_replacements_per_paragraph: int = 3  # Reduced from 10
    min_chars_between_links: int = 1  # Increased from 2 to avoid link clustering
    min_paragraph_length: int = 30  # Increased from 30
    max_paragraph_density: int = 30  # Increased from 30 - min chars per link in paragraph
    skip_existing_links: bool = True
    preserve_case: bool = True
    add_title_attribute: bool = True
    external_link_target: str = "_blank"
    internal_link_target: str = ""
    nofollow_external: bool = True
    track_statistics: bool = True
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'LinkConfig':
        """Create config from dictionary"""
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})


@dataclass
class LinkStatistics:
    """Statistics for link additions"""
    total_files_processed: int = 0
    total_files_modified: int = 0
    total_links_added: int = 0
    links_per_keyword: Dict[str, int] = field(default_factory=dict)
    links_per_url: Dict[str, int] = field(default_factory=dict)
    links_per_file: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    
    def add_link(self, keyword: str, url: str, file_path: str):
        """Record a link addition"""
        self.total_links_added += 1
        self.links_per_keyword[keyword] = self.links_per_keyword.get(keyword, 0) + 1
        self.links_per_url[url] = self.links_per_url.get(url, 0) + 1
        self.links_per_file[file_path] = self.links_per_file.get(file_path, 0) + 1


class LinkBuilder:
    """Main linkbuilding processor - Optimized version"""

    # Elements to skip when processing
    SKIP_TAGS = {
        'a', 'script', 'style', 'code', 'pre', 'button', 'input', 'textarea',
        'select', 'option', 'label', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
        'nav', 'header', 'footer', 'aside', 'meta', 'link', 'img', 'svg',
        'iframe', 'video', 'audio', 'canvas', 'map', 'area', 'form', 'title',
        'head', 'footer'
    }

    # Paths to skip (taxonomy pages, pagination, etc.) - same as precompute_linkbuilding.py
    SKIP_PATHS = {
        '/tags/', '/categories/', '/page/', '/author/',
        '/search/', '/404.html', '/index.xml', '/sitemap.xml'
    }

    def __init__(self, keywords: List[Keyword], config: LinkConfig = None, language: str = None,
                 precomputed_dir: str = None):
        self.keywords = sorted(keywords, key=lambda k: (-k.priority, -len(k.keyword)))
        self.config = config or LinkConfig()
        self.stats = LinkStatistics()
        self.stats_lock = threading.Lock()  # Thread-safe stats updates
        self.current_file = None
        self.language = language or ''  # Language code for progress reporting
        self.reset_page_counters()

        # Build keyword lookup by name (case-insensitive)
        self.keyword_by_name = {}  # Map lowercase keyword to Keyword object
        for kw in self.keywords:
            self.keyword_by_name[kw.keyword.lower()] = kw

        # Precomputed mode support
        self.precomputed_dir = Path(precomputed_dir) if precomputed_dir else None
        self.precomputed_data = {}  # folder -> {file_path: [keywords]}
        self.using_precomputed = False

        # Build Aho-Corasick automaton for fast keyword matching (fallback mode)
        self.automaton = None
        self.keyword_lookup = {}  # Map lowercase keyword to Keyword object
        self._build_automaton()
    
    def _build_automaton(self):
        """Build Aho-Corasick automaton for fast multi-pattern matching"""
        if not AHOCORASICK_AVAILABLE:
            return

        self.automaton = ahocorasick.Automaton()

        for keyword in self.keywords:
            keyword_lower = keyword.keyword.lower()
            self.keyword_lookup[keyword_lower] = keyword
            # Add to automaton with keyword as value
            self.automaton.add_word(keyword_lower, keyword)

        self.automaton.make_automaton()

    def load_precomputed(self):
        """Load precomputed keyword mappings from JSON files.

        Returns True if precomputed data was loaded successfully.
        """
        if not self.precomputed_dir or not self.precomputed_dir.exists():
            return False

        # Find language-specific directory
        lang_dir = self.precomputed_dir / self.language if self.language else self.precomputed_dir

        if not lang_dir.exists():
            return False

        json_files = list(lang_dir.glob('*.json'))
        if not json_files:
            return False

        loaded_files = 0
        for json_file in json_files:
            if json_file.name.startswith('.'):
                continue
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f) or {}
                folder = json_file.stem
                self.precomputed_data[folder] = data
                loaded_files += 1
            except Exception as e:
                print(f"Warning: Failed to load {json_file}: {e}")

        if loaded_files > 0:
            self.using_precomputed = True
            total_mappings = sum(len(d) for d in self.precomputed_data.values())
            print(f"  Loaded precomputed data: {loaded_files} files, {total_mappings} file mappings")
            return True

        return False

    def get_precomputed_keywords_for_file(self, file_path: Path, base_dir: Path) -> Optional[List[Keyword]]:
        """Get precomputed keywords for a specific file.

        Returns list of Keyword objects, or None if not found in precomputed data.
        """
        if not self.using_precomputed:
            return None

        try:
            rel_path = file_path.relative_to(base_dir)
            rel_path_str = '/' + str(rel_path).replace('\\', '/')
        except ValueError:
            return None

        # Determine folder
        parts = rel_path.parts
        if len(parts) <= 1:
            folder = '_root'
        else:
            folder = parts[0]

        # Look up in precomputed data
        folder_data = self.precomputed_data.get(folder, {})
        keyword_names = folder_data.get(rel_path_str)

        if keyword_names is None:
            # Try alternative path format
            alt_path = rel_path_str.replace('/index.html', '/')
            keyword_names = folder_data.get(alt_path)

        if keyword_names is None:
            return None

        # Convert keyword names to Keyword objects
        keywords = []
        for name in keyword_names:
            name_lower = name.lower()
            if name_lower in self.keyword_by_name:
                keywords.append(self.keyword_by_name[name_lower])

        # Sort by priority (descending) then by length (descending)
        keywords.sort(key=lambda k: (-k.priority, -len(k.keyword)))
        return keywords

    def reset_page_counters(self):
        """Reset counters for a new page"""
        self.page_keyword_counts = defaultdict(int)
        self.page_url_counts = defaultdict(int)
        self.page_keyword_url_counts = defaultdict(int)
        self.page_total_links = 0
        self.page_replacements = 0
        self.existing_links = 0
        self.current_page_url = None  # URL path of current page

    def _extract_page_url_from_canonical(self, soup) -> Optional[str]:
        """Extract the URL path from the canonical link in the HTML.

        Looks for <link rel="canonical" href="..."> and extracts the path.
        Returns the URL path that can be compared with keyword URLs.
        """
        canonical = soup.find('link', rel='canonical')
        if not canonical or not canonical.get('href'):
            return None

        href = canonical.get('href', '').strip()
        if not href:
            return None

        # Parse the URL to get just the path
        parsed = urlparse(href)
        url_path = parsed.path

        # Ensure it ends with /
        if url_path and not url_path.endswith('/'):
            url_path = url_path + '/'

        return url_path if url_path else None

    def _is_self_reference(self, keyword_url: str) -> bool:
        """Check if a keyword URL points to the current page (self-reference).

        Returns True if the link would point to the same page, False otherwise.
        """
        if not self.current_page_url or not keyword_url:
            return False

        # Normalize keyword URL
        normalized_keyword_url = keyword_url.strip()
        if not normalized_keyword_url.startswith('/'):
            # Skip external URLs - they're not self-references
            if normalized_keyword_url.startswith(('http://', 'https://', '//')):
                return False
            normalized_keyword_url = '/' + normalized_keyword_url

        if not normalized_keyword_url.endswith('/'):
            normalized_keyword_url = normalized_keyword_url + '/'

        # Compare URLs
        return self.current_page_url == normalized_keyword_url
    
    def should_skip_file(self, file_path: Path) -> bool:
        """Check if a file should be skipped based on its path."""
        path_str = str(file_path).replace('\\', '/')
        
        # Skip paths containing any of the skip patterns
        for skip_pattern in self.SKIP_PATHS:
            if skip_pattern in path_str:
                return True
        
        # Don't skip index.html files - they are important content pages in Hugo
        # Only skip if they match the skip patterns above
        
        return False
    
    def should_skip_for_english(self, file_path: Path, base_dir: Path) -> bool:
        """Check if a file should be skipped when processing English content.
        
        English content is at the root of public/, but other languages are in
        subdirectories like /ar/, /cs/, /de/, etc. We need to skip these.
        """
        path_str = str(file_path).replace('\\', '/')
        base_str = str(base_dir).replace('\\', '/')
        
        # Get relative path from base directory
        try:
            rel_path = file_path.relative_to(base_dir)
            parts = rel_path.parts
            
            # Check if first directory is a 2-letter language code
            if len(parts) > 0:
                first_dir = parts[0]
                # Skip if it's a 2-letter directory name (language code)
                if len(first_dir) == 2 and first_dir.isalpha() and first_dir.islower():
                    return True
        except ValueError:
            pass
        
        return False
    
    def _quick_keyword_check(self, content: str) -> bool:
        """Quick check if content might contain any keywords (optimization).

        Uses Aho-Corasick if available for O(n) check, otherwise samples keywords.
        Returns True if keywords might be present, False if definitely not.
        """
        content_lower = content.lower()

        if AHOCORASICK_AVAILABLE and self.automaton:
            # Use automaton for fast check - just need to find one match
            for _ in self.automaton.iter(content_lower):
                return True
            return False
        else:
            # Fallback: check first 50 keywords (most common/priority)
            for keyword in self.keywords[:50]:
                if keyword.keyword_lower in content_lower:
                    return True
            return False

    def process_directory(self, directory: str, exclude_dirs: List[str] = None,
                         is_english: bool = False, max_workers: int = 4) -> LinkStatistics:
        """Process all HTML files in a directory with parallel processing

        Args:
            directory: Directory to process
            exclude_dirs: List of directory names to exclude
            is_english: True if processing English content (to skip language subdirs)
            max_workers: Number of parallel workers for file processing
        """
        directory = Path(directory)
        exclude_dirs = exclude_dirs or []

        # Try to load precomputed keyword mappings
        if self.precomputed_dir:
            self.load_precomputed()
            if self.using_precomputed:
                print(f"  Using precomputed mode (O(1) keyword lookup)")

        # Find all HTML files
        all_html_files = []
        for root, dirs, files in os.walk(directory):
            # Remove excluded directories from dirs to prevent walking into them
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for file in files:
                if file.endswith('.html'):
                    all_html_files.append(Path(root) / file)

        # Filter out unwanted files
        html_files = []
        skipped_lang = 0
        skipped_other = 0

        for file_path in all_html_files:
            # Skip based on path patterns
            if self.should_skip_file(file_path):
                skipped_other += 1
                continue

            # For English, also skip language subdirectories
            if is_english and self.should_skip_for_english(file_path, directory):
                skipped_lang += 1
                continue

            html_files.append(file_path)

        total_files = len(html_files)
        lang_prefix = f"[{self.language}] " if self.language else ""
        print(f"{lang_prefix}Found {total_files} HTML files to process")
        if skipped_lang > 0:
            print(f"{lang_prefix}  Skipped {skipped_lang} files from other language directories")
        if skipped_other > 0:
            print(f"{lang_prefix}  Skipped {skipped_other} files (categories, tags, pagination, etc.)")

        # Progress tracking
        processed_count = [0]  # Use list for mutable closure
        modified_count = [0]
        skipped_no_keywords = [0]
        progress_lock = threading.Lock()

        def process_with_progress(file_path: Path) -> Tuple[bool, bool]:
            """Process a file and update progress. Returns (was_processed, was_modified)"""
            try:
                # In precomputed mode, check if file has any keywords
                if self.using_precomputed:
                    precomputed_keywords = self.get_precomputed_keywords_for_file(file_path, directory)
                    if precomputed_keywords is None or len(precomputed_keywords) == 0:
                        # No keywords for this file - skip
                        with progress_lock:
                            skipped_no_keywords[0] += 1
                            processed_count[0] += 1
                            with self.stats_lock:
                                self.stats.total_files_processed += 1
                        return (True, False)

                    # Process with specific keywords only
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    modified = self._process_file_content_with_keywords(file_path, content, precomputed_keywords)
                else:
                    # Original mode: quick check and process with all keywords
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    if not self._quick_keyword_check(content):
                        with progress_lock:
                            skipped_no_keywords[0] += 1
                            processed_count[0] += 1
                            with self.stats_lock:
                                self.stats.total_files_processed += 1
                        return (True, False)

                    # Process file with pre-read content
                    modified = self._process_file_content(file_path, content)

                with progress_lock:
                    processed_count[0] += 1
                    if modified:
                        modified_count[0] += 1

                    # Progress update every 500 files (reduced verbosity)
                    if processed_count[0] % 500 == 0:
                        print(f"{lang_prefix}Progress: {processed_count[0]}/{total_files} files "
                              f"({modified_count[0]} modified, {skipped_no_keywords[0]} skipped)")

                return (True, modified)

            except Exception as e:
                with self.stats_lock:
                    self.stats.errors.append(f"Error processing {file_path}: {str(e)}")
                return (False, False)

        # Process files in parallel
        if max_workers > 1 and total_files > 10:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(process_with_progress, f): f for f in html_files}
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        file_path = futures[future]
                        with self.stats_lock:
                            self.stats.errors.append(f"Error processing {file_path}: {str(e)}")
        else:
            # Sequential processing for small file counts
            for html_file in html_files:
                process_with_progress(html_file)

        # Final summary (reduced verbosity - single line)
        print(f"{lang_prefix}Completed: {total_files} files processed, "
              f"{modified_count[0]} modified, {skipped_no_keywords[0]} skipped (no keywords)")

        return self.stats
    
    def process_file(self, file_path: Path) -> bool:
        """Process a single HTML file (legacy method for compatibility)"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return self._process_file_content(file_path, content)
        except Exception as e:
            error_msg = f"Error processing {file_path}: {str(e)}"
            with self.stats_lock:
                self.stats.errors.append(error_msg)
            return False

    def _process_file_content(self, file_path: Path, content: str) -> bool:
        """Process a single HTML file with pre-read content (optimized version)"""
        self.current_file = str(file_path)
        self.reset_page_counters()

        try:
            # Parse HTML using lxml (2-3x faster than html.parser)
            soup = BeautifulSoup(content, 'lxml')

            # Extract current page URL from canonical link for self-reference detection
            self.current_page_url = self._extract_page_url_from_canonical(soup)

            # Count existing links
            self.existing_links = len(soup.find_all('a'))

            # Process the document
            modified = self.process_element(soup)

            with self.stats_lock:
                self.stats.total_files_processed += 1

            if modified and self.page_replacements > 0:
                # Write back modified content
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(str(soup))

                with self.stats_lock:
                    self.stats.total_files_modified += 1
                # Removed per-file output for reduced verbosity
                return True
            else:
                # Removed "No changes" output for reduced verbosity
                return False

        except Exception as e:
            error_msg = f"Error processing {file_path}: {str(e)}"
            with self.stats_lock:
                self.stats.errors.append(error_msg)
            return False

    def _process_file_content_with_keywords(self, file_path: Path, content: str,
                                            keywords_to_use: List[Keyword]) -> bool:
        """Process a file using only the specified keywords (precomputed mode).

        This temporarily switches to using only the specified keywords,
        processes the file, then restores the original state.
        """
        # Save original state
        original_keywords = self.keywords
        original_automaton = self.automaton
        original_keyword_lookup = self.keyword_lookup

        try:
            # Set up for specific keywords only
            self.keywords = keywords_to_use

            # Build temporary automaton for just these keywords
            if AHOCORASICK_AVAILABLE:
                self.automaton = ahocorasick.Automaton()
                self.keyword_lookup = {}
                for keyword in keywords_to_use:
                    keyword_lower = keyword.keyword.lower()
                    self.keyword_lookup[keyword_lower] = keyword
                    self.automaton.add_word(keyword_lower, keyword)
                self.automaton.make_automaton()

            # Process the file
            result = self._process_file_content(file_path, content)

            return result

        finally:
            # Restore original state
            self.keywords = original_keywords
            self.automaton = original_automaton
            self.keyword_lookup = original_keyword_lookup

    def process_element(self, element) -> bool:
        """Process an HTML element recursively"""
        if not element:
            return False
        
        modified = False
        
        # Skip if we've reached limits
        if (self.page_total_links >= self.config.max_links_on_page or
            self.page_replacements >= self.config.max_replacements_per_page):
            return False
        
        # Process children
        if hasattr(element, 'children'):
            # Create a list of children to avoid modification during iteration
            children = list(element.children)
            
            for child in children:
                if isinstance(child, NavigableString):
                    # Process text node
                    if child.parent and child.parent.name not in self.SKIP_TAGS:
                        # Skip if parent is a link with our marker
                        if child.parent.name == 'a' and child.parent.get('data-lb'):
                            continue
                        new_content = self.process_text_node(child, child.parent)
                        if new_content != str(child):
                            # Replace the text node with new content
                            new_soup = BeautifulSoup(new_content, 'html.parser')
                            child.replace_with(new_soup)
                            modified = True
                elif isinstance(child, Tag):
                    # Skip certain tags and existing linkbuilding links
                    if child.name not in self.SKIP_TAGS:
                        # Skip links that already have our marker
                        if child.name == 'a' and child.get('data-lb'):
                            continue
                        if self.process_element(child):
                            modified = True
        
        return modified
    
    def process_text_node(self, text_node: NavigableString, parent_element) -> str:
        """Process a text node and add links - Optimized with Aho-Corasick"""
        text = str(text_node)

        # Skip short text
        if len(text.strip()) < self.config.min_paragraph_length:
            return text

        # Calculate paragraph density limit
        text_length = len(text)
        max_links_density = max(1, text_length // self.config.max_paragraph_density)
        paragraph_links = 0

        # Track positions where we've added links to maintain minimum distance
        link_positions = []

        # Process using Aho-Corasick if available (much faster for many keywords)
        if AHOCORASICK_AVAILABLE and self.automaton:
            return self._process_text_with_automaton(text, max_links_density)
        else:
            return self._process_text_with_regex(text, max_links_density)

    def _process_text_with_automaton(self, text: str, max_links_density: int) -> str:
        """Process text using Aho-Corasick automaton (fast path)"""
        text_lower = text.lower()
        paragraph_links = 0

        # Find all keyword matches in one pass using Aho-Corasick
        matches = []  # List of (start, end, keyword)
        for end_idx, keyword in self.automaton.iter(text_lower):
            start_idx = end_idx - len(keyword.keyword) + 1

            # Verify word boundaries
            if start_idx > 0 and text_lower[start_idx - 1].isalnum():
                continue
            if end_idx + 1 < len(text_lower) and text_lower[end_idx + 1].isalnum():
                continue

            matches.append((start_idx, end_idx + 1, keyword))

        # Sort by priority (higher first), then by position
        matches.sort(key=lambda m: (-m[2].priority, m[0]))

        # Process matches, avoiding overlaps
        used_ranges = []  # List of (start, end) ranges already used
        replacements = []  # List of (start, end, link_html)

        for start, end, keyword in matches:
            # Check page limits
            if (self.page_replacements >= self.config.max_replacements_per_page or
                paragraph_links >= self.config.max_replacements_per_paragraph or
                paragraph_links >= max_links_density):
                break

            # Check keyword-specific limits
            if (self.page_keyword_counts[keyword.keyword] >= self.config.max_replacements_per_keyword or
                self.page_url_counts[keyword.url] >= self.config.max_replacements_per_url):
                continue

            keyword_url_key = f"{keyword.keyword}|{keyword.url}"
            if self.page_keyword_url_counts[keyword_url_key] >= self.config.max_replacements_per_keyword_url:
                continue

            # Skip self-references (don't link a page to itself)
            if self._is_self_reference(keyword.url):
                continue

            # Check for overlaps with already used ranges
            overlaps = False
            for used_start, used_end in used_ranges:
                if not (end <= used_start or start >= used_end):
                    overlaps = True
                    break
                # Check minimum distance
                if abs(start - used_end) < self.config.min_chars_between_links or \
                   abs(used_start - end) < self.config.min_chars_between_links:
                    overlaps = True
                    break

            if overlaps:
                continue

            # Get original case text
            matched_text = text[start:end]

            # Build link HTML
            link_attrs = [f'href="{keyword.url}"']
            if self.config.add_title_attribute and keyword.title:
                link_attrs.append(f'title="{html.escape(keyword.title)}"')
            link_attrs.append('data-lb="1"')

            # Determine if external link
            if keyword.url.startswith(('http://', 'https://', '//')):
                parsed = urlparse(keyword.url)
                if parsed.netloc and parsed.netloc != urlparse(self.current_file).netloc:
                    if self.config.external_link_target:
                        link_attrs.append(f'target="{self.config.external_link_target}"')
                    if self.config.nofollow_external:
                        link_attrs.append('rel="nofollow noopener"')
            elif self.config.internal_link_target:
                link_attrs.append(f'target="{self.config.internal_link_target}"')

            link_html = f'<a {" ".join(link_attrs)}>{html.escape(matched_text)}</a>'

            # Record this replacement
            replacements.append((start, end, link_html))
            used_ranges.append((start, end))

            # Update counters
            self.page_keyword_counts[keyword.keyword] += 1
            self.page_url_counts[keyword.url] += 1
            self.page_keyword_url_counts[keyword_url_key] += 1
            self.page_replacements += 1
            self.page_total_links += 1
            paragraph_links += 1

            # Update statistics (thread-safe)
            with self.stats_lock:
                self.stats.add_link(keyword.keyword, keyword.url, self.current_file)

        # Build result string
        if not replacements:
            return text

        # Sort replacements by position for building result
        replacements.sort(key=lambda r: r[0])

        result_parts = []
        last_end = 0
        for start, end, link_html in replacements:
            if start > last_end:
                result_parts.append(html.escape(text[last_end:start]))
            result_parts.append(link_html)
            last_end = end

        if last_end < len(text):
            result_parts.append(html.escape(text[last_end:]))

        return ''.join(result_parts)

    def _process_text_with_regex(self, text: str, max_links_density: int) -> str:
        """Process text using regex (fallback when Aho-Corasick not available)"""
        paragraph_links = 0
        link_positions = []
        result_parts = []
        last_end = 0

        for keyword in self.keywords:
            # Check if we can add more links
            if (self.page_replacements >= self.config.max_replacements_per_page or
                paragraph_links >= self.config.max_replacements_per_paragraph or
                paragraph_links >= max_links_density):
                break

            # Check keyword-specific limits
            if (self.page_keyword_counts[keyword.keyword] >= self.config.max_replacements_per_keyword or
                self.page_url_counts[keyword.url] >= self.config.max_replacements_per_url):
                continue

            keyword_url_key = f"{keyword.keyword}|{keyword.url}"
            if self.page_keyword_url_counts[keyword_url_key] >= self.config.max_replacements_per_keyword_url:
                continue

            # Skip self-references (don't link a page to itself)
            if self._is_self_reference(keyword.url):
                continue

            # Find keyword in text
            matches = list(keyword.keyword_pattern.finditer(text))

            for match in matches:
                start, end = match.span()

                # Check if this position conflicts with existing links
                too_close = False
                for pos in link_positions:
                    if abs(start - pos) < self.config.min_chars_between_links:
                        too_close = True
                        break

                if too_close:
                    continue

                # Check if we're not overlapping with already processed text
                if start < last_end:
                    continue

                # Add the link
                matched_text = match.group(0)

                # Build link HTML
                link_attrs = [f'href="{keyword.url}"']
                if self.config.add_title_attribute and keyword.title:
                    link_attrs.append(f'title="{html.escape(keyword.title)}"')
                link_attrs.append('data-lb="1"')

                # Determine if external link
                if keyword.url.startswith(('http://', 'https://', '//')):
                    parsed = urlparse(keyword.url)
                    if parsed.netloc and parsed.netloc != urlparse(self.current_file).netloc:
                        if self.config.external_link_target:
                            link_attrs.append(f'target="{self.config.external_link_target}"')
                        if self.config.nofollow_external:
                            link_attrs.append('rel="nofollow noopener"')
                elif self.config.internal_link_target:
                    link_attrs.append(f'target="{self.config.internal_link_target}"')

                link_html = f'<a {" ".join(link_attrs)}>{html.escape(matched_text)}</a>'

                # Add text before the link
                if start > last_end:
                    result_parts.append(html.escape(text[last_end:start]))

                result_parts.append(link_html)
                last_end = end

                # Update counters
                self.page_keyword_counts[keyword.keyword] += 1
                self.page_url_counts[keyword.url] += 1
                self.page_keyword_url_counts[keyword_url_key] += 1
                self.page_replacements += 1
                self.page_total_links += 1
                paragraph_links += 1
                link_positions.append(start)

                # Update statistics (thread-safe)
                with self.stats_lock:
                    self.stats.add_link(keyword.keyword, keyword.url, self.current_file)

                # Move to next keyword after successful replacement
                break

        # Add remaining text
        if last_end < len(text):
            result_parts.append(html.escape(text[last_end:]))

        # Return the result
        if result_parts:
            return ''.join(result_parts)
        else:
            return text


def load_keywords_from_csv(file_path: str) -> List[Keyword]:
    """Load keywords from CSV file"""
    keywords = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            keyword = Keyword(
                keyword=row.get('keyword', '').strip(),
                url=row.get('url', '').strip(),
                title=row.get('title', '').strip(),
                priority=int(row.get('priority', 0)),
                exact_match=row.get('exact_match', '').lower() in ('true', '1', 'yes')
            )
            if keyword.keyword and keyword.url:
                keywords.append(keyword)
    
    return keywords


def load_keywords_from_json(file_path: str) -> List[Keyword]:
    """Load keywords from JSON file (supports both manual and automatic formats)
    
    Both formats use the same structure with capitalized field names:
    {
        "keywords": [
            {
                "Keyword": "AI Tools",
                "URL": "/tools/",
                "Title": "Browse AI Tools",
                "Priority": 10,
                "Exact": false
            }
        ]
    }
    
    Note: The code also supports lowercase field names for backward compatibility.
    """
    keywords = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Check if data has "keywords" wrapper (for backwards compatibility)
    if isinstance(data, dict) and "keywords" in data:
        items = data["keywords"]
    elif isinstance(data, list):
        items = data
    else:
        # Support old automatic format where keywords are dict keys
        # {"AI workflow": {"url": "/flows/ai-workflow", "title": "AI Workflow Automation"}}
        items = []
        for keyword_text, values in data.items():
            if isinstance(values, dict):
                items.append({
                    "keyword": keyword_text,
                    "url": values.get("url", ""),
                    "title": values.get("title", ""),
                    "priority": values.get("priority", 0),
                    "exact_match": values.get("exact_match", False)
                })
    
    # Process each item
    for item in items:
        if isinstance(item, dict):
            # Support both lowercase and capitalized field names for compatibility
            keyword_text = item.get('Keyword', item.get('keyword', '')).strip()
            url = item.get('URL', item.get('url', '')).strip()
            title = item.get('Title', item.get('title', '')).strip()
            priority = item.get('Priority', item.get('priority', 0))
            exact_match = item.get('Exact', item.get('exact_match', item.get('exact', False)))
            
            keyword = Keyword(
                keyword=keyword_text,
                url=url,
                title=title,
                priority=priority,
                exact_match=exact_match
            )
            if keyword.keyword and keyword.url:
                keywords.append(keyword)
    
    return keywords


def load_keywords_from_multiple_sources(manual_file: Optional[str] = None,
                                       automatic_file: Optional[str] = None,
                                       manual_priority_boost: int = 10) -> List[Keyword]:
    """Load keywords from both manual and automatic sources
    
    Args:
        manual_file: Path to manual keywords file (CSV or JSON)
        automatic_file: Path to automatic keywords file (JSON)
        manual_priority_boost: Priority boost for manual keywords over automatic ones
    
    Returns:
        Combined list of keywords with manual keywords having higher priority
    """
    keywords = []
    
    # Load automatic keywords first (lower priority)
    if automatic_file and os.path.exists(automatic_file):
        try:
            auto_keywords = load_keywords_from_json(automatic_file)
            print(f"Loaded {len(auto_keywords)} automatic keywords from {automatic_file}")
            keywords.extend(auto_keywords)
        except Exception as e:
            print(f"Warning: Failed to load automatic keywords from {automatic_file}: {e}")
    
    # Load manual keywords (higher priority)
    if manual_file and os.path.exists(manual_file):
        try:
            if manual_file.endswith('.json'):
                manual_keywords = load_keywords_from_json(manual_file)
            else:
                manual_keywords = load_keywords_from_csv(manual_file)
            
            # Boost priority for manual keywords
            for kw in manual_keywords:
                kw.priority += manual_priority_boost
            
            print(f"Loaded {len(manual_keywords)} manual keywords from {manual_file}")
            keywords.extend(manual_keywords)
        except Exception as e:
            print(f"Warning: Failed to load manual keywords from {manual_file}: {e}")
    
    # Remove duplicates (manual keywords take precedence due to higher priority)
    # Keep the highest priority version of each keyword
    keyword_dict = {}
    for kw in keywords:
        key = kw.keyword.lower()
        if key not in keyword_dict or kw.priority > keyword_dict[key].priority:
            keyword_dict[key] = kw
    
    return list(keyword_dict.values())


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Linkbuilding tool for Hugo static sites',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process English content with manual keywords only
  python linkbuilding.py -k keywords_en.csv -d public/
  
  # Process with both manual and automatic keywords
  python linkbuilding.py -k keywords_en.csv -a automatic/en_automatic.json -d public/en/

  # Process with automatic keywords only
  python linkbuilding.py -a automatic/en_automatic.json -d public/en/

  # Process German content with German keywords, excluding other languages
  python linkbuilding.py -k keywords_de.csv -a automatic/de_automatic.json -d public/de/ --exclude en es fr
  
  # Use JSON configuration
  python linkbuilding.py -k keywords.json -d public/ -c config.json
  
  # Generate report without modifying files
  python linkbuilding.py -k keywords.csv -d public/ --dry-run

CSV Format:
  keyword,url,title,priority,exact_match
  "AI Tools",/tools/,Browse AI Tools,10,false
  "machine learning",/ml/,Learn about ML,5,true

JSON Format (Manual and Automatic - same structure):
  {
    "keywords": [
      {
        "Keyword": "AI Tools",
        "URL": "/tools/",
        "Title": "Browse AI Tools",
        "Priority": 10,
        "Exact": false
      },
      {
        "Keyword": "chatbot",
        "URL": "/flows/chatbot",
        "Title": "Build Custom Chatbots",
        "Priority": 5,
        "Exact": true
      }
    ]
  }
  
Note: Field names are capitalized (Keyword, URL, Title, Priority, Exact) but lowercase versions are also supported for compatibility.
        """
    )
    
    parser.add_argument('-k', '--keywords',
                       help='Path to manual keywords file (CSV or JSON)')
    parser.add_argument('-a', '--automatic',
                       help='Path to automatic keywords file (JSON)')
    parser.add_argument('-d', '--directory', required=True,
                       help='Directory to process HTML files')
    parser.add_argument('-c', '--config',
                       help='Configuration file (JSON)')
    parser.add_argument('--exclude', nargs='+', default=[],
                       help='Directories to exclude from processing')
    parser.add_argument('--language',
                       help='Language code being processed (for progress reporting)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Analyze without modifying files')
    parser.add_argument('--max-links', type=int,
                       help='Override max links per page')
    parser.add_argument('--max-keyword', type=int,
                       help='Override max replacements per keyword')
    parser.add_argument('--max-url', type=int,
                       help='Override max replacements per URL')
    parser.add_argument('--manual-priority-boost', type=int, default=10,
                       help='Priority boost for manual keywords over automatic (default: 10)')
    parser.add_argument('--file-workers', type=int, default=4,
                       help='Number of parallel workers for file processing (default: 4)')
    parser.add_argument('--precomputed-dir',
                       help='Directory containing precomputed JSON files (file-centric mode)')

    args = parser.parse_args()
    
    # Validate that at least one keyword source is provided
    if not args.keywords and not args.automatic:
        parser.error("At least one keyword source is required: use -k/--keywords or -a/--automatic")
    
    # Load keywords from both sources
    keywords = load_keywords_from_multiple_sources(
        manual_file=args.keywords,
        automatic_file=args.automatic,
        manual_priority_boost=args.manual_priority_boost
    )
    
    if not keywords:
        print("Error: No keywords loaded from any source")
        sys.exit(1)
    
    print(f"Loaded {len(keywords)} total keywords")
    
    # Load or create config
    config = LinkConfig()
    if args.config:
        with open(args.config, 'r') as f:
            config_data = json.load(f)
            config = LinkConfig.from_dict(config_data)
    
    # Override config with command line arguments
    if args.max_links:
        config.max_replacements_per_page = args.max_links
    if args.max_keyword:
        config.max_replacements_per_keyword = args.max_keyword
    if args.max_url:
        config.max_replacements_per_url = args.max_url
    
    # Determine language being processed
    language = args.language  # Use explicitly provided language if available
    
    if not language:
        # Try to auto-detect language from directory
        dir_path = Path(args.directory).resolve()
        
        if dir_path.name in ['ar', 'cs', 'da', 'de', 'es', 'fi', 'fr', 'it', 'ja', 'ko',
                             'nl', 'no', 'pl', 'pt', 'ro', 'sk', 'sv', 'tr', 'vi', 'zh']:
            language = dir_path.name
        elif dir_path.name == 'public' or (args.exclude and len(args.exclude) >= 10):
            # Check if we're excluding language directories (indicates English processing)
            common_langs = {'ar', 'cs', 'da', 'de', 'es', 'fi', 'fr', 'it', 'ja', 'ko',
                           'nl', 'no', 'pl', 'pt', 'ro', 'sk', 'sv', 'tr', 'vi', 'zh'}
            if args.exclude:
                excluded_set = set(args.exclude)
                if len(excluded_set.intersection(common_langs)) >= 10:
                    language = 'en'
    
    # Create link builder with language info and precomputed dir
    builder = LinkBuilder(keywords, config, language=language,
                         precomputed_dir=args.precomputed_dir)

    # Process directory
    print(f"Processing directory: {args.directory}")
    if args.exclude:
        print(f"Excluding: {', '.join(args.exclude)}")
    if args.precomputed_dir:
        print(f"Precomputed directory: {args.precomputed_dir}")

    # Determine if we're processing English content
    is_english = (language == 'en')
    if is_english:
        print("  Processing English content - will skip 2-letter language subdirectories")

    # Show optimization status
    if AHOCORASICK_AVAILABLE and not args.precomputed_dir:
        print("  Using Aho-Corasick algorithm for fast keyword matching")
    print(f"  Using {args.file_workers} parallel workers for file processing")

    stats = builder.process_directory(args.directory, args.exclude, is_english, max_workers=args.file_workers)
    
    # Don't generate HTML report file, just output to console
    
    # Print detailed summary
    print(f"\n{'='*60}")
    print(f"LINKBUILDING REPORT SUMMARY")
    print(f"{'='*60}")
    print(f"  Files processed: {stats.total_files_processed}")
    print(f"  Files modified: {stats.total_files_modified}")
    print(f"  Links added: {stats.total_links_added}")
    print(f"  Keywords used: {len(stats.links_per_keyword)}")
    
    # Show top keywords by usage
    if stats.links_per_keyword:
        print(f"\nTop 10 Keywords by Usage:")
        top_keywords = sorted(stats.links_per_keyword.items(), key=lambda x: x[1], reverse=True)[:10]
        for keyword, count in top_keywords:
            print(f"  {count:4} - {keyword}")
    
    # Show top URLs by usage
    if stats.links_per_url:
        print(f"\nTop 10 URLs by Usage:")
        top_urls = sorted(stats.links_per_url.items(), key=lambda x: x[1], reverse=True)[:10]
        for url, count in top_urls:
            print(f"  {count:4} - {url}")
    
    # Don't save JSON report file, summary is already printed to console
    
    if stats.errors:
        print(f"\nErrors encountered: {len(stats.errors)}")
        for error in stats.errors[:5]:
            print(f"  - {error}")
    
    print(f"{'='*60}")
    
    # Exit with appropriate code
    sys.exit(0 if stats.errors == [] else 1)


if __name__ == '__main__':
    main()