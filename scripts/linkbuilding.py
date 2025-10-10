#!/usr/bin/env python3
"""
Linkbuilding Module for Hugo Static Sites
Processes HTML files and adds internal links based on keyword definitions.
Supports multilingual sites with different link definitions per language.
"""

import os
import sys
import json
import csv
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup, NavigableString, Tag
import html


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
    
    def to_html(self) -> str:
        """Generate HTML report"""
        html_parts = [
            '<!DOCTYPE html>',
            '<html lang="en">',
            '<head>',
            '<meta charset="UTF-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">',
            '<title>Linkbuilding Report</title>',
            '<style>',
            'body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; margin: 20px; background: #f5f5f5; }',
            '.container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }',
            'h1 { color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }',
            'h2 { color: #555; margin-top: 30px; }',
            '.stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }',
            '.stat-card { background: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 4px solid #007bff; }',
            '.stat-value { font-size: 2em; font-weight: bold; color: #007bff; }',
            '.stat-label { color: #666; margin-top: 5px; }',
            'table { width: 100%; border-collapse: collapse; margin: 20px 0; }',
            'th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }',
            'th { background: #007bff; color: white; position: sticky; top: 0; }',
            'tr:hover { background: #f8f9fa; }',
            '.error { background: #f8d7da; color: #721c24; padding: 10px; border-radius: 5px; margin: 10px 0; }',
            '.success { background: #d4edda; color: #155724; padding: 10px; border-radius: 5px; margin: 10px 0; }',
            '</style>',
            '</head>',
            '<body>',
            '<div class="container">',
            f'<h1>Linkbuilding Report - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</h1>',
            
            '<div class="stats">',
            '<div class="stat-card">',
            f'<div class="stat-value">{self.total_files_processed}</div>',
            '<div class="stat-label">Files Processed</div>',
            '</div>',
            '<div class="stat-card">',
            f'<div class="stat-value">{self.total_files_modified}</div>',
            '<div class="stat-label">Files Modified</div>',
            '</div>',
            '<div class="stat-card">',
            f'<div class="stat-value">{self.total_links_added}</div>',
            '<div class="stat-label">Links Added</div>',
            '</div>',
            '<div class="stat-card">',
            f'<div class="stat-value">{len(self.links_per_keyword)}</div>',
            '<div class="stat-label">Keywords Used</div>',
            '</div>',
            '</div>',
        ]
        
        if self.total_links_added > 0:
            html_parts.append('<div class="success">✓ Successfully added links to content</div>')
        
        # Links per keyword table
        if self.links_per_keyword:
            html_parts.extend([
                '<h2>Links Added per Keyword</h2>',
                '<table>',
                '<thead><tr><th>Keyword</th><th>Count</th></tr></thead>',
                '<tbody>'
            ])
            for keyword, count in sorted(self.links_per_keyword.items(), key=lambda x: x[1], reverse=True):
                html_parts.append(f'<tr><td>{html.escape(keyword)}</td><td>{count}</td></tr>')
            html_parts.extend(['</tbody>', '</table>'])
        
        # Links per URL table
        if self.links_per_url:
            html_parts.extend([
                '<h2>Links Added per URL</h2>',
                '<table>',
                '<thead><tr><th>URL</th><th>Count</th></tr></thead>',
                '<tbody>'
            ])
            for url, count in sorted(self.links_per_url.items(), key=lambda x: x[1], reverse=True):
                html_parts.append(f'<tr><td>{html.escape(url)}</td><td>{count}</td></tr>')
            html_parts.extend(['</tbody>', '</table>'])
        
        # Modified files table
        if self.links_per_file:
            html_parts.extend([
                '<h2>Modified Files</h2>',
                '<table>',
                '<thead><tr><th>File</th><th>Links Added</th></tr></thead>',
                '<tbody>'
            ])
            for file_path, count in sorted(self.links_per_file.items(), key=lambda x: x[1], reverse=True):
                html_parts.append(f'<tr><td>{html.escape(file_path)}</td><td>{count}</td></tr>')
            html_parts.extend(['</tbody>', '</table>'])
        
        # Errors
        if self.errors:
            html_parts.append('<h2>Errors</h2>')
            for error in self.errors:
                html_parts.append(f'<div class="error">{html.escape(error)}</div>')
        
        html_parts.extend(['</div>', '</body>', '</html>'])
        return '\n'.join(html_parts)


class LinkBuilder:
    """Main linkbuilding processor"""
    
    # Elements to skip when processing
    SKIP_TAGS = {
        'a', 'script', 'style', 'code', 'pre', 'button', 'input', 'textarea',
        'select', 'option', 'label', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
        'nav', 'header', 'footer', 'aside', 'meta', 'link', 'img', 'svg',
        'iframe', 'video', 'audio', 'canvas', 'map', 'area', 'form'
    }
    
    # Paths to skip (taxonomy pages, pagination, etc.) - same as precompute_linkbuilding.py
    SKIP_PATHS = {
        '/tags/', '/categories/', '/page/', '/author/',
        '/search/', '/404.html', '/index.xml', '/sitemap.xml'
    }
    
    def __init__(self, keywords: List[Keyword], config: LinkConfig = None, language: str = None):
        self.keywords = sorted(keywords, key=lambda k: (-k.priority, -len(k.keyword)))
        self.config = config or LinkConfig()
        self.stats = LinkStatistics()
        self.current_file = None
        self.language = language or ''  # Language code for progress reporting
        self.reset_page_counters()
    
    def reset_page_counters(self):
        """Reset counters for a new page"""
        self.page_keyword_counts = defaultdict(int)
        self.page_url_counts = defaultdict(int)
        self.page_keyword_url_counts = defaultdict(int)
        self.page_total_links = 0
        self.page_replacements = 0
        self.existing_links = 0
    
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
    
    def process_directory(self, directory: str, exclude_dirs: List[str] = None, is_english: bool = False) -> LinkStatistics:
        """Process all HTML files in a directory
        
        Args:
            directory: Directory to process
            exclude_dirs: List of directory names to exclude
            is_english: True if processing English content (to skip language subdirs)
        """
        directory = Path(directory)
        exclude_dirs = exclude_dirs or []
        
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
        
        print(f"Found {len(html_files)} HTML files to process")
        if skipped_lang > 0:
            print(f"  Skipped {skipped_lang} files from other language directories")
        if skipped_other > 0:
            print(f"  Skipped {skipped_other} files (categories, tags, pagination, etc.)")
        
        # Process each file with progress reporting
        total_files = len(html_files)
        for index, html_file in enumerate(html_files, 1):
            self.process_file(html_file)
            
            # Report progress every 100 files
            if index % 100 == 0:
                lang_prefix = f"[{self.language}] " if self.language else ""
                print(f"{lang_prefix}Processed {index}/{total_files} files...")
        
        # Final progress if not already shown
        if total_files > 0 and total_files % 100 != 0:
            lang_prefix = f"[{self.language}] " if self.language else ""
            print(f"{lang_prefix}Processed {total_files}/{total_files} files - completed")
        
        return self.stats
    
    def process_file(self, file_path: Path) -> bool:
        """Process a single HTML file"""
        self.current_file = str(file_path)
        self.reset_page_counters()
        
        try:
            # Read file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse HTML
            soup = BeautifulSoup(content, 'html.parser')
            
            # Count existing links
            self.existing_links = len(soup.find_all('a'))
            
            # Process the document
            modified = self.process_element(soup)
            
            self.stats.total_files_processed += 1
            
            if modified and self.page_replacements > 0:
                # Write back modified content
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(str(soup))
                
                self.stats.total_files_modified += 1
                print(f"✓ {file_path}: Added {self.page_replacements} links")
                return True
            else:
                print(f"  {file_path}: No changes")
                return False
                
        except Exception as e:
            error_msg = f"Error processing {file_path}: {str(e)}"
            self.stats.errors.append(error_msg)
            print(f"✗ {error_msg}")
            return False
    
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
        """Process a text node and add links"""
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
        
        # Process each keyword
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
                
                # Add marker attribute for linkbuilding-generated links (short to save space)
                link_attrs.append('data-lb="1"')
                
                # Determine if external link
                if keyword.url.startswith(('http://', 'https://', '//')):
                    parsed = urlparse(keyword.url)
                    if parsed.netloc and parsed.netloc != urlparse(self.current_file).netloc:
                        # External link
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
                
                # Update statistics
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
  python linkbuilding.py -k keywords_en.csv -a en_automatic.json -d public/en/
  
  # Process with automatic keywords only
  python linkbuilding.py -a en_automatic.json -d public/en/
  
  # Process German content with German keywords, excluding other languages
  python linkbuilding.py -k keywords_de.csv -a de_automatic.json -d public/de/ --exclude en es fr
  
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
    
    # Create link builder with language info
    builder = LinkBuilder(keywords, config, language=language)
    
    # Process directory
    print(f"Processing directory: {args.directory}")
    if args.exclude:
        print(f"Excluding: {', '.join(args.exclude)}")
    
    # Determine if we're processing English content
    is_english = (language == 'en')
    if is_english:
        print("  Processing English content - will skip 2-letter language subdirectories")
    
    stats = builder.process_directory(args.directory, args.exclude, is_english)
    
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