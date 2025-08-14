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
    max_replacements_per_keyword: int = 2
    max_replacements_per_url: int = 5
    max_replacements_per_keyword_url: int = 1
    max_links_on_page: int = 500
    max_replacements_per_page: int = 50
    max_replacements_per_paragraph: int = 10
    min_chars_between_links: int = 2
    min_paragraph_length: int = 30
    max_paragraph_density: int = 30  # min chars per link in paragraph
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
    
    def __init__(self, keywords: List[Keyword], config: LinkConfig = None):
        self.keywords = sorted(keywords, key=lambda k: (-k.priority, -len(k.keyword)))
        self.config = config or LinkConfig()
        self.stats = LinkStatistics()
        self.current_file = None
        self.reset_page_counters()
    
    def reset_page_counters(self):
        """Reset counters for a new page"""
        self.page_keyword_counts = defaultdict(int)
        self.page_url_counts = defaultdict(int)
        self.page_keyword_url_counts = defaultdict(int)
        self.page_total_links = 0
        self.page_replacements = 0
        self.existing_links = 0
    
    def process_directory(self, directory: str, exclude_dirs: List[str] = None) -> LinkStatistics:
        """Process all HTML files in a directory"""
        directory = Path(directory)
        exclude_dirs = exclude_dirs or []
        
        # Find all HTML files
        html_files = []
        for root, dirs, files in os.walk(directory):
            # Remove excluded directories from dirs to prevent walking into them
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            for file in files:
                if file.endswith('.html'):
                    html_files.append(Path(root) / file)
        
        print(f"Found {len(html_files)} HTML files to process")
        
        # Process each file
        for html_file in html_files:
            self.process_file(html_file)
        
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
                        new_content = self.process_text_node(child, child.parent)
                        if new_content != str(child):
                            # Replace the text node with new content
                            new_soup = BeautifulSoup(new_content, 'html.parser')
                            child.replace_with(new_soup)
                            modified = True
                elif isinstance(child, Tag):
                    # Skip certain tags
                    if child.name not in self.SKIP_TAGS:
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
    """Load keywords from JSON file"""
    keywords = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for item in data:
        keyword = Keyword(
            keyword=item.get('keyword', '').strip(),
            url=item.get('url', '').strip(),
            title=item.get('title', '').strip(),
            priority=item.get('priority', 0),
            exact_match=item.get('exact_match', False)
        )
        if keyword.keyword and keyword.url:
            keywords.append(keyword)
    
    return keywords


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Linkbuilding tool for Hugo static sites',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process English content with English keywords
  python linkbuilding.py -k keywords_en.csv -d public/
  
  # Process German content with German keywords, excluding other languages
  python linkbuilding.py -k keywords_de.csv -d public/de/ --exclude en es fr
  
  # Use JSON configuration
  python linkbuilding.py -k keywords.json -d public/ -c config.json
  
  # Generate report without modifying files
  python linkbuilding.py -k keywords.csv -d public/ --dry-run

CSV Format:
  keyword,url,title,priority,exact_match
  "AI Tools",/tools/,Browse AI Tools,10,false
  "machine learning",/ml/,Learn about ML,5,true

JSON Format:
  [
    {
      "keyword": "AI Tools",
      "url": "/tools/",
      "title": "Browse AI Tools",
      "priority": 10,
      "exact_match": false
    }
  ]
        """
    )
    
    parser.add_argument('-k', '--keywords', required=True,
                       help='Path to keywords file (CSV or JSON)')
    parser.add_argument('-d', '--directory', required=True,
                       help='Directory to process HTML files')
    parser.add_argument('-c', '--config',
                       help='Configuration file (JSON)')
    parser.add_argument('--exclude', nargs='+', default=[],
                       help='Directories to exclude from processing')
    parser.add_argument('-o', '--output', default='linkbuilding-report.html',
                       help='Output report file (default: linkbuilding-report.html)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Analyze without modifying files')
    parser.add_argument('--max-links', type=int,
                       help='Override max links per page')
    parser.add_argument('--max-keyword', type=int,
                       help='Override max replacements per keyword')
    parser.add_argument('--max-url', type=int,
                       help='Override max replacements per URL')
    
    args = parser.parse_args()
    
    # Load keywords
    if args.keywords.endswith('.json'):
        keywords = load_keywords_from_json(args.keywords)
    else:
        keywords = load_keywords_from_csv(args.keywords)
    
    print(f"Loaded {len(keywords)} keywords")
    
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
    
    # Create link builder
    builder = LinkBuilder(keywords, config)
    
    # Process directory
    print(f"Processing directory: {args.directory}")
    if args.exclude:
        print(f"Excluding: {', '.join(args.exclude)}")
    
    stats = builder.process_directory(args.directory, args.exclude)
    
    # Generate and save report
    report_html = stats.to_html()
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(report_html)
    
    print(f"\nReport saved to: {args.output}")
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Files processed: {stats.total_files_processed}")
    print(f"  Files modified: {stats.total_files_modified}")
    print(f"  Links added: {stats.total_links_added}")
    print(f"  Keywords used: {len(stats.links_per_keyword)}")
    
    if stats.errors:
        print(f"\nErrors encountered: {len(stats.errors)}")
        for error in stats.errors[:5]:
            print(f"  - {error}")
    
    # Exit with appropriate code
    sys.exit(0 if stats.errors == [] else 1)


if __name__ == '__main__':
    main()