#!/usr/bin/env python3
"""
Automatic Link Extraction for Hugo Linkbuilding Module

This script extracts keywords from markdown frontmatter across content directories
and generates automatic link definition files for the linkbuilding system.

Usage:
    python extract_automatic_links.py --content-dir content/en/ --output data/linkbuilding/en_automatic.json

Features:
- Parses TOML frontmatter from markdown files
- Priority 1: Uses ALL keywords from 'linkbuilding' attribute (no limit)
- Priority 2: Uses first 2 keywords from 'keywords' attribute (legacy fallback)
- Determines title from description, shortDescription, or title (in priority order)
- Generates JSON output compatible with linkbuilding.py
- Supports multilingual content processing
- Robust error handling and progress reporting
"""

import os
import sys
import json
import argparse
import toml_frontmatter as frontmatter  # Use robust TOML parser
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LinkExtractor:
    """Main class for extracting automatic links from markdown content"""
    
    def __init__(self, content_dir: str, base_url: str = ""):
        self.content_dir = Path(content_dir).resolve()
        self.base_url = base_url
        self.stats = {
            'files_processed': 0,
            'files_with_keywords': 0,
            'total_keywords': 0,
            'errors': []
        }
    
    def process_directory(self) -> List[Dict]:
        """Process all markdown files in the content directory"""
        logger.info(f"Processing content directory: {self.content_dir}")
        
        if not self.content_dir.exists():
            raise FileNotFoundError(f"Content directory does not exist: {self.content_dir}")
        
        links = []
        
        # Walk through all subdirectories
        for root, dirs, files in os.walk(self.content_dir):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for file in files:
                if file.endswith('.md'):
                    file_path = Path(root) / file
                    try:
                        file_links = self.extract_keywords_from_file(file_path)
                        links.extend(file_links)
                        self.stats['files_processed'] += 1
                        
                        if file_links:
                            self.stats['files_with_keywords'] += 1
                            self.stats['total_keywords'] += len(file_links)
                            
                    except Exception as e:
                        error_msg = f"Error processing {file_path}: {str(e)}"
                        self.stats['errors'].append(error_msg)
                        logger.warning(error_msg)
                        continue
        
        logger.info(f"Processing complete. Files: {self.stats['files_processed']}, "
                   f"Keywords: {self.stats['total_keywords']}, "
                   f"Errors: {len(self.stats['errors'])}")
        
        return links
    
    def extract_keywords_from_file(self, file_path: Path) -> List[Dict]:
        """Extract keywords from a single markdown file"""
        try:
            # Read and parse frontmatter (automatically detects YAML or TOML)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Detect frontmatter type and parse accordingly
            if content.startswith('+++'):
                # TOML frontmatter
                with open(file_path, 'r', encoding='utf-8') as f:
                    post = frontmatter.load(f, handler=frontmatter.TOMLHandler())
            else:
                # YAML frontmatter (default)
                with open(file_path, 'r', encoding='utf-8') as f:
                    post = frontmatter.load(f)
            
            # Priority 1: Use linkbuilding attribute if it exists (no limit on keywords)
            if 'linkbuilding' in post.metadata:
                keywords = post.metadata.get('linkbuilding', [])
                if keywords and isinstance(keywords, list):
                    # Use ALL keywords from linkbuilding attribute (no 2-keyword limit)
                    pass  # keywords already set
                else:
                    return []
            else:
                # Priority 2: Use keywords attribute with 2-keyword limit
                keywords = post.metadata.get('keywords', [])
                if not keywords or not isinstance(keywords, list):
                    return []
                
                # Only take the first 2 keywords for automatic links when using keywords attribute
                keywords = keywords[:2]
            
            # Determine the URL for this file
            file_url = self.determine_url(file_path, post.metadata)
            
            # Get the title (description > shortDescription > title)
            title = self.get_title_from_frontmatter(post.metadata)
            
            # Calculate priority for this page
            priority = self.calculate_priority(file_path, file_url)
            
            # Create link entries for each keyword
            links = []
            for keyword in keywords:
                if isinstance(keyword, str) and keyword.strip():
                    link_entry = {
                        "keyword": keyword.strip(),
                        "url": file_url,
                        "title": title,
                        "priority": priority
                    }
                    links.append(link_entry)
            
            if links:
                logger.debug(f"Extracted {len(links)} keywords (max 2) from {file_path}")
            
            return links
            
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {str(e)}")
            raise
    
    def determine_url(self, file_path: Path, frontmatter_data: Dict) -> str:
        """Determine the URL for a file based on frontmatter and file path
        
        Priority:
        1. Use explicit 'url' from frontmatter (for translated URLs)
        2. For flow templates with 'flow_id', generate from path
        3. Fall back to generating from file path
        """
        
        # 1. Check for explicit URL in frontmatter (this is the translated URL)
        # This ensures we use the same URL that Hugo uses when building the site
        if 'url' in frontmatter_data and frontmatter_data['url']:
            url = frontmatter_data['url']
            # Ensure URL ends with / for consistency (unless it's a file with extension)
            if not url.endswith('/') and '.' not in url.split('/')[-1]:
                url += '/'
            return url
        
        # 2. Check for flow_id (special case for ai-flow-templates)
        if 'flow_id' in frontmatter_data:
            # Generate URL based on file path for flow templates
            relative_path = file_path.relative_to(self.content_dir)
            url_path = self.path_to_url(relative_path)
            return url_path
        
        # 3. Generate URL from file path as fallback
        relative_path = file_path.relative_to(self.content_dir)
        url_path = self.path_to_url(relative_path)
        
        return url_path
    
    def path_to_url(self, relative_path: Path) -> str:
        """Convert a file path to a URL"""
        # Remove .md extension
        path_str = str(relative_path)
        if path_str.endswith('.md'):
            path_str = path_str[:-3]
        
        # Handle index files
        if path_str.endswith('/_index'):
            path_str = path_str[:-7]  # Remove /_index
        elif path_str == '_index':
            path_str = ''  # Root index becomes empty (will become "/" below)
        elif path_str.endswith('/index'):
            path_str = path_str[:-6]  # Remove /index
        
        # Ensure path starts with /
        if not path_str.startswith('/'):
            path_str = '/' + path_str
        
        # Ensure path ends with / for directories (except root)
        if path_str != '/' and not path_str.endswith('/'):
            path_str += '/'
        
        return path_str
    
    def get_title_from_frontmatter(self, frontmatter_data: Dict) -> str:
        """Get title from frontmatter with priority: description > shortDescription > title"""
        
        # Priority 1: description
        if 'description' in frontmatter_data and frontmatter_data['description']:
            return str(frontmatter_data['description']).strip()
        
        # Priority 2: shortDescription
        if 'shortDescription' in frontmatter_data and frontmatter_data['shortDescription']:
            return str(frontmatter_data['shortDescription']).strip()
        
        # Priority 3: title
        if 'title' in frontmatter_data and frontmatter_data['title']:
            return str(frontmatter_data['title']).strip()
        
        # Fallback: empty string
        return ""
    
    def calculate_priority(self, file_path: Path, file_url: str) -> int:
        """Calculate priority based on page hierarchy and type
        
        Priority calculation:
        - Main pages (_index.md): Higher priority (10)
        - Root level pages: High priority (8)
        - Each directory level deeper: -2 priority
        - Minimum priority: 1
        """
        
        # Check if it's a main page (_index.md)
        if file_path.name == '_index.md':
            base_priority = 10
        else:
            base_priority = 8
        
        # Calculate depth by counting URL segments (excluding empty ones)
        url_segments = [segment for segment in file_url.strip('/').split('/') if segment]
        depth = len(url_segments)
        
        # Adjust priority based on depth
        priority = base_priority - (depth * 2)
        
        # Ensure minimum priority of 1
        return max(1, priority)
    
    def get_stats(self) -> Dict:
        """Get processing statistics"""
        return self.stats.copy()


def save_links_to_json(links: List[Dict], output_file: str) -> None:
    """Save links to JSON file in the expected format matching manual linkbuilding format"""
    
    # Ensure output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to proper format with capitalized field names
    formatted_links = []
    for link in links:
        # Calculate priority based on keyword length
        # Longer keywords get higher priority
        keyword_length = len(link['keyword'])
        base_priority = link.get('priority', 1)
        
        # Add bonus priority based on keyword length
        # For example: keywords with 10+ chars get +3, 20+ chars get +6, etc.
        length_bonus = keyword_length // 5  # +1 priority for every 5 characters
        
        # Final priority is base priority plus length bonus
        final_priority = base_priority + length_bonus
        
        formatted_links.append({
            "Keyword": link['keyword'],
            "URL": link['url'],
            "Title": link['title'],
            "Exact": False,  # Default for automatic links
            "Priority": final_priority
        })
    
    # Sort links by keyword for consistent output
    sorted_links = sorted(formatted_links, key=lambda x: x['Keyword'].lower())
    
    # Wrap in "keywords" object to match manual format
    output_data = {"keywords": sorted_links}
    
    # Write JSON file with proper formatting
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved {len(sorted_links)} links to {output_file}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Extract automatic links from Hugo markdown content',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract links from English content
  python extract_automatic_links.py --content-dir content/en/ --output data/linkbuilding/en_automatic.json
  
  # Extract links from Spanish content with base URL
  python extract_automatic_links.py --content-dir content/es/ --output data/linkbuilding/es_automatic.json --base-url "https://example.com"

Output Format:
  [
    {
      "keyword": "AI Tools",
      "url": "/ai-tools/",
      "title": "Browse AI Tools and Templates"
    },
    {
      "keyword": "machine learning",
      "url": "/blog/ml-guide/",
      "title": "Complete Guide to Machine Learning"
    }
  ]
        """
    )
    
    parser.add_argument('--content-dir', required=True,
                       help='Path to content directory (e.g., content/en/)')
    parser.add_argument('--output', required=True,
                       help='Output JSON file path (e.g., data/linkbuilding/en_automatic.json)')
    parser.add_argument('--base-url', default="",
                       help='Base URL for relative links (optional)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--dry-run', action='store_true',
                       help='Process files but do not write output')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Create extractor and process directory
        extractor = LinkExtractor(args.content_dir, args.base_url)
        links = extractor.process_directory()
        
        # Get statistics
        stats = extractor.get_stats()
        
        if not args.dry_run:
            # Save to output file
            save_links_to_json(links, args.output)
        else:
            logger.info("Dry run - no output file written")
        
        # Print summary
        print(f"\nProcessing Summary:")
        print(f"  Files processed: {stats['files_processed']}")
        print(f"  Files with keywords: {stats['files_with_keywords']}")
        print(f"  Total keywords extracted: {stats['total_keywords']}")
        print(f"  Errors encountered: {len(stats['errors'])}")
        
        if stats['errors'] and args.verbose:
            print(f"\nErrors:")
            for error in stats['errors'][:10]:  # Show first 10 errors
                print(f"  - {error}")
            if len(stats['errors']) > 10:
                print(f"  ... and {len(stats['errors']) - 10} more errors")
        
        if not args.dry_run:
            print(f"  Output written to: {args.output}")
        
        # Exit with appropriate code
        sys.exit(0 if len(stats['errors']) == 0 else 1)
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(2)


if __name__ == '__main__':
    main()