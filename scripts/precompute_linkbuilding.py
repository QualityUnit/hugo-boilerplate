#!/usr/bin/env python3
"""
Precompute Linkbuilding Keywords
Analyzes built HTML content to find which keywords are actually present,
then creates optimized linkbuilding files with only the relevant keywords.
This significantly speeds up deployment-time linkbuilding.
"""

import os
import sys
import json
import re
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple
from bs4 import BeautifulSoup
import concurrent.futures
from collections import defaultdict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class ContentAnalyzer:
    """Analyzes HTML content to find which keywords are present."""
    
    # Elements to skip when analyzing
    SKIP_TAGS = {
        'script', 'style', 'code', 'pre', 'meta', 'link', 'svg',
        'iframe', 'video', 'audio', 'canvas', 'map', 'area'
    }
    
    def __init__(self, html_dir: Path):
        self.html_dir = html_dir
        self.found_keywords = set()
        self.file_count = 0
        self.total_text_length = 0
    
    def analyze_file(self, file_path: Path, keywords: List[Tuple[str, re.Pattern]], 
                     already_found: Set[str]) -> Set[str]:
        """Analyze a single HTML file for keyword presence.
        
        Args:
            file_path: Path to HTML file to analyze
            keywords: List of (keyword, compiled_pattern) tuples
            already_found: Set of keywords already found (to skip searching for)
        
        Returns:
            Set of keywords found in this file
        """
        found_in_file = set()
        
        # Skip file if all keywords are already found
        if len(already_found) == len(keywords):
            return found_in_file
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse HTML
            soup = BeautifulSoup(content, 'html.parser')
            
            # Remove script and style elements
            for tag in self.SKIP_TAGS:
                for element in soup.find_all(tag):
                    element.decompose()
            
            # Get text content
            text = soup.get_text(separator=' ', strip=True)
            text_lower = text.lower()
            
            # Check each keyword using precompiled patterns
            # Skip keywords that were already found in previous files
            for keyword, pattern in keywords:
                if keyword not in already_found and pattern.search(text_lower):
                    found_in_file.add(keyword)
            
            self.total_text_length += len(text)
            
        except Exception as e:
            logger.warning(f"Error analyzing {file_path}: {e}")
        
        return found_in_file
    
    def analyze_directory(self, keywords: Dict[str, Dict], 
                         max_workers: int = 8) -> Set[str]:
        """Analyze all HTML files in directory for keyword presence.
        
        Uses an optimized approach that stops searching for keywords once they're found.
        Files are processed in batches with parallel processing within each batch.
        """
        # Find all HTML files
        html_files = list(self.html_dir.rglob('*.html'))
        self.file_count = len(html_files)
        
        logger.info(f"Analyzing {self.file_count} HTML files for keyword presence...")
        
        # Precompile regex patterns for all keywords (much faster)
        keyword_patterns = []
        for keyword in keywords:
            pattern = re.compile(r'\b' + re.escape(keyword.lower()) + r'\b')
            keyword_patterns.append((keyword, pattern))
        
        logger.info(f"  Compiled {len(keyword_patterns)} keyword patterns")
        
        # Process files with early stopping optimization
        all_found_keywords = set()
        batch_size = min(50, max_workers * 2)  # Process in batches
        completed = 0
        
        for batch_start in range(0, len(html_files), batch_size):
            batch_end = min(batch_start + batch_size, len(html_files))
            batch_files = html_files[batch_start:batch_end]
            
            # Check if we've found all keywords already
            if len(all_found_keywords) == len(keyword_patterns):
                logger.info(f"  All {len(keyword_patterns)} keywords found after {completed} files!")
                break
            
            # Process batch in parallel, passing already_found to each worker
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit analysis tasks with current already_found set
                future_to_file = {
                    executor.submit(self.analyze_file, file_path, keyword_patterns, 
                                  all_found_keywords.copy()): file_path
                    for file_path in batch_files
                }
                
                # Collect results from this batch
                for future in concurrent.futures.as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        found = future.result()
                        all_found_keywords.update(found)
                        completed += 1
                        
                        # Progress update
                        if completed % 100 == 0 or completed == self.file_count:
                            remaining_keywords = len(keyword_patterns) - len(all_found_keywords)
                            logger.info(f"  Processed {completed}/{self.file_count} files, "
                                       f"found {len(all_found_keywords)}/{len(keyword_patterns)} keywords "
                                       f"({remaining_keywords} remaining)")
                    except Exception as e:
                        logger.error(f"Error processing {file_path}: {e}")
                        completed += 1
        
        # Final progress if not already shown
        if completed % 100 != 0 and completed != self.file_count:
            logger.info(f"  Processed {completed}/{self.file_count} files, "
                       f"found {len(all_found_keywords)}/{len(keyword_patterns)} keywords")
        
        self.found_keywords = all_found_keywords
        return all_found_keywords


def load_linkbuilding_file(file_path: Path) -> Dict[str, Dict]:
    """Load keywords from linkbuilding JSON file."""
    keywords = {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle both formats
    if isinstance(data, dict) and 'keywords' in data:
        items = data['keywords']
    else:
        items = data if isinstance(data, list) else []
    
    for item in items:
        if isinstance(item, dict):
            # Try both field name formats
            keyword = item.get('Keyword', item.get('keyword', ''))
            url = item.get('URL', item.get('url', ''))
            title = item.get('Title', item.get('title', ''))
            priority = item.get('Priority', item.get('priority', 0))
            exact = item.get('Exact', item.get('exact', False))
            
            if keyword:
                keywords[keyword] = {
                    'url': url,
                    'title': title,
                    'priority': priority,
                    'exact': exact
                }
    
    return keywords


def save_optimized_linkbuilding(keywords: Dict[str, Dict], 
                               found_keywords: Set[str],
                               output_path: Path) -> Dict:
    """Save optimized linkbuilding file with only found keywords."""
    # Filter keywords to only those found
    optimized_items = []
    
    for keyword in found_keywords:
        if keyword in keywords:
            info = keywords[keyword]
            optimized_items.append({
                'Keyword': keyword,
                'URL': info['url'],
                'Title': info['title'],
                'Priority': info['priority'],
                'Exact': info['exact']
            })
    
    # Sort by priority (highest first) then by keyword
    optimized_items.sort(key=lambda x: (-x['Priority'], x['Keyword'].lower()))
    
    # Create output structure
    output_data = {
        'keywords': optimized_items,
        'metadata': {
            'original_keywords': len(keywords),
            'optimized_keywords': len(optimized_items),
            'reduction_percent': round((1 - len(optimized_items) / len(keywords)) * 100, 1) 
                                if keywords else 0
        }
    }
    
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    return output_data['metadata']


def process_language(lang: str, 
                    linkbuilding_dir: Path,
                    public_dir: Path,
                    output_dir: Path) -> Dict:
    """Process a single language."""
    logger.info(f"\nProcessing language: {lang}")
    
    # Determine paths
    manual_file = linkbuilding_dir / f"{lang}.json"
    automatic_file = linkbuilding_dir / f"{lang}_automatic.json"
    
    # English is at root, others in subdirectories
    if lang == 'en':
        html_dir = public_dir
    else:
        html_dir = public_dir / lang
    
    if not html_dir.exists():
        logger.warning(f"HTML directory not found: {html_dir}")
        return {}
    
    # Load all keywords (manual + automatic)
    all_keywords = {}
    
    if automatic_file.exists():
        auto_keywords = load_linkbuilding_file(automatic_file)
        all_keywords.update(auto_keywords)
        logger.info(f"  Loaded {len(auto_keywords)} automatic keywords")
    
    if manual_file.exists():
        manual_keywords = load_linkbuilding_file(manual_file)
        # Manual keywords override automatic ones
        for keyword, info in manual_keywords.items():
            info['priority'] += 10  # Boost manual priority
            all_keywords[keyword] = info
        logger.info(f"  Loaded {len(manual_keywords)} manual keywords")
    
    if not all_keywords:
        logger.warning(f"  No keywords found for {lang}")
        return {}
    
    logger.info(f"  Total keywords to check: {len(all_keywords)}")
    
    # Analyze content
    analyzer = ContentAnalyzer(html_dir)
    found_keywords = analyzer.analyze_directory(all_keywords)
    
    logger.info(f"  Found {len(found_keywords)} keywords in content")
    logger.info(f"  Analyzed {analyzer.file_count} files, "
               f"{analyzer.total_text_length:,} total characters")
    
    # Save optimized file
    output_file = output_dir / f"{lang}_optimized.json"
    metadata = save_optimized_linkbuilding(all_keywords, found_keywords, output_file)
    
    logger.info(f"  Saved optimized file: {output_file}")
    logger.info(f"  Reduction: {metadata['original_keywords']} → "
               f"{metadata['optimized_keywords']} "
               f"({metadata['reduction_percent']}% reduction)")
    
    # Return statistics
    return {
        'language': lang,
        'html_files': analyzer.file_count,
        'total_text_length': analyzer.total_text_length,
        'original_keywords': metadata['original_keywords'],
        'found_keywords': metadata['optimized_keywords'],
        'reduction_percent': metadata['reduction_percent'],
        'output_file': str(output_file)
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Precompute optimized linkbuilding files based on actual content',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script analyzes built HTML content to find which keywords are actually
present, then creates optimized linkbuilding files containing only those keywords.
This significantly reduces processing time during deployment.

Examples:
  # Process all languages
  python precompute_linkbuilding.py \\
    --linkbuilding-dir data/linkbuilding \\
    --public-dir public \\
    --output-dir data/linkbuilding/optimized
  
  # Process specific languages
  python precompute_linkbuilding.py \\
    --linkbuilding-dir data/linkbuilding \\
    --public-dir public \\
    --output-dir data/linkbuilding/optimized \\
    --languages en de fr
        """
    )
    
    parser.add_argument('--linkbuilding-dir', required=True,
                       help='Directory containing original linkbuilding files')
    parser.add_argument('--public-dir', required=True,
                       help='Public directory with built HTML files')
    parser.add_argument('--output-dir', required=True,
                       help='Output directory for optimized linkbuilding files')
    parser.add_argument('--languages', nargs='+',
                       help='Specific languages to process (default: all)')
    parser.add_argument('--max-workers', type=int, default=8,
                       help='Maximum parallel workers for analysis (default: 8)')
    
    args = parser.parse_args()
    
    # Convert paths
    linkbuilding_dir = Path(args.linkbuilding_dir)
    public_dir = Path(args.public_dir)
    output_dir = Path(args.output_dir)
    
    # Validate directories
    if not linkbuilding_dir.exists():
        logger.error(f"Linkbuilding directory not found: {linkbuilding_dir}")
        sys.exit(1)
    if not public_dir.exists():
        logger.error(f"Public directory not found: {public_dir}")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find languages to process
    if args.languages:
        languages = args.languages
    else:
        # Find all automatic linkbuilding files
        auto_files = list(linkbuilding_dir.glob('*_automatic.json'))
        languages = [f.stem.replace('_automatic', '') for f in auto_files]
    
    if not languages:
        logger.error("No languages found to process")
        sys.exit(1)
    
    logger.info(f"Found {len(languages)} languages to process: {', '.join(languages)}")
    logger.info("=" * 60)
    
    # Process each language
    results = []
    for lang in languages:
        try:
            stats = process_language(lang, linkbuilding_dir, public_dir, output_dir)
            if stats:
                results.append(stats)
        except Exception as e:
            logger.error(f"Error processing {lang}: {e}")
    
    # Generate summary report
    logger.info("\n" + "=" * 60)
    logger.info("PRECOMPUTATION SUMMARY")
    logger.info("=" * 60)
    
    total_original = sum(r['original_keywords'] for r in results)
    total_found = sum(r['found_keywords'] for r in results)
    total_files = sum(r['html_files'] for r in results)
    avg_reduction = sum(r['reduction_percent'] for r in results) / len(results) if results else 0
    
    logger.info(f"Languages processed: {len(results)}")
    logger.info(f"Total HTML files analyzed: {total_files:,}")
    logger.info(f"Total original keywords: {total_original:,}")
    logger.info(f"Total keywords found in content: {total_found:,}")
    logger.info(f"Average reduction: {avg_reduction:.1f}%")
    
    logger.info("\nPer-language results:")
    logger.info("-" * 40)
    for r in sorted(results, key=lambda x: x['reduction_percent'], reverse=True):
        logger.info(f"  {r['language'].upper():3} | "
                   f"{r['original_keywords']:6,} → {r['found_keywords']:5,} keywords "
                   f"({r['reduction_percent']:5.1f}% reduction)")
    
    # Save summary report
    summary_file = output_dir / 'precomputation_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            'languages': results,
            'summary': {
                'total_languages': len(results),
                'total_html_files': total_files,
                'total_original_keywords': total_original,
                'total_found_keywords': total_found,
                'average_reduction_percent': avg_reduction
            }
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\nSummary report saved to: {summary_file}")
    logger.info("\nOptimized linkbuilding files are ready for deployment!")
    
    sys.exit(0)


if __name__ == '__main__':
    main()