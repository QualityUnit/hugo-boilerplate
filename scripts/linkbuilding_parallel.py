#!/usr/bin/env python3
"""
Parallel Linkbuilding Runner
Executes linkbuilding.py for multiple languages in parallel for faster processing.
"""

import os
import sys
import json
import argparse
import subprocess
import concurrent.futures
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import time
import logging

# Try to import psutil for memory monitoring, but don't fail if it's not available
try:
    import psutil
    MEMORY_MONITORING = True
except ImportError:
    MEMORY_MONITORING = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def find_language_files(linkbuilding_dir: Path, public_dir: Path) -> List[Dict]:
    """Find all language configurations for linkbuilding.

    Automatically uses optimized keyword files when available (33x faster).
    Falls back to full automatic files if optimized versions don't exist.

    Returns a list of dicts with language info:
    {
        'lang': 'en',
        'manual_file': 'path/to/en.json',
        'automatic_file': 'path/to/en_optimized.json' or 'path/to/en_automatic.json',
        'html_dir': 'path/to/public/en/'
    }
    """
    languages = []
    optimized_dir = linkbuilding_dir / 'optimized'

    # Check if optimized directory exists
    use_optimized = optimized_dir.exists()

    if use_optimized:
        logger.info("✨ Found optimized keyword files - using fast mode (33x faster)")
        automatic_files = list(optimized_dir.glob('*_optimized.json'))
        file_pattern = '_optimized'
    else:
        logger.info("📝 Using full automatic keyword files (slower)")
        automatic_files = list(linkbuilding_dir.glob('*_automatic.json'))
        file_pattern = '_automatic'

    for auto_file in automatic_files:
        # Extract language code from filename (e.g., 'en' from 'en_automatic.json' or 'en_optimized.json')
        lang = auto_file.stem.replace(file_pattern, '')

        # Check for corresponding manual file (always in main linkbuilding dir, not optimized)
        manual_file = linkbuilding_dir / f"{lang}.json"
        if not manual_file.exists():
            manual_file = None
            logger.debug(f"No manual file found for language {lang} - will use automatic only")

        # Determine HTML directory
        # English content is at the root of public, other languages have subdirectories
        if lang == 'en':
            html_dir = public_dir
        else:
            html_dir = public_dir / lang

        if not html_dir.exists():
            # Check if ANY HTML files exist in the public directory for this language
            # Sometimes Hugo might place them differently
            potential_files = list(public_dir.glob(f"**/{lang}/*.html"))[:1]  # Check for at least one file
            if not potential_files:
                logger.warning(f"HTML directory not found for language {lang}: {html_dir} - skipping")
                continue
            logger.info(f"Found alternative location for {lang} content")

        languages.append({
            'lang': lang,
            'manual_file': str(manual_file) if manual_file else None,
            'automatic_file': str(auto_file),
            'html_dir': str(html_dir)
        })

    # If no languages were found, return at least English if it exists
    if not languages and (public_dir / "index.html").exists():
        logger.warning("No language directories found, but found English content at root")

        # Try optimized first, fall back to automatic
        if use_optimized:
            en_auto = optimized_dir / "en_optimized.json"
        else:
            en_auto = linkbuilding_dir / "en_automatic.json"

        en_manual = linkbuilding_dir / "en.json"

        if en_auto.exists():
            languages.append({
                'lang': 'en',
                'manual_file': str(en_manual) if en_manual.exists() else None,
                'automatic_file': str(en_auto),
                'html_dir': str(public_dir)
            })

    return languages


def run_linkbuilding(lang_config: Dict, 
                    script_path: str,
                    config_file: Optional[str] = None,
                    dry_run: bool = False,
                    extra_args: List[str] = None) -> Tuple[str, bool, str]:
    """Run linkbuilding.py for a single language.
    
    Returns: (language, success, output/error message)
    """
    lang = lang_config['lang']
    
    # Build command
    cmd = [sys.executable, script_path]
    
    # Add keyword files
    if lang_config['manual_file']:
        cmd.extend(['-k', lang_config['manual_file']])
    if lang_config['automatic_file']:
        cmd.extend(['-a', lang_config['automatic_file']])
    
    # Add directory
    cmd.extend(['-d', lang_config['html_dir']])
    
    # Add language parameter for progress reporting
    cmd.extend(['--language', lang.upper()])
    
    # For English, exclude other language directories
    if lang == 'en':
        # List of all language codes to exclude
        exclude_langs = ['ar', 'cs', 'da', 'de', 'es', 'fi', 'fr', 'it', 'ja', 'ko',
                        'nl', 'no', 'pl', 'pt', 'ro', 'sk', 'sv', 'tr', 'vi', 'zh']
        cmd.extend(['--exclude'] + exclude_langs)
    
    # Don't add output report file parameter - reports will go to stdout only
    
    # Add optional arguments
    if config_file:
        cmd.extend(['-c', config_file])
    if dry_run:
        cmd.append('--dry-run')
    if extra_args:
        cmd.extend(extra_args)
    
    # Log the command
    logger.info(f"[{lang}] Starting linkbuilding: {' '.join(cmd)}")
    
    try:
        # Run the command with real-time output
        start_time = time.time()
        
        # Use Popen to stream output in real-time
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )
        
        # Collect output while also printing progress messages
        output_lines = []
        for line in process.stdout:
            line = line.rstrip()
            output_lines.append(line)
            
            # Print progress messages in real-time
            if 'Processed' in line and 'files' in line:
                logger.info(line)
        
        # Wait for process to complete
        process.wait()
        elapsed = time.time() - start_time
        
        if process.returncode == 0:
            # Parse summary from output
            
            # Extract detailed stats from output
            stats = {
                'elapsed': elapsed,
                'links_added': 0,
                'files_modified': 0,
                'files_processed': 0,
                'keywords_used': 0
            }
            
            for line in output_lines:
                if 'Links added:' in line:
                    try:
                        stats['links_added'] = int(line.split(':')[-1].strip())
                    except:
                        pass
                elif 'Files modified:' in line:
                    try:
                        stats['files_modified'] = int(line.split(':')[-1].strip())
                    except:
                        pass
                elif 'Files processed:' in line:
                    try:
                        stats['files_processed'] = int(line.split(':')[-1].strip())
                    except:
                        pass
                elif 'Keywords used:' in line:
                    try:
                        stats['keywords_used'] = int(line.split(':')[-1].strip())
                    except:
                        pass
            
            # No JSON report files are generated anymore, stats come from stdout
            
            summary = (f"[{lang}] ✓ {stats['links_added']} links added, "
                      f"{stats['files_modified']} files modified "
                      f"({elapsed:.1f}s)")
            
            logger.info(summary)
            return (lang, True, summary, stats)
        else:
            # Read stderr if there's an error
            stderr_output = process.stderr.read() if process.stderr else ""
            error_msg = f"[{lang}] Failed: {stderr_output or 'Unknown error'}"
            logger.error(error_msg)
            return (lang, False, error_msg, {})
            
    except Exception as e:
        error_msg = f"[{lang}] Error: {str(e)}"
        logger.error(error_msg)
        return (lang, False, error_msg, {})


def main():
    """Main entry point for parallel linkbuilding."""
    parser = argparse.ArgumentParser(
        description='Run linkbuilding.py in parallel for all languages',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run linkbuilding for all languages in parallel
  python linkbuilding_parallel.py --linkbuilding-dir data/linkbuilding --public-dir public
  
  # Run with custom configuration
  python linkbuilding_parallel.py --linkbuilding-dir data/linkbuilding --public-dir public -c config.json
  
  # Dry run to see what would be changed
  python linkbuilding_parallel.py --linkbuilding-dir data/linkbuilding --public-dir public --dry-run
  
  # Run with limited parallelism
  python linkbuilding_parallel.py --linkbuilding-dir data/linkbuilding --public-dir public --max-workers 4
        """
    )
    
    parser.add_argument('--linkbuilding-dir', required=True,
                       help='Directory containing linkbuilding JSON files')
    parser.add_argument('--public-dir', required=True,
                       help='Public directory with HTML files to process')
    parser.add_argument('--script-path', 
                       default='linkbuilding.py',
                       help='Path to linkbuilding.py script (default: linkbuilding.py)')
    parser.add_argument('-c', '--config',
                       help='Configuration file for linkbuilding')
    parser.add_argument('--dry-run', action='store_true',
                       help='Analyze without modifying files')
    parser.add_argument('--max-workers', type=int, default=4,
                       help='Maximum number of parallel workers (default: 4, reduce if memory issues occur)')
    parser.add_argument('--languages', nargs='+',
                       help='Specific languages to process (default: all)')
    parser.add_argument('--exclude-languages', nargs='+',
                       help='Languages to exclude from processing')
    parser.add_argument('--max-links', type=int,
                       help='Override max links per page')
    parser.add_argument('--max-keyword', type=int,
                       help='Override max replacements per keyword')
    parser.add_argument('--max-url', type=int,
                       help='Override max replacements per URL')
    
    args = parser.parse_args()
    
    # Convert paths
    linkbuilding_dir = Path(args.linkbuilding_dir)
    public_dir = Path(args.public_dir)
    
    # Log current working directory and paths for debugging
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Linkbuilding directory path: {linkbuilding_dir.absolute()}")
    logger.info(f"Public directory path: {public_dir.absolute()}")
    
    # Validate directories
    if not linkbuilding_dir.exists():
        logger.error(f"ERROR: Linkbuilding directory not found: {linkbuilding_dir.absolute()}")
        logger.error(f"Expected to find linkbuilding data files at this location")
        logger.error(f"Please ensure the linkbuilding data generation step has been completed")
        sys.exit(1)
    else:
        # Log what we found in the directory
        json_files = list(linkbuilding_dir.glob('*.json'))
        logger.info(f"Found {len(json_files)} JSON files in linkbuilding directory")
        if json_files:
            logger.info(f"Available files: {', '.join(f.name for f in json_files[:5])}...")
    
    if not public_dir.exists():
        logger.error(f"ERROR: Public directory not found: {public_dir.absolute()}")
        logger.error(f"Expected to find Hugo's generated HTML files at this location")
        logger.error(f"Please ensure Hugo build has completed successfully before running linkbuilding")
        sys.exit(1)
    else:
        # Check if there are any HTML files
        html_count = len(list(public_dir.rglob('*.html')))
        logger.info(f"Found {html_count} HTML files in public directory")
    
    # Find script path
    logger.info(f"Looking for linkbuilding.py script...")
    if not Path(args.script_path).exists():
        # Try in same directory as this script
        script_dir = Path(__file__).parent
        script_path = script_dir / 'linkbuilding.py'
        if not script_path.exists():
            logger.error(f"ERROR: linkbuilding.py script not found")
            logger.error(f"  Checked: {Path(args.script_path).absolute()}")
            logger.error(f"  Checked: {script_path.absolute()}")
            logger.error(f"Please ensure linkbuilding.py exists in the scripts directory")
            sys.exit(1)
        else:
            logger.info(f"Found linkbuilding.py at: {script_path.absolute()}")
    else:
        script_path = Path(args.script_path)
        logger.info(f"Using linkbuilding.py at: {script_path.absolute()}")
    
    # Find all language configurations
    logger.info("Discovering language configurations...")
    languages = find_language_files(linkbuilding_dir, public_dir)
    
    if not languages:
        logger.error("ERROR: No language configurations found")
        logger.error(f"  Linkbuilding directory: {linkbuilding_dir.absolute()}")
        logger.error(f"  Public directory: {public_dir.absolute()}")
        logger.error("Possible causes:")
        logger.error("  1. Hugo hasn't built the public directory yet")
        logger.error("  2. Linkbuilding data files (*_automatic.json) are missing")
        logger.error("  3. Language directories don't match between data and public folders")
        
        # Show what's actually in the directories for debugging
        auto_files = list(linkbuilding_dir.glob('*_automatic.json'))
        if auto_files:
            logger.error(f"Found automatic files: {', '.join(f.name for f in auto_files)}")
        else:
            logger.error("No *_automatic.json files found in linkbuilding directory")
        
        # Check public directory structure
        subdirs = [d for d in public_dir.iterdir() if d.is_dir()]
        if subdirs:
            logger.error(f"Public subdirectories: {', '.join(d.name for d in subdirs[:10])}")
        
        sys.exit(1)
    
    # Filter languages if specified
    if args.languages:
        languages = [l for l in languages if l['lang'] in args.languages]
    if args.exclude_languages:
        languages = [l for l in languages if l['lang'] not in args.exclude_languages]
    
    logger.info(f"Found {len(languages)} languages to process: {', '.join(l['lang'] for l in languages)}")
    
    # Build extra arguments
    extra_args = []
    # Override with reasonable limits if not specified
    if args.max_links:
        extra_args.extend(['--max-links', str(args.max_links)])
    else:
        # Default to reasonable limit
        extra_args.extend(['--max-links', '15'])
    
    if args.max_keyword:
        extra_args.extend(['--max-keyword', str(args.max_keyword)])
    else:
        # Default to reasonable limit
        extra_args.extend(['--max-keyword', '1'])
    
    if args.max_url:
        extra_args.extend(['--max-url', str(args.max_url)])
    else:
        # Default to reasonable limit
        extra_args.extend(['--max-url', '3'])
    
    # Run linkbuilding in parallel
    logger.info(f"Starting parallel linkbuilding with {args.max_workers} workers...")
    
    # Report initial memory usage if available
    if MEMORY_MONITORING:
        process = psutil.Process()
        mem_info = process.memory_info()
        logger.info(f"Initial memory usage: {mem_info.rss / 1024 / 1024:.1f} MB")
    
    start_time = time.time()
    
    results = []
    # Use ThreadPoolExecutor for better memory efficiency
    # Threads share memory, which is more efficient for I/O-bound tasks like file processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(
                run_linkbuilding,
                lang_config,
                str(script_path),
                args.config,
                args.dry_run,
                extra_args
            ): lang_config['lang']
            for lang_config in languages
        }
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(futures):
            lang = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"[{lang}] Unexpected error: {str(e)}")
                results.append((lang, False, str(e), {}))
    
    # Calculate statistics
    elapsed = time.time() - start_time
    successful = [r for r in results if r[1]]
    failed = [r for r in results if not r[1]]
    
    # Calculate totals
    total_links = sum(r[3].get('links_added', 0) for r in successful)
    total_files_modified = sum(r[3].get('files_modified', 0) for r in successful)
    total_files_processed = sum(r[3].get('files_processed', 0) for r in successful)
    
    # Print summary
    logger.info("=" * 60)
    logger.info(f"🎯 PARALLEL LINKBUILDING COMPLETED")
    logger.info("=" * 60)
    logger.info(f"⏱️  Total time: {elapsed:.1f} seconds")
    logger.info(f"✅ Languages processed: {len(successful)}/{len(results)}")
    logger.info(f"🔗 Total links added: {total_links:,}")
    logger.info(f"📄 Total files modified: {total_files_modified:,}")
    logger.info(f"📁 Total files processed: {total_files_processed:,}")
    
    # Report final memory usage if available
    if MEMORY_MONITORING:
        process = psutil.Process()
        mem_info = process.memory_info()
        logger.info(f"💾 Final memory usage: {mem_info.rss / 1024 / 1024:.1f} MB")
    
    if successful:
        logger.info("\n📊 Per-Language Statistics:")
        logger.info("-" * 60)
        logger.info("  Lang | Links Added | Files Mod | Keywords | Time")
        logger.info("-" * 60)
        for result in sorted(successful, key=lambda x: x[3].get('links_added', 0), reverse=True):
            lang = result[0]
            stats = result[3]
            logger.info(f"  {lang.upper():4} | {stats.get('links_added', 0):11,} | "
                       f"{stats.get('files_modified', 0):9,} | "
                       f"{stats.get('keywords_used', 0):8,} | "
                       f"{stats.get('elapsed', 0):5.1f}s")
        logger.info("-" * 60)
        
        # Show aggregate top keywords if we have detailed stats
        all_keywords = {}
        for result in successful:
            if 'detailed' in result[3] and 'keywords' in result[3]['detailed']:
                for keyword, count in result[3]['detailed']['keywords'].items():
                    all_keywords[keyword] = all_keywords.get(keyword, 0) + count
        
        if all_keywords:
            logger.info("\n🔝 Top 10 Keywords Across All Languages:")
            logger.info("-" * 40)
            top_keywords = sorted(all_keywords.items(), key=lambda x: x[1], reverse=True)[:10]
            for keyword, count in top_keywords:
                logger.info(f"  {count:5,} - {keyword}")
            logger.info("-" * 40)
    
    if failed:
        logger.error("\n❌ Failed languages:")
        for lang, _, msg, _ in failed:
            logger.error(f"  ✗ {msg}")
    
    # Don't generate combined report files - all output goes to console
    
    # Exit with appropriate code - but be more lenient
    # Only exit with error if ALL languages failed
    if len(successful) == 0 and len(failed) > 0:
        logger.error("All languages failed during linkbuilding")
        sys.exit(1)
    elif len(failed) > 0:
        logger.warning(f"Partial success: {len(successful)} languages succeeded, {len(failed)} failed")
        sys.exit(0)  # Exit successfully to not block deployment
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()