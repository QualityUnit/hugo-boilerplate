#!/usr/bin/env python3
"""
Theme-level script to sync translation URLs with directory structure.
Ensures that files in subdirectories have proper URL paths that match their directory structure.

For example, if content/sk/about/_index.md has url = "/o-nas/", 
then all files in that directory should have URLs like "/o-nas/filename/"

This is a generic theme script that can be used across different Hugo projects.
"""

import os
import re
try:
    import tomllib
except ImportError:
    try:
        import toml as tomllib
    except ImportError:
        import tomli as tomllib
from pathlib import Path
from typing import Dict, Optional, Tuple
import sys
import argparse

def get_hugo_config(hugo_root: Path) -> Dict:
    """Read Hugo configuration to get language settings"""
    config = {}
    
    # Try to read the main hugo.toml config file
    config_paths = [
        hugo_root / 'config' / '_default' / 'hugo.toml',
        hugo_root / 'config.toml',
        hugo_root / 'hugo.toml'
    ]
    
    for config_path in config_paths:
        if config_path.exists():
            try:
                with open(config_path, 'rb') as f:
                    config = tomllib.load(f)
                    break
            except tomllib.TOMLDecodeError as e:
                print(f"Warning: Error parsing config at {config_path}: {e}")
    
    # Also read languages.toml to get baseURLs for each language
    languages_paths = [
        hugo_root / 'config' / '_default' / 'languages.toml',
        hugo_root / 'languages.toml'
    ]
    
    for languages_path in languages_paths:
        if languages_path.exists():
            try:
                with open(languages_path, 'rb') as f:
                    languages_config = tomllib.load(f)
                    # Languages are at the root level of the languages.toml file
                    config['languages'] = languages_config
                    break
            except tomllib.TOMLDecodeError as e:
                print(f"Warning: Error parsing languages config at {languages_path}: {e}")
    
    return config

def get_content_dir(hugo_root: Optional[str] = None) -> Path:
    """Get the content directory path"""
    if hugo_root:
        return Path(hugo_root) / 'content'
    
    # Try to find content dir relative to script location
    # This script is in themes/boilerplate/scripts/
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    
    # Check if we're in a theme directory
    if 'themes' in script_dir.parts:
        # Go up to the Hugo root
        theme_index = script_dir.parts.index('themes')
        hugo_root = Path(*script_dir.parts[:theme_index])
        return hugo_root / 'content'
    
    # Fallback to relative path
    return Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../content')))

# Language-specific directory name translations
# Cache for directory URL lookups
DIRECTORY_URL_CACHE = {}


def extract_front_matter(file_path: Path) -> Tuple[str, Dict, str, str]:
    """Extract TOML front matter from markdown file
    Returns: (original_front_matter_text, parsed_dict, remaining_content, full_content)
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Look for front matter between +++ delimiters
    match = re.match(r'^\+\+\+\s*\n*(.*?)\n*\+\+\+\s*\n*', content, re.DOTALL)
    if match:
        front_matter_text = match.group(1)
        try:
            front_matter = tomllib.loads(front_matter_text)
            remaining_content = content[match.end():]
            return front_matter_text, front_matter, remaining_content, content
        except tomllib.TOMLDecodeError as e:
            print(f"Error parsing front matter in {file_path}: {e}")
            return "", {}, content, content
    return "", {}, content, content


def update_front_matter_url_only(file_path: Path, original_toml: str, new_url: str, remaining_content: str):
    """Update only the URL field in the TOML front matter, preserving all other formatting"""
    # Use regex to replace just the URL line in the original TOML
    url_pattern = r'^url\s*=\s*"[^"]*"'

    # Check if URL exists in the original TOML
    if re.search(url_pattern, original_toml, re.MULTILINE):
        # Replace existing URL
        updated_toml = re.sub(url_pattern, f'url = "{new_url}"', original_toml, flags=re.MULTILINE)
    else:
        # Add URL at the beginning (right after title if it exists)
        lines = original_toml.split('\n')
        new_lines = []
        url_inserted = False

        for line in lines:
            new_lines.append(line)
            # Insert url after title line
            if not url_inserted and line.strip().startswith('title ='):
                new_lines.append(f'url = "{new_url}"')
                url_inserted = True

        # If no title found, insert at the beginning
        if not url_inserted:
            new_lines.insert(0, f'url = "{new_url}"')

        updated_toml = '\n'.join(new_lines)

    # Write the updated content
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('+++\n')
        f.write(updated_toml)
        f.write('\n+++\n')
        f.write(remaining_content)


def get_directory_url_slug(directory_path: Path) -> str:
    """Get the URL slug for a directory by checking its _index.md file or using the directory name"""
    # Check if directory has _index.md file
    index_file = directory_path / '_index.md'
    
    if index_file.exists():
        # Extract URL from _index.md
        _, front_matter, _, _ = extract_front_matter(index_file)
        if 'url' in front_matter:
            url = front_matter['url'].strip('/')
            # Extract the last part of the URL as the slug
            if '/' in url:
                return url.split('/')[-1]
            return url
    
    # Fallback to directory name
    return directory_path.name


def get_directory_url_path(directory_path: Path, language: str, hugo_config: Dict) -> Optional[str]:
    """Get the URL path for a directory by checking its _index.md file or predicting from directory name"""
    # Check if languages have different domains
    languages = hugo_config.get('languages', {})
    has_custom_domains = False
    
    if languages:
        # Get baseURLs for all languages
        base_urls = set()
        for lang_code, lang_config in languages.items():
            if 'baseURL' in lang_config:
                # Extract domain from baseURL
                from urllib.parse import urlparse
                parsed_url = urlparse(lang_config['baseURL'])
                base_urls.add(parsed_url.netloc)
        
        # If we have different domains for different languages, don't use language prefix
        has_custom_domains = len(base_urls) > 1
    
    # Determine if we need to add language prefix
    default_lang = hugo_config.get('defaultContentLanguage', 'en')
    default_in_subdir = hugo_config.get('defaultContentLanguageInSubdir', True)
    
    # Language prefix is needed only if:
    # 1. Languages DON'T have custom domains, AND
    # 2. Either it's not the default language, OR defaultContentLanguageInSubdir is true
    if has_custom_domains:
        # Each language has its own domain, no language prefix needed
        needs_lang_prefix = False
    else:
        # Languages share the same domain, use subdirectories
        needs_lang_prefix = (language != default_lang) or (language == default_lang and default_in_subdir)
    
    # Always build the URL from the directory structure for consistency
    # Get the content directory to calculate relative path
    content_dir = get_content_dir()
    lang_dir = content_dir / language
    
    # Get the relative path from the language directory
    relative_path = directory_path.relative_to(lang_dir)
    
    # Build the URL path by getting URL slugs for each directory
    path_parts = []
    current_path = lang_dir
    for part in relative_path.parts:
        current_path = current_path / part
        url_slug = get_directory_url_slug(current_path)
        path_parts.append(url_slug)
    
    # Construct the final URL
    if needs_lang_prefix:
        return f"/{language}/{'/'.join(path_parts)}"
    else:
        return f"/{'/'.join(path_parts)}"


def ensure_trailing_slash(url: str) -> str:
    """Ensure URL ends with a trailing slash"""
    if not url.endswith('/'):
        return url + '/'
    return url


def process_directory(lang_dir: Path, directory: Path, stats: Dict, hugo_config: Dict, dry_run: bool = False, verbose: bool = False):
    """Process all files in a directory and fix their URLs"""
    language = lang_dir.name
    relative_path = directory.relative_to(lang_dir)
    
    # Get the expected base URL for this directory
    base_url = get_directory_url_path(directory, language, hugo_config)
    
    if not base_url:
        return
    
    if verbose or dry_run:
        print(f"\nProcessing {language}/{relative_path}: base URL = {base_url}")
    
    # Process all .md files in the directory
    for file_path in directory.glob('*.md'):
        if file_path.name == '_index.md':
            # Ensure _index.md has the correct URL
            original_toml, front_matter, remaining_content, _ = extract_front_matter(file_path)
            
            if 'url' not in front_matter:
                # Add the URL if missing
                new_url = ensure_trailing_slash(base_url)
                if not dry_run:
                    update_front_matter_url_only(file_path, original_toml, new_url, remaining_content)
                if verbose or dry_run:
                    content_dir = get_content_dir()
                    rel_file_path = file_path.relative_to(content_dir.parent)
                    print(f"  {'[DRY-RUN] Would add' if dry_run else 'Added'} URL to {rel_file_path}")
                    print(f"    New URL: {new_url}")
                stats['urls_added'] += 1
            else:
                # Check if URL needs fixing
                current_url = front_matter.get('url', '')
                expected_url = ensure_trailing_slash(base_url)
                
                if current_url != expected_url:
                    # Update if the URL doesn't match the expected URL
                    if not dry_run:
                        update_front_matter_url_only(file_path, original_toml, expected_url, remaining_content)
                    if verbose or dry_run:
                        content_dir = get_content_dir()
                        rel_file_path = file_path.relative_to(content_dir.parent)
                        print(f"  {'[DRY-RUN] Would fix' if dry_run else 'Fixed'} {rel_file_path}")
                        print(f"    Current URL: {current_url}")
                        print(f"    New URL:     {expected_url}")
                    stats['urls_fixed'] += 1
                else:
                    stats['urls_correct'] += 1
        else:
            # Process regular files
            original_toml, front_matter, remaining_content, _ = extract_front_matter(file_path)
            
            if 'url' in front_matter:
                current_url = front_matter['url']
                
                # Extract the filename part from the current URL
                url_parts = current_url.rstrip('/').split('/')
                if url_parts:
                    filename_part = url_parts[-1]
                    
                    # Build the correct URL with the base path
                    expected_url = f"{base_url}/{filename_part}/"
                    
                    if current_url != expected_url:
                        if not dry_run:
                            update_front_matter_url_only(file_path, original_toml, expected_url, remaining_content)
                        if verbose or dry_run:
                            content_dir = get_content_dir()
                            rel_file_path = file_path.relative_to(content_dir.parent)
                            print(f"  {'[DRY-RUN] Would fix' if dry_run else 'Fixed'} {rel_file_path}")
                            print(f"    Current URL: {current_url}")
                            print(f"    New URL:     {expected_url}")
                        stats['urls_fixed'] += 1
                    else:
                        stats['urls_correct'] += 1
            else:
                # Add URL if missing
                # Use the filename without extension as the URL slug
                filename_slug = file_path.stem
                new_url = f"{base_url}/{filename_slug}/"
                if not dry_run:
                    update_front_matter_url_only(file_path, original_toml, new_url, remaining_content)
                if verbose or dry_run:
                    content_dir = get_content_dir()
                    rel_file_path = file_path.relative_to(content_dir.parent)
                    print(f"  {'[DRY-RUN] Would add' if dry_run else 'Added'} URL to {rel_file_path}")
                    print(f"    New URL: {new_url}")
                stats['urls_added'] += 1


def main():
    """Main function to process all translation directories"""
    parser = argparse.ArgumentParser(
        description='Sync translation URLs with directory structure in Hugo content'
    )
    parser.add_argument(
        '--hugo-root',
        help='Path to Hugo root directory (default: auto-detect)',
        default=None
    )
    parser.add_argument(
        '--languages',
        help='Comma-separated list of languages to process (default: all except en)',
        default=None
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without modifying files'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed output'
    )
    
    args = parser.parse_args()
    
    # Get content directory
    content_dir = get_content_dir(args.hugo_root)
    
    if not content_dir.exists():
        print(f"Error: Content directory not found at {content_dir}")
        sys.exit(1)
    
    # Get Hugo root directory (parent of content dir)
    hugo_root = content_dir.parent if not args.hugo_root else Path(args.hugo_root)
    
    # Read Hugo configuration
    hugo_config = get_hugo_config(hugo_root)
    
    print(f"Processing content directory: {content_dir}")
    
    # Show config info if verbose
    if args.verbose:
        default_lang = hugo_config.get('defaultContentLanguage', 'en')
        default_in_subdir = hugo_config.get('defaultContentLanguageInSubdir', True)
        print(f"Default language: {default_lang}")
        print(f"Default language in subdir: {default_in_subdir}")
    
    stats = {
        'urls_fixed': 0,
        'urls_added': 0,
        'urls_correct': 0,
        'directories_processed': 0
    }
    
    # Determine which languages to process
    if args.languages:
        languages_to_process = args.languages.split(',')
    else:
        languages_to_process = None
    
    # Process each language directory
    for lang_dir in content_dir.iterdir():
        if lang_dir.is_dir() and lang_dir.name != 'en':  # Skip English
            # Check if we should process this language
            if languages_to_process and lang_dir.name not in languages_to_process:
                continue
                
            print(f"\n=== Processing language: {lang_dir.name} ===")
            
            # Process subdirectories
            for subdir in lang_dir.iterdir():
                if subdir.is_dir() and not subdir.name.startswith('.'):
                    process_directory(lang_dir, subdir, stats, hugo_config, dry_run=args.dry_run, verbose=args.verbose)
                    stats['directories_processed'] += 1
                    
                    # Also process nested subdirectories (e.g., about/team)
                    for nested_dir in subdir.iterdir():
                        if nested_dir.is_dir() and not nested_dir.name.startswith('.'):
                            process_directory(lang_dir, nested_dir, stats, hugo_config, dry_run=args.dry_run, verbose=args.verbose)
                            stats['directories_processed'] += 1
    
    # Print summary
    print("\n=== Summary ===")
    print(f"Directories processed: {stats['directories_processed']}")
    print(f"URLs fixed: {stats['urls_fixed']}")
    print(f"URLs added: {stats['urls_added']}")
    print(f"URLs already correct: {stats['urls_correct']}")
    
    if args.dry_run:
        print("\nDRY RUN - No files were modified")
    else:
        print("\nTranslation URL sync complete.")


if __name__ == "__main__":
    main()