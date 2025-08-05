#!/usr/bin/env python3
"""
Generate Translation URLs Map - Split by Folder

This script creates a mapping of URLs that differ from English for each folder.
It processes all content files in content/en/, identifies their URLs (either from frontmatter 
or from file path), then finds corresponding files in other language directories and only
stores URLs that differ from the English version.

Usage:
    python translation-urls.py

Output:
    Creates /data/translation_urls/[folder].json files with URLs that differ from English

Requirements:
    pip install python-frontmatter
"""

import os
import frontmatter
from frontmatter import TOMLHandler
import json
import argparse
from pathlib import Path
from collections import defaultdict

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate translation URLs mapping split by folder")
    parser.add_argument("--hugo-root", type=str, 
                        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
                        help="Hugo root directory (default: three levels up from script location)")
    parser.add_argument("--content-dir", type=str, default="content",
                        help="Content directory relative to Hugo root (default: content)")
    parser.add_argument("--output-dir", type=str, default="data/translation_urls",
                        help="Output directory relative to Hugo root (default: data/translation_urls)")
    return parser.parse_args()

def get_url_from_file(file_path, lang, relative_path):
    """Extract URL from a markdown file, either from frontmatter or derive from path."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            post = frontmatter.load(f, handler=TOMLHandler())

        # Check if URL is defined in frontmatter
        if 'url' in post.metadata:
            url = post.metadata['url']
            # Ensure URL ends with /
            if not url.endswith('/'):
                url = url + '/'
            return url
        
        # Derive URL from file path
        # Remove .md extension and handle _index.md files
        url_path = relative_path
        if url_path.endswith('.md'):
            url_path = url_path[:-3]
        
        # Handle _index.md files - they represent the directory URL
        if url_path.endswith('/_index'):
            url_path = url_path[:-7]  # Remove /_index (7 characters)
        elif url_path == '_index':
            url_path = ''  # Root index
        
        # Add language prefix for non-English languages
        if lang != 'en':
            # Ensure URL starts with /
            if url_path and not url_path.startswith('/'):
                url_path = '/' + url_path
            elif not url_path:
                url_path = '/'
            
            # Add language prefix
            url_path = f'/{lang}{url_path}'
        else:
            # For English, just ensure proper formatting
            if url_path and not url_path.startswith('/'):
                url_path = '/' + url_path
            elif not url_path:
                url_path = '/'
        
        # Ensure URL ends with /
        if not url_path.endswith('/'):
            url_path = url_path + '/'
        
        return url_path
        
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

def get_all_languages(content_dir):
    """Get list of all language directories."""
    languages = []
    if os.path.exists(content_dir):
        for item in os.listdir(content_dir):
            item_path = os.path.join(content_dir, item)
            if os.path.isdir(item_path) and not item.startswith('_'):
                languages.append(item)
    return sorted(languages)

def get_folder_name(relative_path):
    """Get the top-level folder name from a relative path."""
    parts = relative_path.split(os.sep)
    if len(parts) > 1:
        return parts[0]
    return "_root"  # Files in root directory

def process_content_by_folder(hugo_root, content_dir):
    """Process all content files and organize by folder."""
    languages = get_all_languages(os.path.join(hugo_root, content_dir))
    folders_data = defaultdict(lambda: defaultdict(dict))
    
    print(f"Found languages: {', '.join(languages)}")
    
    # First, collect all files for all languages
    all_files = defaultdict(dict)  # relative_path -> {lang: url}
    
    for lang in languages:
        lang_dir = os.path.join(hugo_root, content_dir, lang)
        if not os.path.exists(lang_dir):
            continue
            
        for root, dirs, files in os.walk(lang_dir):
            for file in files:
                if file.endswith('.md'):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, lang_dir)
                    
                    url = get_url_from_file(file_path, lang, relative_path)
                    if url:
                        all_files[relative_path][lang] = url
    
    # Now organize by folder and only store URLs different from English
    for relative_path, lang_urls in all_files.items():
        folder = get_folder_name(relative_path)
        en_url = lang_urls.get('en', '')
        
        # Only store URLs that differ from English (comparing path structure, not language prefix)
        different_urls = {}
        for lang, url in lang_urls.items():
            if lang != 'en':
                # Strip language prefix for comparison
                lang_prefix = f'/{lang}/'
                if url.startswith(lang_prefix):
                    # Remove language prefix
                    url_without_prefix = url[len(lang_prefix)-1:]  # Keep the leading /
                else:
                    url_without_prefix = url
                
                # Compare the path structure, not the full URL with language prefix
                if url_without_prefix != en_url:
                    different_urls[lang] = url
        
        # Only add to folder data if there are different URLs
        if different_urls:
            # Include English URL as reference for proper fallback calculation
            different_urls['en'] = en_url
            folders_data[folder][relative_path] = different_urls
    
    return folders_data

def generate_folder_files(folders_data, hugo_root, output_dir):
    """Generate JSON files for each folder."""
    output_path = os.path.join(hugo_root, output_dir)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    total_files = 0
    total_size = 0
    folder_stats = []
    
    print(f"\nGenerating translation URL files by folder:")
    print("=" * 60)
    
    for folder, files_data in sorted(folders_data.items()):
        if not files_data:
            continue
            
        # Generate JSON file for this folder
        folder_file = os.path.join(output_path, f"{folder}.json")
        
        with open(folder_file, 'w', encoding='utf-8') as f:
            json.dump(files_data, f, separators=(',', ':'), ensure_ascii=False)
        
        file_size = os.path.getsize(folder_file)
        total_size += file_size
        total_files += 1
        
        # Count unique URLs in this folder
        unique_urls_count = sum(len(urls) for urls in files_data.values())
        
        folder_stats.append({
            'folder': folder,
            'files': len(files_data),
            'unique_urls': unique_urls_count,
            'size': file_size
        })
        
        print(f"  📁 {folder}.json: {len(files_data)} files, {unique_urls_count} unique URLs ({file_size/1024:.1f} KB)")
    
    print("=" * 60)
    print(f"\n📊 Summary:")
    print(f"  Total folders: {total_files}")
    print(f"  Total size: {total_size/1024:.1f} KB")
    print(f"  Average size per folder: {total_size/1024/max(total_files, 1):.1f} KB")
    
    # Show top 5 largest folders
    if folder_stats:
        print(f"\n🔝 Top 5 largest folders:")
        sorted_stats = sorted(folder_stats, key=lambda x: x['size'], reverse=True)[:5]
        for stat in sorted_stats:
            print(f"  {stat['folder']}: {stat['size']/1024:.1f} KB ({stat['unique_urls']} unique URLs)")
    
    return total_files, total_size

def main():
    """Main function."""
    args = parse_args()
    
    print(f"Hugo root: {args.hugo_root}")
    print(f"Processing content files and splitting by folder...")
    
    # Process content organized by folder
    folders_data = process_content_by_folder(args.hugo_root, args.content_dir)
    
    if not folders_data:
        print("No translation differences found!")
        return
    
    # Generate output files
    total_files, total_size = generate_folder_files(folders_data, args.hugo_root, args.output_dir)
    
    print(f"\n✅ Translation URL generation complete!")
    print(f"   Generated {total_files} folder files")
    print(f"   Only storing URLs that differ from English")
    print(f"   This approach significantly reduces file sizes")

if __name__ == "__main__":
    main()