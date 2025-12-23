#!/usr/bin/env python3
"""
Generate AWS Amplify _redirects file from Hugo content URLs.
This script creates redirects from English URL patterns to localized URLs.
For example: /ar/academy/* -> /ar/الأكاديمية/*

AWS Amplify supports both JSON format (for console configuration) and 
_redirects file format (for ZIP deployments).
"""

import os
import json
import yaml
import re
from pathlib import Path
import toml_frontmatter as frontmatter  # Use robust TOML parser
import argparse

def extract_url_from_content(file_path):
    """Extract the URL field from Hugo content file frontmatter."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = frontmatter.load(f)
            return content.metadata.get('url')
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def scan_content_directory(hugo_root):
    """Scan all content directories and extract URL patterns."""
    content_dir = Path(hugo_root) / 'content'
    redirects = []
    
    # Process each language directory
    for lang_dir in content_dir.iterdir():
        if not lang_dir.is_dir() or lang_dir.name == 'en':
            continue
            
        lang_code = lang_dir.name
        
        # Find all _index.md files
        for index_file in lang_dir.rglob('_index.md'):
            localized_url = extract_url_from_content(index_file)
            
            if localized_url:
                # Get the actual directory path
                relative_path = index_file.relative_to(lang_dir).parent
                
                # Build the expected URL based on directory structure
                if relative_path == Path('.'):
                    # Root level _index.md
                    continue
                
                # Construct directory-based URL
                directory_parts = list(relative_path.parts)
                directory_url = f"/{lang_code}/" + "/".join(directory_parts) + "/"
                
                # Only create redirect if the localized URL differs from directory-based URL
                if directory_url != localized_url:
                    print(f"  Redirect needed: {directory_url} -> {localized_url}")
                    
                    # Create redirect for exact match
                    redirects.append({
                        "source": directory_url,
                        "target": localized_url,
                        "status": "301"
                    })
                    
                    # Create redirect for wildcard pattern (subpages)
                    redirects.append({
                        "source": f"{directory_url[:-1]}/*",
                        "target": f"{localized_url[:-1]}/:splat",
                        "status": "301"
                    })
    
    return redirects

def generate_redirects_file(redirects, output_path):
    """Generate _redirects file for AWS Amplify ZIP deployment."""
    with open(output_path, 'w', encoding='utf-8') as f:
        # Write header comment
        f.write("# AWS Amplify Redirects\n")
        f.write("# Generated from Hugo content URLs\n")
        f.write(f"# Total redirects: {len(redirects)}\n\n")
        
        # Write each redirect in the _redirects file format
        for redirect in redirects:
            # Format: source target status
            f.write(f"{redirect['source']} {redirect['target']} {redirect['status']}\n")
    
    print(f"Generated _redirects file with {len(redirects)} rules at {output_path}")

def generate_json_config(redirects, output_path):
    """Generate JSON configuration for AWS Amplify console."""
    # Format redirects for Amplify console JSON configuration
    amplify_redirects = []
    
    for redirect in redirects:
        # Convert wildcard syntax for JSON format
        source = redirect['source'].replace('/*', '/<*>')
        target = redirect['target'].replace('/:splat', '/<*>')
        
        amplify_redirects.append({
            "source": source,
            "target": target,
            "status": redirect['status'],
            "condition": None
        })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(amplify_redirects, f, indent=2, ensure_ascii=False)
    
    print(f"Generated JSON config with {len(amplify_redirects)} rules at {output_path}")

def generate_amplify_redirects(hugo_root, output_dir=None):
    """Generate AWS Amplify redirects in both formats."""
    print(f"Scanning content directory: {hugo_root}/content")
    print("Looking for differences between directory names and localized URLs...\n")
    
    redirects = scan_content_directory(hugo_root)
    
    if output_dir is None:
        output_dir = Path(hugo_root) / 'static'
    
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate _redirects file for ZIP deployment
    redirects_file_path = output_dir / '_redirects'
    generate_redirects_file(redirects, redirects_file_path)
    
    # Generate JSON config for console configuration
    json_path = output_dir / 'amplify-redirects.json'
    generate_json_config(redirects, json_path)
    
    return redirects

def main():
    parser = argparse.ArgumentParser(description='Generate AWS Amplify redirects from Hugo content')
    parser.add_argument('--hugo-root', 
                        default='/Users/viktorzeman/work/FlowHunt-hugo',
                        help='Path to Hugo root directory')
    parser.add_argument('--output-dir', 
                        help='Directory for output files (default: hugo_root/static)')
    parser.add_argument('--format',
                        choices=['both', 'file', 'json'],
                        default='both',
                        help='Output format: both, file (_redirects), or json')
    
    args = parser.parse_args()
    
    # Validate Hugo root exists
    hugo_root = Path(args.hugo_root)
    if not hugo_root.exists():
        print(f"Error: Hugo root directory not found: {hugo_root}")
        return 1
    
    # Generate redirects
    redirects = generate_amplify_redirects(hugo_root, args.output_dir)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"- Generated {len(redirects)} redirect rules")
    
    if len(redirects) > 0:
        print(f"- _redirects file: for ZIP deployment")
        print(f"- amplify-redirects.json: for console configuration")
        print("\nDeployment instructions:")
        print("1. For ZIP deployment: Include _redirects file in the root of your ZIP")
        print("2. For console config: Copy contents of amplify-redirects.json to Amplify console")
    else:
        print("\nNo redirects needed - all URLs match their directory structures!")
        # Clean up empty files if no redirects
        output_path = Path(args.output_dir) if args.output_dir else Path(hugo_root) / 'static'
        if (output_path / '_redirects').exists():
            (output_path / '_redirects').unlink()
        if (output_path / 'amplify-redirects.json').exists():
            (output_path / 'amplify-redirects.json').unlink()
    
    return 0

if __name__ == '__main__':
    exit(main())