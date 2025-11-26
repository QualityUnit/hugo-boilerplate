#!/usr/bin/env python3
"""
Export FAQs from markdown files to CSV format.
Reads all *.md files in a directory and extracts FAQ data from frontmatter.
"""

import os
import sys
import csv
import glob
import yaml
import re

try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # Fallback for older Python
    except ImportError:
        import toml as tomllib  # Fallback to toml


def extract_frontmatter(content):
    """Extract YAML or TOML frontmatter from markdown content."""
    # Try TOML frontmatter first (Hugo uses +++ delimiters)
    toml_pattern = r'^\+\+\+\s*\n(.*?)\n\+\+\+\s*\n'
    toml_match = re.match(toml_pattern, content, re.DOTALL)

    if toml_match:
        try:
            frontmatter = tomllib.loads(toml_match.group(1))
            return frontmatter
        except Exception as e:
            print(f"Error parsing TOML: {e}", file=sys.stderr)
            return None

    # Try YAML frontmatter (--- delimiters)
    yaml_pattern = r'^---\s*\n(.*?)\n---\s*\n'
    yaml_match = re.match(yaml_pattern, content, re.DOTALL)

    if yaml_match:
        try:
            frontmatter = yaml.safe_load(yaml_match.group(1))
            return frontmatter
        except yaml.YAMLError as e:
            print(f"Error parsing YAML: {e}", file=sys.stderr)
            return None

    return None


def extract_faqs_from_file(file_path):
    """Extract FAQ items from a markdown file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        frontmatter = extract_frontmatter(content)

        if frontmatter and 'faq' in frontmatter:
            faqs = frontmatter['faq']
            if isinstance(faqs, list):
                return faqs
    except Exception as e:
        print(f"Error reading {file_path}: {e}", file=sys.stderr)

    return []


def process_directory(directory_path, output_file):
    """Process all markdown files in a directory and export FAQs to CSV."""
    # Find all .md files in the directory
    md_files = glob.glob(os.path.join(directory_path, '*.md'))

    if not md_files:
        print(f"No markdown files found in {directory_path}", file=sys.stderr)
        return

    all_faqs = []

    # Process each markdown file
    for md_file in md_files:
        faqs = extract_faqs_from_file(md_file)
        for faq_item in faqs:
            # Each FAQ item should have 'question' and 'answer' fields
            if isinstance(faq_item, dict):
                question = faq_item.get('question', '').strip()
                answer = faq_item.get('answer', '').strip()

                if question and answer:
                    # Combine question and answer separated by space
                    faq_text = f"{question} {answer}"
                    all_faqs.append(faq_text)

    # Write to CSV file
    if all_faqs:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['faq'])  # Header
            for faq in all_faqs:
                writer.writerow([faq])

        print(f"Exported {len(all_faqs)} FAQs to {output_file}")
    else:
        print(f"No FAQs found in {directory_path}", file=sys.stderr)


def main():
    if len(sys.argv) < 2:
        print("Usage: export-faqs.py <directory_path> [output_file]")
        sys.exit(1)

    directory_path = sys.argv[1]

    if not os.path.isdir(directory_path):
        print(f"Error: {directory_path} is not a valid directory", file=sys.stderr)
        sys.exit(1)

    # Default output file name based on directory name
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    else:
        dir_name = os.path.basename(os.path.normpath(directory_path))
        output_file = f"{dir_name}_faqs.csv"

    process_directory(directory_path, output_file)


if __name__ == "__main__":
    main()
