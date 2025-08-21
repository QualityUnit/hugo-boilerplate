attributes_to_sync = [
    'image',
    'skip_linkbuilding',
    'youtubeVideoID',
    'tags',
    'categories',
    'brands',
    'brandCompanyName',
    'contactAddress',
    'contactEmail',
    'contactPhone',
    'getSupportUrl',
    'brandUrl',
    'youtubeVideoID',
    'date',
    'homePageUrl',
    'images',
    'price',
    'ai-image-generator',
    'effects',
    'models',
    'characterImages',
    'originalCharacterImage',
    'monthlyPrice',
    'annualPrice',
    'ctaHoverBorderColor',
    'ctaBorderColor',
    'featureCategories.features.tiersComparison',
    'tiersComparison'
]

#unset_attributes = [ 'cards' ]
unset_attributes = [  ]

import os
import re
import toml
from pathlib import Path
import datetime

# Base content directory
content_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../content')))
en_content_dir = content_dir / 'en'

def extract_front_matter(file_path):
    """Extract TOML front matter from markdown file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Look for front matter between +++ delimiters
    match = re.match(r'^\+\+\+\s*\n*(.*?)\n*\+\+\+\s*\n*', content, re.DOTALL)
    if match:
        front_matter_text = match.group(1)
        try:
            front_matter = toml.loads(front_matter_text)
            remaining_content = content[match.end():]
            return front_matter, remaining_content
        except toml.TomlDecodeError as e:
            print(f"Error parsing front matter in {file_path}: {e}")
            raise e
    return {}, content

def update_front_matter(file_path, updated_front_matter, remaining_content):
    """Update TOML front matter in markdown file"""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('+++\n')
        f.write(toml.dumps(updated_front_matter))
        f.write('+++\n')
        f.write(remaining_content)
    
def process_file(en_file_path):
    """Process a single English file and update corresponding translations"""
    # Get relative path from content/en
    rel_path = en_file_path.relative_to(en_content_dir)
    
    # Extract front matter from English file
    try:
        en_front_matter, remaining_content = extract_front_matter(en_file_path)
    except toml.TomlDecodeError as e:
        print(f"!!!!! Error processing {en_file_path}: {e}")
        return

    if 'date' not in en_front_matter:
        en_front_matter['date'] = (datetime.datetime.now() - datetime.timedelta(hours=6)).strftime('%Y-%m-%d %H:%M:%S')
        update_front_matter(en_file_path, en_front_matter, remaining_content)

    # unset attributes we don't want
    for attr in unset_attributes:
        if attr in en_front_matter:
            del en_front_matter[attr]
            update_front_matter(en_file_path, en_front_matter, remaining_content)

    # Create a dictionary with only the attributes we want to sync
    sync_attributes = {}
    for attr in attributes_to_sync:
        if attr in en_front_matter:
            sync_attributes[attr] = en_front_matter[attr]
    
    if not sync_attributes:
        return  # Nothing to sync
    
    # Find corresponding files in other language directories
    for lang_dir in content_dir.iterdir():
        if lang_dir.is_dir() and lang_dir.name != 'en':
            translated_file = lang_dir / rel_path

            if translated_file.exists():
                # Extract front matter from translated file
                try:
                    translated_front_matter, remaining_content = extract_front_matter(translated_file)

                    # Update front matter with synced attributes
                    updated = False
                    for attr, value in sync_attributes.items():
                        if translated_front_matter.get(attr) != value:
                            translated_front_matter[attr] = value
                            updated = True

                    # Remove unset attributes
                    for attr in unset_attributes:
                        if attr in translated_front_matter:
                            del translated_front_matter[attr]
                            updated = True

                    if updated:
                        update_front_matter(translated_file, translated_front_matter, remaining_content)
                except Exception as e:
                    print(f"Error processing {translated_file}: {e}")
                    continue

def main():
    # Find all markdown files in the English content directory
    for file_path in en_content_dir.glob('**/*.md'):
        if file_path.is_file():
            process_file(file_path)
    
    print("Content attributes sync complete.")

if __name__ == "__main__":
    main()
