#!/usr/bin/env python3
"""
translate_missing_files.py

This script translates files from /content/en/* to all language variations defined in /content/[lang]/
that don't already exist in the target language directories. If a file already exists in the target
language directory, it will be skipped.

Usage:
    python translate_missing_files.py [--path /path/to/content] [--max-concurrent 10] [--model o3-mini]

Prerequisites:
    - Python 3.6 or higher
    - OpenAI API key (set in .env file or as environment variable OPENAI_API_KEY)
    - Required packages: openai, tqdm, python-dotenv

Examples:
    # Basic usage (will use ../content/ relative to the script location)
    python translate_missing_files.py
    
    # With explicit path
    python translate_missing_files.py --path /Users/username/work/hugo-boilerplate/content
    
    # With API key as environment variable
    export OPENAI_API_KEY="your-api-key"
    python translate_missing_files.py
    
    # With different model
    python translate_missing_files.py --model gpt-3.5-turbo
"""

import os
import sys
import argparse
import openai
import concurrent.futures
import time
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
script_dir = os.path.dirname(os.path.abspath(__file__))
hugo_root = os.path.dirname(script_dir)
env_path = os.path.join(script_dir, '.env')
if os.path.exists(env_path):
    print(f"Loading environment variables from {env_path}")
    load_dotenv(env_path)
else:
    print("No .env file found, using environment variables if available")

# Set your OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    print("Error: OPENAI_API_KEY not found in environment variables or .env file")
    print("Please set the OPENAI_API_KEY environment variable or add it to the .env file")
    sys.exit(1)

# Map of folder names to full language names
LANGUAGE_MAP = {
    # ISO 639-1 language codes
    'af': 'Afrikaans',
    'ar': 'Arabic',
    'bg': 'Bulgarian',
    'bn': 'Bengali',
    'ca': 'Catalan',
    'cs': 'Czech',
    'da': 'Danish',
    'de': 'German',
    'el': 'Greek',
    'en': 'English',
    'es': 'Spanish',
    'et': 'Estonian',
    'fa': 'Persian',
    'fi': 'Finnish',
    'fr': 'French',
    'he': 'Hebrew',
    'hi': 'Hindi',
    'hr': 'Croatian',
    'hu': 'Hungarian',
    'id': 'Indonesian',
    'is': 'Icelandic',
    'it': 'Italian',
    'ja': 'Japanese',
    'ko': 'Korean',
    'lt': 'Lithuanian',
    'lv': 'Latvian',
    'ms': 'Malay',
    'nl': 'Dutch',
    'no': 'Norwegian',
    'pl': 'Polish',
    'pt': 'Portuguese',
    'ro': 'Romanian',
    'ru': 'Russian',
    'sk': 'Slovak',
    'sl': 'Slovenian',
    'sq': 'Albanian',
    'sr': 'Serbian',
    'sv': 'Swedish',
    'sw': 'Swahili',
    'ta': 'Tamil',
    'th': 'Thai',
    'tr': 'Turkish',
    'uk': 'Ukrainian',
    'ur': 'Urdu',
    'vi': 'Vietnamese',
    'zh': 'Chinese',
    'us': 'American English',
    
    # Country-specific language codes
    'pt-br': 'Brazilian Portuguese',
    'zh-cn': 'Simplified Chinese',
    'zh-tw': 'Traditional Chinese',
    'en-gb': 'British English',
    'en-us': 'American English',
    'es-mx': 'Mexican Spanish',
    
    # Special cases that might be confused
    'ch': 'Swiss German',  # Not Chinese, but Swiss domain/German dialect
    'cy': 'Welsh',  # Not Cypriot
    'gl': 'Galician',  # Not Greenlandic
    'mt': 'Maltese',  # Not Montenegrin
    'eu': 'Basque',  # Not European Union
}

def translate_text(text, target_lang, model="gpt-4o"):
    """
    Translate text using OpenAI's API
    
    Args:
        text (str): The text to translate
        target_lang (str): The target language code
        model (str): The OpenAI model to use for translation
        
    Returns:
        str: The translated text
    """
    try:
        # Get the full language name from the map, fallback to the code if not found
        language_name = LANGUAGE_MAP.get(target_lang.lower(), target_lang)
        
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a translation assistant. Translate the text to the target language while preserving the format, front matter, and markdown structure. Do not translate code blocks, URLs, or variable names. Don\'t use word: photo trap. RECOMMENDED translations in each language: EN: trail camera, trail cam, wildlife camera, DE: Wildkamera, HU: Vadkamera, PL: Fotopulapka, Fotopułapka, Fotopułapki, Kamera lesna, RO: Camere vanatoare, BG: Фотокапани, DK: Vildtkamera, EE: rajakaameramuuk, ES: Fototrampeo alebo Cámara de caza, FI: riistakamera, GR: κάμερα παγίδα alebo Κάμερες Κυνηγιού, HR: Lovačka kamera, IT: Fototrappole, Fototrappole da caccia, LV: meža kamera, NL:  wildcamera, LT: Medžioklės kamera, SE: viltkamera"},
                {"role": "user", "content": f"Translate this to {language_name}, preserving all front matter fields, markdown formatting, and HTML tags:\n\n{text}"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Translation error for {target_lang}: {str(e)}")
        # Add a small delay on error to prevent overwhelming the API
        time.sleep(1)
        return text  # Return original text on error

def translate_file(file_path, content, target_lang, target_file, model):
    """
    Translate a single file to a target language and save it
    
    Args:
        file_path (Path): Path to the source file
        content (str): Content of the source file
        target_lang (str): Target language code
        target_file (Path): Path to save the translated file
        model (str): The OpenAI model to use for translation
        
    Returns:
        tuple: (target_file, success_flag)
    """
    try:
        translated_content = translate_text(content, target_lang, model)
        
        # Write translated content to target file
        os.makedirs(target_file.parent, exist_ok=True)
        with open(target_file, 'w', encoding='utf-8') as f:
            f.write(translated_content)
            
        return (target_file, True)
    except Exception as e:
        print(f"Error translating {file_path} to {target_lang}: {str(e)}")
        return (target_file, False)
    
def is_translatable_file(file_path):
    """Check if a file should be translated based on extension"""
    return file_path.suffix.lower() in ['.md', '.markdown', '.yaml', '.yml', '.html', '.txt']

def get_target_languages(content_dir):
    """
    Find all language directories in the content directory except 'en'
    
    Args:
        content_dir (Path): Path to the content directory
        
    Returns:
        list: List of target language directory names
    """
    target_langs = []
    
    for item in content_dir.iterdir():
        if item.is_dir() and item.name != "en":
            target_langs.append(item.name)
            
    return target_langs

def process_files(content_dir, target_langs, max_concurrent=10, model="gpt-4o"):
    """
    Process all translatable files in the English directory
    Only translate files that don't exist in the target language directories
    
    Args:
        content_dir (Path): Path to the content directory
        target_langs (list): List of target language directory names
        max_concurrent (int): Maximum number of concurrent translation requests
        model (str): The OpenAI model to use for translation
    """
    en_dir = content_dir / "en"
    
    # Find all translatable files in the English directory
    translatable_files = []
    
    for root, _, files in os.walk(en_dir):
        for file in files:
            file_path = Path(root) / file
            if is_translatable_file(file_path):
                translatable_files.append(file_path)
    
    total_files = len(translatable_files)
    print(f"Found {total_files} translatable files in the English directory")
    
    if total_files == 0:
        print("No translatable files found in the English directory")
        return
    
    # Create a list of all translation tasks
    translation_tasks = []
    files_already_exist = 0
    
    for file_path in translatable_files:
        # Read the content of the file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Get the relative path from the English directory
        rel_path = file_path.relative_to(en_dir)
        
        # For each target language, create a translation task
        for target_lang in target_langs:
            target_dir = content_dir / target_lang
            target_file = target_dir / rel_path
            
            # Skip if the target file already exists
            if target_file.exists():
                files_already_exist += 1
                continue
            
            # Add the translation task to the list
            translation_tasks.append((file_path, content, target_lang, target_file))
    
    if not translation_tasks:
        print("No files need translation (all files already exist in target languages)")
        return
    
    print(f"Translating {len(translation_tasks)} files to {len(target_langs)} languages")
    
    # Process the translation tasks in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        # Submit all translation tasks
        future_to_file = {
            executor.submit(translate_file, file_path, content, target_lang, target_file, model): (file_path, target_lang)
            for file_path, content, target_lang, target_file in translation_tasks
        }
        
        # Track progress
        files_translated = 0
        files_failed = 0
        
        # Process the results as they complete
        with tqdm(total=len(translation_tasks), desc="Translating files") as progress_bar:
            for future in concurrent.futures.as_completed(future_to_file):
                file_path, target_lang = future_to_file[future]
                try:
                    target_file, success = future.result()
                    if success:
                        files_translated += 1
                        print(f"Translated: {target_file}")
                    else:
                        files_failed += 1
                        print(f"Failed: {file_path} to {target_lang}")
                except Exception as e:
                    print(f"Error processing {file_path} to {target_lang}: {str(e)}")
                    files_failed += 1
                
                progress_bar.update(1)
    
    # Print summary
    print("\nTranslation Summary:")
    print(f"Files translated: {files_translated}")
    print(f"Files failed: {files_failed}")
    print(f"Files skipped (already exist): {files_already_exist}")
    print(f"Total target files: {files_already_exist + files_translated + files_failed}")

def translate_i18n_files(i18n_dir, model="gpt-4o"):
    """
    Translate the en.yaml file to missing languages and save as [lang].yaml
    Skip languages that already have translation files.
    
    Args:
        i18n_dir (Path): Path to the i18n directory
        model (str): The OpenAI model to use for translation
    """
    # Check if the i18n directory exists
    if not i18n_dir.exists() or not i18n_dir.is_dir():
        print(f"Error: i18n directory not found: {i18n_dir}")
        return
    
    # Check if the English YAML file exists
    en_yaml_path = i18n_dir / "en.yaml"
    if not en_yaml_path.exists() or not en_yaml_path.is_file():
        print(f"Error: English YAML file not found: {en_yaml_path}")
        return
    
    # Read the English YAML file
    with open(en_yaml_path, 'r', encoding='utf-8') as f:
        en_yaml_content = f.read()
    
    # Get all language codes from LANGUAGE_MAP except 'en'
    target_langs = [lang for lang in LANGUAGE_MAP.keys() if lang != 'en' and '-' not in lang]
    
    # Filter out languages that already have translation files
    missing_langs = []
    for lang in target_langs:
        target_file = i18n_dir / f"{lang}.yaml"  # Changed from {lang}.en.yaml
        if not target_file.exists():
            missing_langs.append(lang)
    
    if not missing_langs:
        print("All language files already exist in i18n directory. No translation needed.")
        return
    
    print(f"Source file: {en_yaml_path}")
    print(f"Missing languages to translate: {', '.join(missing_langs)}")
    
    # Track statistics
    files_translated = 0
    files_failed = 0
    
    # Create a progress bar
    with tqdm(total=len(missing_langs), desc="Translating i18n files") as progress_bar:
        # Process each missing language
        for target_lang in missing_langs:
            target_file = i18n_dir / f"{target_lang}.yaml"  # Changed from {target_lang}.en.yaml
            
            try:
                # Translate the content
                print(f"Translating to {target_lang} ({LANGUAGE_MAP.get(target_lang, target_lang)})...")
                translated_content = translate_text(en_yaml_content, target_lang, model)
                
                # Save the translated content
                with open(target_file, 'w', encoding='utf-8') as f:
                    f.write(translated_content)
                
                print(f"Saved: {target_file}")
                files_translated += 1
                
            except Exception as e:
                print(f"Error translating to {target_lang}: {str(e)}")
                files_failed += 1
            
            # Update the progress bar
            progress_bar.update(1)
    
    # Print summary
    print("\nI18n Translation Summary:")
    print(f"Files translated: {files_translated}")
    print(f"Files failed: {files_failed}")
    print(f"Total missing files: {len(missing_langs)}")

def main():
    """Main function to parse arguments and process files"""
    parser = argparse.ArgumentParser(
        description="Translate missing files from English to other languages in a Hugo project",
        epilog="""
Examples:
  python translate_missing_files.py
  python translate_missing_files.py --path /path/to/content
  python translate_missing_files.py --model gpt-3.5-turbo
  
Directory Structure:
  The script expects a directory structure where the content directory contains language
  subdirectories (e.g., en, fr, es, de). The script will translate files from the 'en'
  directory to each target language directory, preserving the original directory structure,
  but only if the file doesn't already exist in the target language.
  
Requirements:
  - OpenAI API key (set in .env file or as environment variable OPENAI_API_KEY)
  - Internet connection to access the OpenAI API
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Default path is ../content/ relative to the script location
    default_path = os.path.join(hugo_root, "content")
    default_i18n_path = os.path.join(hugo_root, "i18n")
    
    parser.add_argument(
        "--path",
        help="Path to the content directory containing language subdirectories (default: %(default)s)",
        default=default_path
    )
    parser.add_argument(
        "--i18n-path",
        help="Path to the i18n directory containing translation files (default: %(default)s)",
        default=default_i18n_path
    )
    parser.add_argument(
        "--max-concurrent",
        help="Maximum number of concurrent translation requests (default: %(default)s)",
        type=int,
        default=10
    )
    parser.add_argument(
        "--model",
        help="OpenAI model to use for translation (default: %(default)s)",
        default="gpt-4o"
    )
    parser.add_argument(
        "--i18n-only",
        help="Only translate i18n files, skip content files",
        action="store_true"
    )
    parser.add_argument(
        "--content-only",
        help="Only translate content files, skip i18n files",
        action="store_true"
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    content_dir = Path(args.path)
    i18n_dir = Path(args.i18n_path)
    
    # Check what to translate based on flags
    translate_content = not args.i18n_only
    translate_i18n = not args.content_only
    
    if translate_content:
        # Check if the content directory exists
        if not content_dir.exists() or not content_dir.is_dir():
            print(f"Error: Content directory not found: {content_dir}")
            if not translate_i18n:
                sys.exit(1)
            else:
                print("Skipping content translation.")
                translate_content = False
        
        # Check if the English directory exists
        en_dir = content_dir / "en"
        if not en_dir.exists() or not en_dir.is_dir():
            print(f"Error: English directory not found: {en_dir}")
            if not translate_i18n:
                sys.exit(1)
            else:
                print("Skipping content translation.")
                translate_content = False
    
    if translate_content:
        # Get target languages
        target_langs = get_target_languages(content_dir)
        
        if not target_langs:
            print("No target language directories found. Please create at least one target language directory.")
            if not translate_i18n:
                sys.exit(1)
            else:
                print("Skipping content translation.")
                translate_content = False
        
        print(f"Content directory: {content_dir}")
        print(f"Source language: en")
        print(f"Target languages: {', '.join(target_langs)}")
        print(f"Maximum concurrent requests: {args.max_concurrent}")
        print(f"Using model: {args.model}")
        
        # Process content files
        process_files(content_dir, target_langs, args.max_concurrent, args.model)
    
    if translate_i18n:
        # Check if the i18n directory exists
        if not i18n_dir.exists() or not i18n_dir.is_dir():
            print(f"Error: i18n directory not found: {i18n_dir}")
            if not translate_content:
                sys.exit(1)
            else:
                print("Skipping i18n translation.")
                translate_i18n = False
        
        if translate_i18n:
            print(f"\nTranslating i18n files...")
            print(f"i18n directory: {i18n_dir}")
            print(f"Using model: {args.model}")
            
            # Process i18n files
            translate_i18n_files(i18n_dir, args.model)
    
    if translate_content or translate_i18n:
        print("\nTranslation completed successfully!")
    else:
        print("\nNo translation tasks were performed.")

if __name__ == "__main__":
    main()
