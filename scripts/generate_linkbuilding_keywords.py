#!/usr/bin/env python3
"""
Generate Linkbuilding Keywords

This script generates unique keywords from content files using n-grams, creates
FAISS vectors for each keyword, and then matches documents to their best keywords
for linkbuilding purposes.

Text for vectorization uses only:
- URL attribute from frontmatter (or file path if URL doesn't exist)
- Title from frontmatter
- Description from frontmatter

The actual content of the markdown file is NOT used for vectorization.

Usage:
    python generate_linkbuilding_keywords.py --lang en
    python generate_linkbuilding_keywords.py --lang en --min-ngram 2 --max-ngram 4 --top-k 10

Requirements:
    pip install sentence-transformers faiss-cpu pyyaml frontmatter markdown bs4 tqdm nltk
"""

import os
import re
import argparse
import yaml
import gc
import math
import frontmatter
from frontmatter import TOMLHandler
import markdown
from bs4 import BeautifulSoup
from tqdm import tqdm
from collections import defaultdict, Counter
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Constants
MODEL_NAME = "Alibaba-NLP/gte-multilingual-base"
MAX_TEXT_LENGTH = 1000
DEFAULT_MIN_NGRAM = 2
DEFAULT_MAX_NGRAM = 4
DEFAULT_TOP_K = 10
DEFAULT_MAX_PAGES_PER_KEYWORD = 5  # Limit keyword reuse across pages

# Configuration file path (in project's config directory)
# Script is in themes/boilerplate/scripts/, config is in project_root/config/
SCRIPT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
CONFIG_FILE = os.path.join(PROJECT_ROOT, "config", "linkbuilding_config.yaml")

# Global configuration (loaded from YAML file)
_config = None


def load_config():
    """Load configuration from YAML file."""
    global _config
    if _config is not None:
        return _config

    # Default configuration
    _config = {
        'skip_directories': [
            "affiliate-manager",
            "affiliate-program-directory",
            "gdpr",
            "search",
            "author",
        ],
        'empty_linkbuilding_directories': [
            "affiliate-program-directory",
            "affiliate-manager",
        ],
        'brand_terms': [
            'post affiliate pro',
            'postaffiliatepro',
            'affiliate pro',
            'post affiliate',
            'affiliate tracking',
            'tracking affiliate',
            'affiliate software',
            'affiliate program',
            'affiliate marketing',
            'affiliate network',
            'affiliate commission',
            'affiliate link',
            'affiliate links',
        ],
        'positional_weights': {
            'h1': 3.0,
            'h2': 2.5,
            'h3': 2.0,
            'h4': 1.5,
            'h5': 1.3,
            'h6': 1.2,
            'strong': 1.5,
            'b': 1.5,
            'em': 1.2,
            'title': 2.5,
            'description': 1.5,
            'url': 2.0,
            'body': 1.0,
        },
        'defaults': {
            'min_ngram': DEFAULT_MIN_NGRAM,
            'max_ngram': DEFAULT_MAX_NGRAM,
            'top_k': DEFAULT_TOP_K,
            'min_keyword_freq': 2,
            'min_files': 5,
            'max_pages_per_keyword': DEFAULT_MAX_PAGES_PER_KEYWORD,
        }
    }

    # Try to load from config file
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                file_config = yaml.safe_load(f)
            if file_config:
                # Merge file config with defaults
                for key in file_config:
                    if key in _config:
                        if isinstance(_config[key], dict) and isinstance(file_config[key], dict):
                            _config[key].update(file_config[key])
                        else:
                            _config[key] = file_config[key]
                    else:
                        _config[key] = file_config[key]
            print(f"Loaded configuration from: {CONFIG_FILE}")
        except Exception as e:
            print(f"Warning: Could not load config file {CONFIG_FILE}: {e}")
            print("Using default configuration")
    else:
        print(f"Config file not found: {CONFIG_FILE}")
        print("Using default configuration")

    return _config


def get_config(key, default=None):
    """Get a configuration value."""
    config = load_config()
    return config.get(key, default)


def get_positional_weights():
    """Get positional weights from configuration."""
    return get_config('positional_weights', {})


def get_brand_terms():
    """Get brand terms from configuration as a set."""
    return set(get_config('brand_terms', []))


def get_skip_directories():
    """Get directories to skip from configuration."""
    return get_config('skip_directories', [])


def get_empty_linkbuilding_directories():
    """Get directories that should have empty linkbuilding from configuration."""
    return get_config('empty_linkbuilding_directories', [])

# Global variables for model
_model = None

def load_model(model_name):
    """Load the model once."""
    global _model
    
    if _model is None:
        print(f"Loading model: {model_name}")
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(model_name, trust_remote_code=True)
    else:
        print("Using already loaded model")
    
    return _model

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate linkbuilding keywords")
    parser.add_argument("--lang", type=str, required=True,
                        help="Language to process")
    parser.add_argument(
        "--path",
        type=str,
        default=os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "content")),
        help="Absolute path to the content directory"
    )
    parser.add_argument("--min-ngram", type=int, default=DEFAULT_MIN_NGRAM,
                        help=f"Minimum n-gram size (default: {DEFAULT_MIN_NGRAM})")
    parser.add_argument("--max-ngram", type=int, default=DEFAULT_MAX_NGRAM,
                        help=f"Maximum n-gram size (default: {DEFAULT_MAX_NGRAM})")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                        help=f"Number of top keywords to assign (default: {DEFAULT_TOP_K})")
    parser.add_argument("--model", type=str, default=MODEL_NAME,
                        help=f"Model name to use (default: {MODEL_NAME})")
    parser.add_argument("--min-keyword-freq", type=int, default=2,
                        help="Minimum frequency for keywords to be included (default: 2)")
    parser.add_argument("--min-files", type=int, default=5,
                        help="Minimum number of files a keyword must appear in to be included (default: 2)")
    parser.add_argument("--max-pages-per-keyword", type=int, default=DEFAULT_MAX_PAGES_PER_KEYWORD,
                        help=f"Maximum pages a keyword can be assigned to (default: {DEFAULT_MAX_PAGES_PER_KEYWORD})")
    return parser.parse_args()

def extract_text_from_markdown(content):
    """Extract text content from markdown, removing HTML tags."""
    try:
        html = markdown.markdown(content)
        soup = BeautifulSoup(html, 'html.parser')
        text = soup.get_text(separator=' ', strip=True)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception as e:
        print(f"Error extracting text from markdown: {e}")
        return ""

def generate_ngrams(text, min_n, max_n, lang='english'):
    """Generate n-grams from text."""
    try:
        # Get stopwords for the language
        try:
            stop_words = set(stopwords.words(lang))
        except:
            # Fallback to English if language not supported
            stop_words = set(stopwords.words('english'))

        # Tokenize and clean text
        tokens = word_tokenize(text.lower())
        tokens = [token for token in tokens if token.isalpha() and len(token) > 2 and token not in stop_words]

        ngrams = []
        for n in range(min_n, max_n + 1):
            for i in range(len(tokens) - n + 1):
                ngram = ' '.join(tokens[i:i+n])
                ngrams.append(ngram)

        return ngrams
    except Exception as e:
        print(f"Error generating n-grams: {e}")
        return []


def extract_weighted_ngrams(content, metadata, min_n, max_n, lang='english'):
    """
    Extract n-grams with positional weights from content and metadata.

    Returns:
        dict: keyword -> total_weight mapping
    """
    weighted_ngrams = defaultdict(float)
    positional_weights = get_positional_weights()

    try:
        # Parse markdown to HTML for structure extraction
        html = markdown.markdown(content)
        soup = BeautifulSoup(html, 'html.parser')

        # Extract from headings with weights
        for tag_name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            weight = positional_weights.get(tag_name, 1.0)
            for tag in soup.find_all(tag_name):
                text = tag.get_text(strip=True)
                for ngram in generate_ngrams(text, min_n, max_n, lang):
                    weighted_ngrams[ngram] += weight

        # Extract from bold/strong text
        for tag_name in ['strong', 'b']:
            weight = positional_weights.get(tag_name, 1.0)
            for tag in soup.find_all(tag_name):
                text = tag.get_text(strip=True)
                for ngram in generate_ngrams(text, min_n, max_n, lang):
                    weighted_ngrams[ngram] += weight

        # Extract from emphasized text
        for tag in soup.find_all('em'):
            weight = positional_weights.get('em', 1.0)
            text = tag.get_text(strip=True)
            for ngram in generate_ngrams(text, min_n, max_n, lang):
                weighted_ngrams[ngram] += weight

        # Extract from title (frontmatter)
        title = metadata.get('title', '') if metadata else ''
        if title:
            weight = positional_weights.get('title', 2.5)
            for ngram in generate_ngrams(title, min_n, max_n, lang):
                weighted_ngrams[ngram] += weight

        # Extract from description (frontmatter)
        description = metadata.get('description', '') if metadata else ''
        if description:
            weight = positional_weights.get('description', 1.5)
            for ngram in generate_ngrams(description, min_n, max_n, lang):
                weighted_ngrams[ngram] += weight

        # Extract from URL
        url = metadata.get('url', '') if metadata else ''
        if url:
            weight = positional_weights.get('url', 2.0)
            # Convert URL slashes and dashes to spaces for tokenization
            url_text = url.replace('/', ' ').replace('-', ' ').replace('_', ' ')
            for ngram in generate_ngrams(url_text, min_n, max_n, lang):
                weighted_ngrams[ngram] += weight

        # Extract from body text (lower weight)
        body_text = soup.get_text(separator=' ', strip=True)
        weight = positional_weights.get('body', 1.0)
        for ngram in generate_ngrams(body_text, min_n, max_n, lang):
            # Only add body weight if not already weighted from structural elements
            if ngram not in weighted_ngrams:
                weighted_ngrams[ngram] = weight
            # Small additional weight for body occurrence
            else:
                weighted_ngrams[ngram] += weight * 0.1

    except Exception as e:
        print(f"Error extracting weighted n-grams: {e}")

    return weighted_ngrams

def collect_content_files(content_directory, process_empty_linkbuilding=False):
    """
    Collect all markdown files and extract URL, title, and description from frontmatter.

    Args:
        content_directory: Path to the content directory
        process_empty_linkbuilding: If True, also return files that should have empty linkbuilding

    Returns:
        If process_empty_linkbuilding is False: list of file_data dicts
        If process_empty_linkbuilding is True: tuple of (file_data list, empty_linkbuilding_files list)
    """
    print(f"Collecting content files from: {content_directory}")
    
    if not os.path.exists(content_directory):
        print(f"Content directory not found: {content_directory}")
        if process_empty_linkbuilding:
            return [], []
        return []

    # Get configuration
    skip_directories = get_skip_directories()
    empty_linkbuilding_dirs = get_empty_linkbuilding_directories()

    file_data = []
    empty_linkbuilding_files = []  # Files that should have empty linkbuilding []

    for root, _, files in os.walk(content_directory):
        # Check if current directory or any parent directory is in excluded list
        rel_root = os.path.relpath(root, content_directory)

        # Check if this directory should have empty linkbuilding
        is_empty_linkbuilding_dir = False
        for empty_dir in empty_linkbuilding_dirs:
            if rel_root == empty_dir or rel_root.startswith(empty_dir + os.sep):
                is_empty_linkbuilding_dir = True
                break

        # Skip if the current path starts with any skip directory (unless it's empty linkbuilding dir)
        skip_directory = False
        if not is_empty_linkbuilding_dir:
            for excluded_dir in skip_directories:
                if rel_root == excluded_dir or rel_root.startswith(excluded_dir + os.sep):
                    skip_directory = True
                    break

        if skip_directory:
            print(f"Skipping excluded directory: {rel_root}")
            continue

        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, content_directory)

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    # Try to parse frontmatter
                    try:
                        post = frontmatter.loads(content, handler=TOMLHandler())
                    except:
                        # Fallback to default handler if TOML fails
                        post = frontmatter.loads(content)

                    # Skip files that already have linkbuilding attribute
                    if hasattr(post, 'metadata') and post.metadata and 'linkbuilding' in post.metadata:
                        print(f"Skipping {rel_path} - already has linkbuilding attribute")
                        continue

                    # If this file is in an empty linkbuilding directory, add to that list
                    if is_empty_linkbuilding_dir:
                        empty_linkbuilding_files.append({
                            "path": file_path,
                            "rel_path": rel_path,
                            "post": post
                        })
                        continue

                    # Extract metadata safely
                    metadata = getattr(post, 'metadata', {}) or {}
                    title = metadata.get("title", "") if metadata else ""
                    description = metadata.get("description", "") if metadata else ""

                    # Get URL from frontmatter, or use the relative path if URL doesn't exist
                    url = metadata.get("url", "") if metadata else ""
                    if not url:
                        # Use the relative path without .md extension as fallback
                        url = rel_path.replace('.md', '')

                    # Skip files with no meaningful metadata
                    if not title and not description:
                        continue

                    # Create text for vectorization using only url, title, and description
                    full_text = f"{url} {title} {description}".strip()

                    # Store the markdown content body for weighted n-gram extraction
                    content_body = post.content if hasattr(post, 'content') else ""

                    file_data.append({
                        "path": file_path,
                        "rel_path": rel_path,
                        "url": url,
                        "title": title,
                        "description": description,
                        "content_body": content_body,
                        "full_text": full_text,
                        "metadata": metadata,
                        "post": post
                    })

                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
                    import traceback
                    traceback.print_exc()

    print(f"Found {len(file_data)} content files without linkbuilding attribute")
    if empty_linkbuilding_files:
        print(f"Found {len(empty_linkbuilding_files)} files in empty linkbuilding directories")

    if process_empty_linkbuilding:
        return file_data, empty_linkbuilding_files
    return file_data

def extract_unique_keywords(file_data, min_n, max_n, min_freq=2, min_files=2):
    """
    Extract unique keywords from all content files using weighted n-grams.

    Uses positional weighting (H1, H2, bold, etc.) and tracks per-document weights.

    Returns:
        unique_keywords: List of keywords meeting frequency thresholds
        keyword_total_count: Counter of total keyword occurrences
        keyword_file_count: Dict of keyword -> number of files containing it
        keyword_weights_per_file: Dict of keyword -> {file_path: weight}
    """
    print("Extracting unique keywords with positional weighting...")

    # Track which files contain each keyword and their weights
    keyword_file_count = {}
    keyword_total_count = Counter()
    keyword_weights_per_file = defaultdict(dict)  # keyword -> {file_path: weight}

    for file_info in tqdm(file_data, desc="Processing files for keywords"):
        # Use weighted n-gram extraction from content structure
        content_body = file_info.get('content_body', '')
        metadata = file_info.get('metadata', {})
        file_path = file_info['path']

        # Get weighted n-grams from this file
        weighted_ngrams = extract_weighted_ngrams(content_body, metadata, min_n, max_n)

        # Also add basic n-grams from full_text as fallback
        full_text = file_info['full_text']
        basic_ngrams = generate_ngrams(full_text, min_n, max_n)
        for ngram in basic_ngrams:
            if ngram not in weighted_ngrams:
                weighted_ngrams[ngram] = 1.0

        # Get unique ngrams from this file to count file occurrences
        unique_ngrams_in_file = set(weighted_ngrams.keys())

        # Update total counts
        keyword_total_count.update(weighted_ngrams.keys())

        # Update file counts and per-file weights
        for ngram, weight in weighted_ngrams.items():
            if ngram not in keyword_file_count:
                keyword_file_count[ngram] = 0
            keyword_file_count[ngram] += 1
            keyword_weights_per_file[ngram][file_path] = weight

    # Filter keywords by both total frequency and file frequency
    unique_keywords = []
    for keyword, total_count in keyword_total_count.items():
        file_count = keyword_file_count[keyword]
        if total_count >= min_freq and file_count >= min_files:
            unique_keywords.append(keyword)

    print(f"Generated {len(unique_keywords)} unique keywords with minimum frequency {min_freq} and appearing in at least {min_files} files")
    print(f"Total keywords before file filtering: {len([k for k, c in keyword_total_count.items() if c >= min_freq])}")

    return unique_keywords, keyword_total_count, keyword_file_count, keyword_weights_per_file

def create_keyword_index(keywords, keyword_counts, model_name):
    """Create FAISS index for keywords."""
    print("Creating FAISS index for keywords...")
    
    model = load_model(model_name)
    
    # Generate embeddings for keywords in batches
    batch_size = 100
    keyword_embeddings = []
    
    for i in range(0, len(keywords), batch_size):
        batch = keywords[i:i+batch_size]
        print(f"Processing keyword batch {i//batch_size + 1}/{(len(keywords) + batch_size - 1)//batch_size}")
        
        batch_embeddings = model.encode(batch, show_progress_bar=False)
        keyword_embeddings.extend(batch_embeddings)
    
    keyword_embeddings = np.array(keyword_embeddings).astype('float32')
    
    # Build FAISS index
    import faiss
    dimension = keyword_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(keyword_embeddings)
    
    return index, keyword_embeddings

def find_best_keywords_for_document(
    doc_text,
    keywords,
    keyword_counts,
    keyword_file_count,
    keyword_weights_per_file,
    keyword_index,
    keyword_embeddings,
    model_name,
    top_k,
    total_docs,
    file_path,
    keyword_assignment_count,
    max_pages_per_keyword,
    similarity_threshold=0.1
):
    """
    Find best matching keywords for a single document using improved scoring.

    Improvements:
    - TF-IDF: Penalizes keywords that appear in too many documents
    - Positional weighting: Uses pre-computed weights from H1, H2, bold, etc.
    - Long-tail preference: Bonus for longer, more specific keywords
    - Brand term filtering: Heavy penalty for generic brand terms
    - Uniqueness cap: Respects max_pages_per_keyword limit
    """
    model = load_model(model_name)

    # Generate embedding for the document
    doc_embedding = model.encode([doc_text], show_progress_bar=False)
    doc_embedding = doc_embedding.astype('float32')

    # Search for more candidates than we need
    search_k = min(top_k * 5, len(keywords))

    # Search in FAISS
    distances, indices = keyword_index.search(doc_embedding, search_k)
    distances = distances[0]
    indices = indices[0]

    # Create candidates with improved scoring
    candidates = []
    for i, idx in enumerate(indices):
        if idx < len(keywords):
            keyword = keywords[idx]
            similarity = distances[i]  # Higher is better for cosine similarity

            # Skip if keyword has already been assigned to max pages
            if keyword_assignment_count.get(keyword, 0) >= max_pages_per_keyword:
                continue

            # 1. TF-IDF: Calculate IDF penalty for common keywords
            doc_freq = keyword_file_count.get(keyword, 1)
            idf = math.log(total_docs / (1 + doc_freq))
            # Normalize IDF to 0-1 range (typically IDF ranges from 0 to ~7)
            idf_normalized = max(0, min(1, idf / 5.0))

            # 2. Positional weight bonus from this specific document
            positional_weight = keyword_weights_per_file.get(keyword, {}).get(file_path, 1.0)
            # Normalize positional weight (cap at 5x for very prominent keywords)
            positional_normalized = min(positional_weight / 5.0, 1.0)

            # 3. Long-tail bonus: prefer longer, more specific keywords
            word_count = len(keyword.split())
            # Bonus: 2 words = 0, 3 words = 0.1, 4+ words = 0.2
            length_bonus = min((word_count - 2) * 0.1, 0.2)

            # 4. Brand term penalty
            brand_penalty = 1.0
            keyword_lower = keyword.lower()
            brand_terms = get_brand_terms()
            if keyword_lower in brand_terms:
                brand_penalty = 0.1  # 90% penalty for brand terms
            else:
                # Partial penalty if keyword contains brand terms
                for brand_term in brand_terms:
                    if brand_term in keyword_lower:
                        brand_penalty = 0.5  # 50% penalty
                        break

            # Combined score formula:
            # Base: semantic similarity
            # * IDF factor (penalize common keywords)
            # * Brand penalty
            # + Positional bonus (reward keywords in headings/bold)
            # + Length bonus (reward specific keywords)
            combined_score = (
                similarity
                * (0.5 + idf_normalized)  # IDF factor: 0.5 to 1.5
                * brand_penalty
                + positional_normalized * 0.3  # Up to 0.3 bonus for positional weight
                + length_bonus  # Up to 0.2 bonus for long-tail
            )

            candidates.append({
                'keyword': keyword,
                'similarity': similarity,
                'idf': idf,
                'positional_weight': positional_weight,
                'word_count': word_count,
                'brand_penalty': brand_penalty,
                'combined_score': combined_score
            })

    # Sort by combined score and take top_k
    candidates.sort(key=lambda x: x['combined_score'], reverse=True)

    # Filter by similarity threshold and take top_k
    best_keywords = []
    for candidate in candidates:
        if len(best_keywords) >= top_k:
            break
        if candidate['similarity'] >= similarity_threshold:
            keyword = candidate['keyword']
            best_keywords.append(keyword)
            # Track assignment
            keyword_assignment_count[keyword] = keyword_assignment_count.get(keyword, 0) + 1

    return best_keywords

def update_file_frontmatter(file_info, keywords):
    """Update file with linkbuilding keywords in frontmatter."""
    try:
        post = file_info['post']
        post.metadata['linkbuilding'] = keywords
        
        # Write back to file with proper TOML formatting
        with open(file_info['path'], 'w', encoding='utf-8') as f:
            # Use frontmatter.dumps with TOMLHandler and ensure proper format
            content = frontmatter.dumps(post, handler=TOMLHandler())
            f.write(content)
        
        print(f"Updated {file_info['rel_path']} with {len(keywords)} keywords")
        return True
    except Exception as e:
        print(f"Error updating {file_info['rel_path']}: {e}")
        # Provide more detailed error info
        import traceback
        traceback.print_exc()
        return False

def process_language(args):
    """Process a single language."""
    print(f"\nProcessing language: {args.lang}")

    # Determine content directory
    content_directory = os.path.join(args.path, args.lang)

    # Step 1: Collect content files (including files that need empty linkbuilding)
    file_data, empty_linkbuilding_files = collect_content_files(content_directory, process_empty_linkbuilding=True)

    # Step 1.5: Process files that should have empty linkbuilding []
    empty_updated_count = 0
    if empty_linkbuilding_files:
        print(f"\nSetting empty linkbuilding for {len(empty_linkbuilding_files)} files in excluded directories...")
        for file_info in empty_linkbuilding_files:
            if update_file_frontmatter(file_info, []):
                empty_updated_count += 1
        print(f"Set empty linkbuilding for {empty_updated_count} files")

    if not file_data:
        print(f"✅ No content files need linkbuilding attributes for language: {args.lang}")
        print(f"All files already have linkbuilding attributes or no valid files found.")
        return

    total_docs = len(file_data)
    print(f"Total documents to process: {total_docs}")

    # Step 2: Extract unique keywords from all content with positional weighting
    unique_keywords, keyword_counts, keyword_file_count, keyword_weights_per_file = extract_unique_keywords(
        file_data, args.min_ngram, args.max_ngram, args.min_keyword_freq, args.min_files
    )

    if not unique_keywords:
        print("No keywords found")
        return

    # Show top keywords by frequency (these will be penalized by IDF)
    print(f"\nTop 10 most frequent keywords (will be penalized by IDF):")
    for keyword, count in keyword_counts.most_common(10):
        file_count = keyword_file_count.get(keyword, 0)
        idf = math.log(total_docs / (1 + file_count))
        print(f"  '{keyword}': {count} occurrences in {file_count} files (IDF: {idf:.2f})")

    # Step 3: Create FAISS index for keywords
    keyword_index, keyword_embeddings = create_keyword_index(unique_keywords, keyword_counts, args.model)

    # Step 4: Process documents one by one to find best matching keywords
    print("\nFinding best keywords for all documents with improved scoring...")
    print(f"  - Using TF-IDF to penalize common keywords")
    print(f"  - Using positional weighting (H1, H2, bold, etc.)")
    print(f"  - Preferring long-tail keywords")
    print(f"  - Penalizing generic brand terms")
    print(f"  - Max pages per keyword: {args.max_pages_per_keyword}")

    updated_count = 0

    # Global tracker for keyword assignments across all documents
    keyword_assignment_count = {}

    # Process each file individually
    for idx, file_info in enumerate(file_data):
        print(f"Processing document {idx + 1}/{len(file_data)}: {file_info['rel_path']}")

        # Get document text (only url/path, title, and description)
        doc_text = file_info['full_text']
        file_path = file_info['path']

        # Find best matching keywords for this document with improved scoring
        best_keywords = find_best_keywords_for_document(
            doc_text=doc_text,
            keywords=unique_keywords,
            keyword_counts=keyword_counts,
            keyword_file_count=keyword_file_count,
            keyword_weights_per_file=keyword_weights_per_file,
            keyword_index=keyword_index,
            keyword_embeddings=keyword_embeddings,
            model_name=args.model,
            top_k=args.top_k,
            total_docs=total_docs,
            file_path=file_path,
            keyword_assignment_count=keyword_assignment_count,
            max_pages_per_keyword=args.max_pages_per_keyword
        )

        # Update file with keywords
        if best_keywords and update_file_frontmatter(file_info, best_keywords):
            updated_count += 1
            print(f"Updated {file_info['rel_path']} with keywords: {best_keywords}")

    # Print assignment statistics
    print(f"\nKeyword assignment statistics:")
    assignment_counts = list(keyword_assignment_count.values())
    if assignment_counts:
        print(f"  - Keywords assigned: {len(keyword_assignment_count)}")
        print(f"  - Max assignments per keyword: {max(assignment_counts)}")
        print(f"  - Avg assignments per keyword: {sum(assignment_counts)/len(assignment_counts):.1f}")

        # Show most assigned keywords
        top_assigned = sorted(keyword_assignment_count.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"\nTop 10 most assigned keywords:")
        for keyword, count in top_assigned:
            print(f"  '{keyword}': assigned to {count} pages")

    print(f"\nProcessing complete for {args.lang}")
    print(f"Updated {updated_count} files with linkbuilding keywords")

    # Clean up memory
    gc.collect()

def main():
    """Main function to run the script."""
    print("========== LINKBUILDING KEYWORDS GENERATOR STARTING ===========")
    
    args = parse_args()
    
    print(f"Arguments:")
    print(f"- Language: {args.lang}")
    print(f"- Content path: {args.path}")
    print(f"- N-gram range: {args.min_ngram}-{args.max_ngram}")
    print(f"- Top K keywords: {args.top_k}")
    print(f"- Min keyword frequency: {args.min_keyword_freq}")
    print(f"- Min files per keyword: {args.min_files}")
    print(f"- Max pages per keyword: {args.max_pages_per_keyword}")
    print(f"- Model: {args.model}")
    
    process_language(args)
    
    print("\n========== LINKBUILDING KEYWORDS GENERATOR FINISHED ===========")
    
    # Clean up global model resources
    global _model
    if _model is not None:
        _model = None
    gc.collect()

if __name__ == "__main__":
    main()