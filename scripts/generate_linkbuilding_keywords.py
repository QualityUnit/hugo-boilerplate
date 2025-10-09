#!/usr/bin/env python3
"""
Generate Linkbuilding Keywords

This script generates unique keywords from content files using n-grams, creates
FAISS vectors for each keyword, and then matches documents to their best keywords
for linkbuilding purposes.

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

# Excluded directories - files in these directories will be skipped
EXCLUDED_DIRECTORIES = [
    "affiliate-manager",
    "affiliate-program-directory", 
    "gdpr",
    "search"
]

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
    parser.add_argument("--min-files", type=int, default=2,
                        help="Minimum number of files a keyword must appear in to be included (default: 2)")
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

def collect_content_files(content_directory):
    """Collect all markdown files and their content."""
    print(f"Collecting content files from: {content_directory}")
    
    if not os.path.exists(content_directory):
        print(f"Content directory not found: {content_directory}")
        return []
    
    file_data = []
    
    for root, _, files in os.walk(content_directory):
        # Check if current directory or any parent directory is in excluded list
        rel_root = os.path.relpath(root, content_directory)
        
        # Skip if the current path starts with any excluded directory
        skip_directory = False
        for excluded_dir in EXCLUDED_DIRECTORIES:
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
                    
                    # Extract metadata safely
                    metadata = getattr(post, 'metadata', {}) or {}
                    title = metadata.get("title", "") if metadata else ""
                    description = metadata.get("description", "") if metadata else ""
                    
                    # Extract content safely
                    content_text = getattr(post, 'content', '') or ''
                    text = extract_text_from_markdown(content_text)
                    
                    # Skip files with no meaningful content
                    if not title and not description and not text:
                        continue
                    
                    # Combine title, description and content for full text
                    full_text = f"{title} {description} {text}".strip()
                    
                    if len(text) > MAX_TEXT_LENGTH:
                        text = text[:MAX_TEXT_LENGTH]
                    
                    file_data.append({
                        "path": file_path,
                        "rel_path": rel_path,
                        "title": title,
                        "description": description,
                        "text": text,
                        "full_text": full_text,
                        "post": post
                    })
                    
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
                    import traceback
                    traceback.print_exc()
    
    print(f"Found {len(file_data)} content files without linkbuilding attribute")
    return file_data

def extract_unique_keywords(file_data, min_n, max_n, min_freq=2, min_files=2):
    """Extract unique keywords from all content files using n-grams, filtering by file frequency."""
    print("Extracting unique keywords from content...")
    
    # Track which files contain each keyword
    keyword_file_count = {}
    keyword_total_count = Counter()
    
    for file_info in tqdm(file_data, desc="Processing files for keywords"):
        full_text = file_info['full_text']
        ngrams = generate_ngrams(full_text, min_n, max_n)
        
        # Get unique ngrams from this file to count file occurrences
        unique_ngrams_in_file = set(ngrams)
        
        # Update total counts
        keyword_total_count.update(ngrams)
        
        # Update file counts
        for ngram in unique_ngrams_in_file:
            if ngram not in keyword_file_count:
                keyword_file_count[ngram] = 0
            keyword_file_count[ngram] += 1
    
    # Filter keywords by both total frequency and file frequency
    unique_keywords = []
    for keyword, total_count in keyword_total_count.items():
        file_count = keyword_file_count[keyword]
        if total_count >= min_freq and file_count >= min_files:
            unique_keywords.append(keyword)
    
    print(f"Generated {len(unique_keywords)} unique keywords with minimum frequency {min_freq} and appearing in at least {min_files} files")
    print(f"Total keywords before file filtering: {len([k for k, c in keyword_total_count.items() if c >= min_freq])}")
    
    return unique_keywords, keyword_total_count

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

def find_best_keywords_for_document(doc_text, keywords, keyword_counts, keyword_index, keyword_embeddings, model_name, top_k, similarity_threshold=0.1):
    """Find best matching keywords for a document, preferring higher frequency keywords."""
    model = load_model(model_name)
    
    # Generate embedding for document
    doc_embedding = model.encode([doc_text])
    doc_embedding = doc_embedding.astype('float32')
    
    # Search for more candidates than we need (to allow for frequency-based selection)
    search_k = min(top_k * 3, len(keywords))
    distances, indices = keyword_index.search(doc_embedding, search_k)
    
    # Create candidates with similarity scores and frequencies
    candidates = []
    for i, idx in enumerate(indices[0]):
        if idx < len(keywords):
            keyword = keywords[idx]
            similarity = distances[0][i]  # Higher is better for cosine similarity
            frequency = keyword_counts[keyword]
            
            # Create a combined score: similarity + normalized frequency bonus
            # Normalize frequency to 0-1 range based on max frequency
            max_freq = max(keyword_counts.values())
            freq_bonus = (frequency / max_freq) * 0.3  # 30% weight for frequency
            combined_score = similarity + freq_bonus
            
            candidates.append({
                'keyword': keyword,
                'similarity': similarity,
                'frequency': frequency,
                'combined_score': combined_score
            })
    
    # Sort by combined score (similarity + frequency bonus) and take top_k
    candidates.sort(key=lambda x: x['combined_score'], reverse=True)
    
    # Filter by similarity threshold and take top_k
    best_keywords = []
    for candidate in candidates[:top_k]:
        if candidate['similarity'] >= similarity_threshold:
            best_keywords.append(candidate['keyword'])
    
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
    
    # Step 1: Collect content files
    file_data = collect_content_files(content_directory)
    
    if not file_data:
        print(f"✅ No content files need linkbuilding attributes for language: {args.lang}")
        print(f"All files already have linkbuilding attributes or no valid files found.")
        return
    
    # Step 2: Extract unique keywords from all content
    unique_keywords, keyword_counts = extract_unique_keywords(
        file_data, args.min_ngram, args.max_ngram, args.min_keyword_freq, args.min_files
    )
    
    if not unique_keywords:
        print("No keywords found")
        return
    
    print(f"Top 10 most frequent keywords:")
    for keyword, count in keyword_counts.most_common(10):
        print(f"  '{keyword}': {count} occurrences")
    
    # Step 3: Create FAISS index for keywords
    keyword_index, keyword_embeddings = create_keyword_index(unique_keywords, keyword_counts, args.model)
    
    # Step 4: Process each document and find best matching keywords
    print("Finding best keywords for each document...")
    updated_count = 0
    
    for file_info in tqdm(file_data, desc="Processing documents"):
        # Create document vector from title, description and content
        doc_text = file_info['full_text']
        
        # Find best matching keywords
        best_keywords = find_best_keywords_for_document(
            doc_text, unique_keywords, keyword_counts, keyword_index, keyword_embeddings, args.model, args.top_k
        )
        
        # Update file with keywords
        if best_keywords and update_file_frontmatter(file_info, best_keywords):
            updated_count += 1
    
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