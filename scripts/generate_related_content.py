#!/usr/bin/env python3
"""
Generate Related Content YAML

This script indexes content files in a specific language folder as vectors using FAISS
and a sentence transformer model from Hugging Face. For each file, it finds
the 3 most similar files and generates a YAML file with the related content structure.

Usage:
    python generate_related_content.py --lang en
    python generate_related_content.py --lang en --path /path/to/content

Requirements:
    pip install sentence-transformers faiss-cpu pyyaml frontmatter markdown bs4 tqdm
    
    or install requirements.txt
    pip install -r requirements.txt
    
"""

import os
import re
import argparse
import yaml
import json
import gc
import frontmatter
from frontmatter import TOMLHandler # Changed import
import markdown
from bs4 import BeautifulSoup
from tqdm import tqdm
from collections import defaultdict
import numpy as np
from sklearn.preprocessing import normalize

# Constants
MODEL_NAME = "Alibaba-NLP/gte-multilingual-base"  # Smaller model that works well with sentence-transformers
MAX_TEXT_LENGTH = 1000  # Limit text length to avoid memory issues
TOP_K = 3  # Number of related content items to find

# Global variables for model
_model = None

def load_model(model_name):
    """Load the model once."""
    global _model
    
    # Only load if not already loaded
    if _model is None:
        print(f"Loading model: {model_name}")
        
        # Import here to delay loading these heavy libraries until needed
        from sentence_transformers import SentenceTransformer
        
        # Load the model using sentence_transformers
        _model = SentenceTransformer(model_name, trust_remote_code=True)
    else:
        print("Using already loaded model")
    
    return _model

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate related content YAML")
    parser.add_argument("--lang", type=str,
                        help="Language to process (if only processing one language)")
    parser.add_argument(
        "--path",
        type=str,
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "content")),
        help="Absolute path to the content directory (default: two levels up from script in 'content' folder)"
    )
    parser.add_argument("--content-dir", type=str, default="content",
                        help="Content directory relative to Hugo root (default: content)")
    parser.add_argument("--output-dir", type=str, default="data/related_content",
                        help="Output directory relative to Hugo root (default: data/related_content)")
    parser.add_argument("--exclude-sections", type=str, nargs="+", default=[],
                        help="List of section (directory) names or specific file paths (relative to language content directory) to exclude. For example, 'author' will exclude all content under the 'author/' directory. 'path/to/file.md' will exclude that specific file.")
    parser.add_argument("--model", type=str, default=MODEL_NAME,
                        help=f"Model name to use (default: {MODEL_NAME})")
    parser.add_argument("--hugo-root", type=str, 
                        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
                        help="Hugo root directory (default: two levels up from script location)")
    return parser.parse_args()

def extract_text_from_markdown(content):
    """Extract text content from markdown, removing HTML tags."""
    try:
        # Convert markdown to HTML
        html = markdown.markdown(content)
        # Parse HTML and extract text
        soup = BeautifulSoup(html, 'html.parser')
        # Get text and normalize whitespace
        text = soup.get_text(separator=' ', strip=True)
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception as e:
        print(f"Error extracting text from markdown: {e}")
        return ""

def process_content_files(hugo_root=None, lang=None, content_dir=None, exclude_sections=None, path=None):
    """Process content files and extract relevant information."""
    # Determine the content directory
    if path:
        content_directory = path
    elif content_dir:
        content_directory = os.path.join(hugo_root, content_dir, lang)
    else:
        content_directory = os.path.join(hugo_root, "content", lang)
    
    print(f"Processing content files in: {content_directory}")
    
    # Check if the content directory exists
    if not os.path.exists(content_directory):
        print(f"Content directory not found: {content_directory}")
        return []
    
    # Process content files
    file_data = []
    
    # Walk through the content directory
    for root, _, files in os.walk(content_directory):
        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)
                
                # Extract relative path from content directory
                rel_path = os.path.relpath(file_path, content_directory)
                
                # Skip excluded sections/files
                if exclude_sections:
                    skip_file = False
                    for exclusion_item in exclude_sections:
                        # Case 1: Exact match for a file path (e.g., "blog/specific-post.md")
                        if rel_path == exclusion_item:
                            skip_file = True
                            break
                        # Case 2: Path is within an excluded directory (e.g., exclusion_item is "author")
                        # This checks if rel_path starts with "author/"
                        # exclusion_item.rstrip(os.sep) handles if user inputs "author" or "author/"
                        normalized_dir_pattern = exclusion_item.rstrip(os.sep) + os.sep
                        if rel_path.startswith(normalized_dir_pattern):
                            skip_file = True
                            break
                    if skip_file:
                        # You can uncomment the line below for debugging to see what's being skipped
                        # print(f"Skipping excluded item: {rel_path} due to rule: {exclusion_item}")
                        continue

                # Extract section from path
                path_parts = rel_path.split(os.sep)
                section = path_parts[0] if len(path_parts) > 1 else ""
                
                # Check if this is an index file
                is_index = file.lower() == "_index.md"
                
                try:
                    # Parse frontmatter and content using TOMLHandler
                    with open(file_path, "r", encoding="utf-8") as f:
                        post = frontmatter.load(f, handler=TOMLHandler())

                    # Extract slug - handle index files differently
                    if is_index:
                        # For _index.md files, use the directory path as the slug
                        parent_dir = os.path.dirname(rel_path)
                        if parent_dir:
                            # Get the last part of the directory path
                            slug = os.path.basename(parent_dir)
                        else:
                            # If it's in the root, use the section
                            slug = section if section else "index"
                    else:
                        # For regular files, use the slug from frontmatter or filename
                        slug = post.get("slug", os.path.splitext(file)[0])
                    
                    # Extract title from frontmatter
                    title = post.get("title", "")

                    # Extract text from content
                    body_text = extract_text_from_markdown(post.content)

                    # If body is empty, use frontmatter fields for embedding
                    # This handles files where content is stored in frontmatter (e.g., affiliate directories)
                    if not body_text.strip():
                        frontmatter_texts = []
                        # Add title and description
                        if title:
                            frontmatter_texts.append(title)
                        if post.get("description"):
                            frontmatter_texts.append(post.get("description"))
                        if post.get("shortDescription"):
                            frontmatter_texts.append(post.get("shortDescription"))
                        # Check for nested frontmatter sections (like program_overview)
                        for key, value in post.metadata.items():
                            if isinstance(value, dict):
                                for sub_key, sub_value in value.items():
                                    if sub_key in ("description", "title") and isinstance(sub_value, str):
                                        frontmatter_texts.append(sub_value)
                        text = " ".join(frontmatter_texts)
                    else:
                        text = body_text

                    # Limit text length to avoid memory issues
                    if len(text) > MAX_TEXT_LENGTH:
                        text = text[:MAX_TEXT_LENGTH]
                    
                    # Add to file data
                    file_data.append({
                        "path": rel_path,
                        "section": section,
                        "slug": slug,
                        "title": title,
                        "text": text,
                        "is_index": is_index
                    })
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
    
    print(f"Found {len(file_data)} content files")
    return file_data

def generate_embeddings(file_data, model_name):
    """Generate embeddings for the file data using the specified model."""
    # Load the model (will reuse if already loaded)
    model = load_model(model_name)
    
    # Process files in batches to manage memory
    embeddings = []
    batch_size = 100  # Process multiple files at a time, adjust based on memory constraints
    
    for i in range(0, len(file_data), batch_size):
        batch = file_data[i:i+batch_size]
        texts = [item['text'] for item in batch]
        
        print(f"Processing batch {i//batch_size + 1}/{(len(file_data) + batch_size - 1)//batch_size}")
        
        # Generate embeddings using sentence_transformers
        batch_embeddings = model.encode(texts, show_progress_bar=False)
        
        embeddings.extend(batch_embeddings)
    
    # Convert list to numpy array
    embeddings = np.array(embeddings).astype('float32')
    
    return embeddings

def build_index(embeddings):
    """Build a FAISS index for the embeddings."""
    print("Building FAISS index...")
    
    # Import here to delay loading until needed
    import faiss
    
    # Get the dimension of the embeddings
    dimension = embeddings.shape[1]
    
    # Create a flat index (exact search)
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity with normalized vectors
    
    # Add vectors to the index
    index.add(embeddings)
    
    return index

def find_related_content(file_data, embeddings, top_k=TOP_K):
    """Find related content for each file using FAISS."""
    print("Finding related content...")
    
    # Build the index
    index = build_index(embeddings)
    
    # Find related content for each file
    related_content = defaultdict(lambda: defaultdict(list))
    
    for i, file_info in enumerate(tqdm(file_data)):
        section = file_info['section']
        slug = file_info['slug']
        current_path = file_info['path']
        is_index = file_info.get('is_index', False)
        
        # Normalize current path for comparison
        current_normalized_path = current_path
        if current_normalized_path.endswith('.md'):
            current_normalized_path = current_normalized_path[:-3]
        
        # Skip if no section (like root _index.md)
        if not section:
            continue
        
        # Search for similar files
        query_vector = embeddings[i].reshape(1, -1)
        # Request more results than we need since we'll filter some out
        search_k = top_k + 5  
        distances, indices = index.search(query_vector, min(search_k, len(file_data)))
        
        # Add related content (excluding the file itself)
        added_count = 0
        for j in indices[0]:
            if added_count >= top_k:
                break
                
            if j < len(file_data):  # Check bounds
                related_file = file_data[j]
                related_path = related_file['path']
                related_is_index = related_file.get('is_index', False)
                
                # Convert path format: section/file.md -> section/file
                if related_path.endswith('.md'):
                    related_path = related_path[:-3]
                
                # Special handling for _index.md files - use their directory
                if related_is_index:
                    # Get the directory containing the _index.md file
                    related_dir = os.path.dirname(related_path)
                    if related_dir:
                        related_path = related_dir
                
                # Skip if this is the same file, if the path matches, or if it's an _index file
                if j == i or related_path == current_normalized_path or related_is_index:
                    continue
                
                # Add to related content
                related_content[section][slug].append({
                    'file': related_path
                })
                added_count += 1
    
    # Free memory
    index = None
    gc.collect()
    
    return related_content

def convert_defaultdict_to_dict(d):
    """Convert a defaultdict to a regular dictionary recursively."""
    if isinstance(d, defaultdict):
        d = {k: convert_defaultdict_to_dict(v) for k, v in d.items()}
    return d

def generate_yaml(related_content, hugo_root, output_dir, lang):
    """Generate YAML file with related content structure."""
    print(f"Generating YAML file for language: {lang}")

    # Create output directory if it doesn't exist
    output_path = os.path.join(hugo_root, output_dir)
    os.makedirs(output_path, exist_ok=True)

    # Convert defaultdict to regular dict for YAML serialization
    yaml_data = convert_defaultdict_to_dict(related_content)

    # Write YAML file
    output_file = os.path.join(output_path, f"{lang}.yaml")
    with open(output_file, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)

    print(f"YAML file generated: {output_file}")

def create_clusters(embeddings, max_cluster_size=50, _prev_max_size=None):
    """Create clusters using FAISS index with maximum cluster size constraint."""
    import faiss

    n_samples = len(embeddings)

    # Prevent infinite recursion and division by zero
    if max_cluster_size < 1:
        print("Warning: max_cluster_size reached minimum, falling back to sequential clustering...")
        labels = np.array([i // 50 for i in range(n_samples)])
        return labels

    # Calculate optimal number of clusters to ensure max items per cluster
    n_clusters = max(int(np.ceil(n_samples / max_cluster_size)), 1)

    print(f"Creating {n_clusters} clusters using FAISS (max {max_cluster_size} items per cluster)...")
    print(f"Total items: {n_samples}")

    try:
        # Normalize embeddings for cosine similarity
        normalized_embeddings = normalize(embeddings.astype('float32'))

        # Get embedding dimension
        d = normalized_embeddings.shape[1]

        # Use a simpler approach: select k random centroids and assign points to nearest
        # This avoids the memory-intensive k-means training
        np.random.seed(42)  # For reproducibility
        centroid_indices = np.random.choice(n_samples, size=min(n_clusters, n_samples), replace=False)
        centroids = normalized_embeddings[centroid_indices]

        # Create FAISS index for centroids
        index = faiss.IndexFlatIP(d)  # Inner product for cosine similarity
        index.add(centroids)

        # Assign each point to nearest centroid
        _, labels = index.search(normalized_embeddings, 1)
        labels = labels.flatten()

        # Calculate cluster sizes
        unique_labels, counts = np.unique(labels, return_counts=True)

        print(f"Created {len(unique_labels)} clusters")
        print(f"Cluster sizes: min={min(counts)}, max={max(counts)}, avg={np.mean(counts):.1f}")

        # If any cluster is too large, we need to split it
        max_actual_size = max(counts)
        if max_actual_size > max_cluster_size:
            # Check if we're making progress - if max size didn't change, items have identical embeddings
            if _prev_max_size is not None and max_actual_size >= _prev_max_size:
                print(f"Warning: Clustering not improving (likely identical embeddings). Using sequential fallback...")
                labels = np.array([i // 50 for i in range(n_samples)])
                return labels

            print(f"Warning: Some clusters exceed max size ({max_actual_size} > {max_cluster_size})")
            print(f"Adjusting cluster count...")
            # Recursively increase cluster count if needed, tracking previous max size
            return create_clusters(embeddings, max_cluster_size=max_cluster_size // 2, _prev_max_size=max_actual_size)

        return labels

    except Exception as e:
        print(f"Error during clustering: {e}")
        print("Falling back to simple sequential clustering...")
        # Fallback: simple sequential clusters
        labels = np.array([i // 50 for i in range(n_samples)])
        return labels

def build_hierarchy_data(file_data, labels, lang):
    """Build hierarchical data structure for D3.js pack layout."""
    print("Building hierarchy data for clustering visualization...")

    # Group files by cluster
    clusters = defaultdict(list)
    for idx, label in enumerate(labels):
        clusters[int(label)].append(idx)

    # Build D3-compatible hierarchy
    children = []

    # Sort clusters by ID for consistent ordering
    sorted_clusters = sorted(clusters.items())

    # Process each cluster
    for cluster_id, indices in sorted_clusters:
        # Create cluster node
        cluster_children = []
        cluster_name = f"Cluster {cluster_id + 1}"

        # Find a representative name for the cluster (most common section)
        sections = [file_data[idx]["section"] for idx in indices if file_data[idx]["section"]]
        if sections:
            most_common_section = max(set(sections), key=sections.count)
            if most_common_section:
                cluster_name = most_common_section.replace("-", " ").replace("_", " ").title()

        for idx in indices:
            file_info = file_data[idx]
            section = file_info["section"]
            slug = file_info["slug"]
            is_index = file_info.get("is_index", False)

            # Build URL path
            if is_index:
                url_path = f"/{section}/" if section else "/"
            else:
                url_path = f"/{section}/{slug}/" if section else f"/{slug}/"

            # Add language prefix for non-English
            if lang != "en":
                url_path = f"/{lang}{url_path}"

            cluster_children.append({
                "name": file_info["title"] or file_info["slug"],
                "title": file_info["title"],
                "url": url_path,
                "section": file_info["section"],
                "value": 1,
                "cluster": int(cluster_id)
            })

        children.append({
            "name": cluster_name,
            "cluster": int(cluster_id),
            "children": cluster_children,
            "size": len(cluster_children)
        })

    # Root node
    root = {
        "name": "Website Pages",
        "children": children
    }

    return root

def save_clustering_data(data, hugo_root, lang):
    """Save clustering data as JSON file."""
    output_dir = os.path.join(hugo_root, "data", "clustering")
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f"{lang}.json")

    print(f"Saving clustering data to: {output_file}")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Clustering data saved successfully")

def process_language(args, lang):
    """Process a single language."""
    import time
    print(f"\n[DEBUG] Processing language: {lang}")
    print(f"[DEBUG] Process start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Determine the content directory for this language
    content_dir = os.path.join(args.path, lang) if args.path else os.path.join(args.hugo_root, "content", lang)
    print(f"[DEBUG] Language content directory: {content_dir}")
    
    # Process content files
    print(f"[DEBUG] Collecting content files...")
    file_data = process_content_files(hugo_root=args.hugo_root, path=content_dir, exclude_sections=args.exclude_sections)
    
    if not file_data:
        print(f"[DEBUG] No content files found for language: {lang}")
        return
    
    print(f"[DEBUG] Found {len(file_data)} content files for {lang}")
    
    # Generate embeddings
    print(f"[DEBUG] Generating embeddings (this may take a while)...")
    print(f"[DEBUG] Loading model: {args.model}")
    embeddings = generate_embeddings(file_data, args.model)
    print(f"[DEBUG] Embeddings generated for {len(embeddings)} files")
    
    # Find related content
    print(f"[DEBUG] Finding related content...")
    related_content = find_related_content(file_data, embeddings)
    print(f"[DEBUG] Related content found")

    # Convert defaultdict to regular dict for clean YAML output
    print(f"[DEBUG] Converting data structure...")
    related_content_dict = convert_defaultdict_to_dict(related_content)

    # Generate YAML file
    yaml_dir = os.path.join(args.hugo_root, "data", "related_content")
    os.makedirs(yaml_dir, exist_ok=True)
    yaml_file = os.path.join(yaml_dir, f"{lang}.yaml")

    print(f"[DEBUG] Generating YAML file for language: {lang}")
    print(f"[DEBUG] YAML path: {yaml_file}")
    with open(yaml_file, "w") as f:
        yaml.dump(related_content_dict, f, default_flow_style=False)

    print(f"[DEBUG] YAML file generated: {yaml_file}")

    # Generate clustering data (reusing the same embeddings)
    print(f"[DEBUG] Generating clustering data for visualization...")
    labels = create_clusters(embeddings, max_cluster_size=50)
    hierarchy_data = build_hierarchy_data(file_data, labels, lang)
    save_clustering_data(hierarchy_data, args.hugo_root, lang)
    print(f"[DEBUG] Clustering data generated successfully")

    # Clean up memory for this language
    print(f"[DEBUG] Cleaning up memory...")
    gc.collect()
    print(f"[DEBUG] Process end time for {lang}: {time.strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    """Main function to run the script."""
    import time
    print(f"\n[DEBUG] ========== RELATED CONTENT GENERATOR STARTING ===========")
    print(f"[DEBUG] Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    args = parse_args()
    
    print(f"[DEBUG] Arguments:")
    print(f"[DEBUG] - Hugo root: {args.hugo_root}")
    print(f"[DEBUG] - Exclude sections: {args.exclude_sections}")
    
    # Find all language directories
    if args.path:
        content_dir = args.path
    else:
        content_dir = os.path.join(args.hugo_root, "content")
    
    print(f"[DEBUG] Content directory: {content_dir}")
    
    # Check if content directory exists
    if not os.path.exists(content_dir):
        print(f"[ERROR] Content directory not found: {content_dir}")
        return
    
    # Find all language directories
    print(f"[DEBUG] Scanning for language directories...")
    languages = [d for d in os.listdir(content_dir) 
                if os.path.isdir(os.path.join(content_dir, d)) and not d.startswith('_')]
    
    print(f"[DEBUG] Found {len(languages)} languages: {', '.join(languages)}")
    
    # Process each language
    for idx, lang in enumerate(languages, 1):
        print(f"\n[DEBUG] Processing language {idx}/{len(languages)}: {lang}")
        print(f"[DEBUG] Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        process_language(args, lang)
        print(f"[DEBUG] Completed language: {lang}")
    
    print(f"\n[DEBUG] ========== RELATED CONTENT GENERATOR FINISHED ===========")
    print(f"[DEBUG] End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Clean up global model resources at the end
    if '_model' in globals() and _model is not None:
        del globals()['_model']
    gc.collect()

if __name__ == "__main__":
    main()
