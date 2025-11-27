#!/usr/bin/env python3
"""
Generate Website Clustering Visualization Data

This script creates a hierarchical clustering visualization of website pages
based on semantic similarity using embeddings and clustering.

Usage:
    python generate_clustering.py --lang en
    python generate_clustering.py --lang sk --num-clusters 50
"""

import os
import json
import argparse
import frontmatter
from pathlib import Path
from collections import defaultdict
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN
import markdown
from bs4 import BeautifulSoup
import re
import html

# Constants
MODEL_NAME = "Alibaba-NLP/gte-multilingual-base"
MIN_CLUSTER_SIZE = 5  # Minimum pages per cluster
MIN_SAMPLES = 3  # Minimum samples for core point
MAX_CLUSTER_SIZE = 5  # Maximum pages before creating subclusters (recursive subdivision)
DOMAIN_NAME = "Post Affiliate Pro"  # Domain name for root cluster

# Note: We use URL path + title for embeddings rather than full content
# This provides better semantic clustering based on page structure and purpose

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate clustering visualization data")
    parser.add_argument("--lang", type=str, required=True, help="Language code (e.g., en, sk)")
    parser.add_argument("--hugo-root", type=str, default=".", help="Hugo project root directory")
    parser.add_argument("--min-cluster-size", type=int, default=MIN_CLUSTER_SIZE, help="Minimum pages per cluster")
    parser.add_argument("--min-samples", type=int, default=MIN_SAMPLES, help="Minimum samples for core point")
    parser.add_argument("--exclude-sections", type=str, nargs="+", default=["author"], help="Sections to exclude")
    return parser.parse_args()

def clean_title(title):
    """Clean title by removing HTML tags and unescaping HTML entities."""
    try:
        # Remove HTML tags
        soup = BeautifulSoup(title, 'html.parser')
        clean = soup.get_text(separator=' ', strip=True)
        # Unescape HTML entities
        clean = html.unescape(clean)
        # Remove any remaining problematic characters
        clean = clean.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        # Normalize whitespace
        clean = re.sub(r'\s+', ' ', clean).strip()
        return clean
    except Exception as e:
        print(f"Error cleaning title '{title}': {e}")
        return title

def extract_text_from_markdown(content):
    """Extract text content from markdown."""
    try:
        html = markdown.markdown(content)
        soup = BeautifulSoup(html, 'html.parser')
        text = soup.get_text(separator=' ', strip=True)
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:MAX_TEXT_LENGTH]
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""

def process_content_files(content_dir, lang, exclude_sections):
    """Process content files and extract information."""
    lang_dir = Path(content_dir) / lang

    if not lang_dir.exists():
        print(f"Content directory not found: {lang_dir}")
        return []

    pages = []

    print(f"Processing content files in: {lang_dir}")

    for md_file in lang_dir.rglob("*.md"):
        # Get relative path
        rel_path = md_file.relative_to(lang_dir)

        # Skip excluded sections
        skip = False
        for exclusion in exclude_sections:
            if str(rel_path).startswith(exclusion + "/") or str(rel_path) == exclusion:
                skip = True
                break

        if skip:
            continue

        try:
            # Parse frontmatter
            post = frontmatter.load(md_file)

            # Extract metadata
            title = post.get('title', '')

            # Skip if no title
            if not title:
                continue

            # Clean title to remove HTML and special characters
            title = clean_title(title)

            # Check for custom url in frontmatter
            if 'url' in post.metadata:
                # Use custom URL from frontmatter
                url = post.get('url')
                # Ensure it starts with /
                if not url.startswith('/'):
                    url = '/' + url
                # Extract semantic meaning from custom URL for embedding
                url_semantic = url.replace('/', ' ').replace('-', ' ').replace('_', ' ').strip()
            else:
                # Build URL from file path (without language prefix since each language has its own domain)
                url_path = str(rel_path).replace('.md', '').replace('_index', '').replace('index', '')
                if url_path and not url_path.endswith('/'):
                    url_path += '/'
                # Don't include language code in URL - each language is on separate domain
                url = f"/{url_path}" if url_path else '/'
                # Extract semantic meaning from URL path
                url_semantic = url_path.replace('/', ' ').replace('-', ' ').replace('_', ' ').strip()

            # Get section (first directory in path)
            parts = str(rel_path).split('/')
            section = parts[0] if len(parts) > 1 else ""

            # Combine URL semantic meaning with title for embedding
            # This focuses on structure and key concepts rather than full content
            combined_text = f"{url_semantic} {title}"

            pages.append({
                'title': title,
                'url': url,
                'section': section,
                'text': combined_text,
                'file_path': str(rel_path)
            })

        except Exception as e:
            print(f"Error processing {md_file}: {e}")
            continue

    print(f"Processed {len(pages)} pages")
    return pages

def generate_embeddings(pages, model_name):
    """Generate embeddings for pages."""
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name, trust_remote_code=True)

    print("Generating embeddings...")
    texts = [page['text'] for page in pages]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)

    return embeddings

def cluster_pages_by_section(pages, embeddings, min_cluster_size, min_samples):
    """Organize pages by section/directory, then cluster within each section."""
    print(f"Organizing by sections and clustering within each...")

    # First, group pages by section
    sections = defaultdict(list)
    section_embeddings = defaultdict(list)

    for i, page in enumerate(pages):
        section = page['section'] if page['section'] else 'root'
        page['embedding_index'] = i
        page['value'] = 1
        sections[section].append(page)
        section_embeddings[section].append(embeddings[i])

    print(f"Found {len(sections)} sections")

    # Now cluster within each section
    section_clusters = {}

    for section, section_pages in sections.items():
        print(f"\n  Section '{section}': {len(section_pages)} pages")

        if len(section_pages) < min_cluster_size:
            # Too few pages to cluster, treat as single cluster
            print(f"    Too few pages, keeping as single group")
            section_clusters[section] = {
                'clusters': {0: section_pages},
                'centers': {0: section_pages[0]}
            }
            continue

        # Get embeddings for this section
        sec_embeddings = np.array(section_embeddings[section])

        # Cluster pages within this section
        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',
            cluster_selection_method='eom'
        )
        cluster_labels = clusterer.fit_predict(sec_embeddings)

        # Count clusters
        num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        num_noise = list(cluster_labels).count(-1)

        print(f"    Found {num_clusters} semantic clusters, {num_noise} noise points")

        # Organize pages by cluster within this section
        clusters = defaultdict(list)

        for i, page in enumerate(section_pages):
            label = int(cluster_labels[i])
            # Put noise points in cluster -1
            clusters[label].append(page)

        # Find centroid page for each cluster
        cluster_centers = {}
        for cluster_id in clusters.keys():
            cluster_pages = clusters[cluster_id]
            cluster_embs = np.array([embeddings[p['embedding_index']] for p in cluster_pages])
            centroid = np.mean(cluster_embs, axis=0)

            # Find closest page to centroid
            min_distance = float('inf')
            closest_page = cluster_pages[0]

            for page in cluster_pages:
                embedding = embeddings[page['embedding_index']]
                distance = np.linalg.norm(embedding - centroid)

                if distance < min_distance:
                    min_distance = distance
                    closest_page = page

            cluster_centers[cluster_id] = closest_page

        section_clusters[section] = {
            'clusters': clusters,
            'centers': cluster_centers
        }

    return section_clusters

def create_recursive_subclusters(pages, embeddings, depth=0):
    """Recursively create subclusters until each cluster has <= MAX_CLUSTER_SIZE pages.

    Args:
        pages: List of page dictionaries
        embeddings: Numpy array of embeddings
        depth: Current recursion depth (for logging)

    Returns:
        List of cluster node dictionaries (with potential nested children)
    """
    indent = "  " * depth

    # If small enough, return as leaf pages
    if len(pages) <= MAX_CLUSTER_SIZE:
        print(f"{indent}Cluster with {len(pages)} pages - no subdivision needed")
        return None

    print(f"{indent}Subdividing cluster with {len(pages)} pages (depth {depth})...")

    # Get embeddings for this cluster
    page_embeddings = np.array([embeddings[p['embedding_index']] for p in pages])

    # Use adaptive min_cluster_size based on number of pages
    # For smaller groups, use smaller min_cluster_size to force subdivision
    if len(pages) < 20:
        sub_min_cluster_size = 2
        sub_min_samples = 1
    else:
        sub_min_cluster_size = max(2, MIN_CLUSTER_SIZE // 2)
        sub_min_samples = 2

    clusterer = HDBSCAN(
        min_cluster_size=sub_min_cluster_size,
        min_samples=sub_min_samples,
        metric='euclidean',
        cluster_selection_method='eom'
    )
    sub_labels = clusterer.fit_predict(page_embeddings)

    # Organize into subclusters
    subclusters = defaultdict(list)
    for i, page in enumerate(pages):
        label = int(sub_labels[i])
        subclusters[label].append(page)

    # Count valid clusters (non-noise)
    valid_clusters = {k: v for k, v in subclusters.items() if k != -1}

    # If we didn't get meaningful subclusters, force split by simple division
    if len(valid_clusters) <= 1:
        print(f"{indent}HDBSCAN didn't split, forcing division...")
        # Split pages into chunks of MAX_CLUSTER_SIZE
        chunk_size = MAX_CLUSTER_SIZE
        subclusters = {}
        for i in range(0, len(pages), chunk_size):
            subclusters[i // chunk_size] = pages[i:i + chunk_size]

    print(f"{indent}Created {len(subclusters)} subclusters")

    # Build subcluster nodes recursively
    subcluster_nodes = []
    for sub_id in sorted(subclusters.keys()):
        sub_pages = subclusters[sub_id]

        # Find centroid page for this subcluster
        sub_embs = np.array([embeddings[p['embedding_index']] for p in sub_pages])
        sub_centroid = np.mean(sub_embs, axis=0)

        min_dist = float('inf')
        center_page = sub_pages[0]
        for page in sub_pages:
            emb = embeddings[page['embedding_index']]
            dist = np.linalg.norm(emb - sub_centroid)
            if dist < min_dist:
                min_dist = dist
                center_page = page

        # Recursively subdivide if this subcluster is still too large
        nested_children = None
        if len(sub_pages) > MAX_CLUSTER_SIZE:
            nested_children = create_recursive_subclusters(sub_pages, embeddings, depth + 1)

        if nested_children:
            # This subcluster needs further subdivision
            subcluster_node = {
                "name": center_page['title'],
                "cluster": hash(f"{center_page['url']}_{depth}_{sub_id}") % 100000,
                "children": nested_children,
                "size": len(sub_pages)
            }
        else:
            # This is a leaf cluster - add pages directly
            subcluster_node = {
                "name": center_page['title'],
                "cluster": hash(f"{center_page['url']}_{depth}_{sub_id}") % 100000,
                "children": [],
                "size": len(sub_pages)
            }

            # Add page nodes
            for page in sub_pages:
                page_node = {
                    "name": page['title'],
                    "title": page['title'],
                    "url": page['url'],
                    "section": page['section'],
                    "value": 1,
                    "cluster": hash(f"{page['url']}") % 100000
                }
                subcluster_node["children"].append(page_node)

        subcluster_nodes.append(subcluster_node)

    return subcluster_nodes

def create_hierarchical_structure(section_clusters, embeddings, lang):
    """Create hierarchical JSON structure for D3.js visualization."""
    children = []

    # Iterate through each section
    for section_name in sorted(section_clusters.keys()):
        section_data = section_clusters[section_name]
        clusters = section_data['clusters']
        centers = section_data['centers']

        # Create section node
        # Format section name nicely
        formatted_section = section_name.replace('-', ' ').replace('_', ' ').title()
        if section_name == 'root':
            formatted_section = 'Home Pages'

        section_node = {
            "name": formatted_section,
            "cluster": hash(section_name) % 10000,  # Unique ID for section
            "section": section_name,
            "children": []
        }

        # Add clusters within this section
        for cluster_id in sorted(clusters.keys()):
            cluster_pages = clusters[cluster_id]
            center_page = centers.get(cluster_id, cluster_pages[0])

            # Skip noise cluster if only has a few pages
            if cluster_id == -1 and len(cluster_pages) < 3:
                # Add noise pages directly to section
                for page in cluster_pages:
                    page_node = {
                        "name": page['title'],
                        "title": page['title'],
                        "url": page['url'],
                        "section": page['section'],
                        "value": 1,
                        "cluster": cluster_id
                    }
                    section_node["children"].append(page_node)
                continue

            # Use the title of the page closest to centroid as cluster name
            if cluster_id == -1:
                cluster_name = "Miscellaneous"
            else:
                cluster_name = center_page['title'] if center_page else f"Cluster {cluster_id + 1}"

            # Check if this cluster needs recursive subdivision
            if len(cluster_pages) > MAX_CLUSTER_SIZE:
                print(f"  Cluster '{cluster_name}' has {len(cluster_pages)} pages, creating recursive subclusters...")
                subclusters = create_recursive_subclusters(cluster_pages, embeddings, depth=0)

                if subclusters:
                    # Create wrapper cluster node with subclusters
                    cluster_node = {
                        "name": cluster_name,
                        "cluster": hash(f"{section_name}_{cluster_id}") % 10000,
                        "children": subclusters,
                        "size": len(cluster_pages)
                    }
                else:
                    # Fallback: add pages directly if subdivision failed
                    cluster_node = {
                        "name": cluster_name,
                        "cluster": hash(f"{section_name}_{cluster_id}") % 10000,
                        "children": [],
                        "size": len(cluster_pages)
                    }
                    for page in cluster_pages:
                        page_node = {
                            "name": page['title'],
                            "title": page['title'],
                            "url": page['url'],
                            "section": page['section'],
                            "value": 1,
                            "cluster": cluster_id
                        }
                        cluster_node["children"].append(page_node)
            else:
                # Small cluster - add pages directly
                cluster_node = {
                    "name": cluster_name,
                    "cluster": hash(f"{section_name}_{cluster_id}") % 10000,
                    "children": [],
                    "size": len(cluster_pages)
                }

                for page in cluster_pages:
                    page_node = {
                        "name": page['title'],
                        "title": page['title'],
                        "url": page['url'],
                        "section": page['section'],
                        "value": 1,
                        "cluster": cluster_id
                    }
                    cluster_node["children"].append(page_node)

            section_node["children"].append(cluster_node)

        # Calculate total size for section
        section_node["size"] = sum(len(clusters[cid]) for cid in clusters.keys())
        children.append(section_node)

    # Create root node with domain name and language
    lang_names = {
        'en': 'English',
        'sk': 'Slovak',
        'de': 'German',
        'es': 'Spanish',
        'fr': 'French',
        'hu': 'Hungarian',
        'it': 'Italian',
        'nl': 'Dutch',
        'pl': 'Polish',
        'pt-br': 'Portuguese (Brazil)'
    }

    lang_name = lang_names.get(lang, lang.upper())
    root = {
        "name": f"{DOMAIN_NAME} - {lang_name}",
        "children": children
    }

    return root

def main():
    args = parse_args()

    # Setup paths relative to Hugo root
    hugo_root = Path(args.hugo_root)
    content_dir = hugo_root / "content"
    static_output_dir = hugo_root / "static" / "data" / "clustering"

    # Process content files
    pages = process_content_files(str(content_dir), args.lang, args.exclude_sections)

    if not pages:
        print("No pages found to process")
        return

    # Generate embeddings
    embeddings = generate_embeddings(pages, MODEL_NAME)

    # Cluster pages by section
    section_clusters = cluster_pages_by_section(pages, embeddings, args.min_cluster_size, args.min_samples)

    # Create hierarchical structure
    hierarchy = create_hierarchical_structure(section_clusters, embeddings, args.lang)

    # Save to static directory only
    static_output_dir.mkdir(parents=True, exist_ok=True)
    output_file = static_output_dir / f"{args.lang}.json"

    print(f"\nWriting output to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(hierarchy, f, ensure_ascii=False, indent=2)

    print(f"\nSuccessfully generated clustering data for {args.lang}")
    print(f"Total pages: {len(pages)}")
    print(f"Total sections: {len(section_clusters)}")

    # Count total clusters across all sections
    total_clusters = sum(len(section_clusters[s]['clusters']) for s in section_clusters)
    print(f"Total semantic clusters: {total_clusters}")

if __name__ == "__main__":
    main()
