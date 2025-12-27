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
import toml_frontmatter as frontmatter  # Use robust TOML parser
from pathlib import Path
from collections import defaultdict
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN
import markdown
from bs4 import BeautifulSoup
import re
import html
import faiss
import umap

# Constants
MODEL_NAME = "Alibaba-NLP/gte-multilingual-base"
MIN_CLUSTER_SIZE = 5  # Minimum pages per cluster
MIN_SAMPLES = 3  # Minimum samples for core point
MAX_CLUSTER_SIZE = 5  # Maximum pages before creating subclusters (recursive subdivision)

def get_site_title(hugo_root, lang='en'):
    """Read site title from Hugo config files."""
    import toml

    # Try multiple config locations
    config_paths = [
        hugo_root / "config" / "_default" / "languages.toml",
        hugo_root / f"config_{lang}" / "_default" / "languages.toml",
        hugo_root / "config.toml",
    ]

    for config_path in config_paths:
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = toml.load(f)

                # Try to find title in various locations
                if 'title' in config:
                    return config['title']
                if lang in config and 'title' in config[lang]:
                    return config[lang]['title']
                if 'languages' in config and lang in config['languages']:
                    if 'title' in config['languages'][lang]:
                        return config['languages'][lang]['title']
            except Exception as e:
                print(f"Warning: Could not parse {config_path}: {e}")
                continue

    # Fallback to site name from all_languages.yaml
    all_langs_path = hugo_root / "data" / "all_languages.yaml"
    if all_langs_path.exists():
        try:
            import yaml
            with open(all_langs_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            if 'languages' in data and lang in data['languages']:
                if 'baseDomainName' in data['languages'][lang]:
                    # Extract domain name from URL
                    domain = data['languages'][lang]['baseDomainName']
                    # Remove https:// and www.
                    domain = domain.replace('https://', '').replace('http://', '').replace('www.', '')
                    # Remove trailing slash and .com/.io etc
                    domain = domain.rstrip('/').split('.')[0]
                    return domain.title()
        except Exception as e:
            print(f"Warning: Could not parse {all_langs_path}: {e}")

    return "Website"  # Final fallback

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
    parser.add_argument("--num-clusters", type=int, default=50, help="Number of global clusters for scatterplot (default: 50)")
    parser.add_argument("--similarity-threshold", type=float, default=0.90, help="Similarity threshold for duplicate detection (default: 0.90)")
    parser.add_argument("--k-neighbors", type=int, default=15, help="Number of neighbors to check for similarity (default: 15)")
    parser.add_argument("--exclude-similarity-sections", type=str, nargs="+",
                        default=["affiliate-manager", "affiliate-program-directory"],
                        help="Sections to exclude from similarity report (default: affiliate-manager, affiliate-program-directory)")
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
            with open(md_file, 'r', encoding='utf-8') as f:
                post = frontmatter.load(f)

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

            # Get linkbuilding keywords from frontmatter
            linkbuilding = post.get('linkbuilding', [])
            if linkbuilding is None:
                linkbuilding = []

            pages.append({
                'title': title,
                'url': url,
                'section': section,
                'text': combined_text,
                'file_path': str(rel_path),
                'keywords': linkbuilding
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
            cluster_selection_method='eom',
            copy=True,
            n_jobs=1  # Single-threaded to avoid semaphore leaks
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


def cluster_all_pages_global(pages, embeddings, num_clusters=50):
    """Cluster all pages globally using FAISS k-means.

    Args:
        pages: List of page dictionaries
        embeddings: Numpy array of embeddings (n_pages x embedding_dim)
        num_clusters: Number of clusters to create

    Returns:
        Tuple of (cluster_labels, cluster_centroids)
    """
    print(f"\nPerforming global clustering with FAISS k-means ({num_clusters} clusters)...")

    # Ensure embeddings are float32 for FAISS
    embeddings_float32 = embeddings.astype(np.float32)

    # Get embedding dimension
    d = embeddings_float32.shape[1]

    # Create FAISS k-means
    kmeans = faiss.Kmeans(d=d, k=num_clusters, niter=50, verbose=True, seed=42)
    kmeans.train(embeddings_float32)

    # Get cluster assignments
    _, cluster_labels = kmeans.index.search(embeddings_float32, 1)
    cluster_labels = cluster_labels.flatten()

    # Get centroids
    centroids = kmeans.centroids

    # Count pages per cluster
    unique, counts = np.unique(cluster_labels, return_counts=True)

    print(f"  Cluster sizes: min={min(counts)}, max={max(counts)}, avg={np.mean(counts):.1f}")

    return cluster_labels, centroids


def generate_2d_projections(embeddings, n_neighbors=15, min_dist=0.1, random_state=42):
    """Generate 2D projections using UMAP.

    Args:
        embeddings: Numpy array of embeddings
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        random_state: Random seed for reproducibility

    Returns:
        Numpy array of shape (n_pages, 2) with x, y coordinates
    """
    print(f"\nGenerating 2D projections with UMAP...")
    print(f"  n_neighbors={n_neighbors}, min_dist={min_dist}")

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric='cosine',
        random_state=random_state,
        verbose=True,
        n_jobs=1  # Single-threaded to avoid semaphore leaks
    )

    projections = reducer.fit_transform(embeddings)

    # Normalize to roughly [-1, 1] range for consistent display
    # Use 5th and 95th percentile to avoid outlier influence
    for i in range(2):
        p5, p95 = np.percentile(projections[:, i], [5, 95])
        center = (p5 + p95) / 2
        scale = (p95 - p5) / 2 if p95 != p5 else 1
        projections[:, i] = (projections[:, i] - center) / scale

    print(f"  Projection range: x=[{projections[:, 0].min():.2f}, {projections[:, 0].max():.2f}], "
          f"y=[{projections[:, 1].min():.2f}, {projections[:, 1].max():.2f}]")

    return projections


def generate_scatterplot_data(pages, embeddings, projections, cluster_labels, centroids):
    """Generate scatterplot data structure for visualization.

    Args:
        pages: List of page dictionaries
        embeddings: Numpy array of embeddings
        projections: 2D projections (n_pages x 2)
        cluster_labels: Cluster assignment for each page
        centroids: Cluster centroids in embedding space

    Returns:
        Dictionary with scatterplot data
    """
    print(f"\nGenerating scatterplot data structure...")

    # Build page data with coordinates
    page_data = []
    for i, page in enumerate(pages):
        page_data.append({
            "title": page['title'],
            "url": page['url'],
            "section": page['section'],
            "x": float(projections[i, 0]),
            "y": float(projections[i, 1]),
            "cluster": int(cluster_labels[i]),
            "keywords": page.get('keywords', [])[:10]  # Limit keywords
        })

    # Project centroids to 2D (approximate using mean of cluster points)
    num_clusters = len(centroids)
    cluster_data = []

    for cluster_id in range(num_clusters):
        # Find pages in this cluster
        mask = cluster_labels == cluster_id
        if np.any(mask):
            cluster_points = projections[mask]
            centroid_x = float(np.mean(cluster_points[:, 0]))
            centroid_y = float(np.mean(cluster_points[:, 1]))
            size = int(np.sum(mask))
        else:
            centroid_x = 0.0
            centroid_y = 0.0
            size = 0

        cluster_data.append({
            "id": cluster_id,
            "centroid_x": centroid_x,
            "centroid_y": centroid_y,
            "size": size
        })

    scatterplot_data = {
        "pages": page_data,
        "clusters": cluster_data,
        "num_clusters": num_clusters,
        "total_pages": len(pages)
    }

    return scatterplot_data


def find_similar_articles(pages, embeddings, similarity_threshold=0.90, k_neighbors=10):
    """Find groups of articles that are too similar using FAISS.

    Uses FAISS k-NN search to efficiently find similar articles, then groups
    connected similar articles using Union-Find algorithm.

    Args:
        pages: List of page dictionaries
        embeddings: Numpy array of embeddings
        similarity_threshold: Cosine similarity threshold (0-1). Default 0.90 means
                             articles with >90% similarity are flagged.
        k_neighbors: Number of nearest neighbors to check per article

    Returns:
        List of similarity groups, each containing:
        - pages: List of (page_index, page_info) tuples
        - pairs: List of (idx1, idx2, similarity) tuples showing connections
    """
    print(f"\nFinding similar articles (threshold: {similarity_threshold:.0%})...")

    n_pages = len(pages)
    if n_pages < 2:
        return []

    # Normalize embeddings for cosine similarity (FAISS uses L2, so we normalize)
    embeddings_float32 = embeddings.astype(np.float32)
    faiss.normalize_L2(embeddings_float32)

    # Build FAISS index
    d = embeddings_float32.shape[1]
    index = faiss.IndexFlatIP(d)  # Inner product = cosine similarity after normalization
    index.add(embeddings_float32)

    # Search for k nearest neighbors for each vector
    k = min(k_neighbors + 1, n_pages)  # +1 because each vector is its own neighbor
    similarities, indices = index.search(embeddings_float32, k)

    # Find all pairs above threshold
    similar_pairs = []
    for i in range(n_pages):
        for j_idx in range(1, k):  # Skip first (self)
            neighbor_idx = indices[i, j_idx]
            sim = similarities[i, j_idx]

            if sim >= similarity_threshold and i < neighbor_idx:  # Avoid duplicates
                similar_pairs.append((i, neighbor_idx, float(sim)))

    print(f"  Found {len(similar_pairs)} similar pairs above threshold")

    if not similar_pairs:
        return []

    # Use Union-Find to group connected similar articles
    parent = list(range(n_pages))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Union all similar pairs
    for i, j, _ in similar_pairs:
        union(i, j)

    # Group pages by their root
    groups_dict = {}
    for i in range(n_pages):
        root = find(i)
        if root not in groups_dict:
            groups_dict[root] = []
        groups_dict[root].append(i)

    # Filter to only groups with multiple pages
    similarity_groups = []
    for root, members in groups_dict.items():
        if len(members) > 1:
            # Get pairs within this group
            group_pairs = [(i, j, sim) for i, j, sim in similar_pairs
                          if find(i) == root]
            group_pairs.sort(key=lambda x: -x[2])  # Sort by similarity desc

            # Sort members by most connections
            member_connections = {m: 0 for m in members}
            for i, j, _ in group_pairs:
                member_connections[i] = member_connections.get(i, 0) + 1
                member_connections[j] = member_connections.get(j, 0) + 1

            sorted_members = sorted(members, key=lambda m: -member_connections[m])

            similarity_groups.append({
                'pages': [(idx, pages[idx]) for idx in sorted_members],
                'pairs': group_pairs,
                'max_similarity': max(sim for _, _, sim in group_pairs) if group_pairs else 0
            })

    # Sort groups by max similarity (most similar first)
    similarity_groups.sort(key=lambda g: -g['max_similarity'])

    print(f"  Found {len(similarity_groups)} groups of similar articles")
    total_affected = sum(len(g['pages']) for g in similarity_groups)
    print(f"  Total affected pages: {total_affected}")

    return similarity_groups


def generate_similarity_report_html(similarity_groups, pages, output_path, lang):
    """Generate an HTML report of similar articles.

    Args:
        similarity_groups: Output from find_similar_articles()
        pages: List of page dictionaries
        output_path: Path to write HTML file
        lang: Language code for the report
    """
    print(f"\nGenerating similarity report: {output_path}")

    total_groups = len(similarity_groups)
    total_pages = sum(len(g['pages']) for g in similarity_groups)
    total_pairs = sum(len(g['pairs']) for g in similarity_groups)

    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Similar Articles Report - {lang.upper()}</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{ color: #1a1a1a; border-bottom: 3px solid #3b82f6; padding-bottom: 10px; }}
        h2 {{ color: #374151; margin-top: 30px; }}
        .summary {{
            background: #fff;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .summary-stats {{
            display: flex;
            gap: 30px;
            flex-wrap: wrap;
        }}
        .stat {{
            text-align: center;
        }}
        .stat-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #dc2626;
        }}
        .stat-label {{
            color: #6b7280;
            font-size: 0.9em;
        }}
        .group {{
            background: #fff;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .group-header {{
            background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
            color: white;
            padding: 15px 20px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .group-header:hover {{ background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%); }}
        .group-header h3 {{ margin: 0; font-size: 1.1em; }}
        .group-meta {{ font-size: 0.9em; opacity: 0.9; }}
        .group-content {{ padding: 20px; display: none; }}
        .group.expanded .group-content {{ display: block; }}
        .similarity-badge {{
            display: inline-block;
            padding: 3px 10px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: bold;
        }}
        .sim-high {{ background: #fecaca; color: #991b1b; }}
        .sim-medium {{ background: #fed7aa; color: #9a3412; }}
        .sim-low {{ background: #fef3c7; color: #92400e; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e5e7eb;
        }}
        th {{
            background: #f9fafb;
            font-weight: 600;
            color: #374151;
        }}
        tr:hover {{ background: #f9fafb; }}
        a {{
            color: #2563eb;
            text-decoration: none;
        }}
        a:hover {{ text-decoration: underline; }}
        .section-tag {{
            display: inline-block;
            background: #e5e7eb;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            color: #4b5563;
        }}
        .pairs-section {{ margin-top: 20px; }}
        .pairs-section h4 {{ color: #6b7280; margin-bottom: 10px; }}
        .pair {{
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 8px 12px;
            background: #fef2f2;
            border-radius: 6px;
            margin-bottom: 8px;
            font-size: 0.9em;
        }}
        .pair-arrow {{ color: #9ca3af; }}
        .expand-icon {{
            transition: transform 0.2s;
        }}
        .group.expanded .expand-icon {{
            transform: rotate(180deg);
        }}
        .legend {{
            background: #fff;
            padding: 15px 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
            align-items: center;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.9em;
        }}
        .filter-controls {{
            background: #fff;
            padding: 15px 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            display: flex;
            gap: 20px;
            align-items: center;
            flex-wrap: wrap;
        }}
        .filter-controls label {{ font-weight: 500; }}
        .filter-controls input, .filter-controls select {{
            padding: 8px 12px;
            border: 1px solid #d1d5db;
            border-radius: 6px;
        }}
        #expand-all {{
            padding: 8px 16px;
            background: #3b82f6;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
        }}
        #expand-all:hover {{ background: #2563eb; }}
    </style>
</head>
<body>
    <h1>🔍 Similar Articles Report - {lang.upper()}</h1>

    <div class="summary">
        <div class="summary-stats">
            <div class="stat">
                <div class="stat-value">{total_groups}</div>
                <div class="stat-label">Similar Groups</div>
            </div>
            <div class="stat">
                <div class="stat-value">{total_pages}</div>
                <div class="stat-label">Affected Pages</div>
            </div>
            <div class="stat">
                <div class="stat-value">{total_pairs}</div>
                <div class="stat-label">Similar Pairs</div>
            </div>
        </div>
        <p style="margin-top: 15px; color: #6b7280;">
            This report identifies pages with potentially duplicate or very similar content.
            Consider consolidating, differentiating, or adding canonical tags to these pages.
        </p>
    </div>

    <div class="legend">
        <span style="font-weight: 600;">Similarity Levels:</span>
        <div class="legend-item">
            <span class="similarity-badge sim-high">≥95%</span>
            <span>Nearly identical</span>
        </div>
        <div class="legend-item">
            <span class="similarity-badge sim-medium">90-95%</span>
            <span>Very similar</span>
        </div>
        <div class="legend-item">
            <span class="similarity-badge sim-low">85-90%</span>
            <span>Similar</span>
        </div>
    </div>

    <div class="filter-controls">
        <label>Filter by section:</label>
        <select id="section-filter">
            <option value="">All sections</option>
        </select>
        <label>Min similarity:</label>
        <input type="range" id="sim-filter" min="85" max="99" value="90" style="width: 150px;">
        <span id="sim-filter-value">90%</span>
        <button id="expand-all">Expand All</button>
    </div>

    <div id="groups-container">
'''

    # Generate group HTML
    for group_idx, group in enumerate(similarity_groups):
        max_sim = group['max_similarity']
        sim_class = 'sim-high' if max_sim >= 0.95 else 'sim-medium' if max_sim >= 0.90 else 'sim-low'
        pages_in_group = group['pages']
        pairs = group['pairs']

        # Get sections in this group
        sections = list(set(p['section'] for _, p in pages_in_group))
        sections_str = ', '.join(sections[:3]) + ('...' if len(sections) > 3 else '')

        html_content += f'''
        <div class="group" data-max-sim="{max_sim:.2f}" data-sections="{','.join(sections)}">
            <div class="group-header" onclick="toggleGroup(this.parentElement)">
                <h3>
                    Group {group_idx + 1}: {len(pages_in_group)} similar pages
                    <span class="similarity-badge {sim_class}">{max_sim:.1%} max</span>
                </h3>
                <div class="group-meta">
                    {sections_str}
                    <span class="expand-icon">▼</span>
                </div>
            </div>
            <div class="group-content">
                <table>
                    <thead>
                        <tr>
                            <th>Title</th>
                            <th>Section</th>
                            <th>URL</th>
                        </tr>
                    </thead>
                    <tbody>
'''

        for idx, page in pages_in_group:
            html_content += f'''
                        <tr>
                            <td><strong>{html.escape(page["title"])}</strong></td>
                            <td><span class="section-tag">{html.escape(page["section"])}</span></td>
                            <td><a href="{html.escape(page["url"])}" target="_blank">{html.escape(page["url"])}</a></td>
                        </tr>
'''

        html_content += '''
                    </tbody>
                </table>

                <div class="pairs-section">
                    <h4>Similarity Connections</h4>
'''

        # Show top pairs (limit to avoid huge reports)
        for i, j, sim in pairs[:20]:
            page_i = pages[i]
            page_j = pages[j]
            sim_class = 'sim-high' if sim >= 0.95 else 'sim-medium' if sim >= 0.90 else 'sim-low'
            html_content += f'''
                    <div class="pair">
                        <span class="similarity-badge {sim_class}">{sim:.1%}</span>
                        <a href="{html.escape(page_i["url"])}" target="_blank">{html.escape(page_i["title"][:50])}</a>
                        <span class="pair-arrow">↔</span>
                        <a href="{html.escape(page_j["url"])}" target="_blank">{html.escape(page_j["title"][:50])}</a>
                    </div>
'''

        if len(pairs) > 20:
            html_content += f'<p style="color: #6b7280; font-size: 0.9em;">... and {len(pairs) - 20} more pairs</p>'

        html_content += '''
                </div>
            </div>
        </div>
'''

    # Close HTML
    html_content += '''
    </div>

    <script>
        function toggleGroup(el) {
            el.classList.toggle('expanded');
        }

        // Section filter
        const sections = new Set();
        document.querySelectorAll('.group').forEach(g => {
            g.dataset.sections.split(',').forEach(s => sections.add(s));
        });
        const sectionSelect = document.getElementById('section-filter');
        [...sections].sort().forEach(s => {
            const opt = document.createElement('option');
            opt.value = s;
            opt.textContent = s || '(root)';
            sectionSelect.appendChild(opt);
        });

        sectionSelect.addEventListener('change', filterGroups);

        // Similarity filter
        const simFilter = document.getElementById('sim-filter');
        const simValue = document.getElementById('sim-filter-value');
        simFilter.addEventListener('input', () => {
            simValue.textContent = simFilter.value + '%';
            filterGroups();
        });

        function filterGroups() {
            const section = sectionSelect.value;
            const minSim = parseInt(simFilter.value) / 100;

            document.querySelectorAll('.group').forEach(g => {
                const groupSim = parseFloat(g.dataset.maxSim);
                const groupSections = g.dataset.sections.split(',');

                const matchSection = !section || groupSections.includes(section);
                const matchSim = groupSim >= minSim;

                g.style.display = matchSection && matchSim ? '' : 'none';
            });
        }

        // Expand all
        document.getElementById('expand-all').addEventListener('click', () => {
            const groups = document.querySelectorAll('.group');
            const allExpanded = [...groups].every(g => g.classList.contains('expanded'));
            groups.forEach(g => {
                if (allExpanded) {
                    g.classList.remove('expanded');
                } else {
                    g.classList.add('expanded');
                }
            });
        });
    </script>
</body>
</html>
'''

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"  Report saved: {output_path}")


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
        cluster_selection_method='eom',
        copy=True,
        n_jobs=1  # Single-threaded to avoid semaphore leaks
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
                    "cluster": hash(f"{page['url']}") % 100000,
                    "keywords": page.get('keywords', [])
                }
                subcluster_node["children"].append(page_node)

        subcluster_nodes.append(subcluster_node)

    return subcluster_nodes

def create_hierarchical_structure(section_clusters, embeddings, lang, hugo_root):
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
                        "cluster": cluster_id,
                        "keywords": page.get('keywords', [])
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
                            "cluster": cluster_id,
                            "keywords": page.get('keywords', [])
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
                        "cluster": cluster_id,
                        "keywords": page.get('keywords', [])
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
    site_title = get_site_title(hugo_root, lang)
    root = {
        "name": f"{site_title} - {lang_name}",
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
    hierarchy = create_hierarchical_structure(section_clusters, embeddings, args.lang, hugo_root)

    # Save to static directory only
    static_output_dir.mkdir(parents=True, exist_ok=True)
    output_file = static_output_dir / f"{args.lang}.json"

    print(f"\nWriting output to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(hierarchy, f, ensure_ascii=False, indent=2)

    # Generate scatterplot data with global clustering
    print("\n" + "="*60)
    print("Generating scatterplot visualization data...")
    print("="*60)

    # Global clustering with FAISS k-means
    cluster_labels, centroids = cluster_all_pages_global(pages, embeddings, args.num_clusters)

    # Generate 2D projections with UMAP
    projections = generate_2d_projections(embeddings)

    # Create scatterplot data structure
    scatterplot_data = generate_scatterplot_data(pages, embeddings, projections, cluster_labels, centroids)

    # Save scatterplot data to separate file
    scatterplot_file = static_output_dir / f"{args.lang}_scatterplot.json"
    print(f"\nWriting scatterplot data to: {scatterplot_file}")
    with open(scatterplot_file, 'w', encoding='utf-8') as f:
        json.dump(scatterplot_data, f, ensure_ascii=False, indent=2)

    print(f"\nSuccessfully generated clustering data for {args.lang}")
    print(f"Total pages: {len(pages)}")
    print(f"Total sections: {len(section_clusters)}")

    # Count total clusters across all sections
    total_clusters = sum(len(section_clusters[s]['clusters']) for s in section_clusters)
    print(f"Total semantic clusters (hierarchical): {total_clusters}")
    print(f"Global clusters (scatterplot): {args.num_clusters}")

    # Generate similarity report
    print("\n" + "="*60)
    print("Generating similar articles report...")
    print("="*60)

    # Filter pages for similarity check (exclude specified sections)
    if args.exclude_similarity_sections:
        print(f"Excluding sections from similarity check: {', '.join(args.exclude_similarity_sections)}")
        similarity_indices = [
            i for i, page in enumerate(pages)
            if page.get('section', '') not in args.exclude_similarity_sections
        ]
        similarity_pages = [pages[i] for i in similarity_indices]
        similarity_embeddings = embeddings[similarity_indices]
        print(f"Pages for similarity check: {len(similarity_pages)} (excluded {len(pages) - len(similarity_pages)})")
    else:
        similarity_pages = pages
        similarity_embeddings = embeddings

    similarity_groups = find_similar_articles(
        similarity_pages, similarity_embeddings,
        similarity_threshold=args.similarity_threshold,
        k_neighbors=args.k_neighbors
    )

    if similarity_groups:
        report_file = static_output_dir / f"{args.lang}_similar_articles.html"
        generate_similarity_report_html(similarity_groups, similarity_pages, report_file, args.lang)
        print(f"\nSimilar articles report saved to: {report_file}")
    else:
        print(f"\nNo similar articles found above {args.similarity_threshold:.0%} threshold")

if __name__ == "__main__":
    main()
