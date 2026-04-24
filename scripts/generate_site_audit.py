#!/usr/bin/env python3
"""
Generate Site Audit (SEO-focused).

Computes content-aware embedding metrics over a Hugo site to answer:
  - How topically focused is this site? (siteFocusScore)
  - How wide is its topical spread? (siteRadius)
  - Which pages drift from the site / their section centroid? (outliers)
  - Which pages are near-duplicates and should be merged?
  - Which sections are tight vs. messy?

Outputs land in:
    static/data/audit/{lang}/
        site_metrics.json
        section_report.json
        page_drift.csv
        outliers.csv
        duplicates.csv

Reusable across any Hugo site using the boilerplate theme — just run:

    python themes/boilerplate/scripts/generate_site_audit.py --lang en

Default embedding model matches generate_clustering.py so artifacts are
comparable.
"""

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import csv
import html
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import faiss
import markdown
import toml_frontmatter as frontmatter
import umap
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embedding_cache import cache_path_for, embed_with_cache

DEFAULT_MODEL = "Alibaba-NLP/gte-multilingual-base"
DEFAULT_MAX_CHARS = 2000
DEFAULT_DUP_THRESHOLD = 0.92
DEFAULT_DUP_KNN = 10
DEFAULT_OUTLIER_PERCENTILE = 95.0
DEFAULT_EXCLUDE_SECTIONS = ["author", "affiliate-manager", "affiliate-program-directory", "search"]
DEFAULT_SCATTER_CLUSTERS = 30


def parse_args():
    p = argparse.ArgumentParser(description="Generate SEO site audit metrics")
    p.add_argument("--lang", required=True, help="Language code (e.g. en, sk)")
    p.add_argument("--hugo-root", default=".", help="Hugo project root (default: cwd)")
    p.add_argument("--content-dir", default=None, help="Override content dir (default: {hugo-root}/content)")
    p.add_argument("--output-dir", default=None, help="Override output dir (default: {hugo-root}/static/data/audit/{lang})")
    p.add_argument("--cache-dir", default=None, help="Override cache dir (default: {hugo-root}/data/audit_cache)")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"Sentence-transformer model (default: {DEFAULT_MODEL})")
    p.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS, help="Max body chars fed to embedder")
    p.add_argument("--duplicate-threshold", type=float, default=DEFAULT_DUP_THRESHOLD, help="Cosine similarity above which pages are flagged as near-duplicates")
    p.add_argument("--duplicate-knn", type=int, default=DEFAULT_DUP_KNN, help="kNN depth for duplicate search")
    p.add_argument("--outlier-percentile", type=float, default=DEFAULT_OUTLIER_PERCENTILE, help="Percentile of section drift above which pages are flagged as outliers")
    p.add_argument("--exclude-sections", nargs="+", default=DEFAULT_EXCLUDE_SECTIONS, help="Top-level sections to skip")
    p.add_argument("--scatter-clusters", type=int, default=DEFAULT_SCATTER_CLUSTERS, help="Number of FAISS k-means clusters for scatterplot coloring")
    p.add_argument("--no-cache", action="store_true", help="Ignore embedding cache")
    p.add_argument("--no-scatterplot", action="store_true", help="Skip UMAP 2D projection + scatterplot.json (faster)")
    return p.parse_args()


def clean_title(title):
    try:
        clean = BeautifulSoup(title, "html.parser").get_text(separator=" ", strip=True)
        clean = html.unescape(clean)
        clean = clean.replace("\n", " ").replace("\r", " ").replace("\t", " ")
        return re.sub(r"\s+", " ", clean).strip()
    except Exception:
        return title


def extract_body_text(markdown_content, max_chars):
    try:
        rendered = markdown.markdown(markdown_content or "")
        text = BeautifulSoup(rendered, "html.parser").get_text(separator=" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:max_chars]
    except Exception:
        return ""


def derive_url(rel_path, post):
    custom = post.metadata.get("url") if hasattr(post, "metadata") else post.get("url")
    if custom:
        if not custom.startswith("/"):
            custom = "/" + custom
        return custom
    path = str(rel_path).replace(".md", "").replace("_index", "").replace("index", "")
    if path and not path.endswith("/"):
        path += "/"
    return f"/{path}" if path else "/"


def process_content_files(content_dir, lang, exclude_sections, max_chars):
    lang_dir = Path(content_dir) / lang
    if not lang_dir.exists():
        raise FileNotFoundError(f"Content dir missing: {lang_dir}")

    exclusions = tuple(s.rstrip("/") + "/" for s in exclude_sections)
    pages = []

    print(f"Scanning {lang_dir} …")
    for md_file in lang_dir.rglob("*.md"):
        rel_path = md_file.relative_to(lang_dir)
        rel_str = str(rel_path)
        if any(rel_str.startswith(ex) or rel_str == ex.rstrip("/") for ex in exclusions):
            continue

        try:
            with open(md_file, "r", encoding="utf-8") as f:
                post = frontmatter.load(f)
        except Exception as e:
            print(f"  skip (parse error) {rel_path}: {e}")
            continue

        title = clean_title(post.get("title", "") or "")
        if not title:
            continue

        description = (post.get("description") or "").strip()
        body = extract_body_text(post.content, max_chars)

        parts = rel_str.split("/")
        section = parts[0] if len(parts) > 1 else "root"
        url = derive_url(rel_path, post)

        embed_text = ". ".join(s for s in [title, description, body] if s)

        pages.append({
            "title": title,
            "description": description,
            "url": url,
            "section": section,
            "file_path": rel_str,
            "embed_text": embed_text,
            "word_count": len((body or "").split()),
        })

    print(f"  found {len(pages)} pages across {len({p['section'] for p in pages})} sections")
    return pages


def l2_normalize(vec):
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm


def focus_metrics(embeddings, centroid):
    sims = embeddings @ centroid
    sims = np.clip(sims, -1.0, 1.0)
    distances = 1.0 - sims
    return {
        "focus_score": float(np.mean(sims)),
        "radius": float(np.std(distances)),
        "mean_distance": float(np.mean(distances)),
        "p95_distance": float(np.percentile(distances, 95)),
        "max_distance": float(np.max(distances)),
        "count": int(len(embeddings)),
    }


def build_section_stats(pages, embeddings):
    by_section = defaultdict(list)
    for i, p in enumerate(pages):
        by_section[p["section"]].append(i)

    section_info = {}
    for section, idxs in by_section.items():
        sec_emb = embeddings[idxs]
        centroid = l2_normalize(sec_emb.mean(axis=0))
        metrics = focus_metrics(sec_emb, centroid)
        section_info[section] = {
            "indices": idxs,
            "centroid": centroid,
            "metrics": metrics,
        }
    return section_info


def find_near_duplicates(pages, embeddings, threshold, k):
    n, d = embeddings.shape
    index = faiss.IndexFlatIP(d)
    index.add(embeddings.astype(np.float32))

    k = min(k, n)
    sims, idxs = index.search(embeddings.astype(np.float32), k)

    seen = set()
    pairs = []
    for i in range(n):
        for rank in range(0, k):
            j = int(idxs[i, rank])
            sim = float(sims[i, rank])
            if j < 0 or j == i or sim < threshold:
                continue
            key = (min(i, j), max(i, j))
            if key in seen:
                continue
            seen.add(key)
            pairs.append((key[0], key[1], sim))

    pairs.sort(key=lambda t: t[2], reverse=True)
    return pairs


def recommend_action(page, dist_site, dist_section, section_p95, section_size, has_duplicate):
    reasons = []
    if has_duplicate:
        reasons.append("near-duplicate: merge or canonicalize")
    if section_size < 3:
        reasons.append("orphan section: expand or consolidate")
    if dist_section > section_p95:
        reasons.append("off-topic for its section: refocus or move")
    if dist_site > 0.65:
        reasons.append("off-brand for the whole site: consider removing")
    if page["word_count"] < 200 and dist_section > section_p95 * 0.8:
        reasons.append("thin + off-topic: strong removal candidate")
    return "; ".join(reasons) if reasons else ""


def write_site_metrics(path, site_metrics, section_info, args):
    payload = {
        "lang": args.lang,
        "model": args.model,
        "page_count": site_metrics["count"],
        "site_focus_score": site_metrics["focus_score"],
        "site_radius": site_metrics["radius"],
        "mean_distance_to_centroid": site_metrics["mean_distance"],
        "p95_distance_to_centroid": site_metrics["p95_distance"],
        "max_distance_to_centroid": site_metrics["max_distance"],
        "sections": sorted(
            [
                {
                    "section": s,
                    "page_count": info["metrics"]["count"],
                    "focus_score": info["metrics"]["focus_score"],
                    "radius": info["metrics"]["radius"],
                    "p95_distance": info["metrics"]["p95_distance"],
                }
                for s, info in section_info.items()
            ],
            key=lambda x: x["focus_score"],
            reverse=True,
        ),
        "interpretation": {
            "site_focus_score": "1.0 = all pages share the same meaning; lower = more topical spread. Aim for >0.55 on a focused site; <0.40 suggests fragmented coverage.",
            "site_radius": "Std-dev of per-page cosine distance to the site centroid. Lower is tighter.",
            "section_focus_score": "Same metric restricted to pages inside a section. Sections with the lowest focus_score hold the worst-cohesion content.",
        },
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def write_section_report(path, section_info, pages):
    out = []
    for section, info in section_info.items():
        idxs = info["indices"]
        section_pages = [pages[i] for i in idxs]
        out.append({
            "section": section,
            "page_count": info["metrics"]["count"],
            "focus_score": info["metrics"]["focus_score"],
            "radius": info["metrics"]["radius"],
            "p95_distance_to_section_centroid": info["metrics"]["p95_distance"],
            "example_titles": [p["title"] for p in section_pages[:5]],
        })
    out.sort(key=lambda x: x["focus_score"])  # worst cohesion first
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))


def write_page_drift(path, pages, dist_site, dist_section):
    rows = []
    for i, p in enumerate(pages):
        rows.append({
            "url": p["url"],
            "section": p["section"],
            "title": p["title"],
            "word_count": p["word_count"],
            "distance_to_site_centroid": round(float(dist_site[i]), 4),
            "distance_to_section_centroid": round(float(dist_section[i]), 4),
            "file_path": p["file_path"],
        })
    rows.sort(key=lambda r: r["distance_to_section_centroid"], reverse=True)
    _write_csv(path, rows)


def write_outliers(path, pages, dist_site, dist_section, section_info, duplicate_set):
    rows = []
    section_size = {s: info["metrics"]["count"] for s, info in section_info.items()}
    section_p95 = {s: info["metrics"]["p95_distance"] for s, info in section_info.items()}

    for i, p in enumerate(pages):
        ds = float(dist_section[i])
        d_all = float(dist_site[i])
        sec_p95 = section_p95.get(p["section"], 1.0)
        is_outlier = ds > sec_p95 or d_all > 0.65
        if not is_outlier and i not in duplicate_set:
            continue
        action = recommend_action(
            p,
            dist_site=d_all,
            dist_section=ds,
            section_p95=sec_p95,
            section_size=section_size.get(p["section"], 0),
            has_duplicate=i in duplicate_set,
        )
        if not action:
            continue
        rows.append({
            "url": p["url"],
            "section": p["section"],
            "title": p["title"],
            "word_count": p["word_count"],
            "distance_to_site_centroid": round(d_all, 4),
            "distance_to_section_centroid": round(ds, 4),
            "section_p95_distance": round(sec_p95, 4),
            "recommendation": action,
            "file_path": p["file_path"],
        })
    rows.sort(key=lambda r: r["distance_to_section_centroid"], reverse=True)
    _write_csv(path, rows)
    return rows


def build_scatterplot(embeddings, num_clusters):
    n, d = embeddings.shape
    k = max(2, min(num_clusters, n // 2))

    emb_f32 = embeddings.astype(np.float32)
    kmeans = faiss.Kmeans(d=d, k=k, niter=50, verbose=False, seed=42)
    kmeans.train(emb_f32)
    _, labels = kmeans.index.search(emb_f32, 1)
    cluster_labels = labels.flatten().astype(int)

    n_neighbors = max(2, min(15, n - 1))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    coords = reducer.fit_transform(emb_f32)
    return cluster_labels, coords


def write_scatterplot(path, pages, dist_site, dist_section, coords, cluster_labels, dup_partners, site_metrics):
    rows = []
    max_drift = float(max(site_metrics["max_distance"], np.max(dist_site), np.max(dist_section), 1e-9))
    for i, p in enumerate(pages):
        rows.append({
            "title": p["title"],
            "url": p["url"],
            "section": p["section"],
            "x": float(coords[i, 0]),
            "y": float(coords[i, 1]),
            "cluster": int(cluster_labels[i]),
            "drift_site": round(float(dist_site[i]), 4),
            "drift_section": round(float(dist_section[i]), 4),
            "drift_norm": round(float(dist_section[i]) / max_drift, 4),
            "word_count": p["word_count"],
            "duplicate_of": dup_partners.get(i, []),
        })
    num_clusters = int(max(cluster_labels) + 1) if len(cluster_labels) else 0
    payload = {
        "total_pages": len(pages),
        "num_clusters": num_clusters,
        "max_drift": max_drift,
        "site_focus_score": site_metrics["focus_score"],
        "site_radius": site_metrics["radius"],
        "pages": rows,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False))


def write_duplicates(path, pages, pairs):
    rows = []
    for i, j, sim in pairs:
        a, b = pages[i], pages[j]
        same_section = a["section"] == b["section"]
        if sim >= 0.97:
            action = "merge (duplicate)"
        elif sim >= 0.94:
            action = "consolidate or canonicalize"
        else:
            action = "review — strong overlap"
        rows.append({
            "similarity": round(sim, 4),
            "same_section": same_section,
            "section_a": a["section"],
            "url_a": a["url"],
            "title_a": a["title"],
            "section_b": b["section"],
            "url_b": b["url"],
            "title_b": b["title"],
            "recommendation": action,
            "file_a": a["file_path"],
            "file_b": b["file_path"],
        })
    _write_csv(path, rows)


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    hugo_root = Path(args.hugo_root).resolve()
    content_dir = Path(args.content_dir) if args.content_dir else hugo_root / "content"
    output_dir = Path(args.output_dir) if args.output_dir else hugo_root / "static" / "data" / "audit" / args.lang
    if args.cache_dir:
        cache_path = Path(args.cache_dir) / f"{args.lang}__{args.model.replace('/', '_').replace('-', '_')}.npz"
    else:
        cache_path = cache_path_for(hugo_root, args.lang, args.model)

    pages = process_content_files(content_dir, args.lang, args.exclude_sections, args.max_chars)
    if not pages:
        print("No pages found — aborting.")
        return

    embeddings = embed_with_cache(pages, args.model, cache_path, use_cache=not args.no_cache)

    site_centroid = l2_normalize(embeddings.mean(axis=0))
    site_metrics = focus_metrics(embeddings, site_centroid)
    section_info = build_section_stats(pages, embeddings)

    dist_site = 1.0 - np.clip(embeddings @ site_centroid, -1.0, 1.0)

    dist_section = np.zeros(len(pages), dtype=np.float32)
    for section, info in section_info.items():
        for idx in info["indices"]:
            sim = float(np.clip(embeddings[idx] @ info["centroid"], -1.0, 1.0))
            dist_section[idx] = 1.0 - sim

    dup_pairs = find_near_duplicates(pages, embeddings, args.duplicate_threshold, args.duplicate_knn)
    duplicate_set = {i for pair in dup_pairs for i in pair[:2]}

    dup_partners = defaultdict(list)
    for i, j, sim in dup_pairs:
        dup_partners[i].append({"url": pages[j]["url"], "title": pages[j]["title"], "similarity": round(sim, 4)})
        dup_partners[j].append({"url": pages[i]["url"], "title": pages[i]["title"], "similarity": round(sim, 4)})

    output_dir.mkdir(parents=True, exist_ok=True)

    write_site_metrics(output_dir / "site_metrics.json", site_metrics, section_info, args)
    write_section_report(output_dir / "section_report.json", section_info, pages)
    write_page_drift(output_dir / "page_drift.csv", pages, dist_site, dist_section)
    outliers = write_outliers(output_dir / "outliers.csv", pages, dist_site, dist_section, section_info, duplicate_set)
    write_duplicates(output_dir / "duplicates.csv", pages, dup_pairs)

    if not args.no_scatterplot:
        print(f"Projecting {len(pages)} pages to 2D (UMAP) + k-means coloring…")
        cluster_labels, coords = build_scatterplot(embeddings, args.scatter_clusters)
        write_scatterplot(
            output_dir / "scatterplot.json",
            pages,
            dist_site,
            dist_section,
            coords,
            cluster_labels,
            dup_partners,
            site_metrics,
        )

    print("\n=== Site audit summary ===")
    print(f"Language:            {args.lang}")
    print(f"Pages audited:       {site_metrics['count']}")
    print(f"siteFocusScore:      {site_metrics['focus_score']:.4f}  (1.0 = laser focused, 0.0 = scattered)")
    print(f"siteRadius:          {site_metrics['radius']:.4f}  (std-dev of cosine distance to site centroid)")
    print(f"p95 drift:           {site_metrics['p95_distance']:.4f}")
    print(f"Outlier pages:       {len(outliers)} (see outliers.csv)")
    print(f"Near-duplicate pairs:{len(dup_pairs)} (see duplicates.csv)")
    worst = sorted(section_info.items(), key=lambda kv: kv[1]['metrics']['focus_score'])[:5]
    print("Weakest sections (lowest focus_score):")
    for section, info in worst:
        m = info["metrics"]
        print(f"  {section:40s}  pages={m['count']:<5d}  focus={m['focus_score']:.3f}  radius={m['radius']:.3f}")
    print(f"\nReports written to: {output_dir}")


if __name__ == "__main__":
    main()
