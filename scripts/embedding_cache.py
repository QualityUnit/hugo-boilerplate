"""Shared per-page embedding cache for scripts that embed Hugo pages.

Cache file: {hugo_root}/.audit_cache/{lang}__{model_slug}.npz
Layout:     paths[], hashes[], embeddings[n, d] (float32, L2-normalized).

Hash per page = sha256(embed_text + "|" + model_name). A cached entry is
reused when both the file path and the hash match. Scripts that pick
different page subsets share the same file — misses are per-page, not
per-run — so the cache is re-usable across audit / related-content even
if their exclusion lists differ.
"""

import hashlib
from pathlib import Path

import numpy as np


def cache_path_for(hugo_root, lang, model_name):
    slug = model_name.replace("/", "_").replace("-", "_")
    return Path(hugo_root) / ".audit_cache" / f"{lang}__{slug}.npz"


def _hash(text, model_name):
    return hashlib.sha256((text + "|" + model_name).encode("utf-8")).hexdigest()


def _load(cache_path):
    if not cache_path.exists():
        return {}
    try:
        data = np.load(cache_path, allow_pickle=False)
        paths = data["paths"]
        hashes = data["hashes"]
        embs = data["embeddings"]
        return {str(paths[i]): (str(hashes[i]), embs[i]) for i in range(len(paths))}
    except Exception as e:
        print(f"  cache read failed ({e}); recomputing")
        return {}


def _save(cache_path, cache):
    if not cache:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    paths = np.array(list(cache.keys()))
    hashes = np.array([v[0] for v in cache.values()])
    embs = np.stack([v[1] for v in cache.values()]).astype(np.float32)
    np.savez_compressed(cache_path, paths=paths, hashes=hashes, embeddings=embs)


def embed_with_cache(pages, model_name, cache_path, use_cache=True):
    """Embed pages and persist per-page results.

    pages: list of dicts with keys 'file_path' and 'embed_text'.
    Returns: float32 ndarray of normalized embeddings, same order as pages.
    """
    cache = _load(cache_path) if use_cache else {}

    embeddings = [None] * len(pages)
    to_embed_texts = []
    to_embed_meta = []

    for i, p in enumerate(pages):
        text_hash = _hash(p["embed_text"], model_name)
        entry = cache.get(p["file_path"])
        if entry is not None and entry[0] == text_hash:
            embeddings[i] = entry[1]
        else:
            to_embed_texts.append(p["embed_text"])
            to_embed_meta.append((i, p["file_path"], text_hash))

    hits = len(pages) - len(to_embed_texts)
    if to_embed_texts:
        print(f"Embedding cache: {hits} hits, {len(to_embed_texts)} misses → loading model {model_name}")
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name, trust_remote_code=True)
        new_embs = model.encode(
            to_embed_texts,
            show_progress_bar=True,
            batch_size=32,
            normalize_embeddings=True,
        )
        new_embs = np.asarray(new_embs, dtype=np.float32)

        for k, (i, path, text_hash) in enumerate(to_embed_meta):
            emb = new_embs[k]
            embeddings[i] = emb
            cache[path] = (text_hash, emb)

        _save(cache_path, cache)
        print(f"Embedding cache updated → {cache_path} ({len(cache)} entries)")
    else:
        print(f"Embedding cache: {hits}/{hits} hits (fully cached) → {cache_path}")

    return np.stack(embeddings).astype(np.float32)
