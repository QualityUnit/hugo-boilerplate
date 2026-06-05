"""Shared embedding cache for scripts that embed Hugo pages or assets.

Two backends are provided:
  - Legacy npz (cache_path_for / embed_with_cache): per-lang .npz files,
    keyed by file-path + content hash.
  - SQLite (EmbeddingCache): single shared DB at
    .audit_cache/embedding-cache.sqlite3. Text embeddings are keyed by
    sha256(text); non-text embeddings can provide their own stable keys through
    encode_by_keys().
"""

import hashlib
import os
import sqlite3
from pathlib import Path

import numpy as np

# Load .env from the scripts directory so HF_TOKEN and other vars are available
def _load_env():
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

_load_env()

# Authenticate with HuggingFace if token is present
def _hf_login():
    token = os.environ.get("HF_TOKEN")
    if not token:
        return
    try:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
    except Exception:
        pass

_hf_login()


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

        # Release the model + intermediate tensors so the next caller (e.g. the
        # next language in a multi-language loop) doesn't pile another model on
        # top of this one. gc.collect alone won't free CUDA-cached allocations.
        del model, new_embs
        import gc as _gc
        _gc.collect()
        _free_device_cache()
    else:
        print(f"Embedding cache: {hits}/{hits} hits (fully cached) → {cache_path}")

    return np.stack(embeddings).astype(np.float32)


def shared_sqlite_cache_path(hugo_root):
    """Canonical SQLite cache shared by embedding-based scripts."""
    return Path(hugo_root) / ".audit_cache" / "embedding-cache.sqlite3"


def _free_device_cache():
    """Release cached memory on whichever accelerator is active."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except ImportError:
        pass


def detect_device() -> str:
    """Detect the best available compute device: CUDA > MPS > CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            print(f"[device] CUDA detected: {name}")
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("[device] Apple Silicon MPS detected")
            return "mps"
    except ImportError:
        pass
    print("[device] No GPU detected, using CPU")
    return "cpu"


def default_embedding_device() -> str:
    """Return the best available device, with FLOWHUNT_EMBEDDING_DEVICE override."""
    env = os.environ.get("FLOWHUNT_EMBEDDING_DEVICE")
    if env:
        return env
    return detect_device()


class EmbeddingCache:
    """Persistent text-hash → embedding cache backed by SQLite.

    Shared across generate_clustering, generate_site_audit,
    generate_related_content, generate_paragraph_linkbuilding, and asset
    analysis scripts. Any text or keyed asset embedded by one script is
    immediately available to compatible later runs.

    Vectors are stored as float32 blobs, keyed by sha256(text).
    embedder_or_model in encode() accepts either a loaded
    SentenceTransformer instance, or a model-name string (lazy-loaded only
    on cache misses, then released so the next caller doesn't pile on memory).
    """

    def __init__(
        self,
        path,
        model: str,
        *,
        enabled: bool = True,
        device: str | None = None,
        cache_type: str = "text",
    ):
        self.enabled = enabled
        self.raw_model = model
        self.cache_type = cache_type
        self.model = f"{model}::{cache_type}"
        self.device = device or default_embedding_device()
        self.conn = None
        if not enabled:
            return
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(p))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                model     TEXT NOT NULL,
                text_hash TEXT NOT NULL,
                dim       INTEGER NOT NULL,
                vector    BLOB NOT NULL,
                PRIMARY KEY (model, text_hash)
            )
        """)

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()

    def close(self):
        if self.conn:
            self.conn.commit()
            self.conn.close()
            self.conn = None

    def encode(self, embedder_or_model, texts: list, *, batch_size: int = 32,
               show_progress_bar: bool = True, desc: str = "Embeddings") -> "np.ndarray":
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        if not self.enabled or self.conn is None:
            embedder = self._resolve(embedder_or_model)
            embs = embedder.encode(texts, batch_size=batch_size,
                                   show_progress_bar=show_progress_bar,
                                   normalize_embeddings=True)
            self._release(embedder, embedder_or_model)
            return np.asarray(embs, dtype=np.float32)

        keys = [self._hash(t) for t in texts]
        # Unique keys preserving insertion order
        seen_keys: dict = {}
        for k, t in zip(keys, texts):
            if k not in seen_keys:
                seen_keys[k] = t

        cached: dict = {}
        unique = list(seen_keys)
        for start in range(0, len(unique), 900):
            chunk = unique[start:start + 900]
            placeholders = ",".join("?" for _ in chunk)
            rows = self.conn.execute(
                f"SELECT text_hash, dim, vector FROM embeddings "
                f"WHERE model = ? AND text_hash IN ({placeholders})",
                [self.model, *chunk],
            ).fetchall()
            for key, dim, blob in rows:
                cached[key] = np.frombuffer(blob, dtype=np.float32, count=dim)

        missing = [k for k in unique if k not in cached]
        if missing:
            missing_texts = [seen_keys[k] for k in missing]
            if show_progress_bar or desc != "Anchor embeddings":
                print(f"{desc}: {len(unique) - len(missing)} hits, {len(missing)} misses → embedding")
            embedder = self._resolve(embedder_or_model)
            new_embs = embedder.encode(missing_texts, batch_size=batch_size,
                                       show_progress_bar=show_progress_bar,
                                       normalize_embeddings=True)
            new_embs = np.asarray(new_embs, dtype=np.float32)
            rows_to_insert = []
            for k, vec in zip(missing, new_embs):
                vec32 = vec.astype(np.float32)
                cached[k] = vec32
                rows_to_insert.append((self.model, k, int(vec32.shape[0]), vec32.tobytes()))
            self.conn.executemany(
                "INSERT OR REPLACE INTO embeddings (model, text_hash, dim, vector) VALUES (?, ?, ?, ?)",
                rows_to_insert,
            )
            self.conn.commit()
            self._release(embedder, embedder_or_model)
        else:
            if show_progress_bar or desc != "Anchor embeddings":
                print(f"{desc}: all {len(unique)} texts from cache")

        return np.vstack([cached[k] for k in keys]).astype(np.float32)

    def encode_by_keys(self, keys: list, compute_missing, *, desc: str = "Embeddings") -> "np.ndarray":
        """Return vectors for stable keys, computing and caching misses.

        `compute_missing` receives the list of missing original keys and must
        return a float32-compatible ndarray in the same order. This supports
        non-text embeddings, such as image vectors keyed by image content hash,
        while still sharing the same SQLite cache file.
        """
        if not keys:
            return np.zeros((0, 0), dtype=np.float32)

        if not self.enabled or self.conn is None:
            return np.asarray(compute_missing(keys), dtype=np.float32)

        hashed_keys = [self._hash(k) for k in keys]
        seen_keys: dict = {}
        for hashed, original in zip(hashed_keys, keys):
            if hashed not in seen_keys:
                seen_keys[hashed] = original

        cached: dict = {}
        unique = list(seen_keys)
        for start in range(0, len(unique), 900):
            chunk = unique[start:start + 900]
            placeholders = ",".join("?" for _ in chunk)
            rows = self.conn.execute(
                f"SELECT text_hash, dim, vector FROM embeddings "
                f"WHERE model = ? AND text_hash IN ({placeholders})",
                [self.model, *chunk],
            ).fetchall()
            for key, dim, blob in rows:
                cached[key] = np.frombuffer(blob, dtype=np.float32, count=dim)

        missing = [k for k in unique if k not in cached]
        if missing:
            missing_original = [seen_keys[k] for k in missing]
            print(f"{desc}: {len(unique) - len(missing)} hits, {len(missing)} misses → embedding")
            new_embs = np.asarray(compute_missing(missing_original), dtype=np.float32)
            if new_embs.shape[0] != len(missing):
                raise ValueError(
                    f"{desc}: compute_missing returned {new_embs.shape[0]} vectors "
                    f"for {len(missing)} cache misses"
                )
            rows_to_insert = []
            for k, vec in zip(missing, new_embs):
                vec32 = vec.astype(np.float32)
                cached[k] = vec32
                rows_to_insert.append((self.model, k, int(vec32.shape[0]), vec32.tobytes()))
            self.conn.executemany(
                "INSERT OR REPLACE INTO embeddings (model, text_hash, dim, vector) VALUES (?, ?, ?, ?)",
                rows_to_insert,
            )
            self.conn.commit()
        else:
            print(f"{desc}: all {len(unique)} vectors from cache")

        return np.vstack([cached[k] for k in hashed_keys]).astype(np.float32)

    def _resolve(self, embedder_or_model):
        if isinstance(embedder_or_model, str):
            from sentence_transformers import SentenceTransformer
            return SentenceTransformer(embedder_or_model, trust_remote_code=True, device=self.device)
        return embedder_or_model

    def _release(self, embedder, embedder_or_model):
        if isinstance(embedder_or_model, str):
            import gc
            del embedder
            gc.collect()
            _free_device_cache()
