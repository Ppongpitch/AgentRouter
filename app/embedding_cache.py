"""
embedding_cache.py
──────────────────
On-disk cache mapping seed text -> embedding vector, so a seed that was
already embedded once never gets re-embedded — on restart, on centroid
recompute, or anywhere else. This is the fix for the embedding bottleneck:
building/updating a centroid used to re-encode EVERY seed in that agent's
bucket every time, even ones that hadn't changed.

No print() — pure load/save/lookup functions.
"""

import pickle
from pathlib import Path

import numpy as np


def load_cache(path: str) -> dict:
    """Load the embedding cache from disk. Returns {} if the file doesn't exist yet."""
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, "rb") as f:
        return pickle.load(f)


def save_cache(path: str, cache: dict) -> None:
    """Persist the embedding cache to disk (overwrites the previous file)."""
    with open(path, "wb") as f:
        pickle.dump(cache, f)


def get_or_compute(cache: dict, texts: list[str], embed_fn) -> np.ndarray:
    """
    For each text in `texts`:
      - if it already has a cached embedding, reuse it (no re-embedding)
      - if not, queue it up

    Every text that's missing gets embedded together in ONE batched call
    (not one-by-one) for efficiency, then written into `cache` in-place.

    embed_fn: callable(list[str]) -> np.ndarray of shape (n, dim), normalized.
    Returns embeddings in the same order as `texts`, shape (len(texts), dim).
    """
    missing = [t for t in texts if t not in cache]
    if missing:
        new_embs = embed_fn(missing)
        for t, emb in zip(missing, new_embs):
            cache[t] = emb

    return np.vstack([cache[t] for t in texts])
