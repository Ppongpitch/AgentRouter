"""
embedder.py
───────────
Thin wrapper around the sentence-transformers embedding model.
No print() — errors are raised, not logged, so the caller (main.py)
decides how to surface them.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBED_MODEL


def load_embedder(model_name: str = EMBED_MODEL) -> SentenceTransformer:
    """
    Load and return the sentence-transformers embedding model.
 
    model_kwargs={"use_safetensors": True} forces it to load the
    model.safetensors checkpoint instead of pytorch_model.bin. This avoids
    transformers' torch.load safety check (which requires torch>=2.6) —
    safetensors loading never calls torch.load, so it's unaffected either way.
    """
    return SentenceTransformer(model_name, model_kwargs={"use_safetensors": True})

def embed_texts(embedder: SentenceTransformer, texts: list[str]) -> np.ndarray:
    """
    Encode a list of texts into normalized embeddings.
    Returns shape (len(texts), dim).
    """
    if not texts:
        raise ValueError("embed_texts() received an empty list of texts")
    return embedder.encode(texts, normalize_embeddings=True)


def embed_one(embedder: SentenceTransformer, text: str) -> np.ndarray:
    """Encode a single text into a normalized embedding of shape (1, dim)."""
    return embed_texts(embedder, [text])
