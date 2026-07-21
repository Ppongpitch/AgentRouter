"""
agent_store.py
───────────────
AgentStore keeps THREE representations per agent, all derived from the
SAME cached seed embeddings (so switching semantic mode at request time
is free — no re-embedding, just picking a different pre-built array):

  - seed_embeddings : every seed's individual embedding      -> "per_seed" mode
  - centroid        : mean of all seed embeddings (1 vector)  -> "single_centroid" mode
  - centroids       : k-means sub-centroids (k vectors)       -> "multi_centroid" mode

Every seed's embedding is cached (embedding_cache.py) — building or
updating any of the three representations only ever embeds seeds that
don't already have a cached vector. No print() anywhere — every function
returns data; main.py decides what (if anything) to log.
"""

import copy
import json

import numpy as np
from sklearn.cluster import KMeans

from config import (
    AGENT_THRESHOLDS,
    DEFAULT_THRESHOLD,
    JSON_TO_CAPABILITY,
    KMEANS_RANDOM_STATE,
    N_CLUSTERS_PER_AGENT,
)
from embedding_cache import get_or_compute, save_cache


def load_agents_json(path: str) -> list[dict]:
    """Load the raw agent definitions (capability, model, description, seeds)."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_seeds_from_train_jsonl(
    agents_raw: list[dict],
    train_path: str,
    mapping: dict = JSON_TO_CAPABILITY,
    skip_capabilities: set | None = None,
) -> dict:
    """
    Read train.jsonl (rows of {"text": ..., "agent": ...}), map each row's
    "agent" code to a capability name, and append "text" into that agent's
    seeds list in-place.

    skip_capabilities: capabilities to skip entirely (e.g. ones that
    already have seeds loaded from a persistent store like Postgres —
    merging train.jsonl again on top would create duplicates).

    Returns a stats dict: {"added": int, "skipped": int, "per_agent": {...}}
    instead of printing anything — caller decides how to log it.
    """
    skip_capabilities = skip_capabilities or set()
    cap_lookup = {ag["capability"]: ag for ag in agents_raw}
    added, skipped = 0, 0

    with open(train_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cap = mapping.get(row.get("agent"))
            if cap is None or cap not in cap_lookup or cap in skip_capabilities:
                skipped += 1
                continue
            cap_lookup[cap]["seeds"].append(row["text"])
            added += 1

    return {
        "added": added,
        "skipped": skipped,
        "per_agent": {ag["capability"]: len(ag["seeds"]) for ag in agents_raw},
    }


class AgentStore:
    """
    Holds live agent records with ONE mutable seed list per agent, and
    THREE precomputed representations derived from that list's embeddings:
    per-seed embeddings, a single mean centroid, and k-means sub-centroids.

    - Seeds only grow via update_agent_seeds(), which should be called ONLY
      for LLM-routed queries (semantic hits should not append seeds) —
      that policy is enforced by the caller (router.py), not here.
    - Threshold per agent is fixed via AGENT_THRESHOLDS and never changes
      as the seed list grows.
    - Embedding cache: every seed's embedding is looked up in
      `embedding_cache` before being computed — only genuinely new seeds
      get embedded. This is the fix for the embedding bottleneck.
    """

    def __init__(self, raw_agents: list[dict], embedder, embedding_cache: dict | None = None):
        self.embedder = embedder
        self.embedding_cache: dict = embedding_cache if embedding_cache is not None else {}
        self.agents: list[dict] = []
        self._build(raw_agents)

    # ── build ──────────────────────────────────────────────────────────
    def _build(self, raw_agents: list[dict]) -> None:
        for ag in raw_agents:
            record = copy.deepcopy(ag)
            self._refresh_representations(record)
            record["threshold"] = AGENT_THRESHOLDS.get(
                record["capability"], DEFAULT_THRESHOLD
            )
            self.agents.append(record)

    # ── embedding (cache-aware) ──────────────────────────────────────────
    def _embed_fn(self, texts: list[str]) -> np.ndarray:
        return self.embedder.encode(texts, normalize_embeddings=True)

    def _refresh_representations(self, record: dict) -> None:
        """
        Recompute all three representations for one agent record, reusing
        cached embeddings wherever possible. Called on build and on every
        seed append.
        """
        seed_embeddings = get_or_compute(
            self.embedding_cache, record["seeds"], self._embed_fn
        )  # (n, dim) — only truly new seeds get embedded here
        record["seed_embeddings"] = seed_embeddings
        record["centroid"] = self._mean_centroid(seed_embeddings)
        record["centroids"] = self._kmeans_centroids(seed_embeddings)

    @staticmethod
    def _mean_centroid(seed_embeddings: np.ndarray) -> np.ndarray:
        centroid = seed_embeddings.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-10)
        return centroid.reshape(1, -1)  # (1, dim)

    @staticmethod
    def _kmeans_centroids(seed_embeddings: np.ndarray) -> np.ndarray:
        """
        Cluster seed embeddings into up to N_CLUSTERS_PER_AGENT groups.
        This is pure linear algebra on already-embedded vectors — no
        embedding calls involved, so it's cheap even on every seed append.
        """
        n = len(seed_embeddings)
        k = min(N_CLUSTERS_PER_AGENT, n)

        if k <= 1:
            centroid = seed_embeddings.mean(axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-10)
            return centroid.reshape(1, -1)

        km = KMeans(n_clusters=k, random_state=KMEANS_RANDOM_STATE, n_init=10)
        km.fit(seed_embeddings)
        centers = km.cluster_centers_
        norms = np.linalg.norm(centers, axis=1, keepdims=True) + 1e-10
        return centers / norms

    # ── update ────────────────────────────────────────────────────────
    def update_agent_seeds(self, capability: str, new_seed: str) -> bool:
        """
        Append new_seed to the agent's seed list and refresh all three
        representations. Because of the embedding cache, this only embeds
        `new_seed` itself — every previously-seen seed reuses its cached
        vector; only the (cheap) centroid mean / k-means refit re-runs.
        Threshold is untouched (constant). Returns True if the agent was
        found and updated, False otherwise.
        """
        for ag in self.agents:
            if ag["capability"] == capability:
                ag["seeds"].append(new_seed)
                self._refresh_representations(ag)
                return True
        return False

    # ── cache persistence ────────────────────────────────────────────────
    def save_embedding_cache(self, path: str) -> None:
        """Persist the current embedding cache to disk."""
        save_cache(path, self.embedding_cache)

    def cache_stats(self) -> dict:
        return {"cached_embeddings": len(self.embedding_cache)}

    # ── lookup ────────────────────────────────────────────────────────
    def get_capabilities_str(self) -> str:
        """Formatted list for LLM prompt injection."""
        return "\n".join(
            f"  - {ag['capability']}: {ag['description']}" for ag in self.agents
        )

    def get_capability_names(self) -> list[str]:
        return [ag["capability"] for ag in self.agents]

    def get_agent(self, capability: str) -> dict | None:
        return next(
            (ag for ag in self.agents if ag["capability"] == capability), None
        )