"""
semantic_router.py
───────────────────
Semantic routing with a selectable mode:

  - "per_seed"        : cosine sim vs EVERY individual seed embedding, take max
  - "single_centroid"  : cosine sim vs ONE mean centroid per agent
  - "multi_centroid"   : cosine sim vs k-means sub-centroids per agent, take max

All three modes read precomputed arrays already sitting on each agent
record (agent_store.py keeps all three in sync) — switching modes is just
picking a different array, not recomputing embeddings. No print().
"""

from sklearn.metrics.pairwise import cosine_similarity

from agent_store import AgentStore
from config import DEFAULT_SEMANTIC_MODE
from embedder import embed_one


def _score_agent(q_emb, ag: dict, mode: str) -> float:
    if mode == "per_seed":
        sims = cosine_similarity(q_emb, ag["seed_embeddings"])[0]
        return float(sims.max())
    elif mode == "single_centroid":
        return float(cosine_similarity(q_emb, ag["centroid"])[0][0])
    elif mode == "multi_centroid":
        sims = cosine_similarity(q_emb, ag["centroids"])[0]
        return float(sims.max())
    else:
        raise ValueError(f"Unknown semantic mode: {mode!r}")


def semantic_route(
    query: str,
    embedder,
    store: AgentStore,
    mode: str = DEFAULT_SEMANTIC_MODE,
) -> tuple[dict | None, list[dict]]:
    """
    Compute cosine similarity between the query and every agent, using
    whichever representation `mode` selects, in a single pass.

    Returns (result, scores):
      - result: routing dict if best_sim >= that agent's threshold, else None
      - scores: full ranked list of every agent's score (for debugging /
                for router.py to pass along when falling back to the LLM)
    """
    q_emb = embed_one(embedder, query)  # (1, dim)

    scores = []
    for ag in store.agents:
        sim = _score_agent(q_emb, ag, mode)
        scores.append(
            {
                "capability": ag["capability"],
                "model": ag["model"],
                "sim": sim,
                "threshold": ag["threshold"],
            }
        )
    scores.sort(key=lambda x: x["sim"], reverse=True)
    best = scores[0]

    if best["sim"] >= best["threshold"]:
        result = {
            "capability": best["capability"],
            "model": best["model"],
            "confident": round(best["sim"], 4),
            "router_used": "semantic",
        }
        return result, scores

    return None, scores
