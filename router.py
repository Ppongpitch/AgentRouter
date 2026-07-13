"""
router.py
─────────
adaptive_route(): the single entry point that combines semantic routing
(in a caller-selected mode), LLM fallback routing, AND actually calling
the winning agent's model to get a real answer. No print() anywhere —
this is a library function, not a program. main.py decides what to log.

Seed growth policy: seeds are appended to the winning agent's bucket
ONLY when the LLM router handled the query. Semantic hits never grow
the seed bucket.

Tiers:
  Tier 1 = semantic routing (cosine sim vs. agent representation cleared threshold)
  Tier 2 = LLM routing (semantic wasn't confident enough, LLM router decided)

Timing: both routing time (embed + score + decide, and LLM-router call if
it fires) and generation time (calling the winning agent's endpoint) are
measured separately and returned in milliseconds, so the caller can show
a breakdown of where the total time went.
"""

import time

from agent_executor import run_agent
from agent_store import AgentStore
from config import DEFAULT_SEMANTIC_MODE
from llm_router import llm_route
from semantic_router import semantic_route

TIER_LABELS = {
    1: "Tier 1 — Semantic",
    2: "Tier 2 — LLM Route",
}


def adaptive_route(
    query: str,
    embedder,
    store: AgentStore,
    llm_client,
    mode: str = DEFAULT_SEMANTIC_MODE,
) -> dict:
    """
    1. Try semantic route in the given mode — Tier 1.
    2. If not confident enough, fall back to the LLM router — Tier 2.
    3. Append the query to the winning agent's seed bucket ONLY if the
       LLM router handled it (semantic hits do not grow seeds).
    4. Call the winning agent's actual model with the query to get a real answer.
    5. Return a JSON-serializable routing + answer + timing result.
    """
    routing_start = time.perf_counter()

    sem_result, sem_scores = semantic_route(query, embedder, store, mode=mode)

    if sem_result is not None:
        result = sem_result
        result.setdefault("intent", f"Route to {result['capability']}")
        tier = 1
    else:
        result = llm_route(query, store, llm_client)
        tier = 2

    routing_time_ms = round((time.perf_counter() - routing_start) * 1000, 1)

    seed_appended = False
    if result["router_used"] == "llm":
        seed_appended = store.update_agent_seeds(result["capability"], query)

    agent = store.get_agent(result["capability"])
    description = agent["description"] if agent else ""

    generation_start = time.perf_counter()
    answer = run_agent(query, result["model"], description)
    generation_time_ms = round((time.perf_counter() - generation_start) * 1000, 1)

    return {
        "capability": result["capability"],
        "model": result["model"],
        "intent": result.get("intent", ""),
        "confident": result["confident"],
        "router_used": result["router_used"],
        "tier": tier,
        "tier_label": TIER_LABELS[tier],
        "mode": mode,
        "seed_appended": seed_appended,
        "semantic_scores": {s["capability"]: round(s["sim"], 4) for s in sem_scores},
        "answer": answer,
        "routing_time_ms": routing_time_ms,
        "generation_time_ms": generation_time_ms,
        "total_time_ms": round(routing_time_ms + generation_time_ms, 1),
    }
