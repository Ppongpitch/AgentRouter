"""
handler.py
──────────
RunPod serverless entrypoint. This is the RunPod-flavored twin of main.py —
same startup sequence, same modules (config / db / agent_store / embedder /
router / llm_router / xlmr_router), just wrapped as a `handler(job)`
function instead of FastAPI routes. Keeping the logic in those shared
modules (not duplicated here) means main.py (local/FastAPI) and this file
(RunPod) can never silently drift apart.

Persistence model (this replaces the old local embedding_cache.pkl file):
  - Postgres is the ONLY persistence layer now. On cold start, every seed's
    text + embedding for the current EMBED_MODEL is loaded from the `seeds`
    table. Only seeds NOT already in Postgres get embedded here.
  - Because Postgres is shared across every worker (any region, any cold
    start, any image rebuild), this is strictly better than the old
    image-baked pickle cache: the FIRST cold start anywhere bootstraps it,
    and every worker after that — forever, across deploys — just reads.
  - Every request is logged to query_log: the prompt, the user's stated
    expectation, what actually got routed, and the answer.
  - A Tier-2 (LLM-routed) query that grows an agent's seed bucket gets that
    new seed + embedding written back into Postgres immediately.

Expected environment variables:
  - OPENROUTER_API_KEY : required for any generation / LLM-fallback call
  - POSTGRES_DSN        : required — external reachable Postgres (RunPod
                          containers have no local DB of their own).
                          e.g. "postgresql://user:pass@host:5432/db?sslmode=require"
  - EMBED_MODEL, LLM_ROUTER_MODEL, XLMR_CHECKPOINT_PATH, etc. — optional,
    see config.py for defaults.
"""

import os
import traceback

import runpod

import config
import db
from agent_store import AgentStore, load_agents_json, load_seeds_from_train_jsonl
from embedder import load_embedder
from llm_router import build_llm_client
from router import adaptive_route
from xlmr_router import XLMRClassifier

# ─── COLD-START INIT (runs once per worker boot) ────────────────────────

print(f"[init] Loading embedder: {config.EMBED_MODEL} ...")
embedder = load_embedder(config.EMBED_MODEL)
print("[init] Embedder ready.")

print("[init] Connecting to PostgreSQL ...")
db_pool = db.create_pool(config.POSTGRES_DSN)
db.init_schema(db_pool)
print("[init] PostgreSQL connected, schema ready.")

if os.path.exists(config.AGENTS_JSON_PATH):
    agents_raw = load_agents_json(config.AGENTS_JSON_PATH)
    print(f"[init] Loaded {len(agents_raw)} agents from '{config.AGENTS_JSON_PATH}'.")
else:
    raise FileNotFoundError(
        f"Agents JSON not found at '{config.AGENTS_JSON_PATH}'. "
        f"Set AGENTS_JSON_PATH env var or bake the file into the image next to handler.py."
    )

# ── Load seeds + their embeddings from Postgres (replaces embedding_cache.pkl) ──
db_seeds = db.load_all_seeds(db_pool, config.EMBED_MODEL)
seed_embedding_cache: dict = {}
already_in_db = set()

for ag in agents_raw:
    cap = ag["capability"]
    rows = db_seeds.get(cap)
    if rows:
        # Postgres is the source of truth for this agent once it has rows —
        # replace the inline seeds from capability_agents.json entirely so
        # we don't re-embed or duplicate anything already stored.
        ag["seeds"] = [text for text, _ in rows]
        for text, embedding in rows:
            seed_embedding_cache[text] = embedding
        already_in_db.add(cap)
        print(f"[init]    {cap:<40} loaded {len(rows)} seeds from Postgres")
    else:
        print(f"[init]    {cap:<40} nothing in Postgres yet — will bootstrap")

# For any capability NOT yet in Postgres, merge in train.jsonl seeds too —
# skip capabilities already loaded from the DB to avoid duplicating/re-merging.
if os.path.exists(config.TRAIN_JSONL_PATH):
    stats = load_seeds_from_train_jsonl(
        agents_raw, config.TRAIN_JSONL_PATH, skip_capabilities=already_in_db
    )
    print(f"[init] Merged '{config.TRAIN_JSONL_PATH}' for not-yet-bootstrapped agents: "
          f"+{stats['added']} added, {stats['skipped']} skipped.")
else:
    print(f"[init] No '{config.TRAIN_JSONL_PATH}' found — bootstrapping from inline seeds only.")

print("[init] Building AgentStore (embedding only seeds not already cached)...")
store = AgentStore(agents_raw, embedder, seed_embedding_cache)
print(f"[init] AgentStore built with {len(store.agents)} agents, "
      f"{store.cache_stats()['cached_embeddings']} seed embeddings in memory.")

# One-time bootstrap write: any capability that had nothing in Postgres yet
# gets its freshly-embedded seeds written now, so every future cold start —
# on this worker or any other — loads them straight from the DB instead of
# re-embedding. This is the whole fix for the "re-embeds 800 seeds every
# cold start" problem, and unlike a local file it survives image rebuilds too.
bootstrap_rows = []
for ag in store.agents:
    if ag["capability"] not in already_in_db:
        for seed_text in ag["seeds"]:
            bootstrap_rows.append(
                (ag["capability"], seed_text, seed_embedding_cache[seed_text])
            )
if bootstrap_rows:
    db.insert_seeds_bulk(db_pool, bootstrap_rows, config.EMBED_MODEL)
    print(f"[init] Bootstrapped {len(bootstrap_rows)} seeds into Postgres for the first time.")

llm_client = build_llm_client()
print(f"[init] LLM router ready -> {config.LLM_ROUTER_MODEL} (called only when semantic routing misses)")

if os.path.exists(config.XLMR_CHECKPOINT_PATH):
    print(f"[init] Loading XLM-R classifier from '{config.XLMR_CHECKPOINT_PATH}' ...")
    xlmr_classifier = XLMRClassifier(config.XLMR_CHECKPOINT_PATH, config.XLMR_TOKENIZER_NAME)
    print("[init] XLM-R classifier ready -> mode='xlmr_classifier' (Tier 3) is available.")
else:
    xlmr_classifier = None
    print(f"[init] No XLM-R checkpoint found at '{config.XLMR_CHECKPOINT_PATH}' — "
          f"'xlmr_classifier' mode will return an error if selected.")


# ─── RUNPOD SERVERLESS HANDLER ───────────────────────────────────────────

def handler(job):
    """
    Expected input schema:
    {
        "input": {
            "query": "Write a Python function to parse CSV files",
            "mode": "single_centroid",        # optional, one of config.ROUTING_MODES
            "user_expectation": "code_agent", # optional — now actually persisted, in query_log
            "image_base64": "...",            # optional, for OCR/Vision input
            "image_mime_type": "image/png"    # optional
        }
    }
    """
    job_input = job.get("input", {})

    query = job_input.get("query", "").strip()
    if not query:
        return {"error": "Missing required input 'query'."}

    mode = job_input.get("mode", config.DEFAULT_ROUTING_MODE)
    if mode not in config.ROUTING_MODES:
        mode = config.DEFAULT_ROUTING_MODE

    if mode == "xlmr_classifier" and xlmr_classifier is None:
        return {"error": f"mode='xlmr_classifier' requested but no checkpoint was loaded "
                          f"at cold start (looked for '{config.XLMR_CHECKPOINT_PATH}')."}

    image_base64 = job_input.get("image_base64")
    image_mime_type = job_input.get("image_mime_type")
    user_expectation = job_input.get("user_expectation")

    try:
        result = adaptive_route(
            query,
            embedder,
            store,
            llm_client,
            mode=mode,
            xlmr_classifier=xlmr_classifier,
            image_base64=image_base64,
            image_mime_type=image_mime_type,
        )
    except Exception as e:
        # Surface the real error in the job output instead of a bare crash,
        # and print the full traceback so it's visible in RunPod's logs too.
        print(f"[route] ERROR on query={query!r} mode={mode}: {e}")
        traceback.print_exc()
        return {"error": f"{type(e).__name__}: {e}"}

    print(f"[route] query={query!r} mode={mode} -> {result['capability']} "
          f"({result['tier_label']}, confident={result['confident']}, "
          f"routing={result['routing_time_ms']}ms, generation={result['generation_time_ms']}ms, "
          f"images_returned={len(result.get('answer_images', []))})")

    # Only a Tier-2 (LLM-routed) query grows the seed bucket. When that
    # happens, persist the new seed + its embedding into Postgres immediately
    # (the in-memory cache was already updated inside adaptive_route/AgentStore).
    if result["seed_appended"]:
        embedding = store.embedding_cache.get(query)
        if embedding is not None:
            db.insert_seed(db_pool, result["capability"], query, embedding, config.EMBED_MODEL)
            print(f"[db] seed persisted to Postgres for capability={result['capability']!r}")

    # Log every request: what was asked, what the user expected, what happened.
    db.insert_query_log(
        db_pool,
        user_prompt=query,
        user_expectation=user_expectation,
        routed_capability=result["capability"],
        routed_model=result["model"],
        answer=result.get("answer", ""),
        tier=result["tier"],
        confidence=result["confident"],
        mode=mode,
    )

    result["user_expectation"] = user_expectation
    return result


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
