"""
main.py
───────
The actual program. This is the only place allowed to print() — every
other module is a pure library (no side effects beyond what they're
explicitly asked to do).

Responsibilities:
  1. Load config, embedder, agents once at startup.
  2. Load seed embeddings from PostgreSQL into an in-memory dict (this
     is now the ONLY persistence layer — the old local embedding_cache.pkl
     file is gone). If the DB has nothing yet for a capability, bootstrap
     it from capability_agents.json's inline seeds + train.jsonl, embed
     them, then write them into Postgres so future runs load from there.
  3. Build the AgentStore + LLM client.
  4. Serve a simple HTML page (static/index.html).
  5. Expose POST /api/route so the page's JS can call the router and get
     a JSON result back (routing decision + tier + actual agent answer).
  6. On every request: if a seed got appended (Tier 2 / LLM route), write
     it into Postgres immediately, AND log the request (prompt, user's
     stated expectation, what got routed, the answer) into query_log.
  7. On shutdown: close the DB pool and clear the in-memory seed cache —
     Postgres is the only thing that persists across restarts now.
  8. Run locally via `python main.py` (uvicorn under the hood).
"""

import os
import traceback

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
import db
from agent_store import AgentStore, load_agents_json, load_seeds_from_train_jsonl
from embedder import load_embedder
from llm_router import build_llm_client
from router import adaptive_route
from xlmr_router import XLMRClassifier

# ── Startup: build everything once ──────────────────────────────────────
print(f"Loading embedder: {config.EMBED_MODEL} ...")
embedder = load_embedder(config.EMBED_MODEL)
print("Embedder ready.")

print(f"Connecting to PostgreSQL ...")
db_pool = db.create_pool(config.POSTGRES_DSN)
db.init_schema(db_pool)
print("PostgreSQL connected, schema ready.")

if os.path.exists(config.AGENTS_JSON_PATH):
    agents_raw = load_agents_json(config.AGENTS_JSON_PATH)
    print(f"Loaded {len(agents_raw)} agents from {config.AGENTS_JSON_PATH}")
else:
    raise FileNotFoundError(
        f"Agents JSON not found at '{config.AGENTS_JSON_PATH}'. "
        f"Set AGENTS_JSON_PATH env var or place the file next to main.py."
    )

# ── Load seeds from Postgres — this replaces the old local pickle cache ──
db_seeds = db.load_all_seeds(db_pool, config.EMBED_MODEL)
seed_embedding_cache: dict = {}
already_in_db = set()

for ag in agents_raw:
    cap = ag["capability"]
    rows = db_seeds.get(cap)
    if rows:
        # DB is the source of truth for this agent's seeds — replace the
        # inline seeds from capability_agents.json entirely.
        ag["seeds"] = [text for text, _ in rows]
        for text, embedding in rows:
            seed_embedding_cache[text] = embedding
        already_in_db.add(cap)
        print(f"   {cap:<40} loaded {len(rows)} seeds from Postgres")
    else:
        print(f"   {cap:<40} nothing in Postgres yet — will bootstrap")

# For any capability NOT yet in Postgres, merge in train.jsonl seeds too
# (matches the original bootstrap behavior) — but skip capabilities
# already loaded from the DB, to avoid duplicating/re-merging seeds.
if os.path.exists(config.TRAIN_JSONL_PATH):
    stats = load_seeds_from_train_jsonl(
        agents_raw, config.TRAIN_JSONL_PATH, skip_capabilities=already_in_db
    )
    print(
        f"Loaded seeds from {config.TRAIN_JSONL_PATH} for not-yet-bootstrapped agents: "
        f"+{stats['added']} added, {stats['skipped']} skipped"
    )
else:
    print(f"No train.jsonl found at '{config.TRAIN_JSONL_PATH}' — bootstrapping from inline seeds only.")

print("Building AgentStore (embedding any seeds not already in the cache)...")
store = AgentStore(agents_raw, embedder, seed_embedding_cache)
print(f"AgentStore built with {len(store.agents)} agents.")
print(f"Seed embedding cache now holds {store.cache_stats()['cached_embeddings']} vectors.")

# One-time bootstrap write: any capability that had nothing in Postgres
# yet gets its freshly-embedded seeds written now, so next run loads them
# straight from the DB instead of re-bootstrapping.
bootstrap_rows = []
for ag in store.agents:
    if ag["capability"] not in already_in_db:
        for seed_text in ag["seeds"]:
            bootstrap_rows.append(
                (ag["capability"], seed_text, seed_embedding_cache[seed_text])
            )
if bootstrap_rows:
    db.insert_seeds_bulk(db_pool, bootstrap_rows, config.EMBED_MODEL)
    print(f"Bootstrapped {len(bootstrap_rows)} seeds into Postgres for the first time.")

llm_client = build_llm_client()
print(f"LLM router ready -> {config.LLM_ROUTER_MODEL} (called only when semantic fails)")

if os.path.exists(config.XLMR_CHECKPOINT_PATH):
    print(f"Loading XLM-R classifier from {config.XLMR_CHECKPOINT_PATH} ...")
    xlmr_classifier = XLMRClassifier(config.XLMR_CHECKPOINT_PATH, config.XLMR_TOKENIZER_NAME)
    print("XLM-R classifier ready -> mode='xlmr_classifier' (Tier 3) is available.")
else:
    xlmr_classifier = None
    print(
        f"No XLM-R checkpoint found at '{config.XLMR_CHECKPOINT_PATH}' — "
        f"'xlmr_classifier' mode will return an error if selected. "
        f"Set XLMR_CHECKPOINT_PATH env var if it's somewhere else."
    )


# ── FastAPI app ───────────────────────────────────────────────────────────
app = FastAPI(title="Adaptive Agent Router")

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


class RouteRequest(BaseModel):
    query: str
    mode: str = config.DEFAULT_ROUTING_MODE
    user_expectation: str | None = None
    image_base64: str | None = None
    image_mime_type: str | None = None


@app.get("/")
def serve_index():
    return FileResponse("static/index.html")


@app.post("/api/route")
def route_query(req: RouteRequest):
    mode = req.mode if req.mode in config.ROUTING_MODES else config.DEFAULT_ROUTING_MODE

    if mode == "xlmr_classifier" and xlmr_classifier is None:
        return {
            "error": (
                f"xlmr_classifier mode was selected but no checkpoint was loaded "
                f"at startup (looked for '{config.XLMR_CHECKPOINT_PATH}'). "
                f"Set XLMR_CHECKPOINT_PATH and restart the server."
            )
        }

    try:
        result = adaptive_route(
            req.query,
            embedder,
            store,
            llm_client,
            mode=mode,
            xlmr_classifier=xlmr_classifier,
            image_base64=req.image_base64,
            image_mime_type=req.image_mime_type,
        )
    except Exception as e:
        # Surface the REAL error to the browser instead of a bare 500 with
        # no detail, and print the full traceback here so it's visible in
        # this terminal too. Common cause: an image was attached but the
        # query got routed to a model that doesn't support image input.
        print(f"[route] ERROR on query={req.query!r} mode={mode}: {e}")
        traceback.print_exc()
        return {"error": f"{type(e).__name__}: {e}"}

    print(f"[route] query={req.query!r} mode={mode} -> {result['capability']} "
          f"({result['tier_label']}, confident={result['confident']}, "
          f"routing={result['routing_time_ms']}ms, generation={result['generation_time_ms']}ms, "
          f"images_returned={len(result.get('answer_images', []))})")

    # Only a Tier-2 (LLM-routed) query grows the seed bucket. When that
    # happens, persist the new seed + its embedding into Postgres immediately
    # (the in-memory cache was already updated inside adaptive_route/AgentStore).
    if result["seed_appended"]:
        embedding = store.embedding_cache.get(req.query)
        if embedding is not None:
            db.insert_seed(db_pool, result["capability"], req.query, embedding, config.EMBED_MODEL)
            print(f"[db] seed persisted to Postgres for capability={result['capability']!r}")

    # Log every request: what was asked, what the user expected, what happened.
    db.insert_query_log(
        db_pool,
        user_prompt=req.query,
        user_expectation=req.user_expectation,
        routed_capability=result["capability"],
        routed_model=result["model"],
        answer=result.get("answer", ""),
        tier=result["tier"],
        confidence=result["confident"],
        mode=mode,
    )

    return result


@app.on_event("shutdown")
def on_shutdown():
    """Close the DB pool and drop the in-memory seed cache. Postgres is the
    only thing that persists — nothing local should remain after this."""
    store.embedding_cache.clear()
    db.close_pool(db_pool)
    print("Shutdown: DB pool closed, in-memory seed cache cleared.")


if __name__ == "__main__":
    if config.ENABLE_NGROK:
        try:
            import ngrok
        except ImportError:
            raise SystemExit(
                "ENABLE_NGROK=true but the 'ngrok' package isn't installed.\n"
                "Run: pip install ngrok"
            )

        if config.NGROK_AUTHTOKEN:
            os.environ.setdefault("NGROK_AUTHTOKEN", config.NGROK_AUTHTOKEN)

        listener = ngrok.forward(config.PORT, authtoken_from_env=True)
        public_url = listener.url()
        print(f"\n🌍 Public URL (ngrok): {public_url}")
        print(f"   Paste this into API_BASE_URL near the top of static/index.html,")
        print(f"   then redeploy that one file to your static host.\n")

    print(f"Starting server at http://{config.HOST}:{config.PORT}")
    uvicorn.run(app, host=config.HOST, port=config.PORT)