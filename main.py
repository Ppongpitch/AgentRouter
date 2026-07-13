"""
main.py
───────
The actual program. This is the only place allowed to print() — every
other module is a pure library (no side effects beyond what they're
explicitly asked to do).

Responsibilities:
  1. Load config, embedder, agents, seeds (train.jsonl), and the
     embedding cache once at startup.
  2. Build the AgentStore + LLM client.
  3. Serve a simple HTML page (static/index.html).
  4. Expose POST /api/route so the page's JS can call the router and get
     a JSON result back (routing decision + tier + actual agent answer).
  5. Persist the embedding cache to disk so seeds already embedded once
     are never re-embedded on the next run.
  6. Run locally via `python main.py` (uvicorn under the hood).
"""

import os

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
from agent_store import AgentStore, load_agents_json, load_seeds_from_train_jsonl
from embedder import load_embedder
from embedding_cache import load_cache
from llm_router import build_llm_client
from router import adaptive_route

# ── Startup: build everything once ──────────────────────────────────────
print(f"Loading embedder: {config.EMBED_MODEL} ...")
embedder = load_embedder(config.EMBED_MODEL)
print("Embedder ready.")

embedding_cache = load_cache(config.EMBEDDING_CACHE_PATH)
print(f"Loaded embedding cache: {len(embedding_cache)} cached vectors "
      f"from {config.EMBEDDING_CACHE_PATH}")

if os.path.exists(config.AGENTS_JSON_PATH):
    agents_raw = load_agents_json(config.AGENTS_JSON_PATH)
    print(f"Loaded {len(agents_raw)} agents from {config.AGENTS_JSON_PATH}")
else:
    raise FileNotFoundError(
        f"Agents JSON not found at '{config.AGENTS_JSON_PATH}'. "
        f"Set AGENTS_JSON_PATH env var or place the file next to main.py."
    )

if os.path.exists(config.TRAIN_JSONL_PATH):
    stats = load_seeds_from_train_jsonl(agents_raw, config.TRAIN_JSONL_PATH)
    print(
        f"Loaded seeds from {config.TRAIN_JSONL_PATH}: "
        f"+{stats['added']} added, {stats['skipped']} skipped"
    )
    for cap, n in stats["per_agent"].items():
        print(f"   {cap:<40} seeds={n}")
else:
    print(f"No train.jsonl found at '{config.TRAIN_JSONL_PATH}' — using inline seeds only.")

print("Building AgentStore (embedding any seeds not already in the cache)...")
store = AgentStore(agents_raw, embedder, embedding_cache)
print(f"AgentStore built with {len(store.agents)} agents.")
print(f"Embedding cache now holds {store.cache_stats()['cached_embeddings']} vectors.")

# Persist immediately — the very first run pays the full embedding cost once;
# every run after that reuses the cache and only embeds genuinely new seeds.
store.save_embedding_cache(config.EMBEDDING_CACHE_PATH)
print(f"Embedding cache saved to {config.EMBEDDING_CACHE_PATH}")

llm_client = build_llm_client()
print(f"LLM router ready -> {config.LLM_ROUTER_MODEL} (called only when semantic fails)")


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
    mode: str = config.DEFAULT_SEMANTIC_MODE


@app.get("/")
def serve_index():
    return FileResponse("static/index.html")


@app.post("/api/route")
def route_query(req: RouteRequest):
    mode = req.mode if req.mode in config.SEMANTIC_MODES else config.DEFAULT_SEMANTIC_MODE

    result = adaptive_route(req.query, embedder, store, llm_client, mode=mode)
    print(f"[route] query={req.query!r} mode={mode} -> {result['capability']} "
          f"({result['tier_label']}, confident={result['confident']}, "
          f"routing={result['routing_time_ms']}ms, generation={result['generation_time_ms']}ms)")

    # Only a Tier-2 (LLM-routed) query grows the seed bucket, and only then
    # does the embedding cache actually change — so only save in that case.
    if result["seed_appended"]:
        store.save_embedding_cache(config.EMBEDDING_CACHE_PATH)
        print(f"[cache] embedding cache updated -> "
              f"{store.cache_stats()['cached_embeddings']} vectors total")

    return result


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
