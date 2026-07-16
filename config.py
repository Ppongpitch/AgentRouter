"""
config.py
─────────
Central configuration for the adaptive router. Pure constants only —
no logic, no side effects, no print().
"""

import os

# Load variables from a .env file next to this one, if present. This means
# OPENROUTER_API_KEY (and anything else in .env) survives across terminal
# restarts without needing `$env:OPENROUTER_API_KEY = "..."` every time —
# it's a no-op if python-dotenv isn't installed or .env doesn't exist, so
# this is safe even on machines that still use manual env vars.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ─── Embedding model ─────────────────────────────────────────────────────
EMBED_MODEL = "BAAI/bge-m3"

# ─── Per-agent thresholds (constant — no decay, no growth-based change) ──
# Compared against cosine similarity to each agent's single centroid
# (mean of ALL that agent's seed embeddings).
AGENT_THRESHOLDS = {
    "General Assistant": 0.38,
    "Code Agent": 0.57,
    "Translation / Multilingual Agent": 0.50,
    "Research & Long-document Agent": 0.48,
    "Reasoning Agent": 0.50,
    "Vision / OCR Agent": 0.58,
    "Image Generation Agent": 0.62,
    "Creative Writing Agent": 0.42,
}
DEFAULT_THRESHOLD = 0.50  # fallback if a capability isn't in the dict above

# ─── OpenRouter / LLM router ─────────────────────────────────────────────
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_ROUTER_MODEL = "google/gemini-2.5-flash"  # used only when semantic fails

# ─── Multi-centroid (k-means) settings — used by the "multi_centroid" mode ──
N_CLUSTERS_PER_AGENT = 3
KMEANS_RANDOM_STATE = 42

# ─── Routing modes ─────────────────────────────────────────────────────────
# per_seed         : cosine sim vs EVERY individual seed embedding (max)
# single_centroid  : cosine sim vs ONE mean centroid per agent
# multi_centroid    : cosine sim vs k-means sub-centroids per agent (max)
# xlmr_classifier   : locally-run fine-tuned XLM-R model predicts the agent
#                     directly from the prompt — no embeddings/cosine sim
ROUTING_MODES = ["per_seed", "single_centroid", "multi_centroid", "xlmr_classifier"]
DEFAULT_ROUTING_MODE = "single_centroid"

# ─── XLM-R classifier (Tier 3 routing mode) ────────────────────────────────
XLMR_CHECKPOINT_PATH = os.environ.get("XLMR_CHECKPOINT_PATH", "model/checkpoints/best.ckpt")
XLMR_TOKENIZER_NAME = os.environ.get("XLMR_TOKENIZER_NAME", "xlm-roberta-base")
XLMR_TOP_K = int(os.environ.get("XLMR_TOP_K", "3"))

# ─── File paths ───────────────────────────────────────────────────────────
AGENTS_JSON_PATH = os.environ.get("AGENTS_JSON_PATH", "capability_agents.json")
TRAIN_JSONL_PATH = os.environ.get("TRAIN_JSONL_PATH", "train.jsonl")
# ─── PostgreSQL (persistent store for seeds + their embeddings, and the
# query log — replaces the old local embedding_cache.pkl file entirely) ────
POSTGRES_DSN = os.environ.get(
    "POSTGRES_DSN",
    "postgresql://postgres:postgres@localhost:5432/agentrouter",
)

# ─── train.jsonl "agent" field  →  capability name mapping ───────────────
JSON_TO_CAPABILITY = {
    "image_generation_agent": "Image Generation Agent",
    "reasoning_agent": "Reasoning Agent",
    "research_agent": "Research & Long-document Agent",
    "code_agent": "Code Agent",
    "general_assistant": "General Assistant",
    "creative_writing_agent": "Creative Writing Agent",
    "translation_agent": "Translation / Multilingual Agent",
    "vision_agent": "Vision / OCR Agent",
}

# ─── Server ───────────────────────────────────────────────────────────────
# HOST defaults to 0.0.0.0 so it's reachable from outside the container on
# RunPod (or any host) out of the box. If you want strictly localhost-only
# access during local dev, set ROUTER_HOST=127.0.0.1 explicitly.
HOST = os.environ.get("ROUTER_HOST", "0.0.0.0")
PORT = int(os.environ.get("ROUTER_PORT", "8000"))

# ─── CORS ─────────────────────────────────────────────────────────────────
# Needed because index.html will be hosted on a different domain (your
# static host) than this API (your laptop, exposed via a tunnel). "*" is
# fine for a personal/dev tool like this; tighten it to your static host's
# exact domain if you want to be stricter.
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")

# ─── ngrok (optional, in-process tunnel) ──────────────────────────────────
# If enabled, main.py opens a public tunnel to this server using the
# `ngrok` Python SDK (pip install ngrok) — no separate ngrok.exe/CLI needed.
# Get a free authtoken from https://dashboard.ngrok.com/get-started/your-authtoken
ENABLE_NGROK = os.environ.get("ENABLE_NGROK", "false").lower() == "true"
NGROK_AUTHTOKEN = os.environ.get("NGROK_AUTHTOKEN", "")