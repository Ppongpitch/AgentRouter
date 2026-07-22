"""
db.py
─────
PostgreSQL persistence layer:
  - seeds table: every seed's text + embedding (the durable replacement
    for the old local embedding_cache.pkl), tagged with which embed_model
    produced it — a seed embedded with a different model is treated as
    not-yet-stored, since the vectors aren't comparable across models.
  - query_log table: every request's prompt, the user's stated expectation,
    what actually got routed, and the answer returned.
 
No print() anywhere — pure functions; main.py decides what (if anything)
to log.
 
Uses a small threaded connection pool (psycopg2) since FastAPI runs each
request in a worker thread — a single shared connection would not be
safe to use concurrently.
"""
 
import numpy as np
import psycopg2
import psycopg2.pool
 
from .config import EMBED_MODEL, POSTGRES_DSN
 
_SCHEMA = """
CREATE TABLE IF NOT EXISTS seeds (
    id SERIAL PRIMARY KEY,
    capability TEXT NOT NULL,
    seed_text TEXT NOT NULL,
    embed_model TEXT NOT NULL,
    embedding DOUBLE PRECISION[] NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (capability, seed_text, embed_model)
);
 
CREATE TABLE IF NOT EXISTS query_log (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    user_prompt TEXT NOT NULL,
    user_expectation TEXT,
    routed_capability TEXT NOT NULL,
    routed_model TEXT NOT NULL,
    answer TEXT,
    tier INT,
    confidence DOUBLE PRECISION,
    mode TEXT
);
"""
 
 
def create_pool(dsn: str = POSTGRES_DSN, minconn: int = 1, maxconn: int = 10):
    """Create a small threadsafe connection pool. Call once at startup."""
    return psycopg2.pool.ThreadedConnectionPool(minconn, maxconn, dsn)
 
 
def init_schema(pool) -> None:
    """Create tables if they don't already exist. Safe to call every startup."""
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(_SCHEMA)
        conn.commit()
    finally:
        pool.putconn(conn)
 
 
def load_all_seeds(pool, embed_model: str = EMBED_MODEL) -> dict:
    """
    Load every stored seed + embedding for the CURRENT embed_model only.
    Rows from a different embed_model are ignored on purpose — a model
    change means a different vector space, so those old embeddings aren't
    usable here; that capability is treated the same as having nothing
    stored yet (falls back to bootstrapping from capability_agents.json /
    train.jsonl, and gets its own fresh row set under the new model).
 
    Returns {capability: [(seed_text, embedding_ndarray), ...]}.
    """
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT capability, seed_text, embedding FROM seeds WHERE embed_model = %s",
                (embed_model,),
            )
            rows = cur.fetchall()
    finally:
        pool.putconn(conn)
 
    result: dict = {}
    for capability, seed_text, embedding in rows:
        result.setdefault(capability, []).append((seed_text, np.array(embedding)))
    return result
 
 
def insert_seed(
    pool,
    capability: str,
    seed_text: str,
    embedding: np.ndarray,
    embed_model: str = EMBED_MODEL,
) -> None:
    """
    Persist one seed + its embedding. Idempotent — inserting the same
    (capability, seed_text, embed_model) twice is a no-op, so this is
    safe to call both during initial bootstrap and on every later
    seed-growth event without needing to check existence first.
    """
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO seeds (capability, seed_text, embed_model, embedding)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (capability, seed_text, embed_model) DO NOTHING
                """,
                (capability, seed_text, embed_model, list(map(float, embedding))),
            )
        conn.commit()
    finally:
        pool.putconn(conn)
 
 
def insert_seeds_bulk(
    pool,
    rows: list[tuple],
    embed_model: str = EMBED_MODEL,
) -> None:
    """
    Bulk version of insert_seed for initial bootstrap (many seeds at once).
    rows: list of (capability, seed_text, embedding_ndarray).
    """
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO seeds (capability, seed_text, embed_model, embedding)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (capability, seed_text, embed_model) DO NOTHING
                """,
                [
                    (capability, seed_text, embed_model, list(map(float, embedding)))
                    for capability, seed_text, embedding in rows
                ],
            )
        conn.commit()
    finally:
        pool.putconn(conn)
 
 
def insert_query_log(
    pool,
    user_prompt: str,
    user_expectation: str | None,
    routed_capability: str,
    routed_model: str,
    answer: str,
    tier: int,
    confidence: float,
    mode: str,
) -> None:
    """Log one request: what was asked, what the user expected, what happened."""
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO query_log
                    (user_prompt, user_expectation, routed_capability,
                     routed_model, answer, tier, confidence, mode)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    user_prompt,
                    user_expectation or None,
                    routed_capability,
                    routed_model,
                    answer,
                    tier,
                    confidence,
                    mode,
                ),
            )
        conn.commit()
    finally:
        pool.putconn(conn)
 
 
def close_pool(pool) -> None:
    """Close every connection in the pool. Call this on shutdown."""
    pool.closeall()