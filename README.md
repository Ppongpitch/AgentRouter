# Adaptive Agent Router — local app

Refactor of the single-centroid semantic router notebook into a runnable
local web app: an HTML page calls a local HTTP endpoint that runs the
routing logic and returns JSON.

## File layout

```
config.py            constants only (thresholds, model names, paths, mapping)
embedder.py           load + encode with sentence-transformers — no prints
agent_store.py         AgentStore (single centroid per agent) + seed loaders — no prints
semantic_router.py      cosine-sim vs. centroid — no prints
llm_router.py           OpenRouter LLM fallback via LangChain — no prints
router.py               adaptive_route(): combines semantic + LLM, LLM-only seed growth — no prints
main.py                 the actual program: builds everything once, runs the server, all print()/logging lives here
static/index.html        simple page: textarea + button, calls /api/route
capability_agents.json   agent definitions (replace with your real seed set)
train.jsonl              (you provide this) rows of {"text": ..., "agent": ...}
requirements.txt
```

Every module except `main.py` is a pure library: functions return data,
nothing is printed. `main.py` is "the program" — it owns startup logging
and per-request logging.

## Setup

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY="your-key-here"
```

Place your own `train.jsonl` (rows of `{"text": "...", "agent": "..."}`,
using the same agent codes as before — `code_agent`, `research_agent`, etc.)
next to `main.py`. If it's missing, the app still runs using only the
inline seeds in `capability_agents.json`.

You can also replace `capability_agents.json` with your own file if you
already have a richer seed set — the loader just reads
`capability, model, description, seeds` per agent.

## Run

```bash
python main.py
```

Then open **http://127.0.0.1:8000** in your browser, type a query, and
click "Route Query". The page calls `POST /api/route` and displays the
result (capability, model, confidence, which router handled it, whether
a seed was appended, and the full semantic score breakdown).

## Notes

- Seeds are appended to an agent's bucket **only** when the LLM router
  handles the query — semantic hits never grow the seed list (matches
  the fix from the notebook).
- Threshold per agent is constant (`AGENT_THRESHOLDS` in `config.py`) —
  it does not decay or change as seeds accumulate.
- Centroid is a single mean vector per agent (not k-means, not per-seed
  max-similarity) — recomputed on every seed append.
