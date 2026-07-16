import os
import time
import warnings

import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

import runpod

# ─── 1. GLOBAL COLD-START INITIALIZATION ───────────────────────────────────
# Everything in this section runs ONCE when the container boots. RunPod
# reuses this same warm process across multiple jobs (flashboot), so the
# embedder + seed embeddings stay in memory across jobs — no local disk
# cache and no DB needed for that benefit. Seeds CAN still grow in-memory
# across jobs within one warm worker's lifetime (Tier 2 / LLM route), but
# that growth does NOT survive a full cold restart — there's no persistent
# store in this version.

EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-m3")
AGENTS_JSON_PATH = os.environ.get("AGENTS_JSON_PATH", "capability_agents.json")
TRAIN_JSONL_PATH = os.environ.get("TRAIN_JSONL_PATH", "train.jsonl")

XLMR_CHECKPOINT_PATH = os.environ.get("XLMR_CHECKPOINT_PATH", "best.ckpt")
XLMR_TOKENIZER_NAME = os.environ.get("XLMR_TOKENIZER_NAME", "xlm-roberta-base")
XLMR_TOP_K = int(os.environ.get("XLMR_TOP_K", "3"))

N_CLUSTERS_PER_AGENT = int(os.environ.get("N_CLUSTERS_PER_AGENT", "3"))
KMEANS_RANDOM_STATE = 42

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_IMAGES_URL = "https://openrouter.ai/api/v1/images"
LLM_ROUTER_MODEL = os.environ.get("LLM_ROUTER_MODEL", "google/gemini-2.5-flash")

ROUTING_MODES = ["per_seed", "single_centroid", "multi_centroid", "xlmr_classifier"]
DEFAULT_ROUTING_MODE = os.environ.get("DEFAULT_ROUTING_MODE", "single_centroid")

OCR_CAPABILITY = "Vision / OCR Agent"

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
DEFAULT_THRESHOLD = 0.50

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

# XLM-R classifier class-index -> agent code. MUST match training order —
# see the warning in AgentRouterModel.load_from_checkpoint below.
AGENT_CLASSES = [
    "general_assistant", "code_agent", "translation_agent", "research_agent",
    "reasoning_agent", "vision_agent", "image_generation_agent", "creative_writing_agent",
]
ID2AGENT = {i: agent for i, agent in enumerate(AGENT_CLASSES)}

import json as _json


def _load_agents_json(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return _json.load(f)


def _load_seeds_from_train_jsonl(agents_raw: list, train_path: str) -> None:
    cap_lookup = {ag["capability"]: ag for ag in agents_raw}
    with open(train_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = _json.loads(line)
            cap = JSON_TO_CAPABILITY.get(row.get("agent"))
            if cap in cap_lookup:
                cap_lookup[cap]["seeds"].append(row["text"])


print(f"[init] Loading embedder '{EMBED_MODEL}' ...")
embedder = SentenceTransformer(EMBED_MODEL, model_kwargs={"use_safetensors": True})
print("[init] Embedder ready.")

print(f"[init] Loading agents from '{AGENTS_JSON_PATH}' ...")
AGENTS_RAW = _load_agents_json(AGENTS_JSON_PATH)

if os.path.exists(TRAIN_JSONL_PATH):
    _load_seeds_from_train_jsonl(AGENTS_RAW, TRAIN_JSONL_PATH)
    print(f"[init] Merged seeds from '{TRAIN_JSONL_PATH}'.")
else:
    print(f"[init] No '{TRAIN_JSONL_PATH}' found — using inline seeds only.")


# ─── 2. IN-MEMORY SEED STORE (AgentStore-lite) ─────────────────────────────
# Keeps ONE mutable seed list per agent + all three representations
# (per-seed embeddings, single centroid, k-means sub-centroids), all
# derived from the same in-memory embedding cache — switching routing
# mode is just picking a different pre-built array, never re-embedding.

import copy

embedding_cache: dict = {}  # seed_text -> np.ndarray. In-memory only, no disk/DB.


def _embed_texts(texts: list) -> np.ndarray:
    return embedder.encode(texts, normalize_embeddings=True)


def _get_or_compute_embeddings(texts: list) -> np.ndarray:
    missing = [t for t in texts if t not in embedding_cache]
    if missing:
        new_embs = _embed_texts(missing)
        for t, emb in zip(missing, new_embs):
            embedding_cache[t] = emb
    return np.vstack([embedding_cache[t] for t in texts])


def _mean_centroid(seed_embeddings: np.ndarray) -> np.ndarray:
    centroid = seed_embeddings.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-10)
    return centroid.reshape(1, -1)


def _kmeans_centroids(seed_embeddings: np.ndarray) -> np.ndarray:
    n = len(seed_embeddings)
    k = min(N_CLUSTERS_PER_AGENT, n)
    if k <= 1:
        return _mean_centroid(seed_embeddings)
    km = KMeans(n_clusters=k, random_state=KMEANS_RANDOM_STATE, n_init=10)
    km.fit(seed_embeddings)
    centers = km.cluster_centers_
    norms = np.linalg.norm(centers, axis=1, keepdims=True) + 1e-10
    return centers / norms


def _refresh_agent_representations(agent_record: dict) -> None:
    seed_embeddings = _get_or_compute_embeddings(agent_record["seeds"])
    agent_record["seed_embeddings"] = seed_embeddings
    agent_record["centroid"] = _mean_centroid(seed_embeddings)
    agent_record["centroids"] = _kmeans_centroids(seed_embeddings)


def build_agent_store(raw_agents: list) -> list:
    agents = []
    for ag in raw_agents:
        record = copy.deepcopy(ag)
        _refresh_agent_representations(record)
        record["threshold"] = AGENT_THRESHOLDS.get(record["capability"], DEFAULT_THRESHOLD)
        agents.append(record)
    return agents


def get_agent(agents: list, capability: str):
    return next((ag for ag in agents if ag["capability"] == capability), None)


def update_agent_seeds(agents: list, capability: str, new_seed: str) -> bool:
    """Append new_seed and refresh representations. In-memory only — does
    NOT persist across a cold restart in this no-DB version."""
    ag = get_agent(agents, capability)
    if ag is None:
        return False
    ag["seeds"].append(new_seed)
    _refresh_agent_representations(ag)
    return True


print("[init] Building agent store (embedding all seeds once)...")
AGENTS = build_agent_store(AGENTS_RAW)
print(f"[init] Agent store built with {len(AGENTS)} agents, "
      f"{len(embedding_cache)} seed embeddings cached in memory.")


# ─── 3. LLM ROUTER (Tier 2 fallback) ────────────────────────────────────────

_LLM_SYSTEM = """You are an expert agent router. Given a user query, decide which agent should handle it.

Available agents:
{capabilities}

Rules:
1. Choose EXACTLY ONE capability name from the list above.
2. The capability field must match one of the names EXACTLY (case-sensitive).
3. Estimate confidence as a float 0.00-1.00.
4. Write a one-sentence intent.
5. Respond ONLY with valid JSON - no markdown, no extra text.

Format:
{{"capability": "<exact capability name>", "intent": "<one sentence>", "confident": <float>}}"""

_LLM_PROMPT = ChatPromptTemplate.from_messages([("system", _LLM_SYSTEM), ("human", "{query}")])

llm_client = ChatOpenAI(
    model=LLM_ROUTER_MODEL,
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL,
    temperature=0.0,
    max_tokens=256,
    default_headers={"HTTP-Referer": "http://localhost", "X-Title": "Adaptive Agent Router"},
)
print(f"[init] LLM router ready -> {LLM_ROUTER_MODEL} (called only when semantic routing misses)")


def _get_capabilities_str(agents: list) -> str:
    return "\n".join(f"  - {ag['capability']}: {ag['description']}" for ag in agents)


def llm_route(query: str) -> dict:
    import re

    chain = _LLM_PROMPT | llm_client | StrOutputParser()
    raw = chain.invoke({"capabilities": _get_capabilities_str(AGENTS), "query": query})

    clean = re.sub(r"```(?:json)?", "", raw).strip().strip("`")
    try:
        data = _json.loads(clean)
    except _json.JSONDecodeError:
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        data = _json.loads(match.group()) if match else {}

    valid_caps = [ag["capability"] for ag in AGENTS]
    cap = data.get("capability", "")
    if cap not in valid_caps:
        lower_map = {c.lower(): c for c in valid_caps}
        cap = lower_map.get(cap.lower(), valid_caps[0])

    try:
        confident = round(max(0.0, min(1.0, float(data.get("confident", 0.5)))), 4)
    except (TypeError, ValueError):
        confident = 0.5

    agent = get_agent(AGENTS, cap)
    model = agent["model"] if agent else AGENTS[0]["model"]

    return {
        "capability": cap,
        "model": model,
        "confident": confident,
        "intent": data.get("intent", "General request"),
        "router_used": "llm",
    }


# ─── 4. SEMANTIC ROUTING (Tier 1 — 3 selectable modes) ─────────────────────

def _score_agent(q_emb, ag: dict, mode: str) -> float:
    if mode == "per_seed":
        return float(cosine_similarity(q_emb, ag["seed_embeddings"])[0].max())
    elif mode == "single_centroid":
        return float(cosine_similarity(q_emb, ag["centroid"])[0][0])
    elif mode == "multi_centroid":
        return float(cosine_similarity(q_emb, ag["centroids"])[0].max())
    raise ValueError(f"Unknown semantic mode: {mode!r}")


def semantic_route(query: str, mode: str):
    q_emb = embedder.encode([query], normalize_embeddings=True)
    scores = []
    for ag in AGENTS:
        sim = _score_agent(q_emb, ag, mode)
        scores.append({"capability": ag["capability"], "model": ag["model"],
                        "sim": sim, "threshold": ag["threshold"]})
    scores.sort(key=lambda x: x["sim"], reverse=True)
    best = scores[0]
    if best["sim"] >= best["threshold"]:
        return {
            "capability": best["capability"], "model": best["model"],
            "confident": round(best["sim"], 4), "router_used": "semantic",
        }, scores
    return None, scores


# ─── 5. XLM-R CLASSIFIER (Tier 3, optional) ────────────────────────────────

xlmr_classifier = None

if os.path.exists(XLMR_CHECKPOINT_PATH):
    import torch
    import torch.nn as nn
    from transformers import AutoModel, AutoTokenizer

    class AgentRouterModel(nn.Module):
        def __init__(self, model_name, num_agents, top_k=3):
            super().__init__()
            self.top_k = top_k
            self.encoder = AutoModel.from_pretrained(model_name)
            hidden_size = self.encoder.config.hidden_size
            self.gating_net = nn.Sequential(
                nn.Linear(hidden_size, 256), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(256, num_agents),
            )

        def forward(self, input_ids, attention_mask):
            cls = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
            logits = self.gating_net(cls)
            probs = torch.softmax(logits, dim=1)
            topk_probs, topk_indices = torch.topk(probs, k=self.top_k, dim=1)
            return logits, probs, topk_probs, topk_indices

        @classmethod
        def load_from_checkpoint(cls, checkpoint_path, map_location="cpu"):
            ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
            hparams = ckpt["hyper_parameters"]
            model = cls(model_name=hparams["model_name"], num_agents=hparams["num_agents"],
                         top_k=hparams.get("top_k", 3))
            # strict=False: training-only keys (e.g. a loss-weighting tensor
            # like "agent_weights") aren't part of this inference-only
            # architecture and are safe to ignore. Missing keys (parts of
            # encoder/gating_net NOT found in the checkpoint) are the real
            # red flag — those would load with random weights.
            incompatible = model.load_state_dict(ckpt["state_dict"], strict=False)
            if incompatible.missing_keys:
                warnings.warn(f"XLM-R checkpoint MISSING keys: {incompatible.missing_keys}")
            if incompatible.unexpected_keys:
                warnings.warn(f"XLM-R checkpoint UNEXPECTED keys (ignored): {incompatible.unexpected_keys}")
            return model

    print(f"[init] Loading XLM-R classifier from '{XLMR_CHECKPOINT_PATH}' ...")
    _xlmr_tokenizer = AutoTokenizer.from_pretrained(XLMR_TOKENIZER_NAME)
    _xlmr_model = AgentRouterModel.load_from_checkpoint(XLMR_CHECKPOINT_PATH, map_location="cpu")
    _xlmr_model.eval()
    for p in _xlmr_model.parameters():
        p.requires_grad = False

    @torch.no_grad()
    def _xlmr_predict(prompt: str, top_k: int = None) -> list:
        k = top_k or _xlmr_model.top_k or XLMR_TOP_K
        inputs = _xlmr_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128, padding=True)
        _, probs, topk_probs, topk_indices = _xlmr_model(inputs["input_ids"], inputs["attention_mask"])
        return [
            {"agent_code": ID2AGENT[idx.item()], "confidence": round(prob.item(), 4)}
            for prob, idx in zip(topk_probs[0][:k], topk_indices[0][:k])
        ]

    def xlmr_route(query: str):
        predictions = _xlmr_predict(query)
        top1 = predictions[0]
        capability = JSON_TO_CAPABILITY.get(top1["agent_code"])
        agent = get_agent(AGENTS, capability) if capability else None
        if agent is None:
            agent = AGENTS[0]
            capability = agent["capability"]
        result = {"capability": capability, "model": agent["model"],
                  "confident": top1["confidence"], "router_used": "xlmr_classifier"}
        scores = [{"capability": JSON_TO_CAPABILITY.get(p["agent_code"], p["agent_code"]),
                    "sim": p["confidence"]} for p in predictions]
        return result, scores

    xlmr_classifier = True  # sentinel: xlmr_route is defined and usable
    print("[init] XLM-R classifier ready -> mode='xlmr_classifier' (Tier 3) is available.")
else:
    def xlmr_route(query: str):
        raise RuntimeError(
            f"mode='xlmr_classifier' but no checkpoint found at '{XLMR_CHECKPOINT_PATH}'."
        )
    print(f"[init] No XLM-R checkpoint at '{XLMR_CHECKPOINT_PATH}' — 'xlmr_classifier' mode will error if selected.")


# ─── 6. AGENT EXECUTION (chat completions + dedicated Image API) ──────────

_AGENT_SYSTEM = """You are a specialized AI agent with the following role:
{description}

Answer the user's query directly and helpfully, staying within your role."""
_AGENT_PROMPT = ChatPromptTemplate.from_messages([("system", _AGENT_SYSTEM), ("human", "{query}")])


def _build_agent_client(model: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=model, api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL,
        temperature=0.7, max_tokens=1024,
        default_headers={"HTTP-Referer": "http://localhost", "X-Title": "Adaptive Agent Router"},
    )


def _extract_text_and_images(message) -> tuple:
    text_parts, images = [], []
    content = message.content
    if isinstance(content, str):
        text_parts.append(content)
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif block.get("type") in ("image_url", "image"):
                    image_url = block.get("image_url")
                    url = image_url.get("url") if isinstance(image_url, dict) else image_url
                    if url:
                        images.append(url)
    for img in (getattr(message, "additional_kwargs", {}).get("images") or []):
        if isinstance(img, dict):
            image_url = img.get("image_url")
            url = image_url.get("url") if isinstance(image_url, dict) else img.get("url")
            if url:
                images.append(url)
        elif isinstance(img, str):
            images.append(img)
    return "\n".join(t for t in text_parts if t), images


def run_agent(query: str, model: str, description: str = "",
              image_base64: str = None, image_mime_type: str = None) -> dict:
    client = _build_agent_client(model)
    if image_base64:
        mime = image_mime_type or "image/png"
        messages = [
            SystemMessage(content=description or "General assistant."),
            HumanMessage(content=[
                {"type": "text", "text": query},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_base64}"}},
            ]),
        ]
        response = client.invoke(messages)
    else:
        chain = _AGENT_PROMPT | client
        response = chain.invoke({"description": description or "General assistant.", "query": query})

    text, images = _extract_text_and_images(response)
    if not text and not images:
        warnings.warn(f"run_agent: model '{model}' returned neither text nor images.")
    return {"text": text, "images": images}


def generate_image(prompt: str, model: str) -> dict:
    """Dedicated OpenRouter Image API — NOT the chat-completions path.
    Chat completions + LangChain wasn't reliably surfacing image output
    for image-generation models (missing "modalities" field, and
    non-standard response fields silently dropped by parsing)."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "Adaptive Agent Router",
    }
    resp = requests.post(OPENROUTER_IMAGES_URL, json={"model": model, "prompt": prompt},
                          headers=headers, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    images = []
    for item in data.get("data", []):
        if item.get("b64_json"):
            images.append(f"data:image/png;base64,{item['b64_json']}")
        elif item.get("url"):
            images.append(item["url"])
    if not images:
        warnings.warn(f"generate_image: model '{model}' returned no images. Raw: {data!r}")
    return {"text": "", "images": images}


# ─── 7. ADAPTIVE ROUTE (OCR-first pipeline + tier dispatch + generation) ───

TIER_LABELS = {1: "Tier 1 — Semantic", 2: "Tier 2 — LLM Route", 3: "Tier 3 — XLM-R Classifier"}


def adaptive_route(query: str, mode: str = DEFAULT_ROUTING_MODE,
                    image_base64: str = None, image_mime_type: str = None) -> dict:
    ocr_text = None
    ocr_answer = None
    ocr_generation_ms = 0.0

    if image_base64:
        vision_agent = get_agent(AGENTS, OCR_CAPABILITY)
        if vision_agent is None:
            raise RuntimeError(f"An image was attached but no '{OCR_CAPABILITY}' exists.")
        t0 = time.perf_counter()
        ocr_answer = run_agent(query, vision_agent["model"], vision_agent["description"],
                                image_base64, image_mime_type)
        ocr_generation_ms = round((time.perf_counter() - t0) * 1000, 1)
        ocr_text = ocr_answer["text"]

    routing_start = time.perf_counter()
    if mode == "xlmr_classifier":
        result, sem_scores = xlmr_route(query)
        result.setdefault("intent", f"Route to {result['capability']}")
        tier = 3
    else:
        sem_result, sem_scores = semantic_route(query, mode)
        if sem_result is not None:
            result = sem_result
            result.setdefault("intent", f"Route to {result['capability']}")
            tier = 1
        else:
            result = llm_route(query)
            tier = 2
    routing_time_ms = round((time.perf_counter() - routing_start) * 1000, 1)

    seed_appended = False
    if result["router_used"] == "llm":
        seed_appended = update_agent_seeds(AGENTS, result["capability"], query)

    agent = get_agent(AGENTS, result["capability"])
    description = agent["description"] if agent else ""

    generation_start = time.perf_counter()
    if image_base64 and result["capability"] == OCR_CAPABILITY:
        answer_data = ocr_answer
        result.setdefault("intent", "OCR request — returning extracted text directly")
    elif image_base64:
        combined_query = f"{query}\n\n[Text extracted from the attached image via OCR]:\n{ocr_text}"
        answer_data = run_agent(combined_query, result["model"], description)
    elif result["capability"] == "Image Generation Agent":
        answer_data = generate_image(query, result["model"])
    else:
        answer_data = run_agent(query, result["model"], description)
    generation_time_ms = round((time.perf_counter() - generation_start) * 1000 + ocr_generation_ms, 1)

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
        "answer": answer_data["text"],
        "answer_images": answer_data["images"],
        "ocr_text": ocr_text,
        "routing_time_ms": routing_time_ms,
        "generation_time_ms": generation_time_ms,
        "total_time_ms": round(routing_time_ms + generation_time_ms, 1),
    }


# ─── 8. RUNPOD SERVERLESS HANDLER ──────────────────────────────────────────

def handler(job):
    """
    Expected input schema:
    {
        "input": {
            "query": "Write a Python function to parse CSV files",
            "mode": "single_centroid",       # optional, one of ROUTING_MODES
            "user_expectation": "code_agent", # optional, logged in response only — no DB
            "image_base64": "...",            # optional, for OCR/Vision input
            "image_mime_type": "image/png"    # optional
        }
    }

    Expected environment variables:
    - OPENROUTER_API_KEY  (required for any generation/LLM-fallback call)
    - EMBED_MODEL, LLM_ROUTER_MODEL, XLMR_CHECKPOINT_PATH, etc. — all optional,
      see the constants section at the top of this file for defaults.
    """
    job_input = job.get("input", {})

    query = job_input.get("query", "").strip()
    if not query:
        return {"error": "Missing required input 'query'."}

    mode = job_input.get("mode", DEFAULT_ROUTING_MODE)
    if mode not in ROUTING_MODES:
        mode = DEFAULT_ROUTING_MODE

    if mode == "xlmr_classifier" and xlmr_classifier is None:
        return {"error": f"mode='xlmr_classifier' requested but no checkpoint was loaded "
                          f"at cold start (looked for '{XLMR_CHECKPOINT_PATH}')."}

    image_base64 = job_input.get("image_base64")
    image_mime_type = job_input.get("image_mime_type")
    user_expectation = job_input.get("user_expectation")  # not persisted anywhere in this version

    try:
        result = adaptive_route(query, mode=mode, image_base64=image_base64, image_mime_type=image_mime_type)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

    result["user_expectation"] = user_expectation
    print(f"[route] query={query!r} mode={mode} -> {result['capability']} "
          f"({result['tier_label']}, confident={result['confident']}, "
          f"routing={result['routing_time_ms']}ms, generation={result['generation_time_ms']}ms)")

    return result


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
