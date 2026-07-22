"""
llm_router.py
─────────────
LLM-based fallback routing via OpenRouter (LangChain ChatOpenAI).
No print() — errors and parse failures are handled internally with
sane fallbacks, and the function just returns a routing dict.
"""

import json
import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .agent_store import AgentStore
from .config import LLM_ROUTER_MODEL, OPENROUTER_API_KEY, OPENROUTER_BASE_URL

LLM_SYSTEM = """You are an expert agent router. Given a user query, decide which agent should handle it.

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

_PROMPT = ChatPromptTemplate.from_messages(
    [("system", LLM_SYSTEM), ("human", "{query}")]
)


def build_llm_client(
    model: str = LLM_ROUTER_MODEL,
    api_key: str = OPENROUTER_API_KEY,
    base_url: str = OPENROUTER_BASE_URL,
) -> ChatOpenAI:
    """Construct the LangChain chat client used for LLM-fallback routing."""
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0.0,
        max_tokens=256,
        default_headers={
            "HTTP-Referer": "http://localhost",
            "X-Title": "Adaptive Agent Router",
        },
    )


def _parse_llm_response(raw: str, store: AgentStore) -> dict:
    """Parse + validate the LLM's JSON response, with safe fallbacks."""
    clean = re.sub(r"```(?:json)?", "", raw).strip().strip("`")
    try:
        data = json.loads(clean)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        data = json.loads(match.group()) if match else {}

    valid_caps = store.get_capability_names()
    cap = data.get("capability", "")
    if cap not in valid_caps:
        lower_map = {c.lower(): c for c in valid_caps}
        cap = lower_map.get(cap.lower(), valid_caps[0])  # fallback to first agent

    try:
        confident = round(max(0.0, min(1.0, float(data.get("confident", 0.5)))), 4)
    except (TypeError, ValueError):
        confident = 0.5

    agent = store.get_agent(cap)
    model = agent["model"] if agent else valid_caps and store.agents[0]["model"]

    return {
        "capability": cap,
        "model": model,
        "confident": confident,
        "intent": data.get("intent", "General request"),
        "router_used": "llm",
    }


def llm_route(query: str, store: AgentStore, llm_client: ChatOpenAI) -> dict:
    """Call the LLM router. Only invoked when semantic confidence is insufficient."""
    chain = _PROMPT | llm_client | StrOutputParser()
    raw = chain.invoke({"capabilities": store.get_capabilities_str(), "query": query})
    return _parse_llm_response(raw, store)
