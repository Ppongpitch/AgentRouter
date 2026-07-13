"""
agent_executor.py
──────────────────
Once router.py decides WHICH agent/model should handle a query, this
module actually calls that model via OpenRouter and returns its answer.
No print() — pure function, returns a string.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL

_AGENT_SYSTEM = """You are a specialized AI agent with the following role:
{description}

Answer the user's query directly and helpfully, staying within your role."""

_AGENT_PROMPT = ChatPromptTemplate.from_messages(
    [("system", _AGENT_SYSTEM), ("human", "{query}")]
)


def build_agent_client(
    model: str,
    api_key: str = OPENROUTER_API_KEY,
    base_url: str = OPENROUTER_BASE_URL,
) -> ChatOpenAI:
    """Build a chat client for whichever model the router picked."""
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0.7,
        max_tokens=1024,
        default_headers={
            "HTTP-Referer": "http://localhost",
            "X-Title": "Adaptive Agent Router",
        },
    )


def run_agent(query: str, model: str, description: str = "") -> str:
    """
    Call the routed agent's model with the user's query and return its
    answer as plain text. Raises whatever the underlying client raises
    on failure — main.py decides how to surface errors to the caller.
    """
    client = build_agent_client(model)
    chain = _AGENT_PROMPT | client | StrOutputParser()
    return chain.invoke({"description": description or "General assistant.", "query": query})
