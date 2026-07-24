"""
agent_executor.py
──────────────────
Once router.py decides WHICH agent/model should handle a query, this
module actually calls that model via OpenRouter and returns its answer.
No print() — pure functions, returns structured data.

Handles three things that plain text-only chat completion doesn't:
  1. IMAGE OUTPUT (Image Generation Agent, text-to-image) — via
     generate_image() calling OpenRouter's dedicated Image API
     (POST /api/v1/images), NOT chat completions. Chat completions +
     LangChain wasn't reliably surfacing image output (missing the
     "modalities" field, and non-standard response fields silently
     dropped by LangChain's parser).
  2. IMAGE-TO-IMAGE EDITING (Image Generation Agent, with an attached
     image) — same generate_image() call, but with the attached image
     passed via OpenRouter's documented `input_references` field (a list
     of {"type": "image_url", "image_url": {"url": ...}} objects — base64
     data URLs are explicitly supported, not just http(s) URLs). This is
     genuinely different from just calling generate_image() without an
     image: omitting input_references for an edit-style prompt (e.g.
     "change the style of this image to watercolor") gives the model
     nothing to edit, which is why that case previously returned nothing.
  3. IMAGE INPUT (Vision/OCR Agent) — needs the actual uploaded image
     sent as part of the message, not just the text query. When
     image_base64 is provided, messages are built manually as a
     multimodal HumanMessage (text + image_url content blocks) instead
     of going through the plain ChatPromptTemplate.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

import warnings

import requests

from .config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL

OPENROUTER_IMAGES_URL = "https://openrouter.ai/api/v1/images"

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


def _extract_text_and_images(message) -> tuple[str, list[str]]:
    """
    Pull out both plain text AND any image URLs (as data: URIs or http(s)
    URLs) from a raw AIMessage, checking every shape OpenRouter is known
    to use for image-output models. Returns (text, image_urls).
    """
    text_parts: list[str] = []
    images: list[str] = []

    content = message.content

    if isinstance(content, str):
        text_parts.append(content)
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict):
                btype = block.get("type")
                if btype == "text":
                    text_parts.append(block.get("text", ""))
                elif btype in ("image_url", "image"):
                    image_url = block.get("image_url")
                    url = image_url.get("url") if isinstance(image_url, dict) else image_url
                    if url:
                        images.append(url)

    extra_images = getattr(message, "additional_kwargs", {}).get("images") or []
    for img in extra_images:
        if isinstance(img, dict):
            image_url = img.get("image_url")
            url = image_url.get("url") if isinstance(image_url, dict) else img.get("url")
            if url:
                images.append(url)
        elif isinstance(img, str):
            images.append(img)

    return "\n".join(t for t in text_parts if t), images


def run_agent(
    query: str,
    model: str,
    description: str = "",
    image_base64: str | None = None,
    image_mime_type: str | None = None,
) -> dict:
    """
    Call the routed agent's model with the user's query and return its
    answer. Returns {"text": str, "images": list[str]}.

    If image_base64 is provided (Vision/OCR Agent use case), the query is
    sent alongside the actual image as a multimodal message instead of
    plain text. This is a normal chat-completions call — for IMAGE
    GENERATION/EDITING, use generate_image() below instead, not this.
    """
    client = build_agent_client(model)

    if image_base64:
        mime = image_mime_type or "image/png"
        messages = [
            SystemMessage(content=description or "General assistant."),
            HumanMessage(
                content=[
                    {"type": "text", "text": query},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{image_base64}"},
                    },
                ]
            ),
        ]
        response = client.invoke(messages)
    else:
        chain = _AGENT_PROMPT | client
        response = chain.invoke(
            {"description": description or "General assistant.", "query": query}
        )

    text, images = _extract_text_and_images(response)
    if not text and not images:
        warnings.warn(f"run_agent: model '{model}' returned neither text nor images.")
    return {"text": text, "images": images}


def generate_image(
    prompt: str,
    model: str,
    input_image_base64: str | None = None,
    input_image_mime_type: str | None = None,
) -> dict:
    """
    Calls OpenRouter's DEDICATED Image API (POST /api/v1/images).

    If input_image_base64 is provided, this becomes an IMAGE-TO-IMAGE EDIT
    request (e.g. "change the style of this image to watercolor") instead
    of pure text-to-image generation — the attached image is passed via
    OpenRouter's documented `input_references` field, which explicitly
    supports base64 data URLs (not just http(s) URLs). Without this field,
    an edit-style prompt has nothing to actually edit, which is why that
    case previously returned nothing.

    Returns {"text": "", "images": [data_uri_or_url, ...]}.
    Raises requests.HTTPError on failure — main.py decides how to surface it.
    """
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "Adaptive Agent Router",
    }
    payload = {"model": model, "prompt": prompt}

    if input_image_base64:
        mime = input_image_mime_type or "image/png"
        payload["input_references"] = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{input_image_base64}"},
            }
        ]

    resp = requests.post(OPENROUTER_IMAGES_URL, json=payload, headers=headers, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    images = []
    for item in data.get("data", []):
        if item.get("b64_json"):
            images.append(f"data:image/png;base64,{item['b64_json']}")
        elif item.get("url"):
            images.append(item["url"])

    if not images:
        warnings.warn(
            f"generate_image: model '{model}' returned no images "
            f"(edit_mode={bool(input_image_base64)}). Raw response for debugging: {data!r}"
        )

    return {"text": "", "images": images}
