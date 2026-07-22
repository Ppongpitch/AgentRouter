"""
agent_executor.py
──────────────────
Once router.py decides WHICH agent/model should handle a query, this
module actually calls that model via OpenRouter and returns its answer.
No print() — pure functions, returns structured data.

Handles two things that plain text-only chat completion doesn't:
  1. IMAGE OUTPUT (Image Generation Agent) — OpenRouter image-generation
     models don't return an image as a normal string; the generated
     image(s) come back either as multimodal content blocks in the
     response message, or via a provider-specific "images" field in
     additional_kwargs. Previously this code only ever asked for a plain
     string via StrOutputParser, which silently discarded any image data
     the model actually returned — the "endpoint doesn't give back an
     image" symptom was really "we were throwing the image away before
     it ever reached the response." _extract_text_and_images() below
     checks every place OpenRouter is known to put image data.
  2. IMAGE INPUT (Vision/OCR Agent) — needs the actual uploaded image
     sent as part of the message, not just the text query. When
     image_base64 is provided, messages are built manually as a
     multimodal HumanMessage (text + image_url content blocks) instead
     of going through the plain ChatPromptTemplate.
"""

import warnings

import requests
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

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

    # OpenRouter-specific extension: some image-generation models return
    # generated images via message.images (outside the standard OpenAI
    # schema), which LangChain surfaces under additional_kwargs.
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


def generate_image(prompt: str, model: str) -> dict:
    """
    Calls OpenRouter's DEDICATED Image API (POST /api/v1/images) — a
    separate endpoint from chat completions, purpose-built for image
    generation with a guaranteed response shape. This is OpenRouter's
    own recommended path for image-generation models.

    We deliberately do NOT route image generation through run_agent()'s
    chat-completions + LangChain path: that requires a "modalities":
    ["image", "text"] request field we weren't setting, AND LangChain's
    OpenAI response parser may silently drop the non-standard "images"
    field OpenRouter puts image data in, since it's outside the normal
    chat-completion schema LangChain expects. This function sidesteps
    both problems by calling the Image API directly with `requests`.

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
            f"generate_image: model '{model}' returned no images. "
            f"Raw response for debugging: {data!r}"
        )

    return {"text": "", "images": images}


def run_agent(
    query: str,
    model: str,
    description: str = "",
    image_base64: str | None = None,
    image_mime_type: str | None = None,
) -> dict:
    """
    Call the routed agent's model with the user's query and return its
    answer. Returns {"text": str, "images": list[str]} — "images" is a
    list of data:/http(s) URLs the frontend can drop straight into <img>
    tags; it's empty for ordinary text-only responses.

    If image_base64 is provided (Vision/OCR Agent use case), the query is
    sent alongside the actual image as a multimodal message instead of
    plain text — required for the model to actually be able to look at it.

    Raises whatever the underlying client raises on failure — main.py
    decides how to surface errors to the caller.
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
        # Nothing found in any of the shapes we know to check — dump the
        # raw response so the actual structure can be inspected instead of
        # guessed at. warnings.warn (not print) keeps this a library module
        # per the no-print rule; it still shows up in the terminal running main.py.
        warnings.warn(
            f"run_agent: model '{model}' returned neither text nor images. "
            f"Raw response for debugging — content={response.content!r} "
            f"additional_kwargs={getattr(response, 'additional_kwargs', None)!r} "
            f"response_metadata={getattr(response, 'response_metadata', None)!r}"
        )

    return {"text": text, "images": images}
