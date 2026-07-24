"""
router.py
─────────
adaptive_route(): the single entry point that combines FOUR routing modes
and actually calling the winning agent's model to get a real answer.
No print() anywhere — this is a library function, not a program.
main.py decides what to log.

Modes / tiers:
  per_seed / single_centroid / multi_centroid  -> semantic routing (Tier 1),
                                                    falls back to LLM routing
                                                    (Tier 2) if not confident
  xlmr_classifier                                -> XLM-R local model routing
                                                    (Tier 3), no fallback

Routing ALWAYS happens on the text query alone — an attached image never
affects which agent gets picked, only what happens once an agent is
already chosen. This also means routing happens BEFORE any OCR call, so
a prompt that routes to Image Generation Agent never pays for an
unnecessary OCR call it was never going to use.

Image handling, once the target agent is known:
  - Image Generation Agent + image attached  -> generate_image() as an
    IMAGE-TO-IMAGE EDIT (the attached image is passed via OpenRouter's
    `input_references` field). Without this, an edit-style prompt like
    "change the style of this image to watercolor" has nothing to
    actually edit, which is why that case previously returned nothing.
  - Image Generation Agent + no image        -> generate_image() as
    ordinary text-to-image.
  - Vision/OCR Agent + image attached         -> run_agent() with the
    image, once. That IS the answer directly — no second call needed.
  - Anything else + image attached            -> OCR the image first,
    append the extracted text to the original prompt, send that combined
    TEXT (no image needed anymore) to whichever agent routing picked.
  - Anything else + no image                  -> run_agent() as normal.

Seed growth policy: seeds are appended to the winning agent's bucket
ONLY when the LLM router (Tier 2) handled the routing decision. Semantic
hits (Tier 1) and XLM-R hits (Tier 3) never grow the seed bucket.

Timing: routing_time_ms covers only the text-routing decision. Any OCR
call's time (when one happens) is folded into generation_time_ms, since
it's a real model call, not a routing decision.
"""

import time

from .agent_executor import generate_image, run_agent
from .agent_store import AgentStore
from .config import DEFAULT_ROUTING_MODE
from .llm_router import llm_route
from .semantic_router import semantic_route
from .xlmr_router import XLMRClassifier, xlmr_route

TIER_LABELS = {
    1: "Tier 1 — Semantic",
    2: "Tier 2 — LLM Route",
    3: "Tier 3 — XLM-R Classifier",
}

OCR_CAPABILITY = "Vision / OCR Agent"
IMAGE_GEN_CAPABILITY = "Image Generation Agent"


def adaptive_route(
    query: str,
    embedder,
    store: AgentStore,
    llm_client,
    mode: str = DEFAULT_ROUTING_MODE,
    xlmr_classifier: XLMRClassifier | None = None,
    image_base64: str | None = None,
    image_mime_type: str | None = None,
) -> dict:
    """
    1. Route the TEXT query (image ignored) using the selected mode:
       - per_seed / single_centroid / multi_centroid -> semantic_route()
         (Tier 1), falling back to llm_route() (Tier 2) if not confident.
       - xlmr_classifier -> xlmr_route() (Tier 3), always trusts top-1.
    2. Append the query to the winning agent's seed bucket ONLY if Tier 2
       (LLM router) handled it.
    3. Get the final answer, dispatched on the winning capability AND
       whether an image is attached — see the module docstring above for
       the full dispatch table.
    4. Return a JSON-serializable routing + answer + timing result.
    """
    routing_start = time.perf_counter()

    if mode == "xlmr_classifier":
        if xlmr_classifier is None:
            raise RuntimeError(
                "mode='xlmr_classifier' but no XLMRClassifier was provided. "
                "Check that XLMR_CHECKPOINT_PATH points to a real checkpoint "
                "and that main.py loaded it successfully at startup."
            )
        result, sem_scores = xlmr_route(query, xlmr_classifier, store)
        result.setdefault("intent", f"Route to {result['capability']}")
        tier = 3
    else:
        sem_result, sem_scores = semantic_route(query, embedder, store, mode=mode)
        if sem_result is not None:
            result = sem_result
            result.setdefault("intent", f"Route to {result['capability']}")
            tier = 1
        else:
            result = llm_route(query, store, llm_client)
            tier = 2

    routing_time_ms = round((time.perf_counter() - routing_start) * 1000, 1)

    seed_appended = False
    if result["router_used"] == "llm":
        seed_appended = store.update_agent_seeds(result["capability"], query)

    agent = store.get_agent(result["capability"])
    description = agent["description"] if agent else ""

    # ── Get the final answer — dispatch on capability + image presence ──
    generation_start = time.perf_counter()
    ocr_text = None

    if result["capability"] == IMAGE_GEN_CAPABILITY:
        # Text-to-image (no attachment) OR image-to-image edit (attachment
        # present) — both go through the dedicated Image API. The image,
        # if any, rides along via input_references so an edit prompt has
        # something to actually edit.
        answer_data = generate_image(
            query, result["model"],
            input_image_base64=image_base64,
            input_image_mime_type=image_mime_type,
        )
    elif image_base64 and result["capability"] == OCR_CAPABILITY:
        # The user's prompt itself was an OCR-type request — one call,
        # its result IS the answer.
        answer_data = run_agent(query, result["model"], description, image_base64, image_mime_type)
        ocr_text = answer_data["text"]
    elif image_base64:
        # Not OCR, not image-gen — OCR the image first, then combine the
        # extracted text with the user's original prompt, then send that
        # combined TEXT (no image needed anymore) to the picked agent.
        vision_agent = store.get_agent(OCR_CAPABILITY)
        if vision_agent is None:
            raise RuntimeError(
                f"An image was attached but no '{OCR_CAPABILITY}' exists in "
                f"capability_agents.json — check the capability name matches exactly."
            )
        ocr_answer = run_agent(
            query, vision_agent["model"], vision_agent["description"], image_base64, image_mime_type
        )
        ocr_text = ocr_answer["text"]
        combined_query = f"{query}\n\n[Text extracted from the attached image via OCR]:\n{ocr_text}"
        answer_data = run_agent(combined_query, result["model"], description)
    else:
        answer_data = run_agent(query, result["model"], description)

    generation_time_ms = round((time.perf_counter() - generation_start) * 1000, 1)

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
