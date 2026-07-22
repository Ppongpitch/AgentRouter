"""
router.py
─────────
adaptive_route(): the single entry point that combines FOUR routing modes,
an OCR pre-processing step for attached images, and actually calling the
winning agent's model to get a real answer. No print() anywhere — this is
a library function, not a program. main.py decides what to log.

Modes / tiers:
  per_seed / single_centroid / multi_centroid  -> semantic routing (Tier 1),
                                                    falls back to LLM routing
                                                    (Tier 2) if not confident
  xlmr_classifier                                -> XLM-R local model routing
                                                    (Tier 3), no fallback

Image handling (when image_base64 is provided):
  1. ALWAYS call the Vision/OCR Agent first to extract text from the image.
  2. Route the user's TEXT query alone (image ignored for this decision)
     through the normal tier system above, to find out what the user
     actually wants done.
  3. If that routing lands on Vision/OCR Agent itself, the user's prompt
     WAS an OCR-type request — the OCR call from step 1 already IS the
     answer, so it's reused directly (no second model call).
  4. Otherwise, the OCR'd text is appended to the user's original prompt,
     and that combined text (no image — OCR already extracted what was
     needed) is sent to whichever agent step 2 picked.

Seed growth policy: seeds are appended to the winning agent's bucket
ONLY when the LLM router (Tier 2) handled the TEXT routing decision.
Semantic hits (Tier 1) and XLM-R hits (Tier 3) never grow the seed bucket.
The one-time OCR call itself never grows seeds either.

Timing: routing_time_ms covers only the text-routing decision. The OCR
call's time (when an image is attached) is folded into generation_time_ms,
since it's a real model call, not a routing decision.
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
    1. If an image is attached, OCR it first (always).
    2. Route the TEXT query (image ignored) using the selected mode:
       - per_seed / single_centroid / multi_centroid -> semantic_route()
         (Tier 1), falling back to llm_route() (Tier 2) if not confident.
       - xlmr_classifier -> xlmr_route() (Tier 3), always trusts top-1.
    3. Append the query to the winning agent's seed bucket ONLY if Tier 2
       (LLM router) handled it.
    4. Get the final answer:
       - Image Generation Agent -> generate_image() (dedicated Image API)
       - Image was attached AND routed to Vision/OCR Agent -> reuse the
         OCR result from step 1 directly, no second call
       - Image was attached AND routed elsewhere -> combine OCR text +
         original prompt, send that combined text to the picked agent
       - No image -> run_agent() as normal
    5. Return a JSON-serializable routing + answer + timing result.
    """
    ocr_text = None
    ocr_answer = None
    ocr_generation_ms = 0.0

    if image_base64:
        vision_agent = store.get_agent(OCR_CAPABILITY)
        if vision_agent is None:
            raise RuntimeError(
                f"An image was attached but no '{OCR_CAPABILITY}' exists in "
                f"capability_agents.json — check the capability name matches exactly."
            )
        ocr_call_start = time.perf_counter()
        ocr_answer = run_agent(
            query, vision_agent["model"], vision_agent["description"], image_base64, image_mime_type
        )
        ocr_generation_ms = round((time.perf_counter() - ocr_call_start) * 1000, 1)
        ocr_text = ocr_answer["text"]

    # ── Route the TEXT query alone — the image's content is already
    # captured in ocr_text above (if any), so routing doesn't need it.
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

    # ── Get the final answer ────────────────────────────────────────────
    generation_start = time.perf_counter()

    if image_base64 and result["capability"] == OCR_CAPABILITY:
        # The user's prompt itself was an OCR-type request — the OCR call
        # made above already IS the answer. No second model call needed.
        answer_data = ocr_answer
        result.setdefault("intent", "OCR request — returning extracted text directly")
    elif image_base64:
        # Not an OCR-type request — combine the OCR'd text with the user's
        # original prompt, then send that combined TEXT (no image needed
        # anymore) to whichever agent the routing above picked.
        combined_query = f"{query}\n\n[Text extracted from the attached image via OCR]:\n{ocr_text}"
        answer_data = run_agent(combined_query, result["model"], description)
    elif result["capability"] == "Image Generation Agent":
        # Dedicated Image API — chat completions + LangChain wasn't
        # reliably surfacing image output (see generate_image()'s docstring).
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
