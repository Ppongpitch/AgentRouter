"""
xlmr_router.py
───────────────
XLM-R classifier routing: a locally-run fine-tuned XLM-RoBERTa model
predicts the target agent directly from the prompt — no embeddings, no
cosine similarity, no seed bucket involved. This is Tier 3 in this
project's routing scheme (Tier 1 = semantic, Tier 2 = LLM fallback).

Unlike Tier 1/2, this mode always trusts its own top-1 prediction — no
confidence threshold, no fallback to another tier. If you want that
behavior later, add a threshold check in xlmr_route() and fall back to
llm_route() below it, same pattern as semantic_route() does.

No print() anywhere — pure library code; main.py decides what to log.
"""

import torch
from transformers import AutoTokenizer

from agent_store import AgentStore
from config import JSON_TO_CAPABILITY, XLMR_CHECKPOINT_PATH, XLMR_TOKENIZER_NAME, XLMR_TOP_K
from taxonomy import ID2AGENT
from xlmr_model import AgentRouterModel


class XLMRClassifier:
    """Holds the loaded XLM-R model + tokenizer. Build once at startup."""

    def __init__(
        self,
        checkpoint_path: str = XLMR_CHECKPOINT_PATH,
        tokenizer_name: str = XLMR_TOKENIZER_NAME,
    ):
        self.device = torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = AgentRouterModel.load_from_checkpoint(checkpoint_path, map_location="cpu")
        self.model.to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def predict(self, prompt: str, top_k: int = None) -> list[dict]:
        """Returns top_k [{agent_code, confidence}], highest confidence first."""
        k = top_k or self.model.top_k or XLMR_TOP_K
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=128, padding=True
        ).to(self.device)

        _, probs, topk_probs, topk_indices = self.model(
            inputs["input_ids"], inputs["attention_mask"]
        )

        results = []
        for prob, idx in zip(topk_probs[0][:k], topk_indices[0][:k]):
            results.append(
                {
                    "agent_code": ID2AGENT[idx.item()],
                    "confidence": round(prob.item(), 4),
                }
            )
        return results


def xlmr_route(
    query: str, classifier: XLMRClassifier, store: AgentStore
) -> tuple[dict, list[dict]]:
    """
    Classify the query with the XLM-R model, resolve the predicted agent
    code to a capability + actual model via the SAME AgentStore every
    other routing mode uses — single source of truth for agent -> model,
    so capability_agents.json stays the only place you edit that mapping.

    Returns (result, scores) matching semantic_route()'s shape so router.py
    can treat all routing modes uniformly.
    """
    predictions = classifier.predict(query)
    top1 = predictions[0]

    capability = JSON_TO_CAPABILITY.get(top1["agent_code"])
    agent = store.get_agent(capability) if capability else None

    if agent is None:
        # Unknown/unmapped agent code (e.g. taxonomy.py out of sync with
        # capability_agents.json) — fall back to the first agent rather
        # than crashing the request.
        agent = store.agents[0]
        capability = agent["capability"]

    result = {
        "capability": capability,
        "model": agent["model"],
        "confident": top1["confidence"],
        "router_used": "xlmr_classifier",
    }

    # Reuses the "sim" key name so the frontend's existing scores display
    # (built for cosine-similarity scores) works unmodified for this mode too.
    scores = [
        {
            "capability": JSON_TO_CAPABILITY.get(p["agent_code"], p["agent_code"]),
            "sim": p["confidence"],
        }
        for p in predictions
    ]

    return result, scores
