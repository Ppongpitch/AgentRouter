"""
taxonomy.py
───────────
Maps the XLM-R classifier's output class indices to agent codes.

⚠️ CRITICAL — VERIFY THIS BEFORE TRUSTING ANY PREDICTIONS ⚠️
ID2AGENT's order must match EXACTLY how your model's labels were encoded
during training (e.g. sklearn LabelEncoder.classes_ order, or whatever
list you fed into your Lightning DataModule / label-to-index mapping).
If this order is wrong, the model will still run and return confident-
looking numbers — it will just be confidently pointing at the WRONG agent
for every class whose index doesn't match. There's no runtime error to
catch this; it silently mis-routes.

The order below is a placeholder: it matches the same agent-code order
used elsewhere in this project (config.JSON_TO_CAPABILITY / train.jsonl),
NOT necessarily your actual training order. Check your training script's
label encoder / class list and fix this list to match before relying on
xlmr_classifier mode.
"""

AGENT_CLASSES = [
    "general_assistant",
    "code_agent",
    "translation_agent",
    "research_agent",
    "reasoning_agent",
    "vision_agent",
    "image_generation_agent",
    "creative_writing_agent",
]

ID2AGENT = {i: agent for i, agent in enumerate(AGENT_CLASSES)}

# Only used by model.py's predict_agent() for standalone/debug use outside
# this project's routing pipeline. The actual routing pipeline (xlmr_router.py)
# does NOT use this — it resolves the model via AgentStore/capability_agents.json
# instead, so there's a single source of truth for agent -> model mapping.
AGENT2MODEL = {
    "general_assistant": "openai/gpt-4o-mini-search-preview",
    "code_agent": "anthropic/claude-opus-4.8-fast",
    "translation_agent": "google/gemini-3.5-flash",
    "research_agent": "openai/o4-mini-deep-research",
    "reasoning_agent": "deepseek/deepseek-r1",
    "vision_agent": "google/gemma-4-31b-it",
    "image_generation_agent": "google/gemini-3.1-flash-image",
    "creative_writing_agent": "anthropic/claude-sonnet-4",
}
