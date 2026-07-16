# RunPod's official PyTorch image — PyTorch 2.4.0, Python 3.11, CUDA 12.4,
# pre-validated for all RunPod GPU workers (A4000, A4500, RTX 4000, RTX 2000, etc.)
# Full tag list: https://hub.docker.com/r/runpod/pytorch/tags
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    EMBED_MODEL=BAAI/bge-m3 \
    LLM_ROUTER_MODEL=google/gemini-2.5-flash \
    DEFAULT_ROUTING_MODE=single_centroid

WORKDIR /app

# torch & torchvision are already in the base image.
# Only install what's missing for our workload.
RUN pip install --no-cache-dir \
    transformers==4.44.2 \
    sentence-transformers==3.0.1 \
    sentencepiece \
    scikit-learn \
    numpy \
    langchain-core \
    langchain-openai \
    requests \
    runpod \
    huggingface_hub

# Pre-download the embedding model at build time → zero cold-start model
# download. model_kwargs={"use_safetensors": True} avoids transformers'
# torch.load safety check on the .bin checkpoint (requires torch>=2.6
# otherwise) — safetensors loading sidesteps it entirely regardless of
# torch version.
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('BAAI/bge-m3', model_kwargs={'use_safetensors': False})"

# Pre-download the XLM-R tokenizer too (small, and independent of whether
# you actually have a trained checkpoint yet — see the best.ckpt download below).
RUN python -c "from transformers import AutoTokenizer; \
    AutoTokenizer.from_pretrained('xlm-roberta-base')"

# Ensure curl is available (usually already present on devel images, but
# don't assume) — used below to download the checkpoint from Hugging Face.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY handler.py /app/handler.py
COPY capability_agents.json /app/capability_agents.json

# Optional: your seed dataset for bootstrapping the agents' seed buckets.
# Comment out if you don't have one — handler.py falls back to the inline
# seeds in capability_agents.json when train.jsonl is missing.
COPY train.jsonl /app/train.jsonl

# XLM-R classifier checkpoint (Tier 3 routing mode) — downloaded from
# Hugging Face at BUILD time instead of copied from the repo, since it's
# 3.33GB and can't be committed to GitHub (well over the 100MB hard limit,
# and would blow through Git LFS's free bandwidth/storage quota fast too).
# Saved at repo root (same level as handler.py) — matches
# XLMR_CHECKPOINT_PATH's default of "best.ckpt" in handler.py.
#
# If you don't have a checkpoint yet, comment out this line — the build
# still succeeds, and handler.py just skips loading it at cold start
# ('xlmr_classifier' mode will return a clean error if selected).
RUN python -c "from huggingface_hub import hf_hub_download; \
hf_hub_download(repo_id='tisismark/agent_router_plv3', filename='best.ckpt', local_dir='/app')"
RUN ls -lh /app/best.ckpt

CMD ["python", "/app/handler.py"]
