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
    huggingface_hub \
    psycopg2-binary \
    python-dotenv

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

# Copy the WHOLE repo in one shot instead of naming files one by one — this
# is what actually fixes the recurring "ModuleNotFoundError" pattern: any
# .py file your code imports from (router.py, agent_store.py, taxonomy.py,
# xlmr_model.py, semantic_router.py, agent_executor.py, embedding_cache.py,
# etc.) gets included automatically, even ones added later, instead of
# needing a matching COPY line added by hand every time.
# .dockerignore (same folder) excludes .git, __pycache__, the old
# embedding_cache.pkl, etc. so this doesn't bloat the image.
COPY . /app

# XLM-R classifier checkpoint (Tier 3 routing mode) — downloaded from
# Hugging Face at BUILD time instead of copied from the repo, since it's
# 3.33GB and can't be committed to GitHub (well over the 100MB hard limit,
# and would blow through Git LFS's free bandwidth/storage quota fast too).
# Saved at model/checkpoints/best.ckpt to match config.py's
# XLMR_CHECKPOINT_PATH default exactly ("model/checkpoints/best.ckpt") —
# this was previously mismatched (saved to /app/best.ckpt instead), which
# would have made 'xlmr_classifier' mode silently unavailable.
#
# If you don't have a checkpoint yet, comment out this line — the build
# still succeeds, and handler.py just skips loading it at cold start
# ('xlmr_classifier' mode will return a clean error if selected).
RUN mkdir -p /app/model/checkpoints \
    && python -c "from huggingface_hub import hf_hub_download; \
hf_hub_download(repo_id='tisismark/agent_router_plv3', filename='best.ckpt', local_dir='/app/model/checkpoints')"
RUN ls -lh /app/model/checkpoints/best.ckpt

# Seed embeddings are no longer baked into the image at build time — they
# now live in Postgres (see db.py), shared across every worker/region/
# rebuild instead of frozen into one image. Postgres isn't reachable at
# `docker build` time anyway (external managed DB, no build-time secrets),
# so this step is deliberately gone. The FIRST real cold start after a
# fresh Postgres bootstraps every seed's embedding once; every cold start
# after that — anywhere — just reads them back out. See handler.py's
# "COLD-START INIT" section for the load-from-DB / bootstrap-if-missing logic.
#
# IMPORTANT: set POSTGRES_DSN as a RunPod endpoint environment variable
# (RunPod console -> your endpoint -> Environment Variables), NOT baked in
# here — it must point at a Postgres reachable from RunPod's network (a
# managed host like Neon/Supabase/RDS/etc., not "localhost"), e.g.:
#   postgresql://user:password@your-host.example.com:5432/agentrouter?sslmode=require

CMD ["python", "/app/handler.py"]
