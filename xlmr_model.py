"""
xlmr_model.py
─────────────
Plain nn.Module version of AgentRouterModel — no PyTorch Lightning
dependency at inference time.

A Lightning .ckpt is just a torch.save()'d dict with a "state_dict" key
and a "hyper_parameters" key (from save_hyperparameters() at training
time). We can load both directly with torch.load() and skip installing
pytorch-lightning entirely in the serving image — it's several hundred MB
of dependencies (torchmetrics, fsspec, etc.) we never touch at inference.
"""

import warnings

import torch
import torch.nn as nn
from transformers import AutoModel

from taxonomy import AGENT2MODEL, ID2AGENT


class AgentRouterModel(nn.Module):
    def __init__(self, model_name, num_agents, top_k=3):
        super().__init__()
        self.top_k = top_k

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.gating_net = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_agents),
        )

    def forward(self, input_ids, attention_mask):
        cls = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state[:, 0, :]

        logits = self.gating_net(cls)
        probs = torch.softmax(logits, dim=1)
        topk_probs, topk_indices = torch.topk(probs, k=self.top_k, dim=1)

        return logits, probs, topk_probs, topk_indices

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str, map_location="cpu"):
        """Reads a PyTorch Lightning .ckpt without needing pytorch-lightning
        installed. The file is a plain torch.save()'d dict with
        'state_dict' and 'hyper_parameters' keys."""
        ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        hparams = ckpt["hyper_parameters"]

        model = cls(
            model_name=hparams["model_name"],
            num_agents=hparams["num_agents"],
            top_k=hparams.get("top_k", 3),
        )

        # strict=False: this architecture only defines `encoder` and
        # `gating_net`. Training-time-only parameters/buffers your
        # LightningModule may have had (e.g. a per-class loss-weighting
        # tensor like "agent_weights") don't participate in forward() and
        # aren't part of this inference-only class, so they're expected to
        # show up as "unexpected" here — that's normal, not a sign of a
        # broken checkpoint. Anything under `missing_keys` (encoder/gating_net
        # weights NOT found in the checkpoint) is the real red flag to check.
        incompatible = model.load_state_dict(ckpt["state_dict"], strict=False)
        if incompatible.missing_keys:
            warnings.warn(
                f"AgentRouterModel.load_from_checkpoint: MISSING keys — these "
                f"belong to the architecture but were NOT found in the checkpoint "
                f"(the model is running with random/uninitialized weights for "
                f"these parts): {incompatible.missing_keys}"
            )
        if incompatible.unexpected_keys:
            warnings.warn(
                f"AgentRouterModel.load_from_checkpoint: UNEXPECTED keys — "
                f"present in the checkpoint but not in this architecture, "
                f"safely ignored (usually training-only tensors, e.g. loss "
                f"weighting): {incompatible.unexpected_keys}"
            )
        return model

    @torch.no_grad()
    def predict_agent(self, prompt: str, tokenizer, top_k: int = None):
        """Single-prompt inference. Returns list of {agent, model, confidence}.
        Standalone/debug use only — resolves model via taxonomy.AGENT2MODEL.
        The actual routing pipeline (xlmr_router.py) does NOT call this;
        it uses AgentStore/capability_agents.json as the single source of
        truth for agent -> model mapping instead."""
        k = top_k or self.top_k
        device = next(self.parameters()).device
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True,
            max_length=128, padding=True
        ).to(device)

        _, probs, topk_probs, topk_indices = self(
            inputs["input_ids"], inputs["attention_mask"]
        )

        results = []
        for prob, idx in zip(topk_probs[0][:k], topk_indices[0][:k]):
            agent = ID2AGENT[idx.item()]
            results.append({
                "agent": agent,
                "model": AGENT2MODEL[agent],
                "confidence": round(prob.item(), 4),
            })
        return results
