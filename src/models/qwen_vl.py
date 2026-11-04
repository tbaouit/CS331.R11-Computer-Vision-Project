"""Qwen3-VL vision-language model loader (used for semantic reasoning and target-aware critique)."""
from __future__ import annotations

DEFAULT_VLM_ID = "Qwen/Qwen3-VL-4B-Instruct"


class QwenVLModel:
    """Holds the Qwen3-VL `model`, its `processor`, and the target `device`."""

    def __init__(self, model_id: str = DEFAULT_VLM_ID, device: str = "cuda"):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        print(f"[Models] Loading {model_id} (FP16)...", flush=True)
        self.device = device
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map=None,
        )
        self.model.to(device).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)
        print(f"[Models] {model_id} ready.", flush=True)


    def to(self, device):
        self.model.to(device)
        self.device = str(device)
        return self
