"""Non-invasive trace instrumentation hooks for the upstream SCAmodal baseline."""
from __future__ import annotations

import json
import numpy as np
from PIL import Image


def install_baseline_trace_hooks(backend) -> None:
    """Wrap upstream AmodalCompleter methods to persist intermediate artifacts per sample."""
    vlm = backend.cls._vlm
    reason = vlm.reason_occlusion

    def traced_reason(image, visible_mask=None):
        text = reason(image, visible_mask)
        (backend.job / "reasoning.txt").write_text(text, encoding="utf-8")
        return text

    vlm.reason_occlusion = traced_reason

    critique = vlm.critique

    def traced_critique(completed_image_np, original_image_np=None):
        Image.fromarray(completed_image_np).save(backend.job / f"critic_input_{backend.attempt:02d}.png")
        result = critique(completed_image_np, original_image_np)
        backend.critiques.append(result)
        (backend.job / "critique_history.json").write_text(json.dumps(backend.critiques, indent=2), encoding="utf-8")
        return result

    vlm.critique = traced_critique

    select_prompt = backend.model._select_prompt

    def traced_select(*args, **kwargs):
        prompt, verification = select_prompt(*args, **kwargs)
        (backend.job / "prompt_selection.json").write_text(
            json.dumps(dict(prompt=prompt, clip_verification=verification), indent=2), encoding="utf-8"
        )
        return prompt, verification

    backend.model._select_prompt = traced_select

    detect = backend.model._detect_outpainting

    def traced_detect(*args, **kwargs):
        backend.pad_info = {
            key: (value.item() if isinstance(value, np.generic) else value)
            for key, value in detect(*args, **kwargs).items()
        }
        (backend.job / "outpainting.json").write_text(json.dumps(backend.pad_info, indent=2), encoding="utf-8")
        return backend.pad_info

    backend.model._detect_outpainting = traced_detect

    inpaint = backend.model._inpaint_step

    def traced_inpaint(*args, **kwargs):
        backend.attempt += 1
        prompt = args[2] if len(args) > 2 else kwargs["prompt"]
        (backend.job / f"prompt_{backend.attempt:02d}.txt").write_text(prompt, encoding="utf-8")
        result = inpaint(*args, **kwargs)
        Image.fromarray(result[0]).save(backend.job / f"inpainting_{backend.attempt:02d}.png")
        return result

    backend.model._inpaint_step = traced_inpaint

    shape = backend.cls._shape_predictor.predict_full_shape

    def traced_shape(image, visible):
        result = shape(image, visible)
        Image.fromarray(result.astype(np.uint8) * 255).save(backend.job / "shape_mask_original.png")
        return result

    backend.cls._shape_predictor.predict_full_shape = traced_shape
