"""Modular algorithmic stages for amodal shape consensus, appearance seed, VLM guidance, and SD2 refinement."""
from src.stages.appearance_prior import (
    build_appearance_prior,
    build_oracle_appearance_seed,
)
from src.stages.sd2_refiner import (
    refine_exact_mask_sd2_oracle,
    refine_roi_sd2_blind,
)
from src.stages.shape_consensus import (
    fuse_pix2gestalt_predictions,
    prepare_pix2gestalt_roi_inputs,
)
from src.stages.vlm_guidance import (
    DEFAULT_NEGATIVE_PROMPT,
    build_oracle_vlm_prompts,
    build_vlm_visual_prompts,
    parse_structured_vlm_reasoning,
    run_blind_vlm_reasoning,
    run_oracle_vlm_reasoning,
    run_target_aware_critique,
    truncate_tokens_clean,
    verify_prompt_with_clip,
)

__all__ = [
    "DEFAULT_NEGATIVE_PROMPT",
    "build_appearance_prior",
    "build_oracle_appearance_seed",
    "build_oracle_vlm_prompts",
    "build_vlm_visual_prompts",
    "fuse_pix2gestalt_predictions",
    "parse_structured_vlm_reasoning",
    "prepare_pix2gestalt_roi_inputs",
    "refine_exact_mask_sd2_oracle",
    "refine_roi_sd2_blind",
    "run_blind_vlm_reasoning",
    "run_oracle_vlm_reasoning",
    "run_target_aware_critique",
    "truncate_tokens_clean",
    "verify_prompt_with_clip",
]
