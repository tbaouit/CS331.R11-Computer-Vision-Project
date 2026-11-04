"""CS331 Open-World Amodal Appearance Completion modular source package.

Subpackages:
- `src.data`       : COCO-A dataset loading and polygon/RLE mask decoding.
- `src.models`     : Foundation-model loaders (Pix2Gestalt+CarveKit, Qwen3-VL, SD2 Inpainting, CLIP).
- `src.vision`     : Morphology, ROI geometry, frequency-lock & seam compositing.
- `src.stages`     : Shape consensus, appearance priors, VLM/CLIP guidance & SD2 refinement.
- `src.pipelines`  : High-level Blind-Improved and Oracle-Mask completers & dispatcher.
- `src.evaluation` : Sample runner, SHA-256 resume verification, metrics & run comparison.
"""
