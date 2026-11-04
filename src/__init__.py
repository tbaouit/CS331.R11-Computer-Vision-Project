"""CS331 Amodal Appearance Completion modular source package.

Subpackages:
- `src.data`       : COCO-A dataset loading and polygon/RLE mask decoding.
- `src.vision`     : Morphology, ROI geometry, frequency-lock & seam compositing.
- `src.stages`     : Shape consensus, appearance priors, VLM/CLIP guidance & SD2 refinement.
- `src.pipelines`  : High-level Baseline, Blind-Improved, and Oracle-Mask completers.
- `src.evaluation` : Sample runner, SHA-256 resume verification, metrics & run comparison.
"""
