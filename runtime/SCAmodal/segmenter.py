"""
pipeline/segmenter.py
Wraps SAM3 (Segment Anything Model 3) for auto-segmentation.
Falls back to SAM2 if SAM3 is unavailable.
"""

import numpy as np
import torch
from typing import Optional


class SAMSegmenter:
    """
    Thin wrapper around SAM3 / SAM2 SamAutomaticMaskGenerator.
    Loads the model once and reuses it across requests.
    """

    def __init__(
        self,
        model_type: str = "sam3_large",  # or "vit_h", "vit_l", "vit_b"
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self._generator = None
        self._load_model(checkpoint_path)

    def _load_model(self, checkpoint_path: Optional[str]):
        try:
            # Try SAM3 first (newest)
            from sam3 import build_sam3, SamAutomaticMaskGenerator
            print(f"[SAMSegmenter] Loading SAM3 on {self.device}...")
            sam = build_sam3(checkpoint=checkpoint_path or "sam3_large.pt")
            sam.to(self.device)
            self._generator = SamAutomaticMaskGenerator(sam)
            print("[SAMSegmenter] SAM3 loaded.")

        except ImportError:
            try:
                # Fallback to SAM2
                from sam2.build_sam import build_sam2
                from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
                print(f"[SAMSegmenter] SAM3 unavailable, loading SAM2 on {self.device}...")
                sam2 = build_sam2(
                    config_file="configs/sam2.1/sam2.1_hiera_l.yaml",
                    ckpt_path=checkpoint_path or "sam2.1_hiera_large.pt",
                    device=self.device,
                )
                self._generator = SAM2AutomaticMaskGenerator(sam2)
                print("[SAMSegmenter] SAM2 loaded.")

            except ImportError:
                # Final fallback: original SAM
                from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
                print(f"[SAMSegmenter] SAM2 unavailable, loading SAM1 on {self.device}...")
                sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path or "sam_vit_h_4b8939.pth")
                sam.to(self.device)
                self._generator = SamAutomaticMaskGenerator(sam)
                print("[SAMSegmenter] SAM1 loaded.")

    def segment_everything(
        self,
        image: np.ndarray,
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        min_mask_region_area: int = 100,
    ) -> list[dict]:
        """
        Run auto-segmentation on the full image.

        Args:
            image: HxWx3 uint8 RGB numpy array
            points_per_side: grid density for prompt points
            pred_iou_thresh: IOU confidence threshold
            stability_score_thresh: mask stability threshold
            min_mask_region_area: discard tiny noise masks

        Returns:
            list of SAM mask dicts with keys:
              - segmentation: HxW bool array
              - area: int
              - bbox: [x, y, w, h]
              - predicted_iou: float
              - stability_score: float
              - point_coords: [[x, y]]
              - crop_box: [x, y, w, h]
        """
        # Reconfigure generator params per request
        self._generator.points_per_side = points_per_side
        self._generator.pred_iou_thresh = pred_iou_thresh
        self._generator.stability_score_thresh = stability_score_thresh
        self._generator.min_mask_region_area = min_mask_region_area

        with torch.inference_mode():
            masks = self._generator.generate(image)

        # Sort by area descending (largest = likely background first)
        masks = sorted(masks, key=lambda x: x["area"], reverse=True)
        return masks

    def segment_from_prompt(
        self,
        image: np.ndarray,
        point_coords: list[list[int]],
        point_labels: list[int],
        box: Optional[list[int]] = None,
    ) -> np.ndarray:
        """
        Segment a specific object given point/box prompts.
        Returns the best mask as HxW bool array.
        """
        from segment_anything import SamPredictor

        predictor = SamPredictor(self._generator.predictor.model)
        predictor.set_image(image)

        pt_array = np.array(point_coords)
        lbl_array = np.array(point_labels)
        box_array = np.array(box) if box else None

        masks, scores, _ = predictor.predict(
            point_coords=pt_array,
            point_labels=lbl_array,
            box=box_array,
            multimask_output=True,
        )
        # Return highest-confidence mask
        best_idx = scores.argmax()
        return masks[best_idx]
