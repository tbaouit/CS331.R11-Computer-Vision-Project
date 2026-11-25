"""Attach the supplied amodal mask to decoded RGB."""
import numpy as np


def finalize_rgba(rgb: np.ndarray, amodal_mask: np.ndarray, h: int, w: int) -> np.ndarray:
    """Pack the completed RGB and the amodal mask (as alpha 0/255) into an HxWx4 uint8 image."""
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = rgb
    rgba[:, :, 3] = (amodal_mask * 255).astype(np.uint8)
    return rgba
