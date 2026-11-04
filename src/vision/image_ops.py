"""Low-level image morphology, resizing, hole filling, and filtering primitives."""
from __future__ import annotations

from collections import deque
import numpy as np
from PIL import Image, ImageFilter

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - local CPU data venv fallback
    cv2 = None

try:
    from scipy import ndimage as ndi  # type: ignore
except ImportError:  # pragma: no cover - local CPU data venv fallback
    ndi = None


def resize_array(array: np.ndarray, size_hw: tuple[int, int], mode: str = "linear") -> np.ndarray:
    """Resize a 2D or 3D uint8/float32 array to `(height, width)`."""
    target_h, target_w = int(size_hw[0]), int(size_hw[1])
    if array.shape[:2] == (target_h, target_w):
        return array.copy()

    if cv2 is not None:
        interp_map = {
            "nearest": cv2.INTER_NEAREST,
            "linear": cv2.INTER_LINEAR,
            "cubic": cv2.INTER_CUBIC,
            "area": cv2.INTER_AREA,
        }
        interp = interp_map.get(mode, cv2.INTER_LINEAR)
        return cv2.resize(array, (target_w, target_h), interpolation=interp)

    resample_map = {
        "nearest": Image.Resampling.NEAREST,
        "linear": Image.Resampling.BILINEAR,
        "cubic": Image.Resampling.BICUBIC,
        "area": Image.Resampling.BOX,
    }
    resample = resample_map.get(mode, Image.Resampling.BILINEAR)
    if array.dtype == np.uint8:
        return np.asarray(Image.fromarray(array).resize((target_w, target_h), resample=resample))

    arr_f = array.astype(np.float32)
    if arr_f.ndim == 2:
        pil_f = Image.fromarray(arr_f, mode="F").resize((target_w, target_h), resample=resample)
        return np.asarray(pil_f, dtype=np.float32)

    channels = [
        np.asarray(
            Image.fromarray(arr_f[:, :, c], mode="F").resize((target_w, target_h), resample=resample),
            dtype=np.float32,
        )
        for c in range(arr_f.shape[2])
    ]
    return np.stack(channels, axis=-1)


def binary_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    mask_bool = np.asarray(mask, dtype=bool)
    if radius <= 0 or not mask_bool.any():
        return mask_bool.copy()
    k = int(radius) * 2 + 1
    if cv2 is not None:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        return cv2.dilate(mask_bool.astype(np.uint8), kernel).astype(bool)
    img = Image.fromarray(mask_bool.astype(np.uint8) * 255, mode="L")
    return np.asarray(img.filter(ImageFilter.MaxFilter(k))) > 127


def binary_erode(mask: np.ndarray, radius: int) -> np.ndarray:
    mask_bool = np.asarray(mask, dtype=bool)
    if radius <= 0 or not mask_bool.any():
        return mask_bool.copy()
    r = int(radius)
    k = r * 2 + 1
    if cv2 is not None:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        return cv2.erode(mask_bool.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    padded = np.pad(mask_bool.astype(np.uint8) * 255, pad_width=r, mode="constant", constant_values=0)
    eroded = np.asarray(Image.fromarray(padded, mode="L").filter(ImageFilter.MinFilter(k)))
    return eroded[r:-r, r:-r] > 127


def binary_open(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return np.asarray(mask, dtype=bool).copy()
    return binary_dilate(binary_erode(mask, radius), radius)


def binary_close(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return np.asarray(mask, dtype=bool).copy()
    return binary_erode(binary_dilate(mask, radius), radius)


def mask_outline(mask: np.ndarray, thickness: int = 2) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    return binary_dilate(m, max(1, thickness)) & (~m)


def keep_connected_to_anchor(mask: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    """Keep 8-connected components of `mask` that intersect `anchor`."""
    mask_bool = np.asarray(mask, dtype=bool)
    seeds = mask_bool & np.asarray(anchor, dtype=bool)
    if not mask_bool.any() or not seeds.any():
        return np.zeros_like(mask_bool, dtype=bool)

    if cv2 is not None:
        _, labels = cv2.connectedComponents(mask_bool.astype(np.uint8), connectivity=8)
        valid_ids = np.unique(labels[seeds])
        valid_ids = valid_ids[valid_ids > 0]
        return np.isin(labels, valid_ids)

    if ndi is not None:
        labels, _ = ndi.label(mask_bool, structure=np.ones((3, 3), dtype=int))
        valid_ids = np.unique(labels[seeds])
        valid_ids = valid_ids[valid_ids > 0]
        return np.isin(labels, valid_ids)

    h, w = mask_bool.shape
    visited = seeds.copy()
    q = deque(map(tuple, np.argwhere(seeds)))
    while q:
        y, x = q.popleft()
        y0, y1 = max(0, y - 1), min(h, y + 2)
        x0, x1 = max(0, x - 1), min(w, x + 2)
        sub_mask = mask_bool[y0:y1, x0:x1] & (~visited[y0:y1, x0:x1])
        if sub_mask.any():
            for dy, dx in np.argwhere(sub_mask):
                ny, nx = y0 + int(dy), x0 + int(dx)
                visited[ny, nx] = True
                q.append((ny, nx))
    return visited


def fill_small_internal_holes(mask: np.ndarray, max_hole_pixels: int = 36) -> np.ndarray:
    """Fill tiny enclosed pinholes inside `mask` without altering real openings."""
    mask_bool = np.asarray(mask, dtype=bool)
    inv = ~mask_bool
    if not inv.any():
        return mask_bool.copy()

    h, w = mask_bool.shape
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    exterior_bg = keep_connected_to_anchor(inv, border)
    enclosed_holes = inv & (~exterior_bg)
    if not enclosed_holes.any():
        return mask_bool.copy()

    if cv2 is not None:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(enclosed_holes.astype(np.uint8), connectivity=8)
        small_ids = [idx for idx in range(1, num_labels) if stats[idx, cv2.CC_STAT_AREA] <= max_hole_pixels]
        if not small_ids:
            return mask_bool.copy()
        return mask_bool | np.isin(labels, small_ids)

    if enclosed_holes.sum() <= max_hole_pixels:
        return mask_bool | enclosed_holes
    return mask_bool.copy()


def nearest_foreground_fill(rgb: np.ndarray, known_mask: np.ndarray) -> np.ndarray:
    """Extrapolate `rgb` from `known_mask` to `~known_mask` via nearest pixel."""
    known = np.asarray(known_mask, dtype=bool)
    out = rgb.copy()
    if not known.any() or known.all():
        return out

    if ndi is not None:
        _, indices = ndi.distance_transform_edt(~known, return_indices=True)
        out[~known] = rgb[indices[0][~known], indices[1][~known]]
        return out

    filled = known.copy()
    for _ in range(max(rgb.shape[:2])):
        if filled.all():
            break
        prev = filled.copy()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            shifted_mask = np.zeros_like(filled)
            shifted_rgb = np.zeros_like(out)
            sy0, sy1 = max(0, -dy), filled.shape[0] - max(0, dy)
            dy0, dy1 = max(0, dy), filled.shape[0] - max(0, -dy)
            sx0, sx1 = max(0, -dx), filled.shape[1] - max(0, dx)
            dx0, dx1 = max(0, dx), filled.shape[1] - max(0, -dx)
            shifted_mask[dy0:dy1, dx0:dx1] = prev[sy0:sy1, sx0:sx1]
            shifted_rgb[dy0:dy1, dx0:dx1] = out[sy0:sy1, sx0:sx1]
            newly = shifted_mask & (~filled)
            if newly.any():
                out[newly] = shifted_rgb[newly]
                filled[newly] = True
    return out


def inpaint_holes(rgb: np.ndarray, hole_mask: np.ndarray, radius: int = 7) -> np.ndarray:
    """Inpaint `hole_mask` smoothly from surrounding valid pixels."""
    holes = np.asarray(hole_mask, dtype=bool)
    if not holes.any():
        return rgb.copy()
    if cv2 is not None:
        return cv2.inpaint(rgb, holes.astype(np.uint8) * 255, float(radius), cv2.INPAINT_TELEA)
    filled = nearest_foreground_fill(rgb, ~holes)
    blurred = np.asarray(Image.fromarray(filled).filter(ImageFilter.GaussianBlur(radius=2)))
    out = rgb.copy()
    out[holes] = blurred[holes]
    return out


def gaussian_blur_rgb(rgb: np.ndarray, sigma: float = 4.0) -> np.ndarray:
    """Return float32 Gaussian-blurred RGB image."""
    if cv2 is not None:
        return cv2.GaussianBlur(rgb.astype(np.float32), (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    pil_blurred = Image.fromarray(rgb.astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=float(sigma)))
    return np.asarray(pil_blurred, dtype=np.float32)
