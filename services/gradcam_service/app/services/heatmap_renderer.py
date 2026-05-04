"""
Renders an EigenCAM activation map as a coloured heatmap overlaid on the
original image. Saves the result as a PNG and returns the file path.

Colormap: jet (blue=low → green → red=high) — standard in medical imaging CAM papers.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
import structlog
from PIL import Image

from app.core.config import settings

logger = structlog.get_logger()

# Jet colormap LUT: 256 RGB triplets (uint8)
_JET_LUT: np.ndarray = _build_jet_lut()  # type: ignore[assignment]


def _build_jet_lut() -> np.ndarray:
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        r = np.clip(1.5 - abs(4 * t - 3), 0, 1)
        g = np.clip(1.5 - abs(4 * t - 2), 0, 1)
        b = np.clip(1.5 - abs(4 * t - 1), 0, 1)
        lut[i] = [int(r * 255), int(g * 255), int(b * 255)]
    return lut


_JET_LUT = _build_jet_lut()


def _load_original(image_path: str, target_size: tuple[int, int]) -> Image.Image:
    path = Path(image_path)
    data = path.read_bytes()

    if path.suffix.lower() in {".dcm", ".dicom"}:
        import pydicom

        ds = pydicom.dcmread(io.BytesIO(data))
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        arr = arr * slope + intercept
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim == 3 and arr.shape[0] in {1, 3}:
            arr = arr.transpose(1, 2, 0)
            if arr.shape[2] == 1:
                arr = np.concatenate([arr, arr, arr], axis=2)
        img = Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8)).convert("RGB")
    else:
        img = Image.open(io.BytesIO(data)).convert("RGB")

    return img.resize(target_size, Image.BILINEAR)


def render_and_save(
    cam: np.ndarray,
    image_path: str,
    model_name: str,
    target_class: int | None,
) -> str:
    """
    Overlay the CAM on the original image and save as PNG.

    Parameters
    ----------
    cam : [H, W] float32 in [0, 1]
    image_path : source image (DICOM or raster)
    model_name : used for the output filename
    target_class : used for the output filename

    Returns
    -------
    Absolute path to the saved PNG.
    """
    out_dir = Path(settings.heatmap_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Unique filename based on input hash to avoid collisions
    key = f"{model_name}:{image_path}:{target_class}"
    digest = hashlib.sha1(key.encode()).hexdigest()[:12]
    out_path = out_dir / f"heatmap_{digest}.png"

    # 1. Convert CAM float [0,1] → uint8 → jet RGB
    cam_u8 = (cam * 255).clip(0, 255).astype(np.uint8)
    cam_rgb = _JET_LUT[cam_u8]                          # [H, W, 3]
    cam_h, cam_w = cam.shape

    # 2. Load and resize original image to CAM spatial size
    orig = _load_original(image_path, (cam_w, cam_h))   # PIL (cam_w, cam_h) = (W, H)
    orig_arr = np.asarray(orig, dtype=np.float32)        # [H, W, 3]

    # 3. Alpha-blend: overlay = alpha*heatmap + (1-alpha)*orig
    alpha = settings.heatmap_alpha
    overlay = (alpha * cam_rgb.astype(np.float32) + (1.0 - alpha) * orig_arr).clip(0, 255)
    overlay_img = Image.fromarray(overlay.astype(np.uint8))

    # 4. Tile: [original | heatmap | overlay] side by side for easy inspection
    tile_w = cam_w * 3
    tile = Image.new("RGB", (tile_w, cam_h))
    tile.paste(orig, (0, 0))
    tile.paste(Image.fromarray(cam_rgb), (cam_w, 0))
    tile.paste(overlay_img, (cam_w * 2, 0))

    tile.save(str(out_path), format="PNG", optimize=False)
    logger.info("heatmap saved", path=str(out_path), size=(tile_w, cam_h))
    return str(out_path)
