"""
EigenCAM for ONNX models (gradient-free).

Algorithm:
  1. Find the deepest Conv layer with spatial resolution > 8×8 (backbone/neck boundary).
  2. Expose its output as an extra graph output (in-memory, never written to disk).
  3. Run a single forward pass → get predictions + feature map [1, C, H, W].
  4. Optional class weighting: if target_class is given and detections exist,
     weight spatial positions by IoU with predicted boxes of that class.
  5. EigenCAM: reshape to [C, H*W], compute truncated SVD, take first left-singular
     vector reshaped back to [H, W] as the activation map.
  6. ReLU + min-max normalise → [0, 1] float32 array.

References: EigenCAM (Muhammad & Bhatt, 2021), arXiv:2008.00299
"""

from __future__ import annotations

import io
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import structlog
from onnx import shape_inference

from app.core.config import settings

logger = structlog.get_logger()

_INPUT_SIZE = 640  # YOLOv11 fixed input side
_CONF_THRESHOLD = 0.25


# ─── Internal cache ───────────────────────────────────────────────────────────

@dataclass
class _CachedModel:
    session: ort.InferenceSession
    cam_layer: str          # ONNX value name of the chosen feature-map output
    input_name: str         # e.g. "images"


_cache: dict[str, _CachedModel] = {}
_cache_lock = threading.Lock()


# ─── ONNX graph helpers ───────────────────────────────────────────────────────

def _find_cam_layer(model: onnx.ModelProto) -> str:
    """
    Return the output value-name of the best Conv node for EigenCAM.

    Strategy: collect all Conv output names that have spatial dims > 8 (after
    shape inference). Return the *last* such name by graph order, which sits at
    the neck/backbone boundary of YOLOv11 — rich semantics, decent resolution.
    """
    model = shape_inference.infer_shapes(model)

    # Build value → shape lookup (4-D tensors only)
    shape_map: dict[str, list[int]] = {}
    for vi in list(model.graph.value_info) + list(model.graph.output):
        t = vi.type
        if t.HasField("tensor_type") and t.tensor_type.HasField("shape"):
            dims = [d.dim_value for d in t.tensor_type.shape.dim]
            if len(dims) == 4:
                shape_map[vi.name] = dims

    candidates: list[str] = []
    for node in model.graph.node:
        if node.op_type == "Conv":
            out = node.output[0]
            shape = shape_map.get(out, [])
            # Keep only feature maps with spatial dims > 8
            if len(shape) == 4 and shape[2] > 8 and shape[3] > 8:
                candidates.append(out)

    if candidates:
        # Take the layer at 75 % depth — avoids detection-head convs
        idx = max(0, int(len(candidates) * 0.75) - 1)
        return candidates[idx]

    # Fallback: last Conv node regardless of shape
    last: str = ""
    for node in model.graph.node:
        if node.op_type == "Conv":
            last = node.output[0]
    if not last:
        raise RuntimeError("No Conv node found in ONNX graph")
    return last


def _build_session(model_path: Path) -> _CachedModel:
    logger.info("loading ONNX model for EigenCAM", path=str(model_path))
    base_model = onnx.load(str(model_path))

    cam_layer = _find_cam_layer(base_model)
    logger.debug("selected CAM layer", layer=cam_layer)

    # Clone and add cam_layer as an additional graph output
    augmented = onnx.ModelProto()
    augmented.CopyFrom(base_model)
    vi = onnx.helper.make_tensor_value_info(cam_layer, onnx.TensorProto.FLOAT, None)
    augmented.graph.output.append(vi)

    session = ort.InferenceSession(
        augmented.SerializeToString(),
        providers=settings.ort_providers,
    )

    input_name: str = session.get_inputs()[0].name
    logger.info("ONNX model ready", layer=cam_layer, input=input_name)
    return _CachedModel(session=session, cam_layer=cam_layer, input_name=input_name)


def _get_model(model_name: str) -> _CachedModel:
    with _cache_lock:
        if model_name not in _cache:
            model_path = Path(settings.model_repo_root) / model_name / "1" / "model.onnx"
            if not model_path.exists():
                raise FileNotFoundError(f"ONNX model not found: {model_path}")
            _cache[model_name] = _build_session(model_path)
        return _cache[model_name]


# ─── Image loading (matches triton_client preprocessing) ─────────────────────

def _load_tensor(image_path: str) -> tuple[np.ndarray, int, int]:
    """Returns (float32 BCHW [1,3,640,640], orig_w, orig_h)."""
    import io as _io
    from pathlib import Path as _Path

    from PIL import Image

    path = _Path(image_path)
    data = path.read_bytes()

    if path.suffix.lower() in {".dcm", ".dicom"}:
        import pydicom
        ds = pydicom.dcmread(_io.BytesIO(data))
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
        orig_h, orig_w = arr.shape[:2]
        img = Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8)).convert("RGB")
    else:
        img = Image.open(_io.BytesIO(data)).convert("RGB")
        orig_w, orig_h = img.size

    img_r = img.resize((_INPUT_SIZE, _INPUT_SIZE), Image.BILINEAR)
    tensor = np.asarray(img_r, dtype=np.float32) / 255.0
    tensor = tensor.transpose(2, 0, 1)[np.newaxis]  # [1, 3, 640, 640]
    return tensor, orig_w, orig_h


# ─── EigenCAM core ────────────────────────────────────────────────────────────

def _eigencam(feature_map: np.ndarray, spatial_weights: np.ndarray | None = None) -> np.ndarray:
    """
    feature_map : [C, H, W]
    spatial_weights: [H, W] float mask (optional; from class-discriminative weighting)
    Returns [H, W] activation map, normalised to [0, 1].
    """
    C, H, W = feature_map.shape
    flat = feature_map.reshape(C, H * W)  # [C, N]

    if spatial_weights is not None:
        w = spatial_weights.reshape(1, H * W).clip(0, None)
        w = w / (w.max() + 1e-8)
        flat = flat * w

    # Truncated SVD — only the first singular vector
    try:
        _, _, Vt = np.linalg.svd(flat, full_matrices=False)
        cam = Vt[0].reshape(H, W)
    except np.linalg.LinAlgError:
        # Fallback: channel-wise mean
        cam = flat.mean(axis=0).reshape(H, W)

    cam = np.maximum(cam, 0.0)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.astype(np.float32)


def _class_spatial_weights(
    predictions: np.ndarray, target_class: int, orig_w: int, orig_h: int, map_h: int, map_w: int
) -> np.ndarray:
    """
    Build [map_h, map_w] weight mask from detections of target_class.
    Predictions: [N, 6] in 640-px coords [x1, y1, x2, y2, conf, class_id].
    """
    mask = np.zeros((map_h, map_w), dtype=np.float32)
    scale_x = _INPUT_SIZE / map_w
    scale_y = _INPUT_SIZE / map_h

    for det in predictions:
        x1, y1, x2, y2, conf, cls = det.tolist()
        if int(cls) != target_class or conf < _CONF_THRESHOLD:
            continue
        # Map box to feature-map grid coordinates
        gx1 = max(0, int(x1 / scale_x))
        gy1 = max(0, int(y1 / scale_y))
        gx2 = min(map_w, int(x2 / scale_x) + 1)
        gy2 = min(map_h, int(y2 / scale_y) + 1)
        mask[gy1:gy2, gx1:gx2] += float(conf)

    return mask


# ─── Public API ───────────────────────────────────────────────────────────────

def compute_cam(
    model_name: str,
    image_path: str,
    target_class: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Run EigenCAM for the given model and image.

    Returns
    -------
    cam : np.ndarray  [H_feat, W_feat] float32 in [0, 1]
    meta : dict  {cam_shape, top_regions, num_detections}
    """
    cached = _get_model(model_name)
    tensor, orig_w, orig_h = _load_tensor(image_path)

    # Run inference — get both predictions and feature map
    output_names = [o.name for o in cached.session.get_outputs()]
    outputs = cached.session.run(output_names, {cached.input_name: tensor})

    # Separate cam_layer output from prediction outputs
    cam_idx = output_names.index(cached.cam_layer)
    feature_map: np.ndarray = outputs[cam_idx][0]  # [C, H, W]

    # Gather all other outputs as predictions [N, 6] — detect format
    predictions: np.ndarray | None = None
    for i, name in enumerate(output_names):
        if name == cached.cam_layer:
            continue
        arr = outputs[i]
        if arr.ndim == 3 and arr.shape[-1] >= 6:       # [1, N, ≥6]
            predictions = arr[0]
            break
        if arr.ndim == 2 and arr.shape[-1] >= 6:       # [N, ≥6]
            predictions = arr
            break

    # Build optional spatial weights for class-discriminative CAM
    spatial_weights = None
    if target_class is not None and predictions is not None:
        C, H, W = feature_map.shape
        spatial_weights = _class_spatial_weights(
            predictions[:, :6], target_class, orig_w, orig_h, H, W
        )
        # If no boxes for target_class found → fall back to class-agnostic
        if spatial_weights.max() == 0:
            spatial_weights = None
            logger.debug("no detections for target_class, using class-agnostic CAM",
                         target_class=target_class)

    cam = _eigencam(feature_map, spatial_weights)

    # Summarise top activated regions (2×2 grid quadrants)
    H, W = cam.shape
    top_regions = _summarise_regions(cam, H, W)

    num_detections = 0
    if predictions is not None:
        num_detections = int((predictions[:, 4] > _CONF_THRESHOLD).sum())

    meta: dict[str, Any] = {
        "cam_shape": list(cam.shape),
        "top_regions": top_regions,
        "num_detections": num_detections,
    }
    logger.info(
        "EigenCAM computed",
        model=model_name,
        cam_shape=cam.shape,
        num_detections=num_detections,
    )
    return cam, meta


def _summarise_regions(cam: np.ndarray, H: int, W: int) -> list[dict[str, Any]]:
    """Return top-4 quadrant activations sorted by mean intensity."""
    regions = []
    for qi, (r0, r1, c0, c1) in enumerate([
        (0, H // 2, 0, W // 2),
        (0, H // 2, W // 2, W),
        (H // 2, H, 0, W // 2),
        (H // 2, H, W // 2, W),
    ]):
        patch = cam[r0:r1, c0:c1]
        regions.append({
            "quadrant": qi,
            "mean_activation": round(float(patch.mean()), 4),
            "max_activation": round(float(patch.max()), 4),
        })
    regions.sort(key=lambda x: x["mean_activation"], reverse=True)
    return regions
