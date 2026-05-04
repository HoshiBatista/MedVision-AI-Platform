"""
Synchronous Triton HTTP client used from Celery workers.
All models take [1, 3, 640, 640] FP32 input (BCHW, normalised to [0, 1]).

Detection output  (skin_classification, pneumonia_detection):
    output0: [1, 300, 6]  — [x1, y1, x2, y2, conf, class_id] in 640-px coords

Segmentation output (mri_segmentation):
    output0: [1, 300, 38] — [x1, y1, x2, y2, conf, class_id, coeff*32]
    output1: [1, 32, 160, 160] — prototype masks
"""

import io
import time
from pathlib import Path
from typing import Any

import numpy as np
import pydicom
import structlog
import tritonclient.http as httpclient
from PIL import Image

from app.core.config import settings
from app.core.metrics import TRITON_INFER_DURATION_SECONDS, TRITON_INFER_ERRORS_TOTAL

logger = structlog.get_logger()

_INFER_SIZE = 640
_TASK_TO_MODEL = {
    "classification": "skin_classification",
    "detection": "pneumonia_detection",
    "segmentation": "mri_segmentation",
}


def _load_image(file_path: str) -> tuple[np.ndarray, int, int]:
    """
    Load image from disk (DICOM or raster). Returns (float32 HWC array normalised [0,1], orig_w, orig_h).
    """
    path = Path(file_path)
    data = path.read_bytes()

    if path.suffix.lower() in {".dcm", ".dicom"}:
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

        orig_h, orig_w = arr.shape[:2]
        img = Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8)).convert("RGB")
    else:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        orig_w, orig_h = img.size

    img_resized = img.resize((_INFER_SIZE, _INFER_SIZE), Image.BILINEAR)
    tensor = np.asarray(img_resized, dtype=np.float32) / 255.0
    tensor = tensor.transpose(2, 0, 1)[np.newaxis]  # HWC → BCHW [1,3,640,640]
    return tensor, orig_w, orig_h


def _postprocess_detection(output: np.ndarray, orig_w: int, orig_h: int) -> list[dict[str, Any]]:
    """Parse [1, 300, 6] detection output into a list of finding dicts."""
    preds = output[0]  # [300, 6]
    mask = preds[:, 4] > settings.inference_conf_threshold
    preds = preds[mask]

    findings = []
    scale_x = orig_w / _INFER_SIZE
    scale_y = orig_h / _INFER_SIZE

    for det in preds:
        x1, y1, x2, y2, conf, cls = det.tolist()
        x1, x2 = x1 * scale_x, x2 * scale_x
        y1, y2 = y1 * scale_y, y2 * scale_y
        findings.append({
            "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            "confidence": round(float(conf), 4),
            "class_id": int(cls),
            "area_px2": round((x2 - x1) * (y2 - y1), 1),
        })

    findings.sort(key=lambda d: d["confidence"], reverse=True)
    return findings


def _postprocess_segmentation(
    output0: np.ndarray, output1: np.ndarray, orig_w: int, orig_h: int
) -> list[dict[str, Any]]:
    """
    Parse [1, 300, 38] + [1, 32, 160, 160] segmentation output.
    Format: [x1,y1,x2,y2, conf, class_id, coeff*32]
    Returns findings list; mask reconstruction is delegated to the GradCAM service.
    """
    preds = output0[0]  # [300, 38]
    mask = preds[:, 4] > settings.inference_conf_threshold
    preds = preds[mask]

    if len(preds) == 0:
        return []

    scale_x = orig_w / _INFER_SIZE
    scale_y = orig_h / _INFER_SIZE

    findings = []
    for det in preds:
        x1, y1, x2, y2, conf, cls = det[:6].tolist()
        x1, x2 = x1 * scale_x, x2 * scale_x
        y1, y2 = y1 * scale_y, y2 * scale_y
        area_mm2 = round((x2 - x1) * (y2 - y1), 1)
        findings.append({
            "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            "confidence": round(float(conf), 4),
            "class_id": int(cls),
            "area_px2": area_mm2,
        })

    findings.sort(key=lambda d: d["confidence"], reverse=True)
    return findings


class TritonClient:
    def __init__(self) -> None:
        self._client = httpclient.InferenceServerClient(
            url=settings.triton_http_url,
            verbose=False,
        )

    def run(self, task: str, file_path: str) -> dict[str, Any]:
        model_name = _TASK_TO_MODEL[task]
        tensor, orig_w, orig_h = _load_image(file_path)

        infer_input = httpclient.InferInput("images", list(tensor.shape), "FP32")
        infer_input.set_data_from_numpy(tensor)

        start = time.perf_counter()
        try:
            if task == "segmentation":
                outputs = [
                    httpclient.InferRequestedOutput("output0"),
                    httpclient.InferRequestedOutput("output1"),
                ]
                result = self._client.infer(
                    model_name=model_name, inputs=[infer_input], outputs=outputs
                )
                out0 = result.as_numpy("output0")
                out1 = result.as_numpy("output1")
                findings = _postprocess_segmentation(out0, out1, orig_w, orig_h)
            else:
                outputs = [httpclient.InferRequestedOutput("output0")]
                result = self._client.infer(
                    model_name=model_name, inputs=[infer_input], outputs=outputs
                )
                out0 = result.as_numpy("output0")
                findings = _postprocess_detection(out0, orig_w, orig_h)

        except Exception as exc:
            TRITON_INFER_ERRORS_TOTAL.labels(model=model_name, error_type=type(exc).__name__).inc()
            logger.error("triton infer failed", model=model_name, error=str(exc))
            raise

        elapsed = time.perf_counter() - start
        TRITON_INFER_DURATION_SECONDS.labels(model=model_name).observe(elapsed)
        logger.info(
            "triton infer ok",
            model=model_name,
            num_findings=len(findings),
            duration_ms=round(elapsed * 1000, 1),
        )

        top = findings[0] if findings else {}
        return {
            "model": model_name,
            "findings": findings,
            "num_detections": len(findings),
            "top_confidence": top.get("confidence"),
            "orig_size": [orig_w, orig_h],
        }
