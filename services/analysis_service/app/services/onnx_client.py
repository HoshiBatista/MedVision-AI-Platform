"""
Local ONNX Runtime inference client (CPU) used from Celery workers.

A drop-in alternative to :class:`TritonClient` that runs the exported YOLOv11
models directly with onnxruntime — no GPU or Triton server required. This is the
default backend (``INFERENCE_BACKEND=onnx``) so the dev stack and CI exercise a
real prediction end-to-end.

Models are read from a Triton-style repository laid out as
``<MODEL_REPO_ROOT>/<model_name>/1/*.onnx``. The exported graphs already embed
NMS, so their outputs match the Triton ones exactly and the existing
post-processors are reused verbatim:

Detection  (skin_classification, pneumonia_detection):
    output0: [1, 300, 6]  — [x1, y1, x2, y2, conf, class_id] in 640-px coords
Segmentation (mri_segmentation):
    output0: [1, 300, 38] — [x1, y1, x2, y2, conf, class_id, coeff*32]
    output1: [1, 32, 160, 160] — prototype masks
"""

import threading
import time
from pathlib import Path
from typing import Any

import onnxruntime as ort
import structlog

from app.core.config import settings
from app.core.metrics import ONNX_INFER_DURATION_SECONDS, ONNX_INFER_ERRORS_TOTAL
from app.services.triton_client import (
    _TASK_TO_MODEL,
    _load_image,
    _postprocess_detection,
    _postprocess_segmentation,
)

logger = structlog.get_logger()

# onnxruntime sessions are thread-safe to *run*; guard only the lazy build.
_sessions: dict[str, ort.InferenceSession] = {}
_sessions_lock = threading.Lock()


def _model_path(model_name: str) -> Path:
    """Resolve <repo>/<model_name>/1/*.onnx, preferring the highest version dir."""
    base = Path(settings.model_repo_root) / model_name
    if not base.is_dir():
        raise FileNotFoundError(f"model dir not found: {base}")

    version_dirs = sorted((d for d in base.iterdir() if d.is_dir() and d.name.isdigit()),
                          key=lambda d: int(d.name), reverse=True)
    for vdir in version_dirs:
        onnx_files = sorted(vdir.glob("*.onnx"))
        if onnx_files:
            return onnx_files[0]
    raise FileNotFoundError(f"no .onnx file under {base}")


def _get_session(model_name: str) -> ort.InferenceSession:
    sess = _sessions.get(model_name)
    if sess is not None:
        return sess
    with _sessions_lock:
        sess = _sessions.get(model_name)
        if sess is None:
            path = _model_path(model_name)
            logger.info("loading ONNX model (CPU)", model=model_name, path=str(path))
            sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            _sessions[model_name] = sess
    return sess


class OnnxClient:
    """CPU inference via ONNX Runtime. Mirrors ``TritonClient.run``."""

    def run(self, task: str, file_path: str) -> dict[str, Any]:
        model_name = _TASK_TO_MODEL[task]
        sess = _get_session(model_name)
        tensor, orig_w, orig_h = _load_image(file_path)
        input_name = sess.get_inputs()[0].name

        start = time.perf_counter()
        try:
            if task == "segmentation":
                out0, out1 = sess.run(["output0", "output1"], {input_name: tensor})
                findings = _postprocess_segmentation(out0, out1, orig_w, orig_h)
            else:
                (out0,) = sess.run(["output0"], {input_name: tensor})
                findings = _postprocess_detection(out0, orig_w, orig_h)
        except Exception as exc:
            ONNX_INFER_ERRORS_TOTAL.labels(model=model_name, error_type=type(exc).__name__).inc()
            logger.error("onnx infer failed", model=model_name, error=str(exc))
            raise

        elapsed = time.perf_counter() - start
        ONNX_INFER_DURATION_SECONDS.labels(model=model_name).observe(elapsed)
        logger.info(
            "onnx infer ok",
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
