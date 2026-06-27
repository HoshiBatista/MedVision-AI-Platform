"""Inference backend selection.

Both clients expose ``run(task: str, file_path: str) -> dict``; the worker picks
one via ``settings.inference_backend`` (default ``onnx``: local CPU inference).
"""

from typing import Protocol

from app.core.config import settings


class InferenceClient(Protocol):
    def run(self, task: str, file_path: str) -> dict: ...


def get_inference_client() -> InferenceClient:
    if settings.inference_backend.lower() == "triton":
        from app.services.triton_client import TritonClient

        return TritonClient()

    from app.services.onnx_client import OnnxClient

    return OnnxClient()
