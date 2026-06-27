"""
Clinical report generation via a local Ollama server.

report_service stays a thin client: it builds the prompt (Jinja templates,
see prompt_builder) and delegates text generation to an Ollama instance running a
quantised medical LLM (default OpenBioLLM-8B Q8). No model weights, torch, or
transformers live in this service. A fixed AI-assistance disclaimer is always
appended. No external/cloud LLM API is called.
"""

import re

import httpx
import structlog

from app.core.config import settings

logger = structlog.get_logger()

_DISCLAIMER = (
    "\n\nDISCLAIMER: This report was generated with AI assistance based on "
    "automated image analysis. It must be reviewed and validated by a qualified "
    "radiologist or clinician before clinical use."
)


class ReportGenerator:
    """Thin async client over the Ollama HTTP API."""

    def __init__(self) -> None:
        self._ready = False
        self._model = settings.llm_model_name
        self._base = settings.ollama_url.rstrip("/")

    async def ensure_model(self) -> None:
        """
        Pull the configured model into Ollama (idempotent — returns quickly if the
        weights are already cached). Flips the readiness flag on success. Called as
        a background task at startup so the HTTP server is available immediately.
        """
        try:
            async with httpx.AsyncClient(
                base_url=self._base, timeout=settings.llm_pull_timeout
            ) as client:
                logger.info("ensuring ollama model", model=self._model)
                resp = await client.post(
                    "/api/pull", json={"name": self._model, "stream": False}
                )
                resp.raise_for_status()
            self._ready = True
            logger.info("ollama model ready", model=self._model)
        except Exception as exc:  # noqa: BLE001
            self._ready = False
            logger.error("ollama model unavailable", model=self._model, error=str(exc))

    def is_ready(self) -> bool:
        return self._ready

    async def generate(self, prompt: str) -> str:
        async with httpx.AsyncClient(
            base_url=self._base, timeout=settings.llm_request_timeout
        ) as client:
            resp = await client.post(
                "/api/generate",
                json={
                    "model": self._model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": settings.llm_temperature,
                        "top_k": settings.llm_top_k,
                        "top_p": settings.llm_top_p,
                        "num_predict": settings.llm_max_new_tokens,
                        "repeat_penalty": 1.3,
                    },
                },
            )
            resp.raise_for_status()
            generated: str = resp.json().get("response", "")
        return _post_process(generated)


def _post_process(generated: str) -> str:
    text = re.sub(r"\s+", " ", generated).strip()

    # Trim a trailing incomplete sentence.
    last_terminal = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
    if last_terminal > 0:
        text = text[: last_terminal + 1]

    return text + _DISCLAIMER


# Module-level singleton
report_generator = ReportGenerator()
