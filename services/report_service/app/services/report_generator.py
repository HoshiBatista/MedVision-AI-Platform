import asyncio
import re
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import structlog
import torch
from transformers import BioGptForCausalLM, BioGptTokenizer, pipeline

from app.core.config import settings

logger = structlog.get_logger()

_DISCLAIMER = (
    "\n\nDISCLAIMER: This report was generated with AI assistance (BioGPT) "
    "based on automated image analysis. It must be reviewed and validated by a "
    "qualified radiologist or clinician before clinical use."
)


def _resolve_device() -> str:
    if settings.llm_device != "auto":
        return settings.llm_device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class ReportGenerator:
    def __init__(self) -> None:
        self._pipeline = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._device: str = _resolve_device()

    def load(self) -> None:
        logger.info(
            "loading BioGPT model",
            model=settings.llm_model_name,
            device=self._device,
        )
        tokenizer = BioGptTokenizer.from_pretrained(settings.llm_model_name)
        model = BioGptForCausalLM.from_pretrained(settings.llm_model_name)

        # MPS and CPU run float32; CUDA can use float16 for speed
        if self._device == "cuda":
            model = model.half()

        device_index = -1 if self._device == "cpu" else (0 if self._device in ("cuda", "mps") else -1)

        self._pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=device_index if self._device != "mps" else "mps",
        )
        logger.info("BioGPT model loaded", device=self._device)

    def _generate_sync(self, prompt: str) -> str:
        assert self._pipeline is not None, "Model not loaded"

        outputs = self._pipeline(
            prompt,
            max_new_tokens=settings.llm_max_new_tokens,
            do_sample=True,
            temperature=settings.llm_temperature,
            top_k=settings.llm_top_k,
            top_p=settings.llm_top_p,
            repetition_penalty=1.3,
            pad_token_id=self._pipeline.tokenizer.eos_token_id,
            return_full_text=True,
        )
        generated: str = outputs[0]["generated_text"]

        # Strip the prompt prefix, keep only the generated continuation
        if generated.startswith(prompt):
            generated = generated[len(prompt):]

        return _post_process(prompt, generated)

    async def generate(self, prompt: str) -> str:
        loop = asyncio.get_running_loop()
        fn = partial(self._generate_sync, prompt)
        return await loop.run_in_executor(self._executor, fn)

    def is_ready(self) -> bool:
        return self._pipeline is not None


def _post_process(prompt: str, generated: str) -> str:
    # Trim trailing incomplete sentence
    last_period = max(generated.rfind("."), generated.rfind("!"), generated.rfind("?"))
    if last_period > 0:
        generated = generated[: last_period + 1]

    # Remove repeated whitespace
    generated = re.sub(r"\s+", " ", generated).strip()

    # Reconstruct as structured report
    # The prompt already contains structured context; generated is the continuation
    full_text = f"{prompt.strip()} {generated}"
    return full_text + _DISCLAIMER


# Module-level singleton
report_generator = ReportGenerator()
