import time
import uuid

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = structlog.get_logger()

_SILENT_PATHS = {"/health", "/ready", "/metrics"}


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        if request.url.path in _SILENT_PATHS:
            return await call_next(request)

        request_id = str(uuid.uuid4())[:8]
        client_ip = request.client.host if request.client else "unknown"

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )

        logger.info(
            "request started",
            client_ip=client_ip,
            query=str(request.query_params) or None,
            content_length=request.headers.get("content-length"),
        )

        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception as exc:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            logger.error(
                "request crashed",
                error=repr(exc),
                duration_ms=duration_ms,
                exc_info=True,
            )
            raise

        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        status = response.status_code

        log_fn = logger.warning if status >= 400 else logger.info
        log_fn(
            "request completed",
            status=status,
            duration_ms=duration_ms,
        )

        response.headers["X-Request-ID"] = request_id
        return response
