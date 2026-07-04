"""OpenTelemetry tracing setup."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from fastapi import FastAPI
    from sqlalchemy.engine import Engine

logger = structlog.get_logger()


def _tracing_active(*, enabled: bool, otlp_endpoint: str) -> bool:
    return enabled and bool(otlp_endpoint.strip())


def setup_telemetry(
    *,
    service_name: str,
    enabled: bool,
    otlp_endpoint: str,
    app: FastAPI | None = None,
) -> None:
    """Configure OTLP export and auto-instrumentation."""
    if not _tracing_active(enabled=enabled, otlp_endpoint=otlp_endpoint):
        return

    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    endpoint = otlp_endpoint.rstrip("/")
    if not endpoint.endswith("/v1/traces"):
        endpoint = f"{endpoint}/v1/traces"

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=endpoint)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    if app is not None:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(
            app,
            excluded_urls="/health,/ready,/metrics",
        )

    HTTPXClientInstrumentor().instrument()

    logger.info("opentelemetry configured", service=service_name, endpoint=endpoint)


def instrument_sqlalchemy_engines(
    *engines: Engine,
    enabled: bool,
    otlp_endpoint: str,
) -> None:
    if not _tracing_active(enabled=enabled, otlp_endpoint=otlp_endpoint):
        return

    from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

    instrumentor = SQLAlchemyInstrumentor()
    for engine in engines:
        instrumentor.instrument(engine=engine)


def setup_celery_telemetry(*, service_name: str, enabled: bool, otlp_endpoint: str) -> None:
    """Worker-side tracing (no FastAPI app)."""
    setup_telemetry(service_name=service_name, enabled=enabled, otlp_endpoint=otlp_endpoint)
    if not _tracing_active(enabled=enabled, otlp_endpoint=otlp_endpoint):
        return

    from opentelemetry.instrumentation.celery import CeleryInstrumentor

    CeleryInstrumentor().instrument()


@contextmanager
def traced_span(
    name: str,
    *,
    enabled: bool,
    otlp_endpoint: str,
    attributes: dict[str, str | int | float | bool] | None = None,
) -> Iterator[None]:
    """Manual span for sync code paths (Triton, ONNX). No-op when tracing is off."""
    if not _tracing_active(enabled=enabled, otlp_endpoint=otlp_endpoint):
        yield
        return

    from opentelemetry import trace

    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span(name) as span:
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
        yield
