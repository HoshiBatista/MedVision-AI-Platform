from celery import Celery

from app.core.config import settings
from app.core.telemetry import setup_celery_telemetry

celery = Celery(
    "analysis_service",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["app.workers.tasks"],
)

celery.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_routes={
        "analysis.run": {"queue": "analysis"},
    },
)

setup_celery_telemetry(
    service_name="analysis_worker",
    enabled=settings.otel_traces_enabled,
    otlp_endpoint=settings.otel_exporter_otlp_endpoint,
)
