from prometheus_client import Counter, Histogram

ANALYSIS_JOBS_TOTAL = Counter(
    "analysis_jobs_total",
    "Total analysis jobs submitted",
    ["task", "result"],  # result: queued | failed
)

ANALYSIS_DURATION_SECONDS = Histogram(
    "analysis_duration_seconds",
    "End-to-end analysis task processing time",
    buckets=[1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0],
    labelnames=["task"],
)

TRITON_INFER_DURATION_SECONDS = Histogram(
    "triton_infer_duration_seconds",
    "Triton inference call latency",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
    labelnames=["model"],
)

TRITON_INFER_ERRORS_TOTAL = Counter(
    "triton_infer_errors_total",
    "Triton inference errors",
    ["model", "error_type"],
)
