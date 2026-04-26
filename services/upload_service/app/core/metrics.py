from prometheus_client import Counter, Histogram

UPLOADS_TOTAL = Counter(
    "upload_studies_total",
    "Total study upload attempts",
    ["modality", "result"],  # result: success | validation_error | too_large | unsupported_format
)

UPLOAD_FILE_SIZE_BYTES = Histogram(
    "upload_file_size_bytes",
    "Distribution of uploaded file sizes",
    buckets=[
        100_000,       # 100 KB
        1_000_000,     # 1 MB
        5_000_000,     # 5 MB
        10_000_000,    # 10 MB
        50_000_000,    # 50 MB
        100_000_000,   # 100 MB
        512_000_000,   # 512 MB (max)
    ],
    labelnames=["modality"],
)

UPLOAD_DURATION_SECONDS = Histogram(
    "upload_duration_seconds",
    "End-to-end upload processing time",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
    labelnames=["modality"],
)

DICOM_VALIDATION_TOTAL = Counter(
    "upload_dicom_validation_total",
    "DICOM file validation results",
    ["result"],  # success | error
)
