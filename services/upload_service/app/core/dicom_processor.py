import io
from typing import Any

import numpy as np
import pydicom
import structlog
from PIL import Image

logger = structlog.get_logger()


class DicomValidationError(ValueError):
    pass


def validate_and_extract(data: bytes) -> dict[str, Any]:
    """
    Parse DICOM bytes, apply pixel rescaling, return metadata dict.
    Raises DicomValidationError if the file is not valid DICOM.
    """
    try:
        ds = pydicom.dcmread(io.BytesIO(data))
    except Exception as exc:
        logger.error("DICOM parse failed", error=str(exc), size_bytes=len(data))
        raise DicomValidationError(f"Cannot parse DICOM: {exc}") from exc

    if not hasattr(ds, "PixelData"):
        logger.error("DICOM has no pixel data", modality_tag=str(getattr(ds, "Modality", "")))
        raise DicomValidationError("DICOM file has no pixel data")

    pixel_array = ds.pixel_array.astype(np.float32)

    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    if slope != 1.0 or intercept != 0.0:
        logger.debug("applying pixel rescaling", slope=slope, intercept=intercept)
    pixel_array = pixel_array * slope + intercept

    return {
        "rows": int(getattr(ds, "Rows", 0)),
        "columns": int(getattr(ds, "Columns", 0)),
        "modality_tag": str(getattr(ds, "Modality", "")),
        "patient_id": str(getattr(ds, "PatientID", "")),
        "study_instance_uid": str(getattr(ds, "StudyInstanceUID", "")),
        "series_instance_uid": str(getattr(ds, "SeriesInstanceUID", "")),
        "bits_allocated": int(getattr(ds, "BitsAllocated", 0)),
        "pixel_min": float(pixel_array.min()),
        "pixel_max": float(pixel_array.max()),
        "rescale_slope": slope,
        "rescale_intercept": intercept,
    }


def validate_image(data: bytes) -> dict[str, Any]:
    """Validate PNG/JPEG and return basic metadata."""
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()
    except Exception as exc:
        raise ValueError(f"Invalid image file: {exc}") from exc

    img = Image.open(io.BytesIO(data))
    return {
        "width": img.width,
        "height": img.height,
        "mode": img.mode,
        "format": img.format,
    }
