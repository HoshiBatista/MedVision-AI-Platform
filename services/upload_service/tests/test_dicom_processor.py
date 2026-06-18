"""Unit tests for DICOM/image validation and metadata extraction."""

import io

import pytest
from app.core.dicom_processor import (
    DicomValidationError,
    validate_and_extract,
    validate_image,
)
from PIL import Image


def test_validate_and_extract_returns_metadata(dicom_bytes):
    meta = validate_and_extract(dicom_bytes)
    assert meta["rows"] > 0
    assert meta["columns"] > 0
    assert meta["modality_tag"]  # e.g. "CT"
    assert meta["pixel_min"] <= meta["pixel_max"]
    # rescale tags are read and exposed as floats (the rescale branch runs)
    assert isinstance(meta["rescale_slope"], float)
    assert isinstance(meta["rescale_intercept"], float)


def test_validate_and_extract_rejects_non_dicom():
    with pytest.raises(DicomValidationError):
        validate_and_extract(b"this is definitely not a DICOM file")


def test_validate_and_extract_rejects_dicom_without_pixels(dicom_bytes):
    import pydicom

    ds = pydicom.dcmread(io.BytesIO(dicom_bytes))
    del ds.PixelData
    buf = io.BytesIO()
    ds.save_as(buf, enforce_file_format=True)
    with pytest.raises(DicomValidationError):
        validate_and_extract(buf.getvalue())


def test_validate_image_returns_metadata():
    buf = io.BytesIO()
    Image.new("RGB", (12, 9), "blue").save(buf, format="PNG")
    meta = validate_image(buf.getvalue())
    assert meta["width"] == 12
    assert meta["height"] == 9
    assert meta["format"] == "PNG"
    assert meta["mode"] == "RGB"


def test_validate_image_rejects_garbage():
    with pytest.raises(ValueError):
        validate_image(b"not an image")
