"""Unit tests for Triton I/O helpers (no Triton server involved)."""

import numpy as np
from app.services.triton_client import (
    _TASK_TO_MODEL,
    _load_image,
    _postprocess_detection,
    _postprocess_segmentation,
)
from PIL import Image


def test_task_to_model_mapping():
    assert _TASK_TO_MODEL == {
        "classification": "skin_classification",
        "detection": "pneumonia_detection",
        "segmentation": "mri_segmentation",
    }


def test_postprocess_detection_filters_scales_and_sorts():
    # orig 1280x640 -> scale_x=2, scale_y=1 (infer size 640)
    output = np.array(
        [[
            [10, 20, 30, 40, 0.90, 2],   # keep -> bbox [20,20,60,40], area 800
            [0, 0, 100, 100, 0.50, 1],   # keep (lower conf, sorts second)
            [0, 0, 5, 5, 0.10, 0],       # dropped (conf < 0.25)
        ]],
        dtype=np.float32,
    )
    findings = _postprocess_detection(output, orig_w=1280, orig_h=640)

    assert len(findings) == 2
    assert [f["confidence"] for f in findings] == [0.9, 0.5]  # sorted desc
    top = findings[0]
    assert top["bbox"] == [20.0, 20.0, 60.0, 40.0]
    assert top["class_id"] == 2
    assert top["area_px2"] == 800.0


def test_postprocess_detection_empty_when_below_threshold():
    output = np.array([[[0, 0, 10, 10, 0.05, 0]]], dtype=np.float32)
    assert _postprocess_detection(output, 640, 640) == []


def test_postprocess_segmentation_parses_box_fields():
    row = [10, 10, 20, 20, 0.8, 3] + [0.0] * 32  # 38 cols
    output0 = np.array([[row]], dtype=np.float32)
    output1 = np.zeros((1, 32, 160, 160), dtype=np.float32)

    findings = _postprocess_segmentation(output0, output1, orig_w=640, orig_h=640)
    assert len(findings) == 1
    assert findings[0]["class_id"] == 3
    assert findings[0]["confidence"] == 0.8
    assert findings[0]["bbox"] == [10.0, 10.0, 20.0, 20.0]


def test_postprocess_segmentation_empty():
    row = [0, 0, 1, 1, 0.01, 0] + [0.0] * 32
    output0 = np.array([[row]], dtype=np.float32)
    output1 = np.zeros((1, 32, 160, 160), dtype=np.float32)
    assert _postprocess_segmentation(output0, output1, 640, 640) == []


def test_load_image_png(tmp_path):
    p = tmp_path / "img.png"
    Image.new("RGB", (100, 50), "green").save(p, format="PNG")

    tensor, orig_w, orig_h = _load_image(str(p))
    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32
    assert float(tensor.min()) >= 0.0 and float(tensor.max()) <= 1.0
    assert (orig_w, orig_h) == (100, 50)


def test_load_image_dicom(tmp_path):
    from pydicom.data import get_testdata_file

    with open(get_testdata_file("CT_small.dcm"), "rb") as f:
        data = f.read()
    p = tmp_path / "scan.dcm"
    p.write_bytes(data)

    tensor, orig_w, orig_h = _load_image(str(p))
    assert tensor.shape == (1, 3, 640, 640)
    assert orig_w > 0 and orig_h > 0


def test_load_image_resizes_grayscale_png(tmp_path):
    # single-channel input is broadcast to RGB
    p = tmp_path / "gray.png"
    arr = (np.arange(64 * 64) % 256).astype(np.uint8).reshape(64, 64)
    Image.fromarray(arr, mode="L").save(p, format="PNG")
    tensor, _, _ = _load_image(str(p))
    assert tensor.shape == (1, 3, 640, 640)
