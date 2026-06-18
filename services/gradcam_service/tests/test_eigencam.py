"""Unit tests for the gradient-free EigenCAM helpers."""

import numpy as np
from app.core.gradcam import (
    _class_spatial_weights,
    _eigencam,
    _load_tensor,
    _summarise_regions,
)


def test_eigencam_shape_and_normalisation():
    rng = np.random.RandomState(0)
    feature_map = rng.rand(8, 5, 5).astype(np.float32)
    cam = _eigencam(feature_map)
    assert cam.shape == (5, 5)
    assert cam.dtype == np.float32
    # ReLU + min-max keeps the map within [0, 1] (the SVD sign is arbitrary, so
    # the map may collapse to all-zeros — but it must never leave the range)
    assert float(cam.min()) >= 0.0
    assert float(cam.max()) <= 1.0 + 1e-6
    assert not np.isnan(cam).any()


def test_eigencam_accepts_spatial_weights():
    rng = np.random.RandomState(1)
    feature_map = rng.rand(4, 6, 6).astype(np.float32)
    weights = np.zeros((6, 6), dtype=np.float32)
    weights[0:3, 0:3] = 1.0
    cam = _eigencam(feature_map, weights)
    assert cam.shape == (6, 6)
    assert float(cam.min()) >= 0.0 and float(cam.max()) <= 1.0


def test_class_spatial_weights_marks_target_region():
    # box of class 1 covering top-left; class 0 box ignored; low-conf ignored
    preds = np.array(
        [
            [0, 0, 320, 320, 0.90, 1],
            [0, 0, 640, 640, 0.90, 0],
            [0, 0, 320, 320, 0.10, 1],
        ],
        dtype=np.float32,
    )
    mask = _class_spatial_weights(preds, target_class=1, orig_w=640, orig_h=640, map_h=20, map_w=20)
    assert mask.shape == (20, 20)
    assert mask[0, 0] > 0          # inside the class-1 box
    assert mask[19, 19] == 0       # outside it


def test_class_spatial_weights_empty_when_no_match():
    preds = np.array([[0, 0, 100, 100, 0.9, 5]], dtype=np.float32)
    mask = _class_spatial_weights(preds, target_class=1, orig_w=640, orig_h=640, map_h=10, map_w=10)
    assert mask.max() == 0


def test_summarise_regions_ranks_quadrants():
    cam = np.zeros((4, 4), dtype=np.float32)
    cam[0:2, 0:2] = 1.0  # quadrant 0 (top-left) is hottest
    regions = _summarise_regions(cam, 4, 4)
    assert len(regions) == 4
    assert regions[0]["quadrant"] == 0
    assert regions[0]["mean_activation"] == 1.0
    assert {"quadrant", "mean_activation", "max_activation"} <= regions[0].keys()


def test_load_tensor_png(tmp_path):
    from PIL import Image

    p = tmp_path / "x.png"
    Image.new("RGB", (80, 40), "red").save(p, format="PNG")
    tensor, orig_w, orig_h = _load_tensor(str(p))
    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32
    assert float(tensor.max()) <= 1.0
    assert (orig_w, orig_h) == (80, 40)
