"""Unit tests for the heatmap renderer."""

import numpy as np
from app.core.config import settings
from app.services.heatmap_renderer import _build_jet_lut, _load_original, render_and_save
from PIL import Image


def test_jet_lut_shape_and_endpoints():
    lut = _build_jet_lut()
    assert lut.shape == (256, 3)
    assert lut.dtype == np.uint8
    # low end is blue-dominant, high end is red-dominant
    assert lut[0][2] > lut[0][0]
    assert lut[255][0] > lut[255][2]


def test_load_original_png_resizes(tmp_path):
    p = tmp_path / "src.png"
    Image.new("RGB", (200, 100), "green").save(p, format="PNG")
    img = _load_original(str(p), (32, 16))
    assert img.size == (32, 16)
    assert img.mode == "RGB"


def test_render_and_save_writes_tiled_png(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "heatmap_output_dir", str(tmp_path))

    src = tmp_path / "img.png"
    Image.new("RGB", (64, 64), "white").save(src, format="PNG")
    cam = np.linspace(0, 1, 16 * 16, dtype=np.float32).reshape(16, 16)

    out_path = render_and_save(cam, str(src), "mri_segmentation", target_class=None)

    saved = Image.open(out_path)
    # tile is [original | heatmap | overlay] → 3x width
    assert saved.size == (16 * 3, 16)
    assert saved.format == "PNG"


def test_render_and_save_is_deterministic(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "heatmap_output_dir", str(tmp_path))
    src = tmp_path / "img.png"
    Image.new("RGB", (32, 32), "white").save(src, format="PNG")
    cam = np.zeros((8, 8), dtype=np.float32)

    p1 = render_and_save(cam, str(src), "skin_classification", 3)
    p2 = render_and_save(cam, str(src), "skin_classification", 3)
    assert p1 == p2  # same inputs → same hashed filename
