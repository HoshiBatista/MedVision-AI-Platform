"""API tests for the /explain endpoint (compute_cam / render mocked)."""

import numpy as np

EXPLAIN = "/api/v1/explain"
_META = {
    "cam_shape": [20, 20],
    "top_regions": [{"quadrant": 0, "mean_activation": 0.8, "max_activation": 1.0}],
    "num_detections": 2,
}


def test_explain_rejects_unknown_model(client):
    res = client.post(EXPLAIN, json={"model_name": "unknown_model", "image_path": "/x.png"})
    assert res.status_code == 422


def test_explain_success(client, monkeypatch):
    monkeypatch.setattr(
        "app.api.v1.endpoints.explain.compute_cam",
        lambda **kw: (np.zeros((20, 20), dtype=np.float32), _META),
    )
    monkeypatch.setattr(
        "app.api.v1.endpoints.explain.render_and_save",
        lambda **kw: "/data/heatmaps/heatmap_abc.png",
    )

    res = client.post(
        EXPLAIN,
        json={"model_name": "mri_segmentation", "image_path": "/data/studies/s/original.dcm"},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["heatmap_path"] == "/data/heatmaps/heatmap_abc.png"
    assert body["cam_shape"] == [20, 20]
    assert body["num_detections"] == 2
    assert body["top_regions"][0]["quadrant"] == 0


def test_explain_model_not_found(client, monkeypatch):
    def _raise(**kw):
        raise FileNotFoundError("ONNX model not found: /models/x")

    monkeypatch.setattr("app.api.v1.endpoints.explain.compute_cam", _raise)
    res = client.post(EXPLAIN, json={"model_name": "pneumonia_detection", "image_path": "/x.png"})
    assert res.status_code == 404


def test_explain_cam_failure_returns_500(client, monkeypatch):
    def _raise(**kw):
        raise ValueError("bad input tensor")

    monkeypatch.setattr("app.api.v1.endpoints.explain.compute_cam", _raise)
    res = client.post(EXPLAIN, json={"model_name": "skin_classification", "image_path": "/x.png"})
    assert res.status_code == 500


def test_explain_render_failure_returns_500(client, monkeypatch):
    monkeypatch.setattr(
        "app.api.v1.endpoints.explain.compute_cam",
        lambda **kw: (np.zeros((8, 8), dtype=np.float32), _META),
    )

    def _raise(**kw):
        raise OSError("disk full")

    monkeypatch.setattr("app.api.v1.endpoints.explain.render_and_save", _raise)
    res = client.post(EXPLAIN, json={"model_name": "mri_segmentation", "image_path": "/x.png"})
    assert res.status_code == 500
