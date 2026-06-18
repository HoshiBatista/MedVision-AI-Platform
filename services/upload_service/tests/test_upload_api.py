"""API tests for the upload/studies endpoints."""

from app.core.config import settings

BASE = "/api/v1/studies/"


def test_upload_png_success(client, png_bytes):
    res = client.post(
        BASE + "?modality=DERM",
        files={"file": ("lesion.png", png_bytes, "image/png")},
    )
    assert res.status_code == 201, res.text
    body = res.json()
    assert body["modality"] == "DERM"
    assert body["status"] == "ready"
    assert body["original_filename"] == "lesion.png"
    assert body["meta"]["width"] == 8


def test_upload_dicom_success(client, dicom_bytes):
    res = client.post(
        BASE + "?modality=MRI",
        files={"file": ("scan.dcm", dicom_bytes, "application/dicom")},
    )
    assert res.status_code == 201, res.text
    body = res.json()
    assert body["modality"] == "MRI"
    assert body["meta"]["rows"] > 0
    assert body["meta"]["columns"] > 0


def test_upload_rejects_invalid_modality(client, png_bytes):
    res = client.post(
        BASE + "?modality=XRAY",
        files={"file": ("a.png", png_bytes, "image/png")},
    )
    assert res.status_code == 422


def test_upload_rejects_unsupported_extension(client):
    res = client.post(
        BASE + "?modality=DERM",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert res.status_code == 422


def test_upload_rejects_too_large(client, png_bytes, monkeypatch):
    monkeypatch.setattr(settings, "max_upload_size_mb", 0)
    res = client.post(
        BASE + "?modality=DERM",
        files={"file": ("a.png", png_bytes, "image/png")},
    )
    assert res.status_code == 413


def test_upload_rejects_corrupt_dicom(client):
    res = client.post(
        BASE + "?modality=MRI",
        files={"file": ("bad.dcm", b"garbage-not-dicom", "application/dicom")},
    )
    assert res.status_code == 422


def test_list_and_get_study(client, png_bytes):
    created = client.post(
        BASE + "?modality=DERM",
        files={"file": ("a.png", png_bytes, "image/png")},
    ).json()
    study_id = created["id"]

    listing = client.get(BASE)
    assert listing.status_code == 200
    body = listing.json()
    assert body["total"] >= 1
    assert any(s["id"] == study_id for s in body["items"])

    fetched = client.get(f"{BASE}{study_id}")
    assert fetched.status_code == 200
    assert fetched.json()["id"] == study_id


def test_get_unknown_study_404(client):
    assert client.get(f"{BASE}00000000-0000-0000-0000-000000000000").status_code == 404


def test_upload_requires_authentication(anon_client, png_bytes):
    res = anon_client.post(
        BASE + "?modality=DERM",
        files={"file": ("a.png", png_bytes, "image/png")},
    )
    assert res.status_code == 401
