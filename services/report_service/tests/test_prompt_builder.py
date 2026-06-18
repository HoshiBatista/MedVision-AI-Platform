"""Unit tests for Jinja prompt building."""

import pytest
from app.core.prompt_builder import build_prompt
from app.schemas.report import FindingsInput, PatientContext


def test_build_prompt_for_each_modality():
    findings = FindingsInput(confidence=0.91, bbox_count=2, top_class="tumor")
    for modality in ("MRI", "CXR", "DERM"):
        prompt = build_prompt(modality, findings, None)
        assert isinstance(prompt, str)
        assert prompt.strip()  # non-empty rendered template


def test_build_prompt_is_case_insensitive():
    findings = FindingsInput(confidence=0.5)
    assert build_prompt("mri", findings, None) == build_prompt("MRI", findings, None)


def test_build_prompt_includes_patient_context():
    findings = FindingsInput(confidence=0.8)
    ctx = PatientContext(age=63, sex="F", clinical_indication="headache")
    prompt = build_prompt("MRI", findings, ctx)
    # at least one context value should make it into the rendered prompt
    assert "63" in prompt or "headache" in prompt or "F" in prompt


def test_build_prompt_rejects_unknown_modality():
    with pytest.raises(ValueError):
        build_prompt("ULTRASOUND", FindingsInput(), None)
