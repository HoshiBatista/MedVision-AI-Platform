"""Schema validation for AnalyzeRequest."""

import pytest
from app.schemas.job import AnalyzeRequest
from pydantic import ValidationError


def test_analyze_request_accepts_valid_tasks():
    for task in ("segmentation", "detection", "classification"):
        req = AnalyzeRequest(study_id="s1", task=task)
        assert req.task == task
        assert req.config == {}


def test_analyze_request_rejects_unknown_task():
    with pytest.raises(ValidationError):
        AnalyzeRequest(study_id="s1", task="reconstruction")
