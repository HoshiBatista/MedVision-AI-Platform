from datetime import datetime

from pydantic import BaseModel


class FindingsInput(BaseModel):
    confidence: float | None = None
    labels: list[str] | None = None
    regions: list[dict] | None = None
    max_area_mm2: float | None = None
    bbox_count: int | None = None
    top_class: str | None = None
    raw: dict | None = None


class PatientContext(BaseModel):
    age: int | None = None
    sex: str | None = None
    clinical_indication: str | None = None
    relevant_history: str | None = None


class GenerateReportRequest(BaseModel):
    study_id: str
    modality: str  # MRI | CXR | DERM
    job_ids: list[str]
    findings: FindingsInput
    patient_context: PatientContext | None = None


class ReportResponse(BaseModel):
    report_id: str
    study_id: str
    status: str
    modality: str
    content: str | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
