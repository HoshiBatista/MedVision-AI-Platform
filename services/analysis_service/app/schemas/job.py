from datetime import datetime

from pydantic import BaseModel, Field, field_validator

from app.models.job import VALID_TASKS


class AnalyzeRequest(BaseModel):
    study_id: str
    task: str
    config: dict = {}

    @field_validator("task")
    @classmethod
    def task_must_be_valid(cls, v: str) -> str:
        if v not in VALID_TASKS:
            raise ValueError(f"task must be one of {VALID_TASKS}")
        return v


class JobQueuedResponse(BaseModel):
    job_id: str
    status: str

    model_config = {"from_attributes": True}


class JobSummaryResponse(BaseModel):
    job_id: str = Field(validation_alias="id")
    study_id: str
    task: str
    status: str
    error: str | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True, "populate_by_name": True}


class JobListResponse(BaseModel):
    items: list[JobSummaryResponse]
    total: int


class JobResultResponse(BaseModel):
    # the ORM column is `id`; expose it as `job_id`
    job_id: str = Field(validation_alias="id")
    study_id: str
    task: str
    status: str
    results: dict | None = None
    error: str | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True, "populate_by_name": True}
