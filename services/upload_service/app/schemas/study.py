from datetime import datetime
from typing import Any

from pydantic import BaseModel


class StudyResponse(BaseModel):
    id: str
    user_id: int
    modality: str
    original_filename: str
    file_path: str
    file_size_bytes: int
    status: str
    meta: dict[str, Any] | None
    created_at: datetime

    model_config = {"from_attributes": True}


class StudyListResponse(BaseModel):
    items: list[StudyResponse]
    total: int
