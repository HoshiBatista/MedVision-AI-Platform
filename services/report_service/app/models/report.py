import uuid
from datetime import datetime

from app.core.database import Base
from sqlalchemy import DateTime, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import Mapped, mapped_column

VALID_STATUSES = {"pending", "generating", "completed", "failed"}
VALID_MODALITIES = {"MRI", "CXR", "DERM"}


class Report(Base):
    __tablename__ = "reports"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    study_id: Mapped[str] = mapped_column(UUID(as_uuid=False), nullable=False, index=True)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    modality: Mapped[str] = mapped_column(String(10), nullable=False)
    # pending | generating | completed | failed
    status: Mapped[str] = mapped_column(String(20), default="pending", nullable=False)
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    findings_snapshot: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
