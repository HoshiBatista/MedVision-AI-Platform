from pathlib import Path

import aiofiles

from app.core.config import settings


async def save_file(study_id: str, filename: str, data: bytes) -> str:
    """
    Save raw bytes to /data/studies/{study_id}/{filename}.
    Returns the relative path stored in the DB.
    """
    study_dir = Path(settings.storage_root) / study_id
    study_dir.mkdir(parents=True, exist_ok=True)

    dest = study_dir / filename
    async with aiofiles.open(dest, "wb") as f:
        await f.write(data)

    return str(dest)


def get_file_path(study_id: str, filename: str) -> Path:
    return Path(settings.storage_root) / study_id / filename
