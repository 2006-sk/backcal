import uuid, shutil
from pathlib import Path
from app.core.config import settings

class Storage:
    def __init__(self, upload_dir: str | None = None):
        self.upload_dir = Path(upload_dir or settings.UPLOAD_DIR)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def save_upload(self, fileobj, filename: str) -> tuple[str, Path]:
        fid = str(uuid.uuid4())
        dest = self.upload_dir / f"{fid}__{Path(filename).name}"
        with open(dest, "wb") as out:
            shutil.copyfileobj(fileobj, out)
        return fid, dest

    def get_path(self, file_id_or_name: str) -> Path | None:
        # Lookup by id prefix
        matches = list(self.upload_dir.glob(f"{file_id_or_name}__*"))
        if matches:
            return matches[0]
        # Or treat as direct path
        p = Path(file_id_or_name)
        return p if p.exists() else None
