import json
from pathlib import Path
from typing import Optional, List, Dict, Any

from blogboard.services.storage_base import StorageService


class LocalStorageService(StorageService):
    """
    Filesystem storage backend — mirrors the R2 key structure under a local
    directory. Perfect for local development, testing, and offline runs.
    """

    def __init__(self, root: Optional[str] = None):
        from blogboard.config.settings import app_settings
        self.root = Path(root or app_settings.local_storage_root or "blogboard/web").resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        # Normalise separators and guard against path traversal.
        safe = key.replace("\\", "/").lstrip("/")
        parts = [p for p in safe.split("/") if p not in ("", ".", "..")]
        return self.root.joinpath(*parts)

    def get_object(self, key: str) -> Optional[str]:
        path = self._path(key)
        if not path.is_file():
            return None
        try:
            return path.read_text(encoding="utf-8")
        except OSError as e:
            print(f"[ERROR] Local read failed ({key}): {e}")
            return None

    def put_object(self, key: str, data: str, content_type: str = "text/plain") -> bool:
        try:
            path = self._path(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(data, encoding="utf-8")
            print(f"  ✅ Saved locally: {path}")
            return True
        except OSError as e:
            print(f"[ERROR] Local write failed ({key}): {e}")
            return False

    def list_objects(self, prefix: str = "") -> List[str]:
        base = self._path(prefix) if prefix else self.root
        if not base.exists():
            return []
        if base.is_file():
            return [str(base.relative_to(self.root)).replace("\\", "/")]
        keys: List[str] = []
        for p in sorted(base.rglob("*")):
            if p.is_file():
                keys.append(str(p.relative_to(self.root)).replace("\\", "/"))
        return keys

    # get_json / registry helpers are inherited from StorageService and work
    # because get_object/put_object are implemented.
