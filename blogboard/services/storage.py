from typing import Optional, List, Dict, Any

import boto3
from botocore.exceptions import ClientError

from blogboard.config.settings import app_settings
from blogboard.services.storage_base import StorageService


class R2StorageService(StorageService):
    """Cloudflare R2 storage backend (S3-compatible API)."""

    def __init__(self):
        if not app_settings.is_r2_configured():
            raise RuntimeError(
                "R2 storage is not configured. "
                "Set R2__ACCOUNT_ID, R2__ACCESS_KEY_ID, R2__SECRET_ACCESS_KEY and "
                "R2__BUCKET_NAME in your .env, or use the local storage backend "
                "(STORAGE_BACKEND=local)."
            )
        self.bucket_name = app_settings.r2.BUCKET_NAME.strip(' ="\'')
        self.client = boto3.client(
            service_name="s3",
            endpoint_url=f"https://{app_settings.r2.ACCOUNT_ID.strip()}.r2.cloudflarestorage.com",
            aws_access_key_id=app_settings.r2.ACCESS_KEY_ID.strip(),
            aws_secret_access_key=app_settings.r2.SECRET_ACCESS_KEY.strip(),
            region_name="auto",
        )

    def get_object(self, key: str) -> Optional[str]:
        try:
            response = self.client.get_object(Bucket=self.bucket_name, Key=key)
            return response["Body"].read().decode("utf-8")
        except ClientError as e:
            if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
                return None
            print(f"[ERROR] R2 error in get_object ({key}): {e}")
            return None
        except Exception as e:
            print(f"[ERROR] Unexpected error fetching {key}: {e}")
            return None

    def put_object(self, key: str, data: str, content_type: str = "text/plain") -> bool:
        try:
            self.client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=data.encode("utf-8"),
                ContentType=content_type,
            )
            print(f"  ✅ Uploaded to R2: {self.bucket_name}/{key}")
            return True
        except ClientError as e:
            print(f"[ERROR] Failed to upload {key} to R2: {e}")
            return False
        except Exception as e:
            print(f"[ERROR] Unexpected error uploading {key}: {e}")
            return False

    def list_objects(self, prefix: str = "") -> List[str]:
        try:
            paginator = self.client.get_paginator("list_objects_v2")
            keys: List[str] = []
            for page in paginator.paginate(
                Bucket=self.bucket_name, Prefix=prefix.replace("\\", "/")
            ):
                for obj in page.get("Contents", []):
                    keys.append(obj["Key"])
            return keys
        except ClientError as e:
            print(f"[ERROR] R2 error in list_objects ({prefix}): {e}")
            return []


def get_storage() -> StorageService:
    """Factory: returns the configured storage backend (r2 | local)."""
    backend = (app_settings.storage_backend or "r2").lower().strip()
    if backend == "local":
        from blogboard.services.local_storage import LocalStorageService
        return LocalStorageService()
    if backend == "r2":
        if not app_settings.is_r2_configured():
            print(
                "[WARN] R2 settings missing — falling back to local storage "
                "(set STORAGE_BACKEND=local to silence this warning)."
            )
            from blogboard.services.local_storage import LocalStorageService
            return LocalStorageService()
        return R2StorageService()
    raise ValueError(f"Unknown STORAGE_BACKEND: {backend!r} (expected 'r2' or 'local')")
