"""Small S3-compatible object store helper for durable surface outputs.

Environment variables:
    S3_BUCKET or OBJECT_STORE_BUCKET
    S3_PREFIX or OBJECT_STORE_PREFIX
    S3_ENDPOINT_URL or OBJECT_STORE_ENDPOINT_URL
    AWS_ACCESS_KEY_ID
    AWS_SECRET_ACCESS_KEY
    AWS_DEFAULT_REGION or AWS_REGION
"""

import os
import re
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set


@dataclass
class ObjectStoreConfig:
    bucket: str
    prefix: str = ""
    endpoint_url: Optional[str] = None
    region_name: str = "auto"

    @classmethod
    def from_env(cls) -> Optional["ObjectStoreConfig"]:
        bucket = os.getenv("S3_BUCKET") or os.getenv("OBJECT_STORE_BUCKET")
        if not bucket:
            return None
        prefix = os.getenv("S3_PREFIX") or os.getenv("OBJECT_STORE_PREFIX") or ""
        endpoint_url = os.getenv("S3_ENDPOINT_URL") or os.getenv("OBJECT_STORE_ENDPOINT_URL")
        region = os.getenv("AWS_DEFAULT_REGION") or os.getenv("AWS_REGION") or "auto"
        return cls(bucket=bucket, prefix=prefix.strip("/"), endpoint_url=endpoint_url, region_name=region)


class SurfaceObjectStore:
    """Upload/list averaged surface pickles in an S3-compatible bucket."""

    _SURFACE_RE = re.compile(
        r"averaged_sf1_(?P<sf1>[\d.]+)_sf2_(?P<sf2>[\d.]+)_sp_(?P<sp>[\d.]+)\.pkl$"
    )
    _BUNDLE_MANIFEST_RE = re.compile(r"surface_bundle_.*\.manifest\.json$")

    def __init__(self, config: ObjectStoreConfig):
        import boto3

        self.config = config
        self._client = boto3.client(
            "s3",
            endpoint_url=config.endpoint_url,
            region_name=config.region_name,
        )

    def _key(self, name: str) -> str:
        return f"{self.config.prefix}/{name}" if self.config.prefix else name

    def upload_file(self, path: Path) -> str:
        key = self._key(path.name)
        self._client.upload_file(str(path), self.config.bucket, key)
        return key

    def download_file(self, name: str, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._client.download_file(self.config.bucket, self._key(name), str(path))

    def list_bundle_object_names(self) -> List[str]:
        prefix = f"{self.config.prefix}/surface_bundle_" if self.config.prefix else "surface_bundle_"
        paginator = self._client.get_paginator("list_objects_v2")
        names = []
        for page in paginator.paginate(Bucket=self.config.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                name = Path(obj["Key"]).name
                if name.endswith(".pkl.gz") or name.endswith(".manifest.json"):
                    names.append(name)
        return sorted(names)

    def list_bundle_objects(self) -> List[Dict]:
        prefix = f"{self.config.prefix}/surface_bundle_" if self.config.prefix else "surface_bundle_"
        paginator = self._client.get_paginator("list_objects_v2")
        objects = []
        for page in paginator.paginate(Bucket=self.config.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                name = Path(obj["Key"]).name
                if name.endswith(".pkl.gz") or name.endswith(".manifest.json"):
                    objects.append({
                        "name": name,
                        "key": obj["Key"],
                        "size": obj.get("Size", 0),
                        "last_modified": obj.get("LastModified"),
                    })
        return sorted(objects, key=lambda item: item["name"])

    def list_objects(self) -> List[Dict]:
        prefix = f"{self.config.prefix}/" if self.config.prefix else ""
        paginator = self._client.get_paginator("list_objects_v2")
        objects = []
        for page in paginator.paginate(Bucket=self.config.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                objects.append({
                    "name": Path(obj["Key"]).name,
                    "key": obj["Key"],
                    "size": obj.get("Size", 0),
                    "last_modified": obj.get("LastModified"),
                })
        return sorted(objects, key=lambda item: item["key"])

    def delete_key(self, key: str) -> None:
        self._client.delete_object(Bucket=self.config.bucket, Key=key)

    def upload_surface(self, path: Path) -> str:
        return self.upload_file(path)

    def object_exists(self, name: str) -> bool:
        try:
            self._client.head_object(Bucket=self.config.bucket, Key=self._key(name))
            return True
        except Exception:
            return False

    def delete_surface(self, name: str) -> None:
        self._client.delete_object(Bucket=self.config.bucket, Key=self._key(name))

    def list_completed_surface_ids(self) -> Set[str]:
        prefix = f"{self.config.prefix}/" if self.config.prefix else ""
        paginator = self._client.get_paginator("list_objects_v2")
        completed: Set[str] = set()
        for page in paginator.paginate(Bucket=self.config.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                name = Path(obj["Key"]).name
                match = self._SURFACE_RE.match(name)
                if not match:
                    if self._BUNDLE_MANIFEST_RE.match(name):
                        body = self._client.get_object(
                            Bucket=self.config.bucket, Key=obj["Key"]
                        )["Body"].read()
                        manifest = json.loads(body.decode("utf-8"))
                        completed.update(manifest.get("surface_ids", []))
                    continue
                sf1 = float(match["sf1"])
                sf2 = float(match["sf2"])
                sp = float(match["sp"])
                completed.add(f"{min(sf1, sf2):.1f}|{max(sf1, sf2):.1f}|{sp:.1f}")
        return completed
