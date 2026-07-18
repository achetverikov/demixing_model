from surface_computation.object_store import ObjectStoreConfig, SurfaceObjectStore


class RecordingClient:
    def __init__(self):
        self.calls = []

    def upload_file(self, path, bucket, key):
        self.calls.append((path, bucket, key))


def test_upload_can_publish_pending_file_under_final_object_name(tmp_path):
    pending = tmp_path / "manifest.pending"
    pending.write_text("{}")
    store = SurfaceObjectStore.__new__(SurfaceObjectStore)
    store.config = ObjectStoreConfig(bucket="bucket", prefix="prefix")
    store._client = RecordingClient()

    key = store.upload_file(pending, object_name="manifest.json")

    assert key == "prefix/manifest.json"
    assert store._client.calls == [(str(pending), "bucket", "prefix/manifest.json")]
