import hashlib

import pytest

from face_memory.models import ModelFile, ModelUnavailable, ensure_model


def model_for(tmp_path, content, sha=None):
    source = tmp_path / "source.onnx"
    source.write_bytes(content)
    return ModelFile("m.onnx", source.as_uri(), sha or hashlib.sha256(content).hexdigest())


def test_downloads_and_verifies(tmp_path):
    model = model_for(tmp_path, b"weights")
    path = ensure_model(model, tmp_path / "models")
    assert path.read_bytes() == b"weights"


def test_existing_file_is_used(tmp_path):
    (tmp_path / "m.onnx").write_bytes(b"old")
    model = ModelFile("m.onnx", "file:///nowhere", "0" * 64)
    assert ensure_model(model, tmp_path, download=False).read_bytes() == b"old"


def test_missing_without_download(tmp_path):
    with pytest.raises(ModelUnavailable, match="download-models"):
        ensure_model(ModelFile("m.onnx", "file:///nowhere", "0" * 64), tmp_path, download=False)


def test_checksum_mismatch_leaves_nothing_behind(tmp_path):
    model = model_for(tmp_path, b"tampered", sha="0" * 64)
    models = tmp_path / "models"
    with pytest.raises(ModelUnavailable, match="SHA-256"):
        ensure_model(model, models)
    assert list(models.iterdir()) == []


def test_unreachable_url(tmp_path):
    with pytest.raises(ModelUnavailable, match="Cannot download"):
        ensure_model(ModelFile("m.onnx", (tmp_path / "gone").as_uri(), "0" * 64), tmp_path)
