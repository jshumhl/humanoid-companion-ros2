"""Model files for the default embedder: download once, verify by SHA-256.

Both come from the OpenCV model zoo:
- YuNet face detector, MIT licence.
- SFace face recognizer, Apache 2.0 licence.
"""

import hashlib
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ZOO = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models"
DOWNLOAD_TIMEOUT_SEC = 60


@dataclass(frozen=True)
class ModelFile:
    filename: str
    url: str
    sha256: str


DETECTOR = ModelFile(
    "face_detection_yunet_2023mar.onnx",
    f"{ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
)
RECOGNIZER = ModelFile(
    "face_recognition_sface_2021dec.onnx",
    f"{ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
)


class ModelUnavailable(RuntimeError):
    pass


def ensure_model(model, model_dir, download=True):
    """Return the path of `model` in `model_dir`, downloading it if allowed."""
    model_dir = Path(model_dir).expanduser()
    path = model_dir / model.filename
    if path.is_file():
        return path
    if not download:
        raise ModelUnavailable(
            f"{path} is missing. Fetch it with: python -m face_memory download-models")
    model_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=model_dir, suffix=".part")
    try:
        with os.fdopen(fd, "wb") as out, \
                urllib.request.urlopen(model.url, timeout=DOWNLOAD_TIMEOUT_SEC) as response:
            while chunk := response.read(1 << 20):
                out.write(chunk)
        digest = sha256_of(tmp)
        if digest != model.sha256:
            raise ModelUnavailable(
                f"{model.url} has SHA-256 {digest}, expected {model.sha256}")
        os.replace(tmp, path)
    except OSError as e:
        raise ModelUnavailable(f"Cannot download {model.url}: {e}") from e
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()
