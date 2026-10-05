"""The default SFace embedder on real images.

No photographs of people are stored in or downloaded by this repository. The
recognition tests read a local folder named by the FACE_MEMORY_PHOTOS
environment variable, with one subfolder per person and at least two photos in
each, of people who have agreed to be photographed for this:

    $FACE_MEMORY_PHOTOS/
        alice/  day1.jpg  day2.jpg
        bob/    1.jpg     2.jpg   3.jpg

Each person is enrolled from their first photo (sorted by file name) and must
be recognised in each of their other photos, and never taken for anyone else.
Without the variable those tests are skipped. All tests skip when the models
can't be downloaded.
"""

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from face_memory.config import Phrases
from face_memory.embedder import SFaceEmbedder
from face_memory.memory import KNOWN, FaceMemory, NoFace
from face_memory.models import ModelUnavailable
from face_memory.service import FaceService

MODEL_DIR = Path("~/.cache/face_memory/models")
PHOTO_TYPES = (".jpg", ".jpeg", ".png")


@pytest.fixture(scope="module")
def embedder():
    try:
        return SFaceEmbedder(MODEL_DIR)
    except ModelUnavailable as e:
        pytest.skip(f"models unavailable: {e}")


def load_people():
    root = os.environ.get("FACE_MEMORY_PHOTOS")
    if not root:
        return {}
    people = {}
    for folder in sorted(Path(root).expanduser().iterdir()):
        photos = sorted(p for p in folder.glob("*") if p.suffix.lower() in PHOTO_TYPES)
        if folder.is_dir() and len(photos) >= 2:
            people[folder.name] = photos
    return people


PEOPLE = load_people()
needs_photos = pytest.mark.skipif(
    not PEOPLE, reason="set FACE_MEMORY_PHOTOS to a folder of consented photos")


def read(path):
    frame = cv2.imread(str(path))
    assert frame is not None, f"cannot read {path}"
    return frame


@pytest.fixture(scope="module")
def memory(embedder):
    m = FaceMemory(embedder)       # default thresholds, as shipped in config.yaml
    for name, photos in PEOPLE.items():
        m.enroll(name, [read(photos[0])])
    return m


def test_blank_image_has_no_face(embedder):
    blank = np.full((480, 640, 3), 128, np.uint8)
    assert embedder.faces(blank) == []
    with pytest.raises(NoFace):
        FaceMemory(embedder).enroll("甲", [blank])


def test_noise_image_has_no_face(embedder):
    noise = np.random.default_rng(0).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    assert FaceMemory(embedder).recall(noise) == []


@needs_photos
@pytest.mark.parametrize("name, photo", [
    (name, photo) for name, photos in PEOPLE.items() for photo in photos[1:]])
def test_recognised_in_another_photo(memory, name, photo):
    sightings = memory.recall(read(photo))
    assert sightings, f"no face found in {photo}"
    largest = sightings[0]
    assert (largest.status, largest.name) == (KNOWN, name), \
        f"{photo}: {largest.status} {largest.name} {largest.similarity:.3f}"


@needs_photos
def test_never_taken_for_someone_else(memory):
    for name, photos in PEOPLE.items():
        for photo in photos:
            for s in memory.recall(read(photo)):
                assert s.status != KNOWN or s.name == name, \
                    f"{photo} taken for {s.name} ({s.similarity:.3f})"


@needs_photos
def test_spoken_round_trip(embedder):
    name, photos = next(iter(PEOPLE.items()))

    class Photos:
        def __init__(self, *paths):
            self.paths = list(paths)

        def grab(self, count=1):
            return [read(self.paths.pop(0))]

    service = FaceService(FaceMemory(embedder), Photos(photos[0], photos[1]), Phrases())
    assert service.enroll_face(name) == f"好的，{name}，我记住你了。"
    assert service.who_is_here().startswith(f"我看到了{name}")
