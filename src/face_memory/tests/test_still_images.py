"""The default SFace embedder on real photographs.

The photos are public-domain US government portraits, downloaded on first run
(pinned by commit and SHA-256) and cached; they are not stored in this
repository. Each person is enrolled from one photo and must be recognised in a
different photo taken years apart. The tests are skipped when the models or
photos cannot be downloaded.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from face_memory.config import Phrases
from face_memory.embedder import SFaceEmbedder
from face_memory.memory import KNOWN, UNKNOWN, FaceMemory, NoFace
from face_memory.models import ModelFile, ModelUnavailable, ensure_model
from face_memory.service import FaceService

MODEL_DIR = Path("~/.cache/face_memory/models")
PHOTO_DIR = Path("~/.cache/face_memory/test-images")

# White House photographs, via the face_recognition project's examples.
WHITE_HOUSE = ("https://raw.githubusercontent.com/ageitgey/face_recognition/"
               "9f3061aaeed9a8756d2c970f5dfe066617a8281d/examples")
# Official congressional portraits (Government Publishing Office).
CONGRESS = ("https://raw.githubusercontent.com/unitedstates/images/"
            "aec3e4a88af843b282c0576f420b930f6a9a46ad/congress/450x550")

PHOTOS = {
    "a_white_house": ModelFile("obama.jpg", f"{WHITE_HOUSE}/obama.jpg",
                               "0930e3aa8cae5920329c0c8cbc6a2ab70f47b0e67b432875beaa95cbf7e741f6"),
    "a_low_res": ModelFile("obama-240p.jpg", f"{WHITE_HOUSE}/obama-240p.jpg",
                           "c2ee0d1070d2bdb9da76da876e3312bc8a07d543b6664af5390b8c3612d8cf62"),
    "a_congress": ModelFile("O000167.jpg", f"{CONGRESS}/O000167.jpg",
                            "a4c02fb604b6f2eeddacfb53b4a5336fdac481a51fb29ecea842bca91f646b22"),
    "b_white_house": ModelFile("biden.jpg", f"{WHITE_HOUSE}/biden.jpg",
                               "3c17508bb91554c637a2eabddfae790e5bb1caba93130814fc2ac50be9760c4c"),
    "b_congress": ModelFile("B000444.jpg", f"{CONGRESS}/B000444.jpg",
                            "57472977e9aab56db498141eefef39d5c095236fc9e0c19d2e586f191fd6b567"),
    "a_and_b": ModelFile("two_people.jpg", f"{WHITE_HOUSE}/two_people.jpg",
                         "536b29cd513fb618e4d5aa79d51a279948daa804ce359463449cce5f9dc68aa7"),
    "c": ModelFile("H001075.jpg", f"{CONGRESS}/H001075.jpg",
                   "74820cffc6c80856786b05146792bf7031933a9f277cf3428e2d26d66f1944f5"),
    "d": ModelFile("S000148.jpg", f"{CONGRESS}/S000148.jpg",
                   "4ef69bf51993ef040d54471f6bbf74a840b71b7729a01e971ab25f42aff18ea3"),
}


@pytest.fixture(scope="module")
def embedder():
    try:
        return SFaceEmbedder(MODEL_DIR)
    except ModelUnavailable as e:
        pytest.skip(f"models unavailable: {e}")


@pytest.fixture(scope="module")
def photo():
    try:
        paths = {key: ensure_model(model, PHOTO_DIR) for key, model in PHOTOS.items()}
    except ModelUnavailable as e:
        pytest.skip(f"test photos unavailable: {e}")
    return lambda key: cv2.imread(str(paths[key]))


@pytest.fixture
def memory(embedder, photo):
    m = FaceMemory(embedder)       # default thresholds, as shipped in config.yaml
    m.enroll("甲", [photo("a_white_house")])
    m.enroll("乙", [photo("b_white_house")])
    return m


@pytest.mark.parametrize("key, name", [
    ("a_congress", "甲"), ("a_low_res", "甲"), ("b_congress", "乙"),
])
def test_recognised_in_a_different_photo(memory, photo, key, name):
    (s,) = memory.recall(photo(key))
    assert (s.status, s.name) == (KNOWN, name), s.similarity


@pytest.mark.parametrize("key", ["c", "d"])
def test_strangers_are_not_recognised(memory, photo, key):
    (s,) = memory.recall(photo(key))
    assert s.status == UNKNOWN, (s.name, s.similarity)


def test_two_people_in_one_photo(memory, photo):
    sightings = memory.recall(photo("a_and_b"))
    assert sorted((s.status, s.name) for s in sightings) == [(KNOWN, "乙"), (KNOWN, "甲")]


def test_faces_below_min_face_px_are_ignored(embedder, photo):
    # The face in the 240p photo is 54 pixels wide.
    assert FaceMemory(embedder, min_face_px=60).recall(photo("a_low_res")) == []


def test_photo_without_a_face(memory, embedder):
    blank = np.full((480, 640, 3), 128, np.uint8)
    assert memory.recall(blank) == []
    with pytest.raises(NoFace):
        memory.enroll("丙", [blank])


def test_spoken_round_trip(embedder, photo):
    class Photos:
        def __init__(self, *keys):
            self.keys = list(keys)

        def grab(self, count=1):
            return [photo(self.keys.pop(0))]

    service = FaceService(FaceMemory(embedder), Photos("a_white_house", "a_congress", "c"),
                          Phrases())
    assert service.enroll_face("甲") == "好的，甲，我记住你了。"
    assert service.who_is_here() == "我看到了甲。"
    assert service.who_is_here() == "我还不认识你，可以告诉我你的名字吗？"


def test_name_correction_on_real_faces(memory, photo):
    memory.enroll("丁", [photo("a_congress")])
    assert "甲" not in memory.names()
    (s,) = memory.recall(photo("a_low_res"))
    assert (s.status, s.name) == (KNOWN, "丁")
