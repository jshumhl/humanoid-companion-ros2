import pytest

from face_memory.memory import KNOWN, UNKNOWN, UNSURE, FaceMemory, Sighting
from face_memory.service import FaceService, describe, spoken_count

from .conftest import FakeEmbedder, FakeFrames, at_similarity, face, vec


def service(phrases, *batches):
    memory = FaceMemory(FakeEmbedder(), match_threshold=0.5, unsure_threshold=0.35)
    frames = FakeFrames(*batches)
    return FaceService(memory, frames, phrases, frames_per_enroll=3), frames


def sighting(status, name=None):
    return Sighting(face(vec((0, 1))), status, name, 0.0)


def test_enroll_then_recall(phrases):
    s, frames = service(phrases, [[face(vec((0, 1)))]] * 3, [[face(vec((0, 1)))]])
    assert s.enroll_face("张三") == "好的，张三，我记住你了。"
    assert s.who_is_here() == "我看到了张三。"
    assert frames.counts == [3, 1]


def test_enroll_cleans_the_name(phrases):
    s, _ = service(phrases, [[face(vec((0, 1)))]])
    assert s.enroll_face(" 张三。") == "好的，张三，我记住你了。"
    assert s.memory.names() == ["张三"]


@pytest.mark.parametrize("name", ["", "  ", None, "。"])
def test_enroll_without_a_name_asks_for_it(phrases, name):
    s, frames = service(phrases)
    assert s.enroll_face(name) == phrases.no_name
    assert frames.counts == []


def test_enroll_without_a_face(phrases):
    s, _ = service(phrases, [[], [], []])
    assert s.enroll_face("张三") == phrases.no_face
    assert s.memory.names() == []


def test_camera_errors_are_raised(phrases):
    class Broken:
        def grab(self, count=1):
            raise RuntimeError("Cannot open camera 0")

    s = FaceService(FaceMemory(FakeEmbedder()), Broken(), phrases)
    with pytest.raises(RuntimeError):
        s.who_is_here()


def test_unsure_face_is_asked_about(phrases):
    s, _ = service(phrases, [[face(vec((0, 1)))]], [[face(at_similarity(0, 1, 0.4))]])
    s.enroll_face("张三")
    assert s.who_is_here() == "你是张三吗？"


@pytest.mark.parametrize("sightings, sentence", [
    ([], "我面前好像没有人。"),
    ([sighting(KNOWN, "张三")], "我看到了张三。"),
    ([sighting(KNOWN, "张三"), sighting(KNOWN, "李四")], "我看到了张三和李四。"),
    ([sighting(KNOWN, "甲"), sighting(KNOWN, "乙"), sighting(KNOWN, "丙")], "我看到了甲、乙和丙。"),
    ([sighting(UNKNOWN)], "我还不认识你，可以告诉我你的名字吗？"),
    ([sighting(UNKNOWN), sighting(UNKNOWN)], "我还不认识你们，可以告诉我你们的名字吗？"),
    ([sighting(KNOWN, "张三"), sighting(UNKNOWN), sighting(UNKNOWN)],
     "我看到了张三。还有两位我还不认识。"),
    ([sighting(UNSURE, "张三")], "你是张三吗？"),
    ([sighting(UNSURE, "张三"), sighting(UNSURE, "李四")], "你是张三吗？"),
    ([sighting(KNOWN, "李四"), sighting(UNSURE, "张三")], "我看到了李四。你是张三吗？"),
    ([sighting(UNSURE, "张三"), sighting(UNKNOWN)], "还有一位我还不认识。你是张三吗？"),
])
def test_describe(phrases, sightings, sentence):
    assert describe(sightings, phrases) == sentence


def test_spoken_count():
    assert [spoken_count(n) for n in (1, 2, 10, 11)] == ["一", "两", "十", "11"]
