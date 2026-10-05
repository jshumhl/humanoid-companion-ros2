import pytest

from face_memory.memory import KNOWN, UNKNOWN, UNSURE, FaceMemory, NoFace, clean_name

from .conftest import FakeClock, FakeEmbedder, at_similarity, face, vec


def memory(**kwargs):
    kwargs.setdefault("match_threshold", 0.5)
    kwargs.setdefault("unsure_threshold", 0.35)
    return FaceMemory(FakeEmbedder(), **kwargs)


def test_recall_by_similarity_band():
    m = memory()
    m.enroll("张三", [[face(vec((0, 1)))]])

    assert m.recall([face(at_similarity(0, 1, 0.9))])[0].status == KNOWN
    unsure = m.recall([face(at_similarity(0, 1, 0.4))])[0]
    assert (unsure.status, unsure.name) == (UNSURE, "张三")
    unknown = m.recall([face(at_similarity(0, 1, 0.2))])[0]
    assert (unknown.status, unknown.name) == (UNKNOWN, None)
    assert unknown.similarity == pytest.approx(0.2, abs=1e-6)


def test_thresholds_are_inclusive():
    probe = at_similarity(0, 1, 0.4)
    similarity = float(probe @ vec((0, 1)))
    for threshold, status in ((dict(match_threshold=similarity), KNOWN),
                              (dict(unsure_threshold=similarity), UNSURE)):
        m = memory(**threshold)
        m.enroll("张三", [[face(vec((0, 1)))]])
        assert m.recall([face(probe)])[0].status == status


def test_nobody_remembered_means_unknown():
    (s,) = memory().recall([face(vec((0, 1)))])
    assert (s.status, s.name) == (UNKNOWN, None)


def test_enroll_uses_the_largest_face_and_averages_frames():
    m = memory(match_threshold=0.95)
    frames = [
        [face(vec((0, 1), (1, 1)), size=200), face(vec((5, 1)), size=80)],
        [face(vec((0, 1), (1, -1)), size=200)],
    ]
    used = m.enroll("张三", frames)
    assert used.size == 200
    # The average of the two large faces points along axis 0.
    assert m.recall([face(vec((0, 1)))])[0].status == KNOWN
    assert m.recall([face(vec((5, 1)))])[0].status == UNKNOWN


def test_enroll_skips_frames_without_a_face():
    m = memory()
    m.enroll("张三", [[], [face(vec((0, 1)))]])
    assert m.names() == ["张三"]


def test_enroll_without_a_usable_face_raises():
    m = memory(min_face_px=60)
    with pytest.raises(NoFace):
        m.enroll("张三", [[], [face(vec((0, 1)), size=40)]])
    assert m.names() == []


def test_small_faces_are_ignored_by_recall():
    m = memory(min_face_px=60)
    assert m.recall([face(vec((0, 1)), size=59)]) == []


def test_empty_name_is_rejected():
    with pytest.raises(ValueError):
        memory().enroll(" 。", [[face(vec((0, 1)))]])


def test_enrolling_again_adds_a_sample():
    m = memory()
    m.enroll("张三", [[face(vec((0, 1)))]])
    m.enroll("张三", [[face(vec((2, 1)))]])
    assert m.recall([face(vec((0, 1)))])[0].name == "张三"
    assert m.recall([face(vec((2, 1)))])[0].name == "张三"


def test_samples_per_name_are_capped_oldest_first():
    m = memory(max_samples_per_name=2)
    for axis in (0, 1, 2):
        m.enroll("张三", [[face(vec((axis, 1)))]])
    assert m.recall([face(vec((0, 1)))])[0].status == UNKNOWN
    assert m.recall([face(vec((2, 1)))])[0].status == KNOWN


def test_new_name_for_a_known_face_is_a_correction():
    m = memory()
    m.enroll("张三", [[face(vec((0, 1)))]])
    m.enroll("李四", [[face(at_similarity(0, 1, 0.9))]])
    assert m.names() == ["李四"]
    assert m.recall([face(vec((0, 1)))])[0].name == "李四"


def test_correction_keeps_other_samples_of_the_old_name():
    m = memory()
    m.enroll("张三", [[face(vec((0, 1)))]])
    m.enroll("张三", [[face(vec((3, 1)))]])
    m.enroll("李四", [[face(vec((0, 1)))]])
    assert sorted(m.names()) == ["张三", "李四"]
    assert m.recall([face(vec((3, 1)))])[0].name == "张三"


def test_each_name_goes_to_one_face_largest_first():
    m = memory()
    m.enroll("张三", [[face(vec((0, 1)))]])
    m.enroll("李四", [[face(vec((1, 1)))]])
    sightings = m.recall([
        face(at_similarity(0, 2, 0.7), size=80, x=0),
        face(vec((1, 1)), size=150, x=200),
        face(at_similarity(0, 2, 0.9), size=100, x=400),
    ])
    assert [s.face.size for s in sightings] == [150, 100, 80]
    assert [(s.status, s.name) for s in sightings] == [
        (KNOWN, "李四"), (KNOWN, "张三"), (UNKNOWN, None)]


def test_memory_is_forgotten_after_the_session_goes_idle():
    clock = FakeClock()
    m = memory(session_idle_sec=60, clock=clock)
    m.enroll("张三", [[face(vec((0, 1)))]])
    clock.now += 59
    assert m.recall([face(vec((0, 1)))])[0].status == KNOWN
    clock.now += 59      # each recall keeps the session alive
    assert m.names() == ["张三"]
    clock.now += 61
    assert m.recall([face(vec((0, 1)))])[0].status == UNKNOWN
    assert m.names() == []


def test_forget():
    m = memory()
    m.enroll("张三", [[face(vec((0, 1)))]])
    m.forget()
    assert m.names() == []


def test_invalid_thresholds():
    with pytest.raises(ValueError):
        memory(match_threshold=0.3, unsure_threshold=0.4)


@pytest.mark.parametrize("raw, name", [
    ("张三", "张三"), (" 张三。", "张三"), ("“张三”！", "张三"), ("Anna Li.", "Anna Li"), ("。", ""),
])
def test_clean_name(raw, name):
    assert clean_name(raw) == name
