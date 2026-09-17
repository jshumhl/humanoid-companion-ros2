import pytest

from object_narrator.__main__ import SceneTracker
from object_narrator.config import Config, validate
from object_narrator.sentence import build_sentence, chinese_number, count_classes
from object_narrator.vocab import VOCAB


def test_count_classes_dedupes_and_orders_by_count():
    names = ["chair", "person", "dog", "person", "person", "dog"]
    assert count_classes(names) == [("person", 3), ("dog", 2), ("chair", 1)]


@pytest.mark.parametrize("n, expected", [
    (1, "一"), (2, "两"), (9, "九"), (10, "十"), (12, "十二"),
    (20, "二十"), (35, "三十五"), (100, "100"),
])
def test_chinese_number(n, expected):
    assert chinese_number(n, "两") == expected


def test_sentence_zh_cn():
    counts = [("person", 2), ("dog", 1), ("chair", 1)]
    assert build_sentence(counts, "zh-CN") == "我看到了两个人、一只狗和一把椅子。"


def test_sentence_zh_tw():
    assert build_sentence([("car", 2)], "zh-TW") == "我看到了兩輛汽車。"


def test_sentence_zh_hk():
    counts = [("person", 1), ("cup", 3)]
    assert build_sentence(counts, "zh-HK") == "我見到一個人同三隻杯。"


def test_sentence_empty():
    assert build_sentence([], "zh-CN") == "我没有看到任何东西。"


def test_unknown_class_keeps_english_name():
    assert build_sentence([("forklift", 1)], "zh-CN") == "我看到了一个forklift。"


def test_vocab_covers_all_coco_classes():
    assert len(VOCAB) == 80


def test_tracker_waits_for_stable_scene_and_skips_repeats():
    tracker = SceneTracker(stable_frames=2, announce_empty=False)
    person = [("person", 1)]
    assert tracker.update(person) is False   # first sighting
    assert tracker.update(person) is True    # stable
    assert tracker.update(person) is False   # already spoken
    assert tracker.update([]) is False       # flicker
    assert tracker.update(person) is False   # back to spoken scene
    assert tracker.update([]) is False
    assert tracker.update([]) is False       # stable but empty is silent
    assert tracker.update(person) is False
    assert tracker.update(person) is True    # reappeared after empty scene


def test_validate_rejects_bad_values():
    with pytest.raises(ValueError):
        validate(Config(language="ja-JP"))
    with pytest.raises(ValueError):
        validate(Config(confidence=1.5))


def test_validate_converts_digit_source():
    config = Config(source="1")
    validate(config)
    assert config.source == 1
