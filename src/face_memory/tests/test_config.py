import pytest

from face_memory.config import config_from_dict, load_config

from .conftest import SHIPPED_CONFIG


def test_shipped_config_loads():
    config = load_config(SHIPPED_CONFIG)
    assert config.source == 0
    assert config.unsure_threshold < config.match_threshold
    assert config.phrases.enrolled.format(name="张三") == "好的，张三，我记住你了。"


def test_phrases_can_be_overridden():
    config = config_from_dict({"phrases": {"nobody": "这里没有人。"}})
    assert config.phrases.nobody == "这里没有人。"
    assert config.phrases.unsure == "你是{name}吗？"


@pytest.mark.parametrize("raw", [
    {"match_threshold": 0.3, "unsure_threshold": 0.4},
    {"unsure_threshold": 0},
    {"match_threshold": 1.5},
    {"session_idle_min": 0},
    {"frames_per_enroll": 0},
    {"max_image_side": 100},
    {"source": "ros2:camera/image"},
    {"phrases": {"nobody": " "}},
    {"camera": 0},
    {"phrases": {"hello": "你好"}},
])
def test_invalid_config(raw):
    with pytest.raises(ValueError):
        config_from_dict(raw)


def test_numeric_string_source_becomes_an_index():
    assert config_from_dict({"source": "1"}).source == 1
    assert config_from_dict({"source": "ros2:/camera/image"}).source == "ros2:/camera/image"
