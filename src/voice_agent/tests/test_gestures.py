"""Gesture catalogue, skipping and backends. No robot, ROS, network or microphone."""

import logging

import pytest

from voice_agent.config import GesturesConfig

from .conftest import PACKAGE_DIR
from voice_agent.gestures import (
    Gesture, GestureBackend, GestureBackendUnavailable, GestureCatalogue, GestureCatalogueError,
    GestureController, GestureExample, Ros2Backend, StubBackend, build_controller, load_catalogue,
    make_backend,
)

CATALOGUE = [
    Gesture("hello", "有人跟你打招呼的时候", 2.0),
    Gesture("point", "指出你看到的东西", 1.5),
]


class RecordingBackend(GestureBackend):
    name = "recording"

    def __init__(self, fail=False):
        self.triggered = []
        self.closed = False
        self.fail = fail

    def trigger(self, gesture):
        if self.fail:
            raise RuntimeError("servo bus offline")
        self.triggered.append(gesture.name)

    def close(self):
        self.closed = True


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def make_controller(**kwargs):
    clock = FakeClock()
    backend = RecordingBackend(**kwargs)
    controller = GestureController(GestureCatalogue(CATALOGUE), backend, clock=clock)
    return controller, backend, clock


def test_catalogued_gesture_is_sent():
    controller, backend, _ = make_controller()
    assert controller.request("hello") is True
    assert backend.triggered == ["hello"]


def test_unknown_gesture_is_rejected_and_logged(caplog):
    controller, backend, _ = make_controller()
    with caplog.at_level(logging.WARNING):
        assert controller.request("backflip") is False
    assert backend.triggered == []
    assert "backflip" in caplog.text


def test_request_during_a_gesture_is_skipped_not_queued():
    controller, backend, clock = make_controller()
    assert controller.request("hello") is True      # 2.0 s long
    clock.advance(0.5)
    assert controller.request("point") is False     # still playing
    clock.advance(0.4)
    assert controller.request("point") is False
    assert backend.triggered == ["hello"]           # nothing queued up


def test_next_gesture_allowed_once_the_duration_has_passed():
    controller, backend, clock = make_controller()
    controller.request("hello")
    clock.advance(2.0)
    assert controller.request("point") is True
    assert backend.triggered == ["hello", "point"]


def test_empty_gesture_name_does_nothing():
    controller, backend, _ = make_controller()
    assert controller.request("") is False
    assert backend.triggered == []


def test_backend_failure_does_not_raise():
    controller, backend, clock = make_controller(fail=True)
    assert controller.request("hello") is False
    clock.advance(0.1)
    assert controller.request("hello") is False  # not marked busy by a failure


def test_disabled_controller_sends_nothing():
    backend = RecordingBackend()
    controller = GestureController(GestureCatalogue(CATALOGUE), backend, enabled=False)
    assert controller.request("hello") is False
    assert backend.triggered == []


def test_prompt_section_lists_every_entry_with_its_use_when():
    text = GestureCatalogue(CATALOGUE).prompt_section()
    assert "- hello：有人跟你打招呼的时候" in text
    assert "- point：指出你看到的东西" in text


def test_prompt_section_states_the_rules():
    text = GestureCatalogue(CATALOGUE).prompt_section()
    assert "只能从下列动作中选择。" in text
    assert "如果都不合适，gesture 填 null。" in text
    assert "大多数回复不需要动作，只在自然的时候使用。" in text


def test_prompt_section_includes_the_examples():
    examples = [
        GestureExample("你好！", "你好，我是巴克机器人。", "hello"),
        GestureExample("现在几点？", "现在是下午三点。", None),
    ]
    text = GestureCatalogue(CATALOGUE, examples).prompt_section()
    assert '用户：你好！' in text
    assert '你：{"say": "你好，我是巴克机器人。", "gesture": "hello"}' in text
    assert '你：{"say": "现在是下午三点。", "gesture": null}' in text  # the "no gesture" case


def test_prompt_section_when_empty():
    assert "null" in GestureCatalogue([]).prompt_section()


def test_stub_backend_logs_the_request(caplog):
    with caplog.at_level(logging.INFO, logger="voice_agent.gestures"):
        StubBackend().trigger(CATALOGUE[0])
    assert "would request gesture: hello" in caplog.text


def test_ros2_backend_is_only_imported_when_selected():
    """voice_agent runs with no ROS installed; the stub never touches rclpy."""
    assert make_backend(GesturesConfig(backend="stub")).name == "stub"


def test_ros2_backend_reports_missing_ros():
    pytest.importorskip  # keep the import check explicit below
    try:
        import rclpy  # noqa: F401
    except ImportError:
        with pytest.raises(GestureBackendUnavailable, match="ROS 2 is not available"):
            Ros2Backend("/gesture/request")
    else:
        pytest.skip("ROS 2 is installed here, so the failure path cannot run")


def test_falls_back_to_stub_when_ros_is_unavailable(caplog):
    try:
        import rclpy  # noqa: F401
    except ImportError:
        with caplog.at_level(logging.WARNING):
            backend = make_backend(GesturesConfig(backend="ros2"))
        assert backend.name == "stub"
        assert "unavailable" in caplog.text
    else:
        pytest.skip("ROS 2 is installed here, so the fallback cannot run")


def test_build_controller_disabled_by_config():
    controller = build_controller(GesturesConfig(enabled=False), GestureCatalogue(CATALOGUE))
    assert controller.request("hello") is False


def test_build_controller_with_empty_catalogue():
    controller = build_controller(GesturesConfig(), GestureCatalogue([]))
    assert controller.request("hello") is False
    assert controller.catalogue.names() == []


def test_close_closes_the_backend():
    controller, backend, _ = make_controller()
    controller.close()
    assert backend.closed is True


# --- gestures.yaml ---

SHIPPED_CATALOGUE = PACKAGE_DIR / "gestures.yaml"

VALID_YAML = """
gestures:
  - name: hello
    use_when: 有人跟你打招呼的时候
    duration: 2.0
  - name: point
    use_when: 指出你看到的东西
    duration: 1.5
examples:
  - user: 你好！
    say: 你好，我是巴克机器人。
    gesture: hello
  - user: 现在几点？
    say: 现在是三点。
    gesture: null
"""


def write_catalogue(tmp_path, text):
    path = tmp_path / "gestures.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_shipped_catalogue_loads():
    catalogue = load_catalogue(SHIPPED_CATALOGUE)
    assert catalogue.names() == ["hello", "goodbye", "point", "nod"]
    assert all(g.use_when and g.duration > 0 for g in
               (catalogue.get(n) for n in catalogue.names()))


def test_shipped_catalogue_has_the_three_examples():
    examples = load_catalogue(SHIPPED_CATALOGUE).examples
    assert len(examples) == 3
    assert [e.gesture for e in examples] == ["hello", None, "point"]


def test_shipped_catalogue_entries_all_reach_the_prompt():
    catalogue = load_catalogue(SHIPPED_CATALOGUE)
    text = catalogue.prompt_section()
    for name in catalogue.names():
        assert f"- {name}：{catalogue.get(name).use_when}" in text


def test_catalogue_loads_from_file(tmp_path):
    catalogue = load_catalogue(write_catalogue(tmp_path, VALID_YAML))
    assert catalogue.names() == ["hello", "point"]
    assert catalogue.get("hello").duration == 2.0
    assert catalogue.examples[1].gesture is None


@pytest.mark.parametrize("yaml_text, message", [
    ("gestures:\n  - name: hello\n    use_when: 打招呼\n    duration: 2.0\n"
     "  - name: hello\n    use_when: 再次打招呼\n    duration: 1.0\n",
     "'hello' is used twice"),
    ("gestures:\n  - name: hello\n    use_when: ' '\n    duration: 2.0\n",
     "use_when must be a non-empty string"),
    ("gestures:\n  - name: hello\n    duration: 2.0\n",
     "use_when must be a non-empty string"),
    ("gestures:\n  - name: hello\n    use_when: 打招呼\n    duration: 0\n",
     "duration must be a number greater than 0"),
    ("gestures:\n  - name: hello\n    use_when: 打招呼\n    duration: -1\n",
     "duration must be a number greater than 0"),
    ("gestures:\n  - name: hello\n    use_when: 打招呼\n    duration: fast\n",
     "duration must be a number greater than 0"),
    ("gestures:\n  - name: ' '\n    use_when: 打招呼\n    duration: 2.0\n",
     "name must be a non-empty string"),
    ("gestures: []\n", "'gestures' must be a non-empty list"),
    ("examples: []\n", "'gestures' must be a non-empty list"),
    ("gestures:\n  - name: hello\n    use_when: 打招呼\n    duration: 2.0\n    speed: fast\n",
     "unknown key(s) speed"),
    ("moves:\n  - name: hello\n", "unknown key(s) moves"),
    ("just a string\n", "top level must be a mapping"),
    ("gestures:\n  - name: hello\n    use_when: 打招呼\n    duration: 2.0\nexamples:\n"
     "  - user: 你好\n    say: 你好\n    gesture: wave\n",
     "'wave' is not in the gestures list"),
    ("gestures:\n  - name: hello\n    use_when: 打招呼\n    duration: 2.0\nexamples:\n"
     "  - user: 你好\n    gesture: hello\n", "say must be a non-empty string"),
])
def test_invalid_catalogue_reports_a_clear_error(tmp_path, yaml_text, message):
    with pytest.raises(GestureCatalogueError) as excinfo:
        load_catalogue(write_catalogue(tmp_path, yaml_text))
    assert message in str(excinfo.value)


def test_missing_catalogue_file(tmp_path):
    with pytest.raises(GestureCatalogueError, match="not found"):
        load_catalogue(tmp_path / "absent.yaml")


def test_malformed_yaml(tmp_path):
    with pytest.raises(GestureCatalogueError, match="not valid YAML"):
        load_catalogue(write_catalogue(tmp_path, "gestures: [unclosed"))
