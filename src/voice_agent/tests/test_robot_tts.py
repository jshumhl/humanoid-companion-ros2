"""The robot's offline TTS engine (speech_output.engine: ros2).

No ROS: the service client is a fake, and ROS modules are faked where the
client itself is under test.
"""

import sys
import threading
import time
import types

import pytest

from voice_agent.agent import Reply
from voice_agent.config import ConfigError, RobotTtsConfig, load_config, validate
from voice_agent.robot_tts import (
    MIN_CHARS_PER_SEC,
    RobotTtsOutput,
    RobotTtsUnavailable,
    Ros2TtsClient,
)

from .test_config import write_config
from .test_delivery import FakeAgent, FakeWatcher, make_delivery


class FakeClient:
    """Records calls; can fail, or fire an interrupt while 'speaking'."""

    def __init__(self, error=None, during=None):
        self.calls = []
        self.error = error
        self.during = during          # called while the robot is "talking"

    def speak(self, text, timeout_sec):
        self.calls.append((text, timeout_sec))
        if self.during is not None:
            self.during()
        if self.error is not None:
            raise self.error


def make_output(client=None, **settings):
    return RobotTtsOutput(RobotTtsConfig(**settings), timeout_sec=20, client=client or FakeClient())


# --- speaking ---

def test_speaks_the_text():
    client = FakeClient()
    result = make_output(client).say(" 你好。 ")

    assert result.spoken and not result.interrupted
    assert client.calls[0][0] == "你好。"


def test_timeout_covers_playback_not_just_synthesis():
    """The call returns when the robot stops talking, so longer text needs longer."""
    client = FakeClient()
    output = make_output(client)
    output.say("短。")
    output.say("这是一段比较长的回复，需要更多的时间来播放完。")

    short, long_ = client.calls[0][1], client.calls[1][1]
    assert short == pytest.approx(20 + 2 / MIN_CHARS_PER_SEC)
    assert long_ > short


def test_empty_text_is_not_sent():
    client = FakeClient()
    assert not make_output(client).say("  ").spoken
    assert client.calls == []


def test_long_text_is_cut_to_max_chars():
    client = FakeClient()
    make_output(client, max_chars=5).say("一二三四五六七八")
    assert client.calls[0][0] == "一二三四五"


def test_failure_never_raises():
    client = FakeClient(error=RuntimeError("/tts/speak is not available"))
    result = make_output(client).say("你好。")
    assert not result.spoken and not result.interrupted


def test_gesture_starts_before_the_robot_speaks():
    order = []
    client = FakeClient(during=lambda: order.append("speak"))
    make_output(client).say("你好。", on_playback_start=lambda: order.append("gesture"))
    assert order == ["gesture", "speak"]


def test_a_failing_gesture_does_not_silence_the_robot():
    def broken_gesture():
        raise RuntimeError("gesture backend down")

    client = FakeClient()
    result = make_output(client).say("你好。", on_playback_start=broken_gesture)
    assert result.spoken and len(client.calls) == 1


def test_nothing_to_cache_or_prefetch():
    client = FakeClient()
    output = make_output(client)
    output.prepare(["我现在连不上网络，请稍后再试。"])
    output.prefetch("下一句。")
    assert client.calls == []


# --- interruption ---

def test_interrupt_before_the_sentence_cancels_it():
    client, interrupt = FakeClient(), threading.Event()
    interrupt.set()
    gestures = []

    result = make_output(client).say("你好。", on_playback_start=lambda: gestures.append(1),
                                     interrupt=interrupt)

    assert result.interrupted and not result.spoken
    assert client.calls == [] and gestures == []


def test_interrupt_while_speaking_finishes_the_sentence_then_reports_it():
    """The service cannot stop playback, so the sentence is heard in full."""
    interrupt = threading.Event()
    client = FakeClient(during=interrupt.set)

    result = make_output(client).say("你好。", interrupt=interrupt)

    assert result.spoken and result.interrupted
    assert result.started_at is not None and result.stopped_at >= result.started_at


def test_delivery_drops_the_rest_after_an_interrupt_during_speech(config):
    """After the unstoppable sentence ends, the turn ends as after any
    interruption: no 还要继续吗, no held-back sentences, and the model is told."""
    config.conversation.max_reply_sentences = 1
    watcher, agent = FakeWatcher(), FakeAgent()
    client = FakeClient(during=lambda: watcher.interrupt(at=time.monotonic()))
    delivery, _ = make_delivery(config, make_output(client), watcher=watcher, agent=agent)

    outcome = delivery.respond_and_speak(lambda cancel: Reply("第一句。第二句。"))

    assert outcome.interrupted and not outcome.paused
    assert not delivery.has_pending
    assert [call[0] for call in client.calls] == ["第一句。"]
    # The whole first sentence was heard, and the model is told exactly that.
    assert agent.interrupted_with == "第一句。"
    assert outcome.stop_latency_ms is not None and outcome.stop_latency_ms >= 0


# --- ROS 2 client ---

SERVICE_TYPE = "fake_tts_interfaces/srv/Speak"


def ros2_settings(**overrides):
    values = {"service": "/tts/speak", "type": SERVICE_TYPE,
              "request": {"voice": 1, "speed": 0.9}}
    values.update(overrides)
    return RobotTtsConfig(**values)


def install_fake_ros(monkeypatch, available=True, done=True, response=None):
    """Fake rclpy and a service type package; returns a dict of what was called."""
    seen = {"requests": [], "cancelled": False, "inited": False}
    if response is None:
        response = types.SimpleNamespace(success=True, message="ok")

    class Future:
        def done(self):
            return done

        def result(self):
            return response

        def cancel(self):
            seen["cancelled"] = True

    class Client:
        def wait_for_service(self, timeout_sec):
            seen["wait_timeout"] = timeout_sec
            return available

        def call_async(self, request):
            seen["requests"].append(request)
            return Future()

    class Node:
        def create_client(self, srv_type, service):
            seen["type"] = srv_type
            seen["service"] = service
            return Client()

    rclpy = types.ModuleType("rclpy")
    rclpy.ok = lambda: seen["inited"]
    rclpy.init = lambda args=None: seen.update(inited=True)
    rclpy.create_node = lambda name: Node()
    rclpy.spin_until_future_complete = lambda node, future, timeout_sec: seen.update(
        spin_timeout=timeout_sec)

    class Speak:
        class Request:
            """Like a rosidl request: fixed fields, and it lists them."""

            __slots__ = ("content", "speed", "text", "voice")

            def __init__(self):
                self.text, self.content, self.voice, self.speed = "", "", 0, 1.0

            @classmethod
            def get_fields_and_field_types(cls):
                return {"text": "string", "content": "string", "voice": "int32",
                        "speed": "float"}

    package = types.ModuleType("fake_tts_interfaces")
    srv = types.ModuleType("fake_tts_interfaces.srv")
    srv.Speak = Speak
    package.srv = srv
    monkeypatch.setitem(sys.modules, "rclpy", rclpy)
    monkeypatch.setitem(sys.modules, "fake_tts_interfaces", package)
    monkeypatch.setitem(sys.modules, "fake_tts_interfaces.srv", srv)
    seen["service_class"] = Speak
    return seen


def test_client_imports_the_configured_type_and_sends_the_request(monkeypatch):
    seen = install_fake_ros(monkeypatch)
    Ros2TtsClient(ros2_settings()).speak("你好。", timeout_sec=25)

    request = seen["requests"][0]
    assert (request.text, request.voice, request.speed) == ("你好。", 1, 0.9)
    assert seen["type"] is seen["service_class"]
    assert seen["service"] == "/tts/speak"
    assert seen["spin_timeout"] == 25


def test_the_text_goes_in_the_configured_field(monkeypatch):
    seen = install_fake_ros(monkeypatch)
    Ros2TtsClient(ros2_settings(text_field="content", request={})).speak("你好。", 25)

    request = seen["requests"][0]
    assert (request.content, request.text) == ("你好。", "")


@pytest.mark.parametrize("overrides", [
    {"text_field": "words"},
    {"request": {"volume": 3}},
])
def test_a_field_the_type_lacks_fails_at_startup(monkeypatch, overrides):
    install_fake_ros(monkeypatch)
    with pytest.raises(ValueError, match="no field"):
        Ros2TtsClient(ros2_settings(**overrides))


def test_client_reports_a_missing_service(monkeypatch):
    install_fake_ros(monkeypatch, available=False)
    with pytest.raises(RuntimeError, match="not available"):
        Ros2TtsClient(ros2_settings()).speak("你好。", timeout_sec=25)


def test_client_times_out_and_abandons_the_call(monkeypatch):
    seen = install_fake_ros(monkeypatch, done=False)
    with pytest.raises(TimeoutError, match="still be speaking"):
        Ros2TtsClient(ros2_settings()).speak("你好。", timeout_sec=25)
    assert seen["cancelled"]


def test_client_reports_a_failed_playback(monkeypatch):
    install_fake_ros(monkeypatch, response=types.SimpleNamespace(success=False, message="播放失败"))
    with pytest.raises(RuntimeError, match="播放失败"):
        Ros2TtsClient(ros2_settings()).speak("你好。", timeout_sec=25)


def test_a_response_without_success_counts_as_spoken(monkeypatch):
    install_fake_ros(monkeypatch, response=types.SimpleNamespace())
    Ros2TtsClient(ros2_settings()).speak("你好。", timeout_sec=25)


def test_client_without_ros_says_how_to_fix_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "rclpy", None)
    with pytest.raises(RobotTtsUnavailable, match="edge-tts"):
        Ros2TtsClient(ros2_settings())


def test_client_without_the_service_type_says_how_to_fix_it(monkeypatch):
    install_fake_ros(monkeypatch)
    with pytest.raises(RobotTtsUnavailable, match="missing_interfaces/srv/Speak"):
        Ros2TtsClient(ros2_settings(type="missing_interfaces/srv/Speak"))


def test_a_type_the_package_lacks_is_unavailable(monkeypatch):
    install_fake_ros(monkeypatch)
    with pytest.raises(RobotTtsUnavailable, match="no service named Say"):
        Ros2TtsClient(ros2_settings(type="fake_tts_interfaces/srv/Say"))


def test_agent_falls_back_to_edge_tts_without_ros(config, monkeypatch):
    from voice_agent.__main__ import make_speech_output
    from voice_agent.speech import SpeechOutput

    monkeypatch.setitem(sys.modules, "rclpy", None)
    config.speech_output.engine = "ros2"
    config.speech_output.ros2 = ros2_settings()
    assert isinstance(make_speech_output(config, provider=None), SpeechOutput)


# --- config ---

def test_shipped_config_names_no_robot_service(config):
    """The service is the deployment's to set; the module ships without one."""
    assert config.speech_output.ros2.service == ""
    assert config.speech_output.ros2.type == ""
    assert config.speech_output.ros2.text_field == "text"


def test_ros2_engine_needs_a_service_and_type(config):
    config.speech_output.engine = "ros2"
    with pytest.raises(ConfigError, match="speech_output.ros2.service"):
        validate(config)
    config.speech_output.ros2 = ros2_settings()
    validate(config)


@pytest.mark.parametrize("field, value, message", [
    ("service", "tts/speak", "service"),
    ("service", "/tts/", "service"),
    ("type", "fake_tts_interfaces/msg/Speak", "type"),
    ("type", "Speak", "type"),
    ("text_field", "", "text_field"),
    ("text_field", "voice", "must not set 'voice'"),
    ("request", {"bad-name": 1}, "request keys"),
    ("max_chars", 0, "max_chars"),
])
def test_bad_robot_tts_settings_are_rejected(config, field, value, message):
    config.speech_output.engine = "ros2"
    config.speech_output.ros2 = ros2_settings(**{field: value})
    with pytest.raises(ConfigError, match=message):
        validate(config)


def test_ros2_settings_load_from_yaml(tmp_path):
    path = write_config(tmp_path, speech_output={
        "engine": "ros2",
        "ros2": {"service": "/tts/speak", "type": SERVICE_TYPE, "text_field": "content",
                 "request": {"voice": 2}},
    })
    settings = load_config(path).speech_output.ros2
    assert (settings.service, settings.type, settings.text_field, settings.request) == (
        "/tts/speak", SERVICE_TYPE, "content", {"voice": 2})
