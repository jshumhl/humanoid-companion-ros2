import json
import logging
import threading

from voice_agent.agent import Agent
from voice_agent.providers import ProviderError, ProviderUnavailable
from voice_agent.tools import Tool, ToolRegistry, look_around_tool

SCENE = "我看到了两个人和一把椅子。"


def make_agent(config, provider, describe_scene=lambda _: SCENE):
    tools = ToolRegistry([look_around_tool(config.narrator, describe_scene=describe_scene)])
    return Agent(provider, tools, config, config.gesture_catalogue)


def test_who_are_you(config, fake_provider):
    provider = fake_provider(['{"say": "我是巴克机器人。很高兴认识你！"}'])
    reply = make_agent(config, provider).respond_to_text("你是谁？")

    assert reply.text == "我是巴克机器人。很高兴认识你！"
    assert not reply.tool and not reply.fallback


def test_what_do_you_see_speaks_tool_result_as_is(config, fake_provider):
    provider = fake_provider(['{"tool": "look_around", "args": {}}'])
    reply = make_agent(config, provider).respond_to_text("你看见什么？")

    assert reply.text == SCENE
    assert reply.tool == "look_around"
    assert len(provider.chat_calls) == 1  # no second model call to rephrase


def test_system_prompt_has_persona_and_tool_catalog(config, fake_provider):
    provider = fake_provider(['{"say": "你好。"}'])
    make_agent(config, provider).respond_to_text("你好")

    system = provider.chat_calls[0][0]
    assert system["role"] == "system"
    assert "巴克机器人" in system["content"]
    assert "- look_around()：" in system["content"]
    assert "{tools}" not in system["content"]


def test_catalogued_gesture_rides_along_with_the_reply(config, fake_provider):
    provider = fake_provider(['{"say": "你好，我是巴克机器人。", "gesture": "hello"}'])
    reply = make_agent(config, provider).respond_to_text("你好")

    assert reply.text == "你好，我是巴克机器人。"
    assert reply.gesture == "hello"


def test_unknown_gesture_is_dropped_but_the_reply_is_still_spoken(config, fake_provider, caplog):
    provider = fake_provider(['{"say": "你好。", "gesture": "backflip"}'])
    with caplog.at_level(logging.WARNING):
        reply = make_agent(config, provider).respond_to_text("你好")

    assert reply.text == "你好。"   # the reply survives
    assert reply.gesture == ""     # the made-up gesture does not
    assert not reply.fallback
    assert "backflip" in caplog.text


def test_reply_without_a_gesture_is_normal(config, fake_provider):
    provider = fake_provider(['{"say": "今天天气不错。"}'])
    reply = make_agent(config, provider).respond_to_text("今天天气怎么样？")

    assert reply.text == "今天天气不错。"
    assert reply.gesture == ""


def test_tool_reply_can_carry_a_gesture(config, fake_provider):
    provider = fake_provider(['{"tool": "look_around", "gesture": "point"}'])
    reply = make_agent(config, provider).respond_to_text("你看见什么？")

    assert reply.text == SCENE
    assert (reply.tool, reply.gesture) == ("look_around", "point")


def test_gesture_catalogue_is_in_the_system_prompt(config, fake_provider):
    provider = fake_provider(['{"say": "你好。"}'])
    make_agent(config, provider).respond_to_text("你好")

    system = provider.chat_calls[0][0]["content"]
    assert "- hello：第一次见到人，或者有人跟你打招呼的时候" in system
    assert "如果都不合适，gesture 填 null。" in system
    assert '"gesture": null' in system      # the example that chooses no gesture
    assert "{gestures}" not in system


def test_gesture_is_dropped_when_the_agent_has_no_catalogue(config, fake_provider):
    tools = ToolRegistry([look_around_tool(config.narrator, describe_scene=lambda _: SCENE)])
    provider = fake_provider(['{"say": "你好。", "gesture": "hello"}'])
    reply = Agent(provider, tools, config).respond_to_text("你好")

    assert reply.text == "你好。"
    assert reply.gesture == ""


class RecordingGestureLog:
    def __init__(self):
        self.entries = []

    def record(self, user_text, gesture, reply="", tool="", fallback=""):
        self.entries.append((user_text, gesture, fallback))


def test_every_turn_is_logged_with_its_gesture_or_none(config, fake_provider):
    gesture_log = RecordingGestureLog()
    provider = fake_provider(['{"say": "你好。", "gesture": "hello"}',
                              '{"say": "现在是下午三点。"}',
                              ProviderUnavailable("DNS failure")])
    tools = ToolRegistry([look_around_tool(config.narrator, describe_scene=lambda _: SCENE)])
    agent = Agent(provider, tools, config, config.gesture_catalogue, gesture_log)

    agent.respond_to_text("你好！")
    agent.respond_to_text("现在几点？")
    agent.respond_to_text("你是谁？")

    assert gesture_log.entries == [
        ("你好！", "hello", ""),
        ("现在几点？", "", ""),          # no gesture is recorded too
        ("你是谁？", "", "offline"),     # so are failed turns
    ]


def test_history_kept_across_turns(config, fake_provider):
    provider = fake_provider(['{"say": "我是巴克机器人。"}', '{"tool": "look_around"}', '{"say": "好的。"}'])
    agent = make_agent(config, provider)
    agent.respond_to_text("你是谁？")
    agent.respond_to_text("你看见什么？")
    agent.respond_to_text("谢谢")

    third_call = provider.chat_calls[2]
    assert [m["role"] for m in third_call] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert json.loads(third_call[4]["content"]) == {"say": SCENE}


def test_history_is_trimmed(config, fake_provider):
    config.conversation.max_history_turns = 2
    provider = fake_provider(['{"say": "嗯。"}'] * 5)
    agent = make_agent(config, provider)
    for i in range(5):
        agent.respond_to_text(f"第{i}句")
    assert len(agent.history) == 4
    assert agent.history[0]["content"] == "第3句"


def test_long_reply_cut_to_max_sentences(config, fake_provider):
    provider = fake_provider(['{"say": "一。二。三。四。五。"}'])
    assert make_agent(config, provider).respond_to_text("讲个故事").text == "一。二。三。"


def test_offline_uses_fallback_and_keeps_history_clean(config, fake_provider):
    provider = fake_provider([ProviderUnavailable("DashScope unreachable: [Errno 8] nodename")])
    agent = make_agent(config, provider)
    reply = agent.respond_to_text("你是谁？")

    assert reply.text == config.fallback_phrases.offline
    assert reply.fallback == "offline"
    assert agent.history == []


def test_hanging_provider_times_out_to_offline(config, fake_provider):
    config.timeouts.llm_sec = 0.2
    release = threading.Event()
    provider = fake_provider([lambda: release.wait(5) or '{"say": "太晚了。"}'])
    try:
        reply = make_agent(config, provider).respond_to_text("你是谁？")
    finally:
        release.set()
    assert reply.fallback == "offline"


def test_provider_error_never_reaches_speech(config, fake_provider):
    provider = fake_provider([ProviderError("DashScope 403: AccessDenied.Unpurchased")])
    reply = make_agent(config, provider).respond_to_text("你是谁？")
    assert reply.text == config.fallback_phrases.error
    assert "403" not in reply.text


def test_malformed_json_uses_fallback(config, fake_provider):
    provider = fake_provider(['{"say": "我是'])
    assert make_agent(config, provider).respond_to_text("你是谁？").fallback == "error"


def test_unknown_tool_uses_fallback(config, fake_provider):
    provider = fake_provider(['{"tool": "fly"}'])
    assert make_agent(config, provider).respond_to_text("飞起来").fallback == "error"


def test_camera_failure_uses_tool_failed(config, fake_provider):
    def broken(_):
        raise RuntimeError("Cannot open camera 0")

    provider = fake_provider(['{"tool": "look_around"}'])
    reply = make_agent(config, provider, describe_scene=broken).respond_to_text("你看见什么？")
    assert reply.text == config.fallback_phrases.tool_failed


def test_audio_turn_includes_transcript(config, fake_provider):
    provider = fake_provider(['{"tool": "look_around"}'], transcript="你看见什么？")
    reply = make_agent(config, provider).respond_to_audio(b"RIFF", 16000)
    assert reply.heard == "你看见什么？"
    assert reply.text == SCENE


def test_empty_transcript_is_not_heard(config, fake_provider):
    provider = fake_provider([], transcript="  ")
    assert make_agent(config, provider).respond_to_audio(b"RIFF", 16000).fallback == "not_heard"


def test_asr_offline(config, fake_provider):
    provider = fake_provider([], transcribe_error=ProviderUnavailable("timeout"))
    assert make_agent(config, provider).respond_to_audio(b"RIFF", 16000).fallback == "offline"
