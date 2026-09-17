import pytest

from voice_agent.protocol import ToolCall
from voice_agent.tools import Tool, ToolRegistry, UnknownTool, look_around_tool


def test_look_around_calls_describe_scene_with_narrator_config():
    seen = []

    def describe_scene(narrator_config):
        seen.append(narrator_config)
        return "我看到了两个人。"

    narrator = object()
    registry = ToolRegistry([look_around_tool(narrator, describe_scene=describe_scene)])

    assert registry.dispatch(ToolCall("look_around")) == "我看到了两个人。"
    assert seen == [narrator]


def test_dispatch_ignores_unexpected_args():
    registry = ToolRegistry([look_around_tool(None, describe_scene=lambda _: "我没有看到任何东西。")])
    assert registry.dispatch(ToolCall("look_around", {"direction": "left"})) == "我没有看到任何东西。"


def test_dispatch_unknown_tool():
    registry = ToolRegistry([look_around_tool(None, describe_scene=lambda _: "")])
    with pytest.raises(UnknownTool):
        registry.dispatch(ToolCall("open_door"))


def test_dispatch_propagates_tool_errors():
    def broken(_):
        raise RuntimeError("Cannot open camera 0")

    registry = ToolRegistry([Tool("look_around", "look_around()", "看", broken)])
    with pytest.raises(RuntimeError):
        registry.dispatch(ToolCall("look_around"))


def test_catalog_lists_tools_for_prompt():
    registry = ToolRegistry([look_around_tool(None, describe_scene=lambda _: "")])
    assert registry.names() == ["look_around"]
    assert registry.catalog().startswith("- look_around()：")
