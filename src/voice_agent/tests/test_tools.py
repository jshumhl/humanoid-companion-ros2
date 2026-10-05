import pytest

from voice_agent.protocol import ToolCall
from voice_agent.tools import (Tool, ToolRegistry, UnknownTool, default_tools, enroll_face_tool,
                               look_around_tool, who_is_here_tool)


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


class FakeFaces:
    def __init__(self):
        self.enrolled = []

    def enroll_face(self, name):
        self.enrolled.append(name)
        return f"好的，{name}，我记住你了。" if name else "可以再告诉我一次你的名字吗？"

    def who_is_here(self):
        return "我看到了张三。"


def test_enroll_face_passes_the_name():
    faces = FakeFaces()
    registry = ToolRegistry([enroll_face_tool(faces)])
    assert registry.dispatch(ToolCall("enroll_face", {"name": "张三"})) == "好的，张三，我记住你了。"
    assert faces.enrolled == ["张三"]


@pytest.mark.parametrize("args", [{}, {"name": None}, {"name": 3}, {"who": "张三"}])
def test_enroll_face_without_a_usable_name(args):
    faces = FakeFaces()
    registry = ToolRegistry([enroll_face_tool(faces)])
    assert registry.dispatch(ToolCall("enroll_face", args)) == "可以再告诉我一次你的名字吗？"
    assert faces.enrolled == [""]


def test_who_is_here():
    registry = ToolRegistry([who_is_here_tool(FakeFaces())])
    assert registry.dispatch(ToolCall("who_is_here")) == "我看到了张三。"


def test_face_tools_are_registered_only_with_a_face_service():
    assert default_tools(None).names() == ["look_around"]
    registry = default_tools(None, FakeFaces())
    assert registry.names() == ["look_around", "enroll_face", "who_is_here"]
    catalog = registry.catalog()
    assert "- enroll_face(name)：" in catalog
    assert '{"tool": "enroll_face", "args": {"name": "张三"}}' in catalog
