"""Tools the model can call, and their dispatch."""

import logging
from dataclasses import dataclass
from typing import Callable

log = logging.getLogger(__name__)


class UnknownTool(LookupError):
    pass


@dataclass(frozen=True)
class Tool:
    name: str
    signature: str       # shown to the model, e.g. "look_around()"
    description: str     # Chinese, shown to the model
    run: Callable[[dict], str]


class ToolRegistry:
    def __init__(self, tools):
        self._tools = {tool.name: tool for tool in tools}

    def names(self):
        return list(self._tools)

    def catalog(self):
        """Tool list inserted into the system prompt in place of {tools}."""
        return "\n".join(f"- {t.signature}：{t.description}" for t in self._tools.values())

    def dispatch(self, call):
        """Run a ToolCall and return its sentence. Raises UnknownTool, or whatever the tool raises."""
        tool = self._tools.get(call.name)
        if tool is None:
            raise UnknownTool(call.name)
        log.debug("Calling %s with %s", call.name, call.args)
        return tool.run(call.args)


def look_around_tool(narrator_config, describe_scene=None):
    """look_around(): describe what the camera currently sees."""

    def run(args):
        nonlocal describe_scene
        if describe_scene is None:
            # Imported on first use: it pulls in OpenCV and the detector.
            from object_narrator import describe_scene
        return describe_scene(narrator_config)

    return Tool(
        name="look_around",
        signature="look_around()",
        description="用摄像头看一下周围，返回一句描述看到了什么的中文句子。",
        run=run,
    )


def enroll_face_tool(faces):
    """enroll_face(name): remember the face in front of the camera as `name`.

    `faces` is a face_memory FaceService. A missing name is answered by asking
    for it, not treated as a failure.
    """

    def run(args):
        name = args.get("name")
        return faces.enroll_face(name if isinstance(name, str) else "")

    return Tool(
        name="enroll_face",
        signature="enroll_face(name)",
        description=(
            '有人告诉你他自己或面前的人叫什么时调用（比如"我叫张三""这是张三"），'
            '记住镜头前这个人的脸。你问过"你是某某吗？"而对方说是的时候，也用这个名字调用；'
            '对方说不是并说了自己的名字时，用新名字调用。'
            '调用格式：{"tool": "enroll_face", "args": {"name": "张三"}}。返回的句子会原样朗读。'),
        run=run,
    )


def who_is_here_tool(faces):
    """who_is_here(): say who in front of the camera has been enrolled."""
    return Tool(
        name="who_is_here",
        signature="who_is_here()",
        description=("有人问你认不认识他、记不记得他、面前是谁时调用，"
                     "返回一句说明认出了谁的中文句子，会原样朗读。不要猜别人的名字。"),
        run=lambda args: faces.who_is_here(),
    )


def default_tools(narrator_config, faces=None):
    """look_around, plus enroll_face and who_is_here when `faces` is given."""
    tools = [look_around_tool(narrator_config)]
    if faces is not None:
        tools += [enroll_face_tool(faces), who_is_here_tool(faces)]
    return ToolRegistry(tools)
