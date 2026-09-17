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
        if call.args:
            log.info("Ignoring arguments for %s: %s", call.name, call.args)
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


def default_tools(narrator_config):
    return ToolRegistry([look_around_tool(narrator_config)])
