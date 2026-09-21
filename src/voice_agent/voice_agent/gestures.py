"""Gestures the robot can play while it speaks.

The model may attach at most one gesture to a reply, chosen from the catalogue
in gestures.yaml. The gesture is requested when speech playback starts, so the
movement and the voice run together.

The gesture section of the system prompt is generated from that file, examples
included, so gesture behavior is tuned by editing YAML rather than code.

Requests go through a backend. The default one only logs, so the agent runs on
a laptop with no robot and no ROS installed. The ROS 2 backend is imported only
when it is selected.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

log = logging.getLogger(__name__)

# Rules shown above the catalogue in the system prompt.
PROMPT_RULES = (
    "只能从下列动作中选择。\n"
    "如果都不合适，gesture 填 null。\n"
    "大多数回复不需要动作，只在自然的时候使用。"
)


class GestureCatalogueError(ValueError):
    """gestures.yaml is missing, malformed, or inconsistent."""


@dataclass(frozen=True)
class Gesture:
    name: str
    use_when: str      # the situation to use it in, in Chinese
    duration: float    # seconds; requests during it are skipped


@dataclass(frozen=True)
class GestureExample:
    user: str
    say: str
    gesture: Optional[str]   # None means "no gesture", shown as null


class GestureBackendUnavailable(RuntimeError):
    """The selected backend cannot be used, e.g. ROS 2 is not installed."""


class GestureBackend:
    name = "base"

    def trigger(self, gesture):
        raise NotImplementedError

    def close(self):
        pass


class StubBackend(GestureBackend):
    """Default backend: records the request in the log, moves nothing."""

    name = "stub"

    def trigger(self, gesture):
        log.info("would request gesture: %s", gesture.name)
        print(f"   [gesture: {gesture.name}]", flush=True)


class Ros2Backend(GestureBackend):
    """Publishes the gesture name as std_msgs/String, for the robot to act on."""

    name = "ros2"

    def __init__(self, topic):
        try:
            import rclpy
            from std_msgs.msg import String
        except ImportError as e:
            raise GestureBackendUnavailable(
                f"ROS 2 is not available ({e}). Use gestures.backend: stub, or run inside "
                f"a sourced ROS 2 environment."
            ) from e

        self._rclpy = rclpy
        self._string = String
        if not rclpy.ok():
            rclpy.init(args=None)
        self._node = rclpy.create_node("voice_agent_gestures")
        self._publisher = self._node.create_publisher(String, topic, 10)
        self.topic = topic
        log.info("Publishing gestures on %s", topic)

    def trigger(self, gesture):
        self._publisher.publish(self._string(data=gesture.name))

    def close(self):
        self._node.destroy_node()
        if self._rclpy.ok():
            self._rclpy.shutdown()


def make_backend(gestures_config):
    """Build the configured backend, falling back to the stub if it is unavailable."""
    if gestures_config.backend == "ros2":
        try:
            return Ros2Backend(gestures_config.ros2_topic)
        except GestureBackendUnavailable as e:
            log.warning("Gesture backend 'ros2' unavailable, using 'stub': %s", e)
    return StubBackend()


class GestureCatalogue:
    """The gestures the model is allowed to ask for, and how to ask."""

    def __init__(self, gestures, examples=()):
        self._by_name = {g.name: g for g in gestures}
        self.examples = list(examples)

    def __len__(self):
        return len(self._by_name)

    def __contains__(self, name):
        return name in self._by_name

    def get(self, name):
        return self._by_name.get(name)

    def names(self):
        return list(self._by_name)

    def prompt_section(self):
        """The whole gesture section, inserted in place of {gestures}.

        Rules, then the catalogue, then examples, so the model sees what it may
        choose from and what choosing nothing looks like.
        """
        if not self._by_name:
            return "（暂无可用动作，gesture 一律填 null。）"

        lines = [PROMPT_RULES, "", "可用动作："]
        lines += [f"- {g.name}：{g.use_when}" for g in self._by_name.values()]
        if self.examples:
            lines += ["", "例子："]
            for example in self.examples:
                gesture = "null" if example.gesture is None else f'"{example.gesture}"'
                lines.append(f"用户：{example.user}")
                lines.append(f'你：{{"say": "{example.say}", "gesture": {gesture}}}')
        return "\n".join(lines)


def load_catalogue(path):
    """Read and validate gestures.yaml. Raises GestureCatalogueError."""
    path = Path(path).expanduser()
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError as e:
        raise GestureCatalogueError(f"Gesture catalogue not found: {path}") from e
    except yaml.YAMLError as e:
        raise GestureCatalogueError(f"{path} is not valid YAML: {e}") from e
    if not isinstance(raw, dict):
        raise GestureCatalogueError(f"{path}: top level must be a mapping with a 'gestures' list")

    unknown = set(raw) - {"gestures", "examples"}
    if unknown:
        raise GestureCatalogueError(
            f"{path}: unknown key(s) {', '.join(sorted(unknown))}. Allowed: gestures, examples")

    gestures = _load_gestures(raw.get("gestures"), path)
    examples = _load_examples(raw.get("examples") or [], {g.name for g in gestures}, path)
    return GestureCatalogue(gestures, examples)


def _load_gestures(entries, path):
    if not isinstance(entries, list) or not entries:
        raise GestureCatalogueError(f"{path}: 'gestures' must be a non-empty list")

    gestures, names = [], set()
    for i, entry in enumerate(entries):
        where = f"{path}: gestures[{i}]"
        if not isinstance(entry, dict):
            raise GestureCatalogueError(f"{where} must be a mapping")
        unknown = set(entry) - {"name", "use_when", "duration"}
        if unknown:
            raise GestureCatalogueError(
                f"{where}: unknown key(s) {', '.join(sorted(unknown))}. "
                f"Allowed: name, use_when, duration")

        name = entry.get("name")
        use_when = entry.get("use_when")
        duration = entry.get("duration")
        if not isinstance(name, str) or not name.strip():
            raise GestureCatalogueError(f"{where}.name must be a non-empty string")
        if not isinstance(use_when, str) or not use_when.strip():
            raise GestureCatalogueError(
                f"{where}.use_when must be a non-empty string describing when to use {name!r}")
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration <= 0:
            raise GestureCatalogueError(f"{where}.duration must be a number greater than 0")
        if name.strip() in names:
            raise GestureCatalogueError(f"{where}.name {name.strip()!r} is used twice")

        names.add(name.strip())
        gestures.append(Gesture(name.strip(), use_when.strip(), float(duration)))
    return gestures


def _load_examples(entries, known_names, path):
    if not isinstance(entries, list):
        raise GestureCatalogueError(f"{path}: 'examples' must be a list")

    examples = []
    for i, entry in enumerate(entries):
        where = f"{path}: examples[{i}]"
        if not isinstance(entry, dict):
            raise GestureCatalogueError(f"{where} must be a mapping")
        unknown = set(entry) - {"user", "say", "gesture"}
        if unknown:
            raise GestureCatalogueError(
                f"{where}: unknown key(s) {', '.join(sorted(unknown))}. "
                f"Allowed: user, say, gesture")

        user, say, gesture = entry.get("user"), entry.get("say"), entry.get("gesture")
        for field_name, value in (("user", user), ("say", say)):
            if not isinstance(value, str) or not value.strip():
                raise GestureCatalogueError(f"{where}.{field_name} must be a non-empty string")
        if gesture is not None:
            if not isinstance(gesture, str) or not gesture.strip():
                raise GestureCatalogueError(
                    f"{where}.gesture must be a gesture name or null")
            if gesture.strip() not in known_names:
                raise GestureCatalogueError(
                    f"{where}.gesture {gesture.strip()!r} is not in the gestures list")
            gesture = gesture.strip()
        examples.append(GestureExample(user.strip(), say.strip(), gesture))
    return examples


class GestureController:
    """Sends gesture requests, dropping any that arrive while one is playing."""

    def __init__(self, catalogue, backend, enabled=True, clock=time.monotonic):
        self._catalogue = catalogue
        self._backend = backend
        self._enabled = enabled
        self._clock = clock
        self._busy_until = 0.0

    @property
    def catalogue(self):
        return self._catalogue

    def request(self, name):
        """Ask for a gesture. Returns True if it was sent.

        Unknown names and requests during another gesture are dropped: a
        gesture is never queued, because by the time it would run the sentence
        it belonged to is over.
        """
        if not name or not self._enabled:
            return False

        gesture = self._catalogue.get(name)
        if gesture is None:
            log.warning("Ignoring unknown gesture %r; known: %s",
                        name, ", ".join(self._catalogue.names()) or "(none)")
            return False

        now = self._clock()
        if now < self._busy_until:
            log.info("Skipping gesture %s: another gesture is still playing for %.1fs",
                     name, self._busy_until - now)
            return False

        try:
            self._backend.trigger(gesture)
        except Exception as e:
            # A gesture must never interrupt speech.
            log.warning("Gesture %s failed: %s", name, e)
            return False

        self._busy_until = now + gesture.duration
        return True

    def close(self):
        self._backend.close()


def build_controller(gestures_config, catalogue, clock=time.monotonic):
    if not gestures_config.enabled or len(catalogue) == 0:
        return GestureController(catalogue, StubBackend(), enabled=False, clock=clock)
    return GestureController(catalogue, make_backend(gestures_config), clock=clock)
